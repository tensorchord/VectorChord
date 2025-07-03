// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

pub mod am_build;

use crate::index::fetcher::*;
use crate::index::gucs;
use crate::index::storage::PostgresRelation;
use crate::index::vchordg::opclass::opfamily;
use crate::index::vchordg::scanners::*;
use pgrx::datum::Internal;
use pgrx::pg_sys::Datum;
use std::cell::LazyCell;
use std::ffi::CStr;
use std::num::NonZero;
use std::ops::DerefMut;
use std::ptr::NonNull;
use std::sync::OnceLock;

#[repr(C)]
struct Reloption {
    vl_len_: i32,
    pub options: i32,
}

impl Reloption {
    unsafe fn options<'a>(this: *const Self) -> &'a CStr {
        unsafe {
            let ptr = this
                .cast::<u8>()
                .add((&raw const (*this).options).read() as _);
            CStr::from_ptr(ptr.cast())
        }
    }
}

const TABLE: &[pgrx::pg_sys::relopt_parse_elt] = &[pgrx::pg_sys::relopt_parse_elt {
    optname: c"options".as_ptr(),
    opttype: pgrx::pg_sys::relopt_type::RELOPT_TYPE_STRING,
    offset: std::mem::offset_of!(Reloption, options) as i32,
    #[cfg(feature = "pg18")]
    isset_offset: 0,
}];

static RELOPT_KIND: OnceLock<pgrx::pg_sys::relopt_kind::Type> = OnceLock::new();

pub fn init() {
    RELOPT_KIND.get_or_init(|| {
        let kind;
        unsafe {
            kind = pgrx::pg_sys::add_reloption_kind();
            pgrx::pg_sys::add_string_reloption(
                kind as _,
                c"options".as_ptr(),
                c"Vector index options, represented as a TOML string.".as_ptr(),
                c"".as_ptr(),
                None,
                pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
            );
        }
        kind
    });
}

#[pgrx::pg_extern(sql = "")]
fn _vchordg_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
    type T = pgrx::pg_sys::IndexAmRoutine;
    unsafe {
        let index_am_routine = pgrx::pg_sys::palloc0(size_of::<T>()) as *mut T;
        index_am_routine.write(AM_HANDLER);
        Internal::from(Some(Datum::from(index_am_routine)))
    }
}

const AM_HANDLER: pgrx::pg_sys::IndexAmRoutine = const {
    let mut am_routine = unsafe { std::mem::zeroed::<pgrx::pg_sys::IndexAmRoutine>() };

    am_routine.type_ = pgrx::pg_sys::NodeTag::T_IndexAmRoutine;

    am_routine.amsupport = 1;
    am_routine.amcanorderbyop = true;

    #[cfg(any(feature = "pg17", feature = "pg18"))]
    {
        am_routine.amcanbuildparallel = true;
    }

    // Index access methods that set `amoptionalkey` to `false`
    // must index all tuples, even if the first column is `NULL`.
    // However, PostgreSQL does not generate a path if there is no
    // index clauses, even if there is a `ORDER BY` clause.
    // So we have to set it to `true` and set costs of every path
    // for vector index scans without `ORDER BY` clauses a large number
    // and throw errors if someone really wants such a path.
    am_routine.amoptionalkey = true;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuildphasename = Some(am_build::ambuildphasename);
    am_routine.ambuild = Some(am_build::ambuild);
    am_routine.ambuildempty = Some(am_build::ambuildempty);
    am_routine.aminsert = Some(aminsert);
    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.ambeginscan = Some(ambeginscan);
    am_routine.amrescan = Some(amrescan);
    am_routine.amgettuple = Some(amgettuple);
    am_routine.amendscan = Some(amendscan);

    am_routine
};

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvalidate(_opclass_oid: pgrx::pg_sys::Oid) -> bool {
    true
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amoptions(
    reloptions: Datum,
    validate: bool,
) -> *mut pgrx::pg_sys::bytea {
    let relopt_kind = RELOPT_KIND.get().copied().expect("init is not called");
    let rdopts = unsafe {
        pgrx::pg_sys::build_reloptions(
            reloptions,
            validate,
            relopt_kind,
            size_of::<Reloption>(),
            TABLE.as_ptr(),
            TABLE.len() as _,
        )
    };
    rdopts as *mut pgrx::pg_sys::bytea
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amcostestimate(
    _root: *mut pgrx::pg_sys::PlannerInfo,
    _path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        use pgrx::pg_sys::disable_cost;
        if !gucs::vchordg_enable_scan() {
            *index_startup_cost = disable_cost;
            *index_total_cost = disable_cost;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 1.0;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = 1.0;
        *index_correlation = 0.0;
        *index_pages = 1.0;
    }
}

#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, heap_relation, values, is_null, heap_tid) }
}

#[cfg(any(
    feature = "pg14",
    feature = "pg15",
    feature = "pg16",
    feature = "pg17",
    feature = "pg18"
))]
#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, heap_relation, values, is_null, heap_tid) }
}

unsafe fn aminsertinner(
    index_relation: pgrx::pg_sys::Relation,
    _heap_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    ctid: pgrx::pg_sys::ItemPointer,
) -> bool {
    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let datum = unsafe { (!is_null.add(0).read()).then_some(values.add(0).read()) };
    let ctid = unsafe { ctid.read() };
    if let Some(store) = unsafe { datum.and_then(|x| opfamily.store(x)) } {
        for (vector, extra) in store {
            let key = ctid_to_key(ctid);
            let payload = kv_to_pointer((key, extra));
            crate::index::vchordg::algo::insert(opfamily, &index, payload, vector);
        }
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
    callback: pgrx::pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        #[cfg(any(
            feature = "pg13",
            feature = "pg14",
            feature = "pg15",
            feature = "pg16",
            feature = "pg17"
        ))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pgrx::pg_sys::vacuum_delay_point(false);
    };
    let callback = callback.expect("null function pointer");
    let callback = |pointer: NonZero<u64>| {
        let (key, _) = pointer_to_kv(pointer);
        let mut ctid = key_to_ctid(key);
        unsafe { callback(&mut ctid, callback_state) }
    };
    crate::index::vchordg::algo::bulkdelete(opfamily, &index, check, callback);
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        #[cfg(any(
            feature = "pg13",
            feature = "pg14",
            feature = "pg15",
            feature = "pg16",
            feature = "pg17"
        ))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pgrx::pg_sys::vacuum_delay_point(false);
    };
    crate::index::vchordg::algo::maintain(opfamily, &index, check);
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index_relation, n_keys, n_orderbys) };
    let scanner: Scanner = Scanner {
        hack: None,
        scanning: LazyCell::new(Box::new(|| Box::new(std::iter::empty()))),
        bump: Box::new(bumpalo::Bump::new()),
    };
    unsafe {
        (*scan).opaque = CurrentMemoryContext.leak_and_drop_on_delete(scanner).cast();
    }
    scan
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pgrx::pg_sys::IndexScanDesc,
    keys: pgrx::pg_sys::ScanKey,
    _n_keys: std::os::raw::c_int,
    orderbys: pgrx::pg_sys::ScanKey,
    _n_orderbys: std::os::raw::c_int,
) {
    unsafe {
        use crate::index::vchordg::opclass::Opfamily;
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as _);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(orderbys, (*scan).orderByData, (*scan).numberOfOrderBys as _);
        }
        if (*scan).numberOfOrderBys == 0 && (*scan).numberOfKeys == 0 {
            pgrx::error!(
                "vector search with no WHERE clause and no ORDER BY clause is not supported"
            );
        }
        let scanner = &mut *(*scan).opaque.cast::<Scanner>();
        scanner.scanning = LazyCell::new(Box::new(|| Box::new(std::iter::empty())));
        scanner.bump.reset();
        let opfamily = opfamily((*scan).indexRelation);
        let index = PostgresRelation::new((*scan).indexRelation);
        let options = SearchOptions {
            ef_search: gucs::vchordg_ef_search(),
            beam_search: gucs::vchordg_beam_search(),
            max_scan_tuples: gucs::vchordg_max_scan_tuples(),
            io_search: gucs::vchordg_io_search(),
            io_rerank: gucs::vchordg_io_rerank(),
        };
        let fetcher = {
            let hack = scanner.hack;
            LazyCell::new(move || {
                HeapFetcher::new(
                    (*scan).indexRelation,
                    (*scan).heapRelation,
                    (*scan).xs_snapshot,
                    if let Some(hack) = hack {
                        hack.as_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                )
            })
        };
        // PAY ATTENTATION: `scanning` references `bump`, so `scanning` must be dropped before `bump`.
        let bump = scanner.bump.as_ref();
        scanner.scanning = match opfamily {
            Opfamily::VectorL2
            | Opfamily::VectorCosine
            | Opfamily::HalfvecL2
            | Opfamily::HalfvecCosine
            | Opfamily::VectorIp
            | Opfamily::HalfvecIp => {
                let mut builder = DefaultBuilder::new(opfamily);
                for i in 0..(*scan).numberOfOrderBys {
                    let data = (*scan).orderByData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                for i in 0..(*scan).numberOfKeys {
                    let data = (*scan).keyData.add(i as usize);
                    let value = (*data).sk_argument;
                    let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                    builder.add((*data).sk_strategy, (!is_null).then_some(value));
                }
                LazyCell::new(Box::new(move || {
                    // only do this since `PostgresRelation` has no destructor
                    let index = bump.alloc(index.clone());
                    builder.build(index, options, fetcher, bump)
                }))
            }
        };
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pgrx::pg_sys::IndexScanDesc,
    direction: pgrx::pg_sys::ScanDirection::Type,
) -> bool {
    if direction != pgrx::pg_sys::ScanDirection::ForwardScanDirection {
        pgrx::error!("vector search without a forward scan direction is not supported");
    }
    // https://www.postgresql.org/docs/current/index-locking.html
    // If heap entries referenced physical pointers are deleted before
    // they are consumed by PostgreSQL, PostgreSQL will received wrong
    // physical pointers: no rows or irreverent rows are referenced.
    if unsafe { (*(*scan).xs_snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
    {
        pgrx::error!("scanning with a non-MVCC-compliant snapshot is not supported");
    }
    let scanner = unsafe { (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked() };
    if let Some((_, key, recheck)) = scanner.scanning.deref_mut().next() {
        unsafe {
            (*scan).xs_heaptid = key_to_ctid(key);
            (*scan).xs_recheck = recheck;
            (*scan).xs_recheckorderby = false;
        }
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pgrx::pg_sys::IndexScanDesc) {
    let scanner = unsafe { &mut *(*scan).opaque.cast::<Scanner>() };
    scanner.scanning = LazyCell::new(Box::new(|| Box::new(std::iter::empty())));
    scanner.bump.reset();
}

type Iter = Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;

pub struct Scanner {
    pub hack: Option<NonNull<pgrx::pg_sys::IndexScanState>>,
    scanning: LazyCell<Iter, Box<dyn FnOnce() -> Iter>>,
    bump: Box<bumpalo::Bump>,
}
