pub mod am_build;

use super::algorithm::BumpAlloc;
use super::gucs::enable_prefilter;
use crate::index::gucs;
use crate::index::lazy_cell::LazyCell;
use crate::index::opclass::{Opfamily, opfamily};
use crate::index::scanners::*;
use crate::index::storage::PostgresRelation;
use algorithm::Bump;
use pgrx::datum::Internal;
use pgrx::pg_sys::{BlockIdData, Datum, ItemPointerData};
use std::ffi::CStr;
use std::num::NonZero;
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
}];

static RELOPT_KIND: OnceLock<pgrx::pg_sys::relopt_kind::Type> = OnceLock::new();

pub fn init() {
    RELOPT_KIND.get_or_init(|| {
        let kind;
        unsafe {
            kind = pgrx::pg_sys::add_reloption_kind();
            pgrx::pg_sys::add_string_reloption(
                kind,
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
fn _vchordrq_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
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

    #[cfg(feature = "pg17")]
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

#[allow(clippy::too_many_arguments)]
#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amcostestimate(
    root: *mut pgrx::pg_sys::PlannerInfo,
    path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        use pgrx::pg_sys::disable_cost;
        let index_opt_info = (*path).indexinfo;
        // do not use index, if there are no orderbys or clauses
        if (*path).indexorderbys.is_null() && (*path).indexclauses.is_null() {
            *index_startup_cost = disable_cost;
            *index_total_cost = disable_cost;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 1.0;
            return;
        }
        let selectivity = {
            use pgrx::pg_sys::{
                JoinType, add_predicate_to_index_quals, clauselist_selectivity,
                get_quals_from_indexclauses,
            };
            let index_quals = get_quals_from_indexclauses((*path).indexclauses);
            let selectivity_quals = add_predicate_to_index_quals(index_opt_info, index_quals);
            clauselist_selectivity(
                root,
                selectivity_quals,
                (*(*index_opt_info).rel).relid as _,
                JoinType::JOIN_INNER,
                std::ptr::null_mut(),
            )
        };
        // index exists
        if !(*index_opt_info).hypothetical {
            let relation = Index::open((*index_opt_info).indexoid, pgrx::pg_sys::NoLock as _);
            let opfamily = opfamily(relation.raw());
            if !matches!(
                opfamily,
                Opfamily::HalfvecCosine
                    | Opfamily::HalfvecIp
                    | Opfamily::HalfvecL2
                    | Opfamily::VectorCosine
                    | Opfamily::VectorIp
                    | Opfamily::VectorL2
            ) {
                *index_startup_cost = 0.0;
                *index_total_cost = 0.0;
                *index_selectivity = 1.0;
                *index_correlation = 0.0;
                *index_pages = 1.0;
                return;
            }
            let index = PostgresRelation::new(relation.raw());
            let probes = gucs::probes();
            let cost = algorithm::cost(index);
            if cost.cells.len() != 1 + probes.len() {
                panic!(
                    "need {} probes, but {} probes provided",
                    cost.cells.len() - 1,
                    probes.len()
                );
            }
            let node_count = {
                let tuples = (*index_opt_info).tuples as u32;
                let mut count = 0.0;
                let r = cost.cells.iter().copied().rev();
                let numerator = std::iter::once(1).chain(probes.clone());
                let denumerator = r.clone();
                let scale = r.skip(1).chain(std::iter::once(tuples));
                for (scale, (numerator, denumerator)) in scale.zip(numerator.zip(denumerator)) {
                    count += (scale as f64) * ((numerator as f64) / (denumerator as f64));
                }
                count
            };
            let page_count = {
                let mut pages = 0_f64;
                pages += 1.0;
                pages += node_count * cost.dims as f64 / 60000.0;
                pages += probes.iter().sum::<u32>() as f64 * {
                    let x = opfamily.vector_kind().element_size() * cost.dims;
                    x.div_ceil(3840 * x.div_ceil(5120).min(2)) as f64
                };
                pages += cost.cells[0] as f64;
                pages
            };
            let next_count =
                f64::max(1.0, (*root).limit_tuples) * f64::min(1000.0, 1.0 / selectivity);
            *index_startup_cost = 0.001 * node_count;
            *index_total_cost = 0.001 * node_count + next_count;
            *index_selectivity = selectivity;
            *index_correlation = 0.0;
            *index_pages = page_count;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = selectivity;
        *index_correlation = 0.0;
        *index_pages = 1.0;
    }
}

#[cfg(feature = "pg13")]
#[allow(clippy::too_many_arguments)]
#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, values, is_null, heap_tid) }
}

#[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
#[allow(clippy::too_many_arguments)]
#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap_relation: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    unsafe { aminsertinner(index_relation, values, is_null, heap_tid) }
}

unsafe fn aminsertinner(
    index_relation: pgrx::pg_sys::Relation,
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
            crate::index::algorithm::insert(opfamily, index.clone(), payload, vector);
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
        pgrx::pg_sys::vacuum_delay_point();
    };
    let callback = callback.expect("null function pointer");
    let callback = |pointer: NonZero<u64>| {
        let (key, _) = pointer_to_kv(pointer);
        let mut ctid = key_to_ctid(key);
        unsafe { callback(&mut ctid, callback_state) }
    };
    crate::index::algorithm::bulkdelete(opfamily, index, check, callback);
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
        pgrx::pg_sys::vacuum_delay_point();
    };
    crate::index::algorithm::maintain(opfamily, index, check);
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
        bump: Box::new(BumpAlloc::new()),
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
        use crate::index::opclass::Opfamily;
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
            epsilon: gucs::epsilon(),
            probes: gucs::probes(),
            max_scan_tuples: gucs::max_scan_tuples(),
            maxsim_refine: gucs::maxsim_refine(),
            maxsim_threshold: gucs::maxsim_threshold(),
            io_rerank: gucs::io_rerank(),
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
            | Opfamily::VectorIp
            | Opfamily::VectorCosine
            | Opfamily::HalfvecL2
            | Opfamily::HalfvecIp
            | Opfamily::HalfvecCosine => {
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
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim => {
                let mut builder = MaxsimBuilder::new(opfamily);
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
    if let Some((_, key, recheck)) = LazyCell::force_mut(&mut scanner.scanning).next() {
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
    bump: Box<BumpAlloc>,
}

struct HeapFetcher {
    index_info: *mut pgrx::pg_sys::IndexInfo,
    estate: *mut pgrx::pg_sys::EState,
    econtext: *mut pgrx::pg_sys::ExprContext,
    heap_relation: pgrx::pg_sys::Relation,
    snapshot: pgrx::pg_sys::Snapshot,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    values: [Datum; 32],
    is_nulls: [bool; 32],
    hack: *mut pgrx::pg_sys::IndexScanState,
}

impl HeapFetcher {
    unsafe fn new(
        index_relation: pgrx::pg_sys::Relation,
        heap_relation: pgrx::pg_sys::Relation,
        snapshot: pgrx::pg_sys::Snapshot,
        hack: *mut pgrx::pg_sys::IndexScanState,
    ) -> Self {
        unsafe {
            let index_info = pgrx::pg_sys::BuildIndexInfo(index_relation);
            let estate = pgrx::pg_sys::CreateExecutorState();
            let econtext = pgrx::pg_sys::MakePerTupleExprContext(estate);
            Self {
                index_info,
                estate,
                econtext,
                heap_relation,
                snapshot,
                slot: pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut()),
                values: [Datum::null(); 32],
                is_nulls: [true; 32],
                hack,
            }
        }
    }
}

impl Drop for HeapFetcher {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            // free common resources
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            pgrx::pg_sys::FreeExecutorState(self.estate);
        }
    }
}

impl SearchFetcher for HeapFetcher {
    fn filter_fetch(&mut self, key: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])> {
        unsafe {
            let mut ctid = key_to_ctid(key);
            let table_am = (*self.heap_relation).rd_tableam;
            let fetch_row_version = (*table_am)
                .tuple_fetch_row_version
                .expect("unsupported heap access method");
            if !fetch_row_version(self.heap_relation, &mut ctid, self.snapshot, self.slot) {
                return None;
            }
            if !enable_prefilter() || self.hack.is_null() {
                if let Some(qual) = NonNull::new((*self.hack).ss.ps.qual) {
                    use pgrx::datum::FromDatum;
                    use pgrx::memcxt::PgMemoryContexts;
                    assert!(qual.as_ref().flags & pgrx::pg_sys::EEO_FLAG_IS_QUAL as u8 != 0);
                    let evalfunc = qual.as_ref().evalfunc.expect("no evalfunc for qual");
                    if !(*self.hack).ss.ps.ps_ExprContext.is_null() {
                        let econtext = (*self.hack).ss.ps.ps_ExprContext;
                        (*econtext).ecxt_scantuple = self.slot;
                        pgrx::pg_sys::MemoryContextReset((*econtext).ecxt_per_tuple_memory);
                        let result = PgMemoryContexts::For((*econtext).ecxt_per_tuple_memory)
                            .switch_to(|_| {
                                let mut is_null = true;
                                let datum = evalfunc(qual.as_ptr(), econtext, &mut is_null);
                                bool::from_datum(datum, is_null)
                            });
                        if result != Some(true) {
                            return None;
                        }
                    }
                }
            }
            (*self.econtext).ecxt_scantuple = self.slot;
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            pgrx::pg_sys::FormIndexDatum(
                self.index_info,
                self.slot,
                self.estate,
                self.values.as_mut_ptr(),
                self.is_nulls.as_mut_ptr(),
            );
            Some((&self.values, &self.is_nulls))
        }
    }

    fn filter_only(&self, key: [u16; 3]) -> bool {
        if !enable_prefilter() || self.hack.is_null() {
            return true;
        }
        unsafe {
            let mut ctid = key_to_ctid(key);
            let table_am = (*self.heap_relation).rd_tableam;
            let fetch_row_version = (*table_am)
                .tuple_fetch_row_version
                .expect("unsupported heap access method");
            if !fetch_row_version(self.heap_relation, &mut ctid, self.snapshot, self.slot) {
                return false;
            }
            if let Some(qual) = NonNull::new((*self.hack).ss.ps.qual) {
                use pgrx::datum::FromDatum;
                use pgrx::memcxt::PgMemoryContexts;
                assert!(qual.as_ref().flags & pgrx::pg_sys::EEO_FLAG_IS_QUAL as u8 != 0);
                let evalfunc = qual.as_ref().evalfunc.expect("no evalfunc for qual");
                if !(*self.hack).ss.ps.ps_ExprContext.is_null() {
                    let econtext = (*self.hack).ss.ps.ps_ExprContext;
                    (*econtext).ecxt_scantuple = self.slot;
                    pgrx::pg_sys::MemoryContextReset((*econtext).ecxt_per_tuple_memory);
                    let result = PgMemoryContexts::For((*econtext).ecxt_per_tuple_memory)
                        .switch_to(|_| {
                            let mut is_null = true;
                            let datum = evalfunc(qual.as_ptr(), econtext, &mut is_null);
                            bool::from_datum(datum, is_null)
                        });
                    if result != Some(true) {
                        return false;
                    }
                }
            }
            true
        }
    }
}

struct Index {
    raw: *mut pgrx::pg_sys::RelationData,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl Index {
    fn open(indexrelid: pgrx::pg_sys::Oid, lockmode: pgrx::pg_sys::LOCKMASK) -> Self {
        Self {
            raw: unsafe { pgrx::pg_sys::index_open(indexrelid, lockmode) },
            lockmode,
        }
    }
    fn raw(&self) -> *mut pgrx::pg_sys::RelationData {
        self.raw
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::index_close(self.raw, self.lockmode);
        }
    }
}

pub const fn ctid_to_key(
    ItemPointerData {
        ip_blkid: BlockIdData { bi_hi, bi_lo },
        ip_posid,
    }: ItemPointerData,
) -> [u16; 3] {
    [bi_hi, bi_lo, ip_posid]
}

pub const fn key_to_ctid([bi_hi, bi_lo, ip_posid]: [u16; 3]) -> ItemPointerData {
    ItemPointerData {
        ip_blkid: BlockIdData { bi_hi, bi_lo },
        ip_posid,
    }
}

pub const fn pointer_to_kv(pointer: NonZero<u64>) -> ([u16; 3], u16) {
    let value = pointer.get();
    let bi_hi = ((value >> 48) & 0xffff) as u16;
    let bi_lo = ((value >> 32) & 0xffff) as u16;
    let ip_posid = ((value >> 16) & 0xffff) as u16;
    let extra = value as u16;
    ([bi_hi, bi_lo, ip_posid], extra)
}

pub const fn kv_to_pointer((key, value): ([u16; 3], u16)) -> NonZero<u64> {
    let x = (key[0] as u64) << 48 | (key[1] as u64) << 32 | (key[2] as u64) << 16 | value as u64;
    NonZero::new(x).expect("invalid key")
}
