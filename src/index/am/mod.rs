pub mod am_build;
pub mod am_scan;

use crate::index::projection::RandomProject;
use crate::index::storage::PostgresRelation;
use algorithm::operator::{Dot, L2, Op, Vector};
use algorithm::types::*;
use half::f16;
use pgrx::datum::Internal;
use pgrx::pg_sys::Datum;
use std::ffi::CStr;
use std::num::NonZeroU64;
use std::sync::OnceLock;
use vector::VectorOwned;
use vector::vect::VectOwned;

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

    am_routine.ambuild = Some(am_build::ambuild);
    am_routine.ambuildempty = Some(am_build::ambuildempty);
    am_routine.aminsert = Some(aminsert);
    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.ambeginscan = Some(am_scan::ambeginscan);
    am_routine.amrescan = Some(am_scan::amrescan);
    am_routine.amgettuple = Some(am_scan::amgettuple);
    am_routine.amendscan = Some(am_scan::amendscan);

    am_routine
};

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvalidate(_opclass_oid: pgrx::pg_sys::Oid) -> bool {
    true
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amoptions(reloptions: Datum, validate: bool) -> *mut pgrx::pg_sys::bytea {
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
pub unsafe extern "C" fn amcostestimate(
    _root: *mut pgrx::pg_sys::PlannerInfo,
    path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        if (*path).indexorderbys.is_null() && (*path).indexclauses.is_null() {
            *index_startup_cost = f64::MAX;
            *index_total_cost = f64::MAX;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 0.0;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = 1.0;
        *index_correlation = 1.0;
        *index_pages = 0.0;
    }
}

#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
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
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
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
    heap_tid: pgrx::pg_sys::ItemPointer,
) -> bool {
    let opfamily = unsafe { crate::index::opclass::opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let payload = ctid_to_pointer(unsafe { heap_tid.read() });
    let vector = unsafe { opfamily.input_vector(*values.add(0), *is_null.add(0)) };
    let Some(vector) = vector else { return false };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => algorithm::insert::<Op<VectOwned<f32>, L2>>(
            index,
            payload,
            RandomProject::project(VectOwned::<f32>::from_owned(vector).as_borrowed()),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => algorithm::insert::<Op<VectOwned<f32>, Dot>>(
            index,
            payload,
            RandomProject::project(VectOwned::<f32>::from_owned(vector).as_borrowed()),
        ),
        (VectorKind::Vecf16, DistanceKind::L2) => algorithm::insert::<Op<VectOwned<f16>, L2>>(
            index,
            payload,
            RandomProject::project(VectOwned::<f16>::from_owned(vector).as_borrowed()),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => algorithm::insert::<Op<VectOwned<f16>, Dot>>(
            index,
            payload,
            RandomProject::project(VectOwned::<f16>::from_owned(vector).as_borrowed()),
        ),
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambulkdelete(
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
    let opfamily = unsafe { crate::index::opclass::opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        pgrx::pg_sys::vacuum_delay_point();
    };
    let callback = callback.expect("null function pointer");
    let callback = |p: NonZeroU64| unsafe { callback(&mut pointer_to_ctid(p), callback_state) };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, L2>>(index, check, callback);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, Dot>>(index, check, callback);
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, L2>>(index, check, callback);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, Dot>>(index, check, callback);
        }
    }
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { crate::index::opclass::opfamily((*info).index) };
    let index = unsafe { PostgresRelation::new((*info).index) };
    let check = || unsafe {
        pgrx::pg_sys::vacuum_delay_point();
    };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f32>, L2>>(index, check);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f32>, Dot>>(index, check);
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f16>, L2>>(index, check);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f16>, Dot>>(index, check);
        }
    }
    stats
}

const fn pointer_to_ctid(pointer: NonZeroU64) -> pgrx::pg_sys::ItemPointerData {
    let value = pointer.get();
    pgrx::pg_sys::ItemPointerData {
        ip_blkid: pgrx::pg_sys::BlockIdData {
            bi_hi: ((value >> 32) & 0xffff) as u16,
            bi_lo: ((value >> 16) & 0xffff) as u16,
        },
        ip_posid: (value & 0xffff) as u16,
    }
}

const fn ctid_to_pointer(ctid: pgrx::pg_sys::ItemPointerData) -> NonZeroU64 {
    let mut value = 0;
    value |= (ctid.ip_blkid.bi_hi as u64) << 32;
    value |= (ctid.ip_blkid.bi_lo as u64) << 16;
    value |= ctid.ip_posid as u64;
    NonZeroU64::new(value).expect("invalid pointer")
}

#[test]
const fn soundness_check() {
    let a = pgrx::pg_sys::ItemPointerData {
        ip_blkid: pgrx::pg_sys::BlockIdData { bi_hi: 1, bi_lo: 2 },
        ip_posid: 3,
    };
    let b = ctid_to_pointer(a);
    let c = pointer_to_ctid(b);
    assert!(a.ip_blkid.bi_hi == c.ip_blkid.bi_hi);
    assert!(a.ip_blkid.bi_lo == c.ip_blkid.bi_lo);
    assert!(a.ip_posid == c.ip_posid);
}
