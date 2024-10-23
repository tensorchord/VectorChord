use crate::algorithm;
use crate::algorithm::build::{HeapRelation, Reporter};
use crate::index::am_options::{Opfamily, Reloption};
use crate::index::am_scan::Scanner;
use crate::index::utils::{ctid_to_pointer, pointer_to_ctid};
use crate::index::{am_options, am_scan};
use crate::postgres::Relation;
use crate::utils::cells::PgCell;
use base::search::Pointer;
use pgrx::datum::Internal;
use pgrx::pg_sys::Datum;

static RELOPT_KIND_RABBITHOLE: PgCell<pgrx::pg_sys::relopt_kind::Type> = unsafe { PgCell::new(0) };

pub unsafe fn init() {
    unsafe {
        RELOPT_KIND_RABBITHOLE.set(pgrx::pg_sys::add_reloption_kind());
        pgrx::pg_sys::add_string_reloption(
            RELOPT_KIND_RABBITHOLE.get(),
            c"options".as_ptr(),
            c"Vector index options, represented as a TOML string.".as_ptr(),
            c"".as_ptr(),
            None,
            pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
        );
    }
}

#[pgrx::pg_extern(sql = "\
CREATE FUNCTION _rabbithole_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';")]
fn _rabbithole_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
    type T = pgrx::pg_sys::IndexAmRoutine;
    unsafe {
        let index_am_routine = pgrx::pg_sys::palloc0(size_of::<T>()) as *mut T;
        index_am_routine.write(AM_HANDLER);
        Internal::from(Some(Datum::from(index_am_routine)))
    }
}

const AM_HANDLER: pgrx::pg_sys::IndexAmRoutine = {
    let mut am_routine =
        unsafe { std::mem::MaybeUninit::<pgrx::pg_sys::IndexAmRoutine>::zeroed().assume_init() };

    am_routine.type_ = pgrx::pg_sys::NodeTag::T_IndexAmRoutine;

    am_routine.amcanorderbyop = true;

    // Index access methods that set `amoptionalkey` to `false`
    // must index all tuples, even if the first column is `NULL`.
    // However, PostgreSQL does not generate a path if there is no
    // index clauses, even if there is a `ORDER BY` clause.
    // So we have to set it to `true` and set costs of every path
    // for vector index scans without `ORDER BY` clauses a large number
    // and throw errors if someone really wants such a path.
    am_routine.amoptionalkey = true;
    am_routine.amcanmulticol = true;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuild = Some(ambuild);
    am_routine.ambuildempty = Some(ambuildempty);
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
pub unsafe extern "C" fn amvalidate(opclass_oid: pgrx::pg_sys::Oid) -> bool {
    if am_options::convert_opclass_to_vd(opclass_oid).is_some() {
        pgrx::info!("Vector indexes can only be built on built-in operator classes.");
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amoptions(reloptions: Datum, validate: bool) -> *mut pgrx::pg_sys::bytea {
    let rdopts = unsafe {
        pgrx::pg_sys::build_reloptions(
            reloptions,
            validate,
            RELOPT_KIND_RABBITHOLE.get(),
            size_of::<Reloption>(),
            Reloption::TAB.as_ptr(),
            Reloption::TAB.len() as _,
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

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambuild(
    heap: pgrx::pg_sys::Relation,
    index: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
) -> *mut pgrx::pg_sys::IndexBuildResult {
    pub struct Heap {
        heap: pgrx::pg_sys::Relation,
        index: pgrx::pg_sys::Relation,
        index_info: *mut pgrx::pg_sys::IndexInfo,
        opfamily: Opfamily,
    }
    impl HeapRelation for Heap {
        fn traverse<F>(&self, callback: F)
        where
            F: FnMut((Pointer, Option<u32>, Vec<f32>)),
        {
            pub struct State<'a, F> {
                pub this: &'a Heap,
                pub callback: F,
            }
            #[pgrx::pg_guard]
            unsafe extern "C" fn call<F>(
                index: pgrx::pg_sys::Relation,
                ctid: pgrx::pg_sys::ItemPointer,
                values: *mut Datum,
                is_null: *mut bool,
                _tuple_is_alive: bool,
                state: *mut core::ffi::c_void,
            ) where
                F: FnMut((Pointer, Option<u32>, Vec<f32>)),
            {
                pgrx::check_for_interrupts!();
                use base::vector::OwnedVector;
                let state = unsafe { &mut *state.cast::<State<F>>() };
                let vector = unsafe {
                    state
                        .this
                        .opfamily
                        .datum_to_vector(*values.add(0), *is_null.add(0))
                };
                if let Some(vector) = vector {
                    let pointer = unsafe { ctid_to_pointer(ctid.read()) };
                    let extra = unsafe {
                        match (*(*index).rd_att).natts {
                            1 => None,
                            2 => {
                                if !is_null.add(1).read() {
                                    Some(values.add(1).read().value() as u32)
                                } else {
                                    None
                                }
                            }
                            _ => unreachable!(),
                        }
                    };
                    let vector = match vector {
                        OwnedVector::Vecf32(x) => x,
                        OwnedVector::Vecf16(_) => unreachable!(),
                        OwnedVector::SVecf32(_) => unreachable!(),
                        OwnedVector::BVector(_) => unreachable!(),
                    };
                    (state.callback)((pointer, extra, vector.into_vec()));
                }
            }
            let table_am = unsafe { &*(*self.heap).rd_tableam };
            let mut state = State {
                this: self,
                callback,
            };
            unsafe {
                table_am.index_build_range_scan.unwrap()(
                    self.heap,
                    self.index,
                    self.index_info,
                    true,
                    false,
                    true,
                    0,
                    pgrx::pg_sys::InvalidBlockNumber,
                    Some(call::<F>),
                    (&mut state) as *mut State<F> as *mut _,
                    std::ptr::null_mut(),
                );
            }
        }
    }
    pub struct PgReporter {}
    impl Reporter for PgReporter {
        fn tuples_total(&mut self, tuples_total: usize) {
            unsafe {
                pgrx::pg_sys::pgstat_progress_update_param(
                    pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_TOTAL as _,
                    tuples_total as _,
                );
            }
        }
        fn tuples_done(&mut self, tuples_done: usize) {
            unsafe {
                pgrx::pg_sys::pgstat_progress_update_param(
                    pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_DONE as _,
                    tuples_done as _,
                );
            }
        }
    }
    let (vector_options, rabbithole_options) = unsafe { am_options::options(index) };
    let heap_relation = Heap {
        heap,
        index,
        index_info,
        opfamily: unsafe { am_options::opfamily(index) },
    };
    let index_relation = unsafe { Relation::new(index) };
    algorithm::build::build(
        vector_options,
        rabbithole_options,
        heap_relation,
        index_relation,
        PgReporter {},
    );
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambuildempty(_index: pgrx::pg_sys::Relation) {
    pgrx::error!("Unlogged indexes are not supported.");
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
    index: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    use base::vector::OwnedVector;
    let opfamily = unsafe { am_options::opfamily(index) };
    let vector = unsafe { opfamily.datum_to_vector(*values.add(0), *is_null.add(0)) };
    if let Some(vector) = vector {
        let pointer = ctid_to_pointer(unsafe { heap_tid.read() });
        let extra = unsafe {
            match (*(*index).rd_att).natts {
                1 => None,
                2 => {
                    if !is_null.add(1).read() {
                        Some(values.add(1).read().value() as u32)
                    } else {
                        None
                    }
                }
                _ => unreachable!(),
            }
        };
        let vector = match vector {
            OwnedVector::Vecf32(x) => x,
            OwnedVector::Vecf16(_) => unreachable!(),
            OwnedVector::SVecf32(_) => unreachable!(),
            OwnedVector::BVector(_) => unreachable!(),
        };
        algorithm::insert::insert(
            unsafe { Relation::new(index) },
            pointer,
            extra,
            vector.into_vec(),
        );
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambeginscan(
    index: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index, n_keys, n_orderbys) };
    unsafe {
        let scanner = am_scan::scan_make(None, None, Vec::new(), false);
        (*scan).opaque = CurrentMemoryContext.leak_and_drop_on_delete(scanner).cast();
    }
    scan
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amrescan(
    scan: pgrx::pg_sys::IndexScanDesc,
    keys: pgrx::pg_sys::ScanKey,
    _n_keys: std::os::raw::c_int,
    orderbys: pgrx::pg_sys::ScanKey,
    _n_orderbys: std::os::raw::c_int,
) {
    unsafe {
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as _);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(orderbys, (*scan).orderByData, (*scan).numberOfOrderBys as _);
        }
        let opfamily = am_options::opfamily((*scan).indexRelation);
        let (orderbys, spheres, filters) = {
            let mut orderbys = Vec::new();
            let mut spheres = Vec::new();
            let mut filters = Vec::new();
            if (*scan).numberOfOrderBys == 0 && (*scan).numberOfKeys == 0 {
                pgrx::error!(
                    "vector search with no WHERE clause and no ORDER BY clause is not supported"
                );
            }
            for i in 0..(*scan).numberOfOrderBys {
                let data = (*scan).orderByData.add(i as usize);
                let value = (*data).sk_argument;
                let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                match (*data).sk_strategy {
                    1 => orderbys.push(opfamily.datum_to_vector(value, is_null)),
                    _ => unreachable!(),
                }
            }
            for i in 0..(*scan).numberOfKeys {
                let data = (*scan).keyData.add(i as usize);
                let value = (*data).sk_argument;
                let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                match (*data).sk_strategy {
                    2 => spheres.push(opfamily.datum_to_sphere(value, is_null)),
                    11 => {
                        if is_null {
                            filters.push(0);
                            filters.push(1);
                        } else {
                            filters.push(value.value() as u32);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (orderbys, spheres, filters)
        };
        let (vector, threshold, filters, recheck) =
            am_scan::scan_build(orderbys, spheres, filters, opfamily);
        let scanner = (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked();
        let scanner = std::mem::replace(
            scanner,
            am_scan::scan_make(vector, threshold, filters, recheck),
        );
        am_scan::scan_release(scanner);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amgettuple(
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
    let relation = unsafe { Relation::new((*scan).indexRelation) };
    if let Some((pointer, recheck)) = am_scan::scan_next(scanner, relation) {
        let ctid = pointer_to_ctid(pointer);
        unsafe {
            (*scan).xs_heaptid = ctid;
            (*scan).xs_recheckorderby = false;
            (*scan).xs_recheck = recheck;
        }
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amendscan(scan: pgrx::pg_sys::IndexScanDesc) {
    unsafe {
        let scanner = (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked();
        let scanner = std::mem::replace(scanner, am_scan::scan_make(None, None, Vec::new(), false));
        am_scan::scan_release(scanner);
    }
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
    let callback = callback.unwrap();
    let callback = |p: Pointer| unsafe { callback(&mut pointer_to_ctid(p), callback_state) };
    algorithm::vacuum::vacuum(unsafe { Relation::new((*info).index) }, callback);
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvacuumcleanup(
    _info: *mut pgrx::pg_sys::IndexVacuumInfo,
    _stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    std::ptr::null_mut()
}
