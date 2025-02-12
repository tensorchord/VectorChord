use crate::index::am::pointer_to_ctid;
use crate::index::gucs::{epsilon, max_scan_tuples, probes};
use crate::index::opclass::{Opfamily, Sphere, opfamily};
use crate::index::projection::RandomProject;
use crate::index::storage::PostgresRelation;
use algorithm::RerankMethod;
use algorithm::operator::{Dot, L2, Op, Vector};
use algorithm::types::*;
use half::f16;
use std::num::NonZeroU64;
use vector::VectorOwned;
use vector::vect::VectOwned;

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambeginscan(
    index_relation: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index_relation, n_keys, n_orderbys) };
    unsafe {
        let scanner = Scanner {
            opfamily: opfamily(index_relation),
            scanning: Scanning::Empty {},
        };
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
        let opfamily = opfamily((*scan).indexRelation);
        let (orderbys, spheres) = {
            let mut orderbys = Vec::new();
            let mut spheres = Vec::new();
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
                    1 => orderbys.push(opfamily.input_vector(value, is_null)),
                    _ => unreachable!(),
                }
            }
            for i in 0..(*scan).numberOfKeys {
                let data = (*scan).keyData.add(i as usize);
                let value = (*data).sk_argument;
                let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                match (*data).sk_strategy {
                    2 => spheres.push(opfamily.input_sphere(value, is_null)),
                    _ => unreachable!(),
                }
            }
            (orderbys, spheres)
        };
        let scanning;
        if let Some((vector, threshold, recheck)) = scanner_build(orderbys, spheres) {
            scanning = Scanning::Initial {
                vector,
                threshold,
                recheck,
            };
        } else {
            scanning = Scanning::Empty {};
        };
        let scanner = &mut *(*scan).opaque.cast::<Scanner>();
        scanner.scanning = scanning;
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
    let relation = unsafe { PostgresRelation::new((*scan).indexRelation) };
    let heap = unsafe { (*scan).heapRelation };
    if let Some((pointer, recheck)) = scanner_next(scanner, relation, heap) {
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
    let scanner = unsafe { &mut *(*scan).opaque.cast::<Scanner>() };
    scanner.scanning = Scanning::Empty {};
}

unsafe fn get_attribute_number_from_index(
    index: pgrx::pg_sys::Relation,
) -> pgrx::pg_sys::AttrNumber {
    unsafe {
        let a = (*index).rd_index;
        let natts = (*a).indnatts;
        assert!(natts == 1);
        (*a).indkey.values.as_slice(natts as _)[0]
    }
}

unsafe fn fetch_vector(
    opfamily: Opfamily,
    heap_relation: pgrx::pg_sys::Relation,
    attnum: i16,
    payload: NonZeroU64,
) -> Option<OwnedVector> {
    unsafe {
        let slot = pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut());
        let table_am = (*heap_relation).rd_tableam;
        let fetch_row_version = (*table_am).tuple_fetch_row_version.unwrap();
        let mut ctid = pointer_to_ctid(payload);
        fetch_row_version(
            heap_relation,
            &mut ctid,
            &raw mut pgrx::pg_sys::SnapshotAnyData,
            slot,
        );
        assert!(attnum > 0);
        if attnum > (*slot).tts_nvalid {
            pgrx::pg_sys::slot_getsomeattrs(slot, attnum as i32);
        }
        let result = opfamily.input_vector(
            (*slot).tts_values.add(attnum as usize - 1).read(),
            (*slot).tts_isnull.add(attnum as usize - 1).read(),
        );
        if !slot.is_null() {
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(slot);
        }
        result
    }
}

struct Scanner {
    opfamily: Opfamily,
    scanning: Scanning,
}

enum Scanning {
    Initial {
        vector: OwnedVector,
        threshold: Option<f32>,
        recheck: bool,
    },
    Vbase {
        vbase: Box<dyn Iterator<Item = (f32, NonZeroU64)>>,
        recheck: bool,
    },
    Empty {},
}

fn scanner_build(
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
) -> Option<(OwnedVector, Option<f32>, bool)> {
    let mut vector = None;
    let mut threshold = None;
    let mut recheck = false;
    for orderby_vector in orderbys.into_iter().flatten() {
        if vector.is_none() {
            vector = Some(orderby_vector);
        } else {
            pgrx::error!("vector search with multiple vectors is not supported");
        }
    }
    for Sphere { center, radius } in spheres.into_iter().flatten() {
        if vector.is_none() {
            (vector, threshold) = (Some(center), Some(radius));
        } else {
            recheck = true;
        }
    }
    Some((vector?, threshold, recheck))
}

fn scanner_next(
    scanner: &mut Scanner,
    relation: PostgresRelation,
    heap: *mut pgrx::pg_sys::RelationData,
) -> Option<(NonZeroU64, bool)> {
    if let Scanning::Initial {
        vector,
        threshold,
        recheck,
    } = &scanner.scanning
    {
        let opfamily = scanner.opfamily;
        let vector = vector.clone();
        let threshold = *threshold;
        let recheck = *recheck;
        let max_scan_tuples = max_scan_tuples();
        let probes = probes();
        let epsilon = epsilon();
        scanner.scanning = Scanning::Vbase {
            vbase: match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2) => {
                    let vector = RandomProject::project(
                        VectOwned::<f32>::from_owned(vector.clone()).as_borrowed(),
                    );
                    let (method, results) = algorithm::search::<Op<VectOwned<f32>, L2>>(
                        relation.clone(),
                        vector.clone(),
                        probes,
                        epsilon,
                    );
                    match (method, max_scan_tuples, threshold) {
                        (RerankMethod::Index, None, None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Index, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                        (RerankMethod::Heap, None, None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Heap, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                    }
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    let vector = RandomProject::project(
                        VectOwned::<f32>::from_owned(vector.clone()).as_borrowed(),
                    );
                    let (method, results) = algorithm::search::<Op<VectOwned<f32>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        probes,
                        epsilon,
                    );
                    match (method, max_scan_tuples, threshold) {
                        (RerankMethod::Index, None, None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Index, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f32>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                        (RerankMethod::Heap, None, None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Heap, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f32>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f32>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::L2) => {
                    let vector = RandomProject::project(
                        VectOwned::<f16>::from_owned(vector.clone()).as_borrowed(),
                    );
                    let (method, results) = algorithm::search::<Op<VectOwned<f16>, L2>>(
                        relation.clone(),
                        vector.clone(),
                        probes,
                        epsilon,
                    );
                    match (method, max_scan_tuples, threshold) {
                        (RerankMethod::Index, None, None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Index, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, L2>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                        (RerankMethod::Heap, None, None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Heap, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, L2>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    let vector = RandomProject::project(
                        VectOwned::<f16>::from_owned(vector.clone()).as_borrowed(),
                    );
                    let (method, results) = algorithm::search::<Op<VectOwned<f16>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        probes,
                        epsilon,
                    );
                    match (method, max_scan_tuples, threshold) {
                        (RerankMethod::Index, None, None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Index, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Index, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_index::<Op<VectOwned<f16>, Dot>>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                        (RerankMethod::Heap, None, None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.fuse()) as Box<dyn Iterator<Item = (f32, NonZeroU64)>>
                        }
                        (RerankMethod::Heap, None, Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take_while(move |(x, _)| *x < threshold))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), None) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(vbase.take(max_scan_tuples as _))
                        }
                        (RerankMethod::Heap, Some(max_scan_tuples), Some(threshold)) => {
                            let vbase = algorithm::rerank_heap::<Op<VectOwned<f16>, Dot>, _>(
                                vector,
                                results,
                                move |payload| unsafe {
                                    Some(RandomProject::project(
                                        VectOwned::<f16>::from_owned(fetch_vector(
                                            opfamily,
                                            heap,
                                            get_attribute_number_from_index(relation.raw()),
                                            payload,
                                        )?)
                                        .as_borrowed(),
                                    ))
                                },
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload));
                            Box::new(
                                vbase
                                    .take_while(move |(x, _)| *x < threshold)
                                    .take(max_scan_tuples as _),
                            )
                        }
                    }
                }
            },
            recheck,
        };
    }
    match &mut scanner.scanning {
        Scanning::Initial { .. } => unreachable!(),
        Scanning::Vbase { vbase, recheck } => vbase.next().map(|(_, x)| (x, *recheck)),
        Scanning::Empty {} => None,
    }
}
