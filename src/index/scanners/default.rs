use super::{SearchBuilder, SearchFetcher, SearchIo, SearchOptions};
use crate::index::algorithm::RandomProject;
use crate::index::am::pointer_to_kv;
use crate::index::gucs::enable_prefilter;
use crate::index::opclass::{Opfamily, Sphere};
use algorithm::operator::{Dot, L2, Op, Operator};
use algorithm::types::{DistanceKind, OwnedVector, VectorKind};
use algorithm::*;
use always_equal::AlwaysEqual;
use distance::Distance;
use half::f16;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecIp
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorIp
                | Opfamily::VectorL2
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
            spheres: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            1 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vector(x)) };
                self.orderbys.push(x);
            }
            2 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_sphere(x)) };
                self.spheres.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'a>(
        self,
        relation: &'a (impl RelationPrefetch + RelationReadStream),
        options: SearchOptions,
        mut fetcher: impl SearchFetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a> {
        let mut vector = None;
        let mut threshold = None;
        let mut recheck = false;
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(orderby_vector);
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for Sphere { center, radius } in self.spheres.into_iter().flatten() {
            if vector.is_none() {
                (vector, threshold) = (Some(center), Some(radius));
            } else {
                recheck = true;
            }
        }
        let opfamily = self.opfamily;
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        // let prefilter = move |result: &Result| {
        //     if !enable_prefilter() {
        //         return true;
        //     }
        //     let (key, _) = pointer_to_kv(result.1.0.0);
        //     fetcher.filter(key)
        // };
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> = match (
            opfamily.vector_kind(),
            opfamily.distance_kind(),
        ) {
            (VectorKind::Vecf32, DistanceKind::L2) => {
                let original_vector = if let OwnedVector::Vecf32(vector) = vector {
                    vector
                } else {
                    unreachable!()
                };
                let vector = RandomProject::project(original_vector.as_borrowed());
                let results = default_search::<_, Op<VectOwned<f32>, L2>, _>(
                    relation.clone(),
                    vector.clone(),
                    options.probes,
                    options.epsilon,
                    bump,
                    {
                        let index = relation.clone();
                        move |results| {
                            PlainPrefetcher::<_, BinaryHeap<_>, _, Op<VectOwned<f32>, L2>>::new(
                                index.clone(),
                                results,
                                |_, _| (true, None),
                                false,
                            )
                        }
                    },
                );
                let prefilter = move |result: &Result, with_fetch: bool| {
                    let (key, _) = pointer_to_kv(result.1.0.0);
                    match (enable_prefilter(), with_fetch) {
                        (false, false) => (true, None),
                        (true, false) => (fetcher.filter_only(key), None),
                        (_, true) => match fetcher.filter_fetch(key) {
                            Some((datums, is_nulls)) => {
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                (true, Some(raw))
                            }
                            None => (false, None),
                        },
                    }
                };
                let method = how(relation.clone());
                rerank_wrapper::<Op<VectOwned<f32>, L2>>(
                    original_vector,
                    relation,
                    prefilter,
                    opfamily,
                    results,
                    method,
                    options.io_rerank,
                )
            }
            (VectorKind::Vecf32, DistanceKind::Dot) => {
                let original_vector = if let OwnedVector::Vecf32(vector) = vector {
                    vector
                } else {
                    unreachable!()
                };
                let vector = RandomProject::project(original_vector.as_borrowed());
                let results = default_search::<_, Op<VectOwned<f32>, Dot>, _>(
                    relation.clone(),
                    vector.clone(),
                    options.probes,
                    options.epsilon,
                    bump,
                    {
                        let index = relation.clone();
                        move |results| {
                            PlainPrefetcher::<_, BinaryHeap<_>, _, Op<VectOwned<f32>, Dot>>::new(
                                index.clone(),
                                results,
                                |_, _| (true, None),
                                false,
                            )
                        }
                    },
                );
                let prefilter = move |result: &Result, with_fetch: bool| {
                    let (key, _) = pointer_to_kv(result.1.0.0);
                    match (enable_prefilter(), with_fetch) {
                        (false, false) => (true, None),
                        (true, false) => (fetcher.filter_only(key), None),
                        (_, true) => match fetcher.filter_fetch(key) {
                            Some((datums, is_nulls)) => {
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                (true, Some(raw))
                            }
                            None => (false, None),
                        },
                    }
                };
                let method = how(relation.clone());
                rerank_wrapper::<Op<VectOwned<f32>, Dot>>(
                    original_vector,
                    relation,
                    prefilter,
                    opfamily,
                    results,
                    method,
                    options.io_rerank,
                )
            }
            (VectorKind::Vecf16, DistanceKind::L2) => {
                let original_vector = if let OwnedVector::Vecf16(vector) = vector {
                    vector
                } else {
                    unreachable!()
                };
                let vector = RandomProject::project(original_vector.as_borrowed());
                let results = default_search::<_, Op<VectOwned<f16>, L2>, _>(
                    relation.clone(),
                    vector.clone(),
                    options.probes,
                    options.epsilon,
                    bump,
                    {
                        let index = relation.clone();
                        move |results| {
                            PlainPrefetcher::<_, BinaryHeap<_>, _, Op<VectOwned<f16>, L2>>::new(
                                index.clone(),
                                results,
                                |_, _| (true, None),
                                false,
                            )
                        }
                    },
                );
                let prefilter = move |result: &Result, with_fetch: bool| {
                    let (key, _) = pointer_to_kv(result.1.0.0);
                    match (enable_prefilter(), with_fetch) {
                        (false, false) => (true, None),
                        (true, false) => (fetcher.filter_only(key), None),
                        (_, true) => match fetcher.filter_fetch(key) {
                            Some((datums, is_nulls)) => {
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                (true, Some(raw))
                            }
                            None => (false, None),
                        },
                    }
                };
                let method = how(relation.clone());
                rerank_wrapper::<Op<VectOwned<f16>, L2>>(
                    original_vector,
                    relation,
                    prefilter,
                    opfamily,
                    results,
                    method,
                    options.io_rerank,
                )
            }
            (VectorKind::Vecf16, DistanceKind::Dot) => {
                let original_vector = if let OwnedVector::Vecf16(vector) = vector {
                    vector
                } else {
                    unreachable!()
                };
                let vector = RandomProject::project(original_vector.as_borrowed());
                let results = default_search::<_, Op<VectOwned<f16>, Dot>, _>(
                    relation.clone(),
                    vector.clone(),
                    options.probes,
                    options.epsilon,
                    bump,
                    {
                        let index = relation.clone();
                        move |results| {
                            PlainPrefetcher::<_, BinaryHeap<_>, _, Op<VectOwned<f16>, Dot>>::new(
                                index.clone(),
                                results,
                                |_, _| (true, None),
                                false,
                            )
                        }
                    },
                );
                let prefilter = move |result: &Result, with_fetch: bool| {
                    let (key, _) = pointer_to_kv(result.1.0.0);
                    match (enable_prefilter(), with_fetch) {
                        (false, false) => (true, None),
                        (true, false) => (fetcher.filter_only(key), None),
                        (_, true) => match fetcher.filter_fetch(key) {
                            Some((datums, is_nulls)) => {
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                (true, Some(raw))
                            }
                            None => (false, None),
                        },
                    }
                };
                let method = how(relation.clone());
                rerank_wrapper::<Op<VectOwned<f16>, Dot>>(
                    original_vector,
                    relation,
                    prefilter,
                    opfamily,
                    results,
                    method,
                    options.io_rerank,
                )
            }
        };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(x, _)| *x < threshold))
        } else {
            iter
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        Box::new(iter.map(move |(distance, pointer)| {
            let (key, _) = pointer_to_kv(pointer);
            (distance, key, recheck)
        }))
    }
}

type Extra<'b> = &'b mut (NonZero<u64>, u16, &'b mut [u32]);
type Result<'b> = ((Reverse<Distance>, AlwaysEqual<()>), AlwaysEqual<Extra<'b>>);

#[allow(clippy::too_many_arguments)]
fn rerank_wrapper<'a, O: Operator>(
    vector: O::Vector,
    relation: &'a (impl RelationPrefetch + RelationReadStream),
    prefilter: impl FnMut(&Result, bool) -> (bool, Option<O::Vector>) + 'a,
    opfamily: Opfamily,
    results: Vec<Result<'a>>,
    method: RerankMethod,
    io_rerank: SearchIo,
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'a> {
    match (method, io_rerank) {
        (RerankMethod::Index, SearchIo::ReadBuffer) => {
            let prefetcher = PlainPrefetcher::<_, BinaryHeap<_>, _, O>::new(
                relation.clone(),
                results,
                prefilter,
                false,
            );
            Box::new(
                rerank_index::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
        (RerankMethod::Index, SearchIo::PrefetchBuffer) => {
            let prefetcher = SimplePrefetcher::new(relation.clone(), results, prefilter, false);
            Box::new(
                rerank_index::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
        (RerankMethod::Index, SearchIo::ReadStream) => {
            let prefetcher = StreamPrefetcher::new(relation, results, prefilter, false);
            Box::new(
                rerank_index::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
        (RerankMethod::Heap, SearchIo::ReadBuffer) => {
            let prefetcher = PlainPrefetcher::<_, BinaryHeap<_>, _, O>::new(
                relation.clone(),
                results,
                prefilter,
                true,
            );
            Box::new(
                rerank_heap::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
        (RerankMethod::Heap, SearchIo::PrefetchBuffer) => {
            let prefetcher = SimplePrefetcher::new(relation.clone(), results, prefilter, true);
            Box::new(
                rerank_heap::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
        (RerankMethod::Heap, SearchIo::ReadStream) => {
            let prefetcher = StreamPrefetcher::new(relation, results, prefilter, true);
            Box::new(
                rerank_heap::<O, _, _>(vector, prefetcher)
                    .map(move |(distance, payload)| (opfamily.output(distance), payload)),
            )
        }
    }
}
