use super::{SearchBuilder, SearchFetcher, SearchOptions};
use crate::index::algorithm::RandomProject;
use crate::index::am::pointer_to_kv;
use crate::index::opclass::Opfamily;
use crate::index::scanners::SearchIo;
use algorithm::operator::Dot;
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

pub struct MaxsimBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<Vec<OwnedVector>>>,
}

impl SearchBuilder for MaxsimBuilder {
    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            3 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vectors(x)) };
                self.orderbys.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'a>(
        self,
        relation: &'a (impl RelationPrefetch + RelationReadStream),
        options: SearchOptions,
        _: impl SearchFetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a> {
        let mut vectors = None;
        for orderby_vectors in self.orderbys.into_iter().flatten() {
            if vectors.is_none() {
                vectors = Some(orderby_vectors);
            } else {
                pgrx::error!("maxsim search with multiple vectors is not supported");
            }
        }
        if let Some(_max_scan_tuples) = options.max_scan_tuples {
            pgrx::error!("maxsim search with max_scan_tuples is not supported");
        }
        let maxsim_refine = options.maxsim_refine;
        let maxsim_threshold = options.maxsim_threshold;
        let opfamily = self.opfamily;
        let Some(vectors) = vectors else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        let method = how(relation.clone());
        if !matches!(method, RerankMethod::Index) {
            pgrx::error!("maxsim search with rerank_in_table is not supported");
        }
        assert!(matches!(opfamily.distance_kind(), DistanceKind::Dot));
        let n = vectors.len();
        let accu_map = |(Reverse(distance), AlwaysEqual(payload))| (distance, payload);
        let rough_map = |((_, AlwaysEqual(rough)), AlwaysEqual(&mut (payload, ..))): (
            _,
            AlwaysEqual<&mut (NonZero<u64>, _, _)>,
        )| (rough, payload);
        let fake_prefilter = |_| true;
        let iter: Box<dyn Iterator<Item = _>> = match opfamily.vector_kind() {
            VectorKind::Vecf32 => {
                type Op = operator::Op<VectOwned<f32>, Dot>;
                let vectors = vectors
                    .into_iter()
                    .map(|vector| {
                        RandomProject::project(
                            if let OwnedVector::Vecf32(vector) = vector {
                                vector
                            } else {
                                unreachable!()
                            }
                            .as_borrowed(),
                        )
                    })
                    .collect::<Vec<_>>();
                Box::new(vectors.into_iter().map(|vector| {
                    let (results, estimation_by_threshold) = maxsim_search::<_, Op, _>(
                        relation.clone(),
                        vector.clone(),
                        options.probes.clone(),
                        options.epsilon,
                        maxsim_threshold,
                        bump,
                        {
                            let index = relation.clone();
                            move |results| {
                                PlainPrefetcher::<_, BinaryHeap<_>>::new(index.clone(), results)
                            }
                        },
                    );
                    let (mut accu_set, mut rough_set) = (Vec::new(), Vec::new());
                    if maxsim_refine != 0 && !results.is_empty() {
                        match options.io_rerank {
                            SearchIo::ReadBuffer => {
                                let prefetcher = PlainPrefetcher::<_, BinaryHeap<_>>::new(
                                    relation.clone(),
                                    results,
                                );
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            SearchIo::PrefetchBuffer => {
                                let prefetcher = SimplePrefetcher::new(relation.clone(), results);
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            SearchIo::ReadStream => {
                                let prefetcher = StreamPrefetcher::new(relation, results);
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                        }
                    } else {
                        let rough_iter = results.into_iter();
                        rough_set.extend(rough_iter.map(rough_map));
                    }
                    (accu_set, rough_set, estimation_by_threshold)
                }))
            }
            VectorKind::Vecf16 => {
                type Op = operator::Op<VectOwned<f16>, Dot>;
                let vectors = vectors
                    .into_iter()
                    .map(|vector| {
                        RandomProject::project(
                            if let OwnedVector::Vecf16(vector) = vector {
                                vector
                            } else {
                                unreachable!()
                            }
                            .as_borrowed(),
                        )
                    })
                    .collect::<Vec<_>>();
                Box::new(vectors.into_iter().map(|vector| {
                    let (results, estimation_by_threshold) = maxsim_search::<_, Op, _>(
                        relation.clone(),
                        vector.clone(),
                        options.probes.clone(),
                        options.epsilon,
                        maxsim_threshold,
                        bump,
                        {
                            let index = relation.clone();
                            move |results| {
                                PlainPrefetcher::<_, BinaryHeap<_>>::new(index.clone(), results)
                            }
                        },
                    );
                    let (mut accu_set, mut rough_set) = (Vec::new(), Vec::new());
                    if maxsim_refine != 0 && !results.is_empty() {
                        match options.io_rerank {
                            SearchIo::ReadBuffer => {
                                let prefetcher = PlainPrefetcher::<_, BinaryHeap<_>>::new(
                                    relation.clone(),
                                    results,
                                );
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            SearchIo::PrefetchBuffer => {
                                let prefetcher = SimplePrefetcher::new(relation.clone(), results);
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            SearchIo::ReadStream => {
                                let prefetcher = StreamPrefetcher::new(relation, results);
                                let mut reranker = rerank_index::<Op, _, _>(
                                    vector.clone(),
                                    prefetcher,
                                    fake_prefilter,
                                );
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                        }
                    } else {
                        let rough_iter = results.into_iter();
                        rough_set.extend(rough_iter.map(rough_map));
                    }
                    (accu_set, rough_set, estimation_by_threshold)
                }))
            }
        };
        let mut updates = Vec::new();
        let mut estimations = Vec::new();
        for (query_id, (accu_set, rough_set, estimation_by_threshold)) in iter.enumerate() {
            updates.reserve(accu_set.len() + rough_set.len());
            let is_empty = accu_set.is_empty() && rough_set.is_empty();
            let mut estimation_by_scope = Distance::NEG_INFINITY;
            for (distance, payload) in accu_set {
                estimation_by_scope = std::cmp::max(estimation_by_scope, distance);
                let (key, _) = pointer_to_kv(payload);
                updates.push((key, query_id, distance));
            }
            for (distance, payload) in rough_set {
                let (key, _) = pointer_to_kv(payload);
                updates.push((key, query_id, distance));
            }
            estimations.push(if !is_empty {
                std::cmp::max(estimation_by_scope, estimation_by_threshold)
            } else {
                Distance::ZERO
            });
        }
        updates.sort_unstable_by_key(|&(key, ..)| key);
        let iter = updates
            .chunk_by(|(kl, ..), (kr, ..)| kl == kr)
            .map(|chunk| {
                let key = chunk[0].0;
                let mut value = vec![None; n];
                for &(_, query_id, distance) in chunk {
                    let this = value[query_id].get_or_insert(Distance::INFINITY);
                    *this = std::cmp::min(*this, distance);
                }
                let mut maxsim = 0.0f32;
                for (query_id, distance) in value.into_iter().enumerate() {
                    let d = distance.unwrap_or(estimations[query_id]);
                    maxsim += Distance::to_f32(d);
                }
                (Reverse(Distance::from_f32(maxsim)), AlwaysEqual(key))
            })
            .collect::<BinaryHeap<_>>()
            .into_iter_sorted_polyfill()
            .map(|(Reverse(distance), AlwaysEqual(key))| {
                let distance = distance.to_f32();
                let recheck = false;
                (distance, key, recheck)
            });
        let iter: Box<dyn Iterator<Item = _>> = Box::new(iter);
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        #[allow(clippy::let_and_return)]
        iter
    }
}

// Emulate unstable library feature `binary_heap_into_iter_sorted`.
// See https://github.com/rust-lang/rust/issues/59278.

pub trait IntoIterSortedPolyfill<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T>;
}

impl<T> IntoIterSortedPolyfill<T> for BinaryHeap<T> {
    fn into_iter_sorted_polyfill(self) -> IntoIterSorted<T> {
        IntoIterSorted { inner: self }
    }
}

#[derive(Clone, Debug)]
pub struct IntoIterSorted<T> {
    inner: BinaryHeap<T>,
}

impl<T: Ord> Iterator for IntoIterSorted<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

impl<T: Ord> ExactSizeIterator for IntoIterSorted<T> {}

impl<T: Ord> std::iter::FusedIterator for IntoIterSorted<T> {}
