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

use crate::index::fetcher::*;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::algo::*;
use crate::index::vchordrq::filter::filter;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::SearchOptions;
use crate::recorder::Recorder;
use algo::accessor::Dot;
use algo::prefetcher::*;
use algo::*;
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use distance::Distance;
use simd::f16;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::*;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct MaxsimBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<Vec<OwnedVector>>>,
}

impl SearchBuilder for MaxsimBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordrq::Opaque;

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

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        mut fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        _sender: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
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
        let method = how(index);
        if !matches!(method, RerankMethod::Index) {
            pgrx::error!("maxsim search with rerank_in_table is not supported");
        }
        assert!(matches!(opfamily.distance_kind(), DistanceKind::Dot));
        let make_h1_plain_prefetcher = MakeH1PlainPrefetcher { index };
        let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
        let make_h0_simple_prefetcher = MakeH0SimplePrefetcher { index };
        let make_h0_stream_prefetcher = MakeH0StreamPrefetcher {
            index,
            hints: Hints::default().full(true),
        };
        let n = vectors.len();
        let accu_map = |(Reverse(distance), AlwaysEqual(payload))| (distance, payload);
        let rough_map =
            |((_, AlwaysEqual(rough)), AlwaysEqual(PackedRefMut8(&mut (payload, ..)))): (
                _,
                AlwaysEqual<PackedRefMut8<(NonZero<u64>, _, _)>>,
            )| (rough, payload);
        let iter: Box<dyn Iterator<Item = _>> = match opfamily.vector_kind() {
            VectorKind::Vecf32 => {
                type Op = operator::Op<VectOwned<f32>, Dot>;
                let unprojected = vectors
                    .into_iter()
                    .map(|vector| {
                        if let OwnedVector::Vecf32(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>();
                let projected = unprojected
                    .iter()
                    .map(|vector| RandomProject::project(vector.as_borrowed()))
                    .collect::<Vec<_>>();
                Box::new((0..n).map(move |i| {
                    let (results, estimation_by_threshold) = match options.io_search {
                        Io::Plain => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_plain_prefetcher.clone(),
                        ),
                        Io::Simple => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_simple_prefetcher.clone(),
                        ),
                        Io::Stream => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_stream_prefetcher.clone(),
                        ),
                    };
                    let (mut accu_set, mut rough_set) = (Vec::new(), Vec::new());
                    if maxsim_refine != 0 && !results.is_empty() {
                        let sequence = Heap::from(results);
                        match (options.io_rerank, options.prefilter) {
                            (Io::Plain, false) => {
                                let prefetcher = PlainPrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Plain, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher = PlainPrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Simple, false) => {
                                let prefetcher = SimplePrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Simple, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher = SimplePrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Stream, false) => {
                                let prefetcher =
                                    StreamPrefetcher::new(index, sequence, Hints::default());
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Stream, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher =
                                    StreamPrefetcher::new(index, sequence, Hints::default());
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
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
                let unprojected = vectors
                    .into_iter()
                    .map(|vector| {
                        if let OwnedVector::Vecf16(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                    })
                    .collect::<Vec<_>>();
                let projected = unprojected
                    .iter()
                    .map(|vector| RandomProject::project(vector.as_borrowed()))
                    .collect::<Vec<_>>();
                Box::new((0..n).map(move |i| {
                    let (results, estimation_by_threshold) = match options.io_search {
                        Io::Plain => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_plain_prefetcher.clone(),
                        ),
                        Io::Simple => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_simple_prefetcher.clone(),
                        ),
                        Io::Stream => maxsim_search::<_, Op>(
                            index,
                            projected[i].as_borrowed(),
                            options.probes.clone(),
                            options.epsilon,
                            maxsim_threshold,
                            bump,
                            make_h1_plain_prefetcher.clone(),
                            make_h0_stream_prefetcher.clone(),
                        ),
                    };
                    let (mut accu_set, mut rough_set) = (Vec::new(), Vec::new());
                    if maxsim_refine != 0 && !results.is_empty() {
                        let sequence = Heap::from(results);
                        match (options.io_rerank, options.prefilter) {
                            (Io::Plain, false) => {
                                let prefetcher = PlainPrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Plain, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher = PlainPrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Simple, false) => {
                                let prefetcher = SimplePrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Simple, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher = SimplePrefetcher::new(index, sequence);
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Stream, false) => {
                                let prefetcher =
                                    StreamPrefetcher::new(index, sequence, Hints::default());
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
                                accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                let (rough_iter, accu_iter) = reranker.finish();
                                accu_set.extend(accu_iter.map(accu_map));
                                rough_set.extend(rough_iter.into_iter().map(rough_map));
                            }
                            (Io::Stream, true) => {
                                let predicate =
                                    id_0(|(_, AlwaysEqual(PackedRefMut8((pointer, _, _))))| {
                                        let (key, _) = pointer_to_kv(*pointer);
                                        let Some(mut tuple) = fetcher.fetch(key) else {
                                            return false;
                                        };
                                        tuple.filter()
                                    });
                                let sequence = filter(sequence, predicate);
                                let prefetcher =
                                    StreamPrefetcher::new(index, sequence, Hints::default());
                                let mut reranker =
                                    rerank_index::<Op, _, _, _>(unprojected[i].clone(), prefetcher);
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

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut8<'a, (B, C, D)>>)) -> R,
{
    f
}
