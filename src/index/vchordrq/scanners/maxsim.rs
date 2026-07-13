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
// Copyright (c) 2025-2026 TensorChord Inc.

mod candidate;
mod external;
mod gpu;
mod rerank;
mod search;

use self::candidate::{LegacyPageCandidateGenerator, PageCandidateGenerator};
use self::gpu::{GpuTileMaxsimBackend, UnixSocketTransport, report_gpu_fallback};
use self::rerank::{CpuExactMaxsimBackend, ExactMaxsimBackend, HeapArrayTensorSource};
use crate::index::fetcher::*;
use crate::index::gucs::PostgresMaxsimBackend;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::dispatch::*;
use crate::index::vchordrq::filter::filter;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::SearchOptions;
use crate::recorder::Recorder;
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use index::bump::Bump;
use index::packed::PackedRefMut8;
use index::prefetcher::*;
use index::relation::{Hints, Page, RelationPrefetch, RelationRead, RelationReadStream};
use index_accessor::Dot;
use simd::f16;
use std::cmp::Reverse;
use std::num::NonZero;
use std::time::Duration;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::{RerankMethod, how, maxsim_search, rerank_index};
use vector::VectorOwned;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
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
            Opfamily::VectorMaxsim
                | Opfamily::HalfvecMaxsim
                | Opfamily::Rabitq8Maxsim
                | Opfamily::Rabitq4Maxsim
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
        let maxsim_candidate_limit = options
            .maxsim_candidate_limit
            .map_or(usize::MAX, |value| value as usize);
        let has_maxsim_candidate_limit = options.maxsim_candidate_limit.is_some();
        let maxsim_backend = options.maxsim_backend;
        let maxsim_gpu_endpoint = options
            .maxsim_gpu_endpoint
            .as_deref()
            .map(|endpoint| endpoint.to_string_lossy().into_owned())
            .unwrap_or_default();
        let maxsim_gpu_timeout = Duration::from_millis(options.maxsim_gpu_timeout_ms as u64);
        let maxsim_gpu_max_batch_tokens = options.maxsim_gpu_max_batch_tokens as usize;
        let maxsim_gpu_max_batch_bytes = options.maxsim_gpu_max_batch_bytes as usize;
        if !matches!(maxsim_backend, PostgresMaxsimBackend::CoarseOnly)
            && (!has_maxsim_candidate_limit || maxsim_candidate_limit == 0)
        {
            pgrx::error!("exact MaxSim requires a positive vchordrq.maxsim_candidate_limit");
        }
        let opfamily = self.opfamily;
        let Some(vectors) = vectors else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        let expected_dim = vchordrq::cost(index).dim;
        if vectors.iter().any(|vector| vector.dim() != expected_dim) {
            pgrx::error!("dimension is not matched");
        }
        let rerank_query = vectors.clone();
        let method = how(index);
        if !matches!(method, RerankMethod::Index) {
            pgrx::error!("maxsim search with rerank_in_table is not supported");
        }
        assert!(matches!(opfamily.distance_kind(), DistanceKind::Dot));
        let search_hints = Hints::default().full(true);
        let rerank_hints = Hints::default().full(false);
        let make_h1_plain_prefetcher = MakeH1PlainPrefetcher { index };
        let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
        let make_h0_simple_prefetcher = MakeH0SimplePrefetcher { index };
        let make_h0_stream_prefetcher = MakeH0StreamPrefetcher {
            index,
            hints: search_hints,
        };
        let n = vectors.len();
        let accu_map = |(Reverse(distance), AlwaysEqual(payload))| (distance, payload);
        let rough_map =
            |((_, AlwaysEqual(rough)), AlwaysEqual(PackedRefMut8(&mut (payload, ..)))): (
                _,
                AlwaysEqual<PackedRefMut8<(NonZero<u64>, _, _)>>,
            )| (rough, payload);
        let mut token_searches = {
            let fetcher = &mut fetcher;
            let iter: Box<dyn Iterator<Item = _>> = match opfamily.vector_kind() {
                VectorKind::Vecf32 => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, Dot>;
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Simple, false) => {
                                    let prefetcher = SimplePrefetcher::new(index, sequence);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Stream, false) => {
                                    let prefetcher =
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
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
                    type Op = vchordrq::operator::Op<VectOwned<f16>, Dot>;
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Simple, false) => {
                                    let prefetcher = SimplePrefetcher::new(index, sequence);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Stream, false) => {
                                    let prefetcher =
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
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
                VectorKind::Rabitq8 => {
                    type Op = vchordrq::operator::Op<Rabitq8Owned, Dot>;
                    let unprojected = vectors
                        .into_iter()
                        .map(|vector| {
                            if let OwnedVector::Rabitq8(vector) = vector {
                                vector
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>();
                    Box::new((0..n).map(move |i| {
                        let (results, estimation_by_threshold) = match options.io_search {
                            Io::Plain => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
                                options.probes.clone(),
                                options.epsilon,
                                maxsim_threshold,
                                bump,
                                make_h1_plain_prefetcher.clone(),
                                make_h0_plain_prefetcher.clone(),
                            ),
                            Io::Simple => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
                                options.probes.clone(),
                                options.epsilon,
                                maxsim_threshold,
                                bump,
                                make_h1_plain_prefetcher.clone(),
                                make_h0_simple_prefetcher.clone(),
                            ),
                            Io::Stream => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Simple, false) => {
                                    let prefetcher = SimplePrefetcher::new(index, sequence);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Stream, false) => {
                                    let prefetcher =
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
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
                VectorKind::Rabitq4 => {
                    type Op = vchordrq::operator::Op<Rabitq4Owned, Dot>;
                    let unprojected = vectors
                        .into_iter()
                        .map(|vector| {
                            if let OwnedVector::Rabitq4(vector) = vector {
                                vector
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>();
                    Box::new((0..n).map(move |i| {
                        let (results, estimation_by_threshold) = match options.io_search {
                            Io::Plain => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
                                options.probes.clone(),
                                options.epsilon,
                                maxsim_threshold,
                                bump,
                                make_h1_plain_prefetcher.clone(),
                                make_h0_plain_prefetcher.clone(),
                            ),
                            Io::Simple => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
                                options.probes.clone(),
                                options.epsilon,
                                maxsim_threshold,
                                bump,
                                make_h1_plain_prefetcher.clone(),
                                make_h0_simple_prefetcher.clone(),
                            ),
                            Io::Stream => maxsim_search::<_, Op>(
                                index,
                                unprojected[i].as_borrowed(),
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Simple, false) => {
                                    let prefetcher = SimplePrefetcher::new(index, sequence);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
                                    accu_set.extend(reranker.by_ref().take(maxsim_refine as _));
                                    let (rough_iter, accu_iter) = reranker.finish();
                                    accu_set.extend(accu_iter.map(accu_map));
                                    rough_set.extend(rough_iter.into_iter().map(rough_map));
                                }
                                (Io::Stream, false) => {
                                    let prefetcher =
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
                                    );
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
                                        StreamPrefetcher::new(index, sequence, rerank_hints);
                                    let mut reranker = rerank_index::<Op, _, _, _>(
                                        unprojected[i].clone(),
                                        prefetcher,
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
            iter
        };
        let mut candidates =
            LegacyPageCandidateGenerator.generate(n, &mut token_searches, maxsim_candidate_limit);
        drop(token_searches);
        let iter: Box<dyn Iterator<Item = _>> = match maxsim_backend {
            PostgresMaxsimBackend::CoarseOnly => Box::new(candidates.map(|candidate| {
                let distance = candidate.approximate_distance.to_f32();
                let recheck = false;
                (distance, candidate.heap_key, recheck)
            })),
            PostgresMaxsimBackend::CpuExact => {
                let mut source = HeapArrayTensorSource::new(&mut fetcher, opfamily);
                let exact =
                    CpuExactMaxsimBackend.rerank(&rerank_query, &mut candidates, &mut source);
                let exact = exact.unwrap_or_else(|error| pgrx::error!("{error}"));
                Box::new(exact.map(|result| {
                    let distance = result.distance.to_f32();
                    let recheck = false;
                    (distance, result.heap_key, recheck)
                }))
            }
            PostgresMaxsimBackend::Gpu => {
                let candidates = candidates.collect::<Vec<_>>();
                let mut candidate_iter = candidates.iter().copied();
                let mut source = HeapArrayTensorSource::new(&mut fetcher, opfamily);
                let transport = UnixSocketTransport::new(maxsim_gpu_endpoint);
                let mut backend = GpuTileMaxsimBackend::new(
                    transport,
                    maxsim_gpu_timeout,
                    maxsim_gpu_max_batch_tokens,
                    maxsim_gpu_max_batch_bytes,
                );
                let exact = backend.rerank(&rerank_query, &mut candidate_iter, &mut source);
                let exact = exact.unwrap_or_else(|error| pgrx::error!("{error}"));
                Box::new(exact.map(|result| {
                    let distance = result.distance.to_f32();
                    let recheck = false;
                    (distance, result.heap_key, recheck)
                }))
            }
            PostgresMaxsimBackend::Auto => {
                let candidates = candidates.collect::<Vec<_>>();
                let mut candidate_iter = candidates.iter().copied();
                let mut source = HeapArrayTensorSource::new(&mut fetcher, opfamily);
                let transport = UnixSocketTransport::new(maxsim_gpu_endpoint);
                let mut backend = GpuTileMaxsimBackend::new(
                    transport,
                    maxsim_gpu_timeout,
                    maxsim_gpu_max_batch_tokens,
                    maxsim_gpu_max_batch_bytes,
                );
                let exact = backend.rerank(&rerank_query, &mut candidate_iter, &mut source);
                let exact = match exact {
                    Ok(exact) => exact,
                    Err(error) => {
                        report_gpu_fallback(&error);
                        let mut candidate_iter = candidates.iter().copied();
                        let mut source = HeapArrayTensorSource::new(&mut fetcher, opfamily);
                        CpuExactMaxsimBackend
                            .rerank(&rerank_query, &mut candidate_iter, &mut source)
                            .unwrap_or_else(|error| pgrx::error!("{error}"))
                    }
                };
                Box::new(exact.map(|result| {
                    let distance = result.distance.to_f32();
                    let recheck = false;
                    (distance, result.heap_key, recheck)
                }))
            }
        };
        #[allow(clippy::let_and_return)]
        iter
    }
}

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut8<'a, (B, C, D)>>)) -> R,
{
    f
}
