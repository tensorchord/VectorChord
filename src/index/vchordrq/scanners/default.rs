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
use crate::index::opclass::Sphere;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::algo::*;
use crate::index::vchordrq::filter::filter;
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::SearchOptions;
use algo::accessor::{Dot, L2S};
use algo::prefetcher::*;
use algo::*;
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use simd::f16;
use std::num::NonZero;
use vchordrq::operator::{self};
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::*;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordrq::Opaque;

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

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        mut fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
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
        let make_h1_plain_prefetcher = MakeH1PlainPrefetcher { index };
        let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
        let make_h0_simple_prefetcher = MakeH0SimplePrefetcher { index };
        let make_h0_stream_prefetcher = MakeH0StreamPrefetcher {
            index,
            hints: Hints::default().full(true),
        };
        let f = move |(distance, payload)| (opfamily.output(distance), payload);
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2S) => {
                    type Op = operator::Op<VectOwned<f32>, L2S>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    type Op = operator::Op<VectOwned<f32>, Dot>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::L2S) => {
                    type Op = operator::Op<VectOwned<f16>, L2S>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    type Op = operator::Op<VectOwned<f16>, Dot>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => default_search::<_, Op>(
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher = SimplePrefetcher::new(index, sequence);
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let predicate =
                                id_0(move |(_, AlwaysEqual(PackedRefMut4((pointer, _, _))))| {
                                    let (key, _) = pointer_to_kv(*pointer);
                                    let Some(mut tuple) = fetcher.fetch(key) else {
                                        return false;
                                    };
                                    tuple.filter()
                                });
                            let sequence = filter(sequence, predicate);
                            let prefetcher =
                                StreamPrefetcher::new(index, sequence, Hints::default());
                            Box::new(rerank_index::<Op, _, _, _>(unprojected, prefetcher).map(f))
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap::<Op, _, _, _>(unprojected, prefetcher, fetch).map(f),
                            )
                        }
                    }
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

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut4<'a, (B, C, D)>>)) -> R,
{
    f
}
