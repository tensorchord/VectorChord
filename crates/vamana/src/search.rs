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

use crate::Opaque;
use crate::candidates::Candidates;
use crate::operator::{Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::visited::Visited;
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Page, RelationRead};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    ef_search: u32,
    beam_search: u32,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) -> Box<dyn Iterator<Item = (Distance, NonZero<u64>)> + 'b>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let start = meta_tuple.start();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let ef = ef_search;
    let beam = beam_search;
    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return Box::new(std::iter::empty());
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return Box::new(std::iter::empty());
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointer_s = vertex_tuple.pointer().into_inner();
        let score_s = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
        candidates.push((Reverse(score_s), AlwaysEqual(pointer_s)));
    }
    let mut iter = std::iter::from_fn(move || {
        while let Some(((_, AlwaysEqual(pointer_u)), guards)) = candidates.pop() {
            let vector_guard = {
                let mut guards = guards;
                let r = guards.next().expect("internal");
                assert!(guards.next().is_none(), "internal");
                drop(guards);
                r
            };
            let Some(vector_bytes) = vector_guard.get(pointer_u.1) else {
                // the link is broken
                continue;
            };
            let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
            let outs_u = vector_tuple
                .neighbours()
                .iter()
                .flat_map(|neighbour| neighbour.into_inner())
                .map(|(v, _)| v)
                .filter(|&v| !visited.contains(v))
                .collect::<VecDeque<_>>();
            let dis_u = O::distance(
                O::Vector::pack(vector_tuple.elements().to_vec(), *vector_tuple.metadata())
                    .as_borrowed(),
                vector,
            );
            let payload_u = vector_tuple.payload();
            drop(vector_guard);
            let mut iterator = prefetch_vertices.prefetch(outs_u);
            while let Some((v, guards)) = iterator.next() {
                visited.insert(v);
                let vertex_guard = {
                    let mut guards = guards;
                    let r = guards.next().expect("internal");
                    assert!(guards.next().is_none(), "internal");
                    drop(guards);
                    r
                };
                let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                    // the link is broken
                    continue;
                };
                let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                let pointer_v = vertex_tuple.pointer().into_inner();
                let score_v = O::process((vertex_tuple.metadata(), vertex_tuple.elements()), &lut);
                candidates.push((Reverse(score_v), AlwaysEqual(pointer_v)));
            }
            return Some((Reverse(dis_u), AlwaysEqual(payload_u)));
        }
        None
    });
    let mut results = Results::new(ef as _);
    let search = std::iter::from_fn(move || {
        for element @ (Reverse(dis_c), _) in iter.by_ref() {
            results.push(element);
            if results
                .peek_ef_th()
                .map(|dis_e| dis_e < dis_c)
                .unwrap_or_default()
            {
                break;
            }
        }
        results.pop_min()
    });
    Box::new(search.filter_map(|(dis_u, AlwaysEqual(payload_u))| Some((dis_u, payload_u?))))
}
