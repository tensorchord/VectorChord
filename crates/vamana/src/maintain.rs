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
use crate::operator::{Operator, Vector};
use crate::tuples::{MetaTuple, OptionNeighbour, VectorTuple, VertexTuple, WithReader, WithWriter};
use algo::{Page, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use std::cmp::Reverse;
use std::iter::{repeat, zip};
use vector::VectorOwned;

pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(index: &R, check: impl Fn())
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha();
    let start = meta_tuple.start().into_inner();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    // do it's best to remove broken edges
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let vertex_guard = index.read(current);
            let mut members = Vec::new();
            for i in 1..=vertex_guard.len() {
                if let Some(vertex_bytes) = vertex_guard.get(i) {
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    if let Some(payload) = vertex_tuple.payload() {
                        let pointer = vertex_tuple.pointer().into_inner();
                        let vector;
                        let neighbours;
                        {
                            let vector_guard = index.read(pointer.0);
                            let vector_bytes =
                                vector_guard.get(pointer.1).expect("data corruption");
                            let vector_tuple =
                                VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                            vector = O::Vector::pack(
                                vector_tuple.elements().to_vec(),
                                *vector_tuple.metadata(),
                            );
                            neighbours = vector_tuple
                                .neighbours()
                                .iter()
                                .flat_map(|neighbour| neighbour.into_inner())
                                .collect::<Vec<_>>();
                        }
                        members.push((i, payload, pointer, vector, neighbours));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            while !members.is_empty() {
                for (i, payload_u, pointer_u, vector_u, neighbours_u) in
                    std::mem::take(&mut members)
                {
                    let u = (current, i);
                    let mut trace = Vec::new();
                    let mut bad = Vec::new();
                    for &(v, dis_v) in neighbours_u.iter() {
                        let vertex_guard = index.read(v.0);
                        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                            // the link is broken
                            continue;
                        };
                        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                        let pointer_v = vertex_tuple.pointer().into_inner();
                        let payload_v = vertex_tuple.payload();
                        let vector_guard = index.read(pointer_v.0);
                        let vector_bytes = vector_guard.get(pointer_v.1).expect("data corruption");
                        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                        let vector_v = O::Vector::pack(
                            vector_tuple.elements().to_vec(),
                            *vector_tuple.metadata(),
                        );
                        drop(vertex_guard);
                        if payload_v.is_some() {
                            trace.push((v, dis_v, vector_v));
                        } else {
                            let outs_v = vector_tuple
                                .neighbours()
                                .iter()
                                .flat_map(|neighbour| neighbour.into_inner())
                                .map(|(v, _)| v)
                                .collect::<Vec<_>>();
                            drop(vector_guard);
                            bad.extend(outs_v);
                        }
                    }
                    for v in bad {
                        let vertex_guard = index.read(v.0);
                        let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                            // the link is broken
                            continue;
                        };
                        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                        let pointer_v = vertex_tuple.pointer().into_inner();
                        let payload_v = vertex_tuple.payload();
                        let vector_guard = index.read(pointer_v.0);
                        let vector_bytes = vector_guard.get(pointer_v.1).expect("data corruption");
                        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
                        let vector_v = O::Vector::pack(
                            vector_tuple.elements().to_vec(),
                            *vector_tuple.metadata(),
                        );
                        drop(vertex_guard);
                        if payload_v.is_some() {
                            let dis_v = O::distance(vector_u.as_borrowed(), vector_v.as_borrowed());
                            trace.push((v, dis_v, vector_v));
                        }
                    }
                    let outs = crate::prune::robust_prune(
                        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                        (pointer_u, u),
                        trace.into_iter().flat_map(|item| {
                            let (u, dis_u, vector_u) = item;
                            let vertex_guard = index.read(u.0);
                            let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                                // the link is broken
                                return None;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointer_u = vertex_tuple.pointer().into_inner();
                            Some(((Reverse(dis_u), AlwaysEqual((pointer_u, u))), vector_u))
                        }),
                        m,
                        alpha,
                    );
                    let mut vector_guard = index.write(pointer_u.0, false);
                    let vector_bytes = vector_guard.get_mut(pointer_u.1).expect("data corruption");
                    let mut vector_tuple = VectorTuple::<O::Vector>::deserialize_mut(vector_bytes);
                    let check_payload = *vector_tuple.payload() == Some(payload_u);
                    let check_neighbours = vector_tuple
                        .neighbours()
                        .iter()
                        .flat_map(|neighbour| neighbour.into_inner())
                        .eq(neighbours_u.clone());
                    if check_payload && check_neighbours {
                        let iterator = outs
                            .iter()
                            .map(|&(Reverse(dis_v), AlwaysEqual((_, v)))| {
                                OptionNeighbour::some(v, dis_v)
                            })
                            .chain(repeat(OptionNeighbour::NONE));
                        for (hole, fill) in zip(vector_tuple.neighbours().iter_mut(), iterator) {
                            *hole = fill;
                        }
                    } else {
                        members.push((i, payload_u, pointer_u, vector_u, neighbours_u));
                    }
                }
            }
            current = next;
        }
    }
    // remove vertices and vectors
    {
        let mut current = link;
        while current != u32::MAX {
            check();
            let mut vertex_guard = index.write(current, true);
            for i in 1..=vertex_guard.len() {
                if let Some(bytes) = vertex_guard.get(i) {
                    let tuple = VertexTuple::deserialize_ref(bytes);
                    let p = tuple.payload();
                    if p.is_none() && Some((current, i)) != start {
                        vertex_guard.free(i);
                    }
                };
            }
            current = vertex_guard.get_opaque().next;
            {
                let mut current = { vertex_guard }.get_opaque().link;
                while current != u32::MAX {
                    check();
                    let mut vector_tuple = index.write(current, true);
                    for i in 1..=vector_tuple.len() {
                        if let Some(bytes) = vector_tuple.get(i) {
                            let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                            let p = tuple.payload();
                            if p.is_none() && Some((current, i)) != start {
                                vector_tuple.free(i);
                            }
                        };
                    }
                    current = vector_tuple.get_opaque().next;
                }
            }
        }
    }
}
