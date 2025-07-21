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
use crate::operator::{CloneAccessor, Operator};
use crate::tuples::{MetaTuple, VectorTuple, VertexTuple, WithReader};
use crate::types::DistanceKind;
use crate::vectors::{by_read, copy_all, copy_nothing, copy_outs, update};
use algo::{Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use std::cmp::Reverse;
use vector::VectorOwned;

pub fn maintain<R: RelationRead + RelationWrite, O: Operator>(index: &R, check: impl Fn())
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let m = meta_tuple.m();
    let alpha = meta_tuple.alpha().to_vec();
    let start = meta_tuple.start();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let Some(s) = start.into_inner() else {
        return;
    };
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
                    if vertex_tuple.payload().is_some() {
                        let pointers_u = vertex_tuple.pointers().to_vec();
                        members.push((pointers_u, (vertex_guard.id(), i)));
                    }
                }
            }
            let next = { vertex_guard }.get_opaque().next;
            for (pointers_u, u) in members {
                'occ: loop {
                    let Ok((vector_u, neighbours_u, _, version)) = crate::vectors::read::<R, O, _, _>(
                        by_read::<R>(index, pointers_u.iter().copied()),
                        CloneAccessor::<O::Vector>::default(),
                        copy_all,
                    ) else {
                        // the link is broken
                        break 'occ;
                    };
                    let trace = {
                        let mut trace = Vec::new();
                        let mut extend = Vec::new();
                        for &(v, dis_v) in neighbours_u.iter() {
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                // the link is broken
                                continue;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointers_v = vertex_tuple.pointers().to_vec();
                            let payload_v = vertex_tuple.payload();
                            drop(vertex_guard);
                            let Ok((vector_v, outs_v, _, _)) = crate::vectors::read::<R, O, _, _>(
                                by_read::<R>(index, pointers_v.iter().copied()),
                                CloneAccessor::<O::Vector>::default(),
                                copy_outs,
                            ) else {
                                // the link is broken
                                continue;
                            };
                            if payload_v.is_some() {
                                trace.push((
                                    (Reverse(dis_v), AlwaysEqual((pointers_v, v))),
                                    vector_v,
                                ));
                            } else {
                                extend.extend(outs_v.iter().copied());
                            }
                        }
                        for v in extend {
                            let vertex_guard = index.read(v.0);
                            let Some(vertex_bytes) = vertex_guard.get(v.1) else {
                                // the link is broken
                                continue;
                            };
                            let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                            let pointers_v = vertex_tuple.pointers().to_vec();
                            let payload_v = vertex_tuple.payload();
                            drop(vertex_guard);
                            let Ok((vector_v, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                                by_read::<R>(index, pointers_v.iter().copied()),
                                CloneAccessor::<O::Vector>::default(),
                                copy_nothing,
                            ) else {
                                // the link is broken
                                continue;
                            };
                            if payload_v.is_some() {
                                let dis_v =
                                    O::distance(vector_u.as_borrowed(), vector_v.as_borrowed());
                                trace.push((
                                    (Reverse(dis_v), AlwaysEqual((pointers_v, v))),
                                    vector_v,
                                ));
                            }
                        }
                        trace
                    };
                    let outs = crate::prune::prune(
                        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                        (pointers_u.to_vec(), u),
                        trace.into_iter(),
                        m,
                        &alpha,
                        |(_, u)| *u,
                        O::DISTANCE == DistanceKind::L2S,
                    );
                    if update::<R, O>(
                        (index, pointers_u.as_slice()),
                        (version, neighbours_u),
                        outs.iter()
                            .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
                    ) != Ok(false)
                    {
                        break 'occ;
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
                    if p.is_none() && (current, i) != s {
                        vertex_guard.free(i);
                    }
                };
            }
            current = vertex_guard.get_opaque().next;
            {
                let mut current = { vertex_guard }.get_opaque().link;
                while current != u32::MAX {
                    check();
                    let mut vector_guard = index.write(current, false);
                    for i in 1..=vector_guard.len() {
                        if let Some(bytes) = vector_guard.get(i) {
                            use crate::tuples::VectorTupleReader;
                            let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                            match tuple {
                                VectorTupleReader::_0(tuple) => {
                                    let p = tuple.payload();
                                    if p.is_none() && (current, i) != s {
                                        vector_guard.free(i);
                                    }
                                }
                                VectorTupleReader::_1(tuple) => {
                                    let p = tuple.payload();
                                    if p.is_none() && (current, i) != s {
                                        vector_guard.free(i);
                                    }
                                }
                            }
                        };
                    }
                    current = vector_guard.get_opaque().next;
                }
            }
        }
    }
}
