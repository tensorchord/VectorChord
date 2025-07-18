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
use crate::operator::{CloneAccessor, Operator, Vector};
use crate::results::Results;
use crate::tuples::*;
use crate::types::DistanceKind;
use crate::vectors::{by_prefetch, by_read, copy_all, copy_nothing, copy_outs, update};
use crate::visited::Visited;
use algo::accessor::LAccess;
use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Bump, Page, PageGuard, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use rabitq::bits::Bits;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::num::{NonZero, Wrapping};
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<'b, R: RelationRead + RelationWrite, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'b>,
    payload: NonZero<u64>,
    bump: &'b impl Bump,
    mut prefetch_vertices: impl PrefetcherSequenceFamily<'b, R> + 'b,
    prefetch_vectors: impl PrefetcherSequenceFamily<'b, R> + 'b,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let start = meta_tuple.start();
    let bits = Bits::try_from(meta_tuple.bits()).expect("data corruption");
    let m = meta_tuple.m();
    let max_alpha = meta_tuple.max_alpha();
    let ef = meta_tuple.ef_construction();
    let beam = meta_tuple.beam_construction();
    let skip = meta_tuple.skip();
    drop(meta_guard);
    let version_t = Wrapping(rand::random());
    let (pointers_t, t) = {
        let list_of_vector_bytes = {
            let (left, right) = O::Vector::split(vector, m as _);
            let left = left.into_iter().enumerate().map(|(index, elements)| {
                VectorTuple::serialize(&VectorTuple::<O::Vector>::_1 {
                    payload: Some(payload),
                    elements: elements.to_vec(),
                    index: index as u32,
                })
            });
            let right = VectorTuple::serialize(&VectorTuple::<O::Vector>::_0 {
                payload: Some(payload),
                elements: right.0.to_vec(),
                metadata: right.1,
                neighbours: vec![OptionNeighbour::NONE; m as usize],
                version: version_t,
            });
            left.chain(std::iter::once(right)).collect::<Vec<_>>()
        };
        let mut vertex_bytes = {
            let code = O::Vector::code(bits, vector);
            VertexTuple::serialize(&VertexTuple {
                metadata: code.0.into_array(),
                payload: Some(payload),
                elements: rabitq::bits::pack_code(bits, &code.1),
                pointers: vec![Pointer::new((u32::MAX, 0)); list_of_vector_bytes.len()], // a sentinel value
            })
        };
        let mut vertex_guard = if let Some(guard) = index.search(vertex_bytes.len()) {
            guard
        } else {
            append_vertex_tuple(index, skip, vertex_bytes.len())
        };
        let link = vertex_guard.get_opaque().link;
        let pointers_t = list_of_vector_bytes
            .iter()
            .map(|vector_bytes| {
                let mut vector_guard = append_vector_tuple(index, link, vector_bytes.len());
                let i = vector_guard
                    .alloc(vector_bytes)
                    .expect("implementation: a free page cannot accommodate a single tuple");
                Pointer::new((vector_guard.id(), i))
            })
            .collect::<Vec<_>>();
        {
            let mut vertex_tuple = VertexTuple::deserialize_mut(&mut vertex_bytes);
            vertex_tuple.pointers().copy_from_slice(&pointers_t);
        }
        let t = {
            let i = vertex_guard
                .alloc(&vertex_bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            (vertex_guard.id(), i)
        };
        drop(vertex_guard);
        if skip < t.0 {
            let mut meta_guard = index.write(0, false);
            let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
            let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
            *meta_tuple.skip() = (*meta_tuple.skip()).max(t.0);
        }
        (pointers_t, t)
    };
    let start = if start.into_inner().is_none() {
        let mut meta_guard = index.write(0, false);
        let meta_bytes = meta_guard.get_mut(1).expect("data corruption");
        let mut meta_tuple = MetaTuple::deserialize_mut(meta_bytes);
        let dst = meta_tuple.start();
        if dst.into_inner().is_none() {
            *dst = OptionPointer::some(t);
            return;
        } else {
            *dst
        }
    } else {
        start
    };
    let lut = O::Vector::preprocess(vector);
    let mut visited = Visited::new();
    let mut candidates = Candidates::new(beam as usize, prefetch_vectors);
    let Some(s) = start.into_inner() else {
        return;
    };
    {
        visited.insert(s);
        let vertex_guard = index.read(s.0);
        let Some(vertex_bytes) = vertex_guard.get(s.1) else {
            // the link is broken
            return;
        };
        let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
        let pointers_s = bump.alloc_slice(vertex_tuple.pointers());
        let score_s = O::process(
            bits,
            dims,
            (vertex_tuple.metadata(), vertex_tuple.elements()),
            &lut,
        );
        candidates.push((Reverse(score_s), AlwaysEqual((pointers_s, s))));
    }
    let mut iter = std::iter::from_fn(|| {
        while let Some(((_, AlwaysEqual((pointers_u, u))), guards)) = candidates.pop() {
            let Ok((dis_u, outs_u, _, _)) = crate::vectors::read::<R, O, _, _>(
                by_prefetch::<R>(guards, pointers_u.iter().copied()),
                LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                copy_outs,
            ) else {
                // the link is broken
                continue;
            };
            let mut iterator = prefetch_vertices.prefetch(
                outs_u
                    .into_iter()
                    .filter(|&x| !visited.contains(x))
                    .collect::<VecDeque<_>>(),
            );
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
                let pointers_v = bump.alloc_slice(vertex_tuple.pointers());
                let score_v = O::process(
                    bits,
                    dims,
                    (vertex_tuple.metadata(), vertex_tuple.elements()),
                    &lut,
                );
                candidates.push((Reverse(score_v), AlwaysEqual((pointers_v, v))));
            }
            return Some((Reverse(dis_u), AlwaysEqual((pointers_u, u))));
        }
        None
    });
    let mut results = Results::new(ef as _);
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
    let trace = {
        let (left, right) = results.into_inner();
        left.into_iter()
            .map(|(dis_v, v)| (Reverse(dis_v), v))
            .chain(right)
            .flat_map(|item| {
                let (Reverse(dis_u), AlwaysEqual((pointers_u, u))) = item;
                let Ok((vector_u, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                    by_read(index, pointers_u.iter().copied()),
                    CloneAccessor::<O::Vector>::default(),
                    copy_nothing,
                ) else {
                    // the link is broken
                    return None;
                };
                Some(((Reverse(dis_u), AlwaysEqual((pointers_u, u))), vector_u))
            })
            .collect::<Vec<_>>()
    };
    let outs = crate::prune::prune(
        |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
        (bump.alloc_slice(&pointers_t), t),
        trace.into_iter(),
        m,
        max_alpha,
        |(_, u)| *u,
        O::DISTANCE == DistanceKind::L2S,
    );
    let _ = update::<R, O>(
        (index, pointers_t.as_slice()),
        (version_t, VecDeque::new()),
        outs.iter()
            .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
    );
    for (Reverse(dis_t), AlwaysEqual((pointers_u, u))) in outs {
        'occ: loop {
            let Ok((neighbours_u, _, version)) =
                crate::vectors::read_without_accessor::<R, O, _>((index, pointers_u), copy_all)
            else {
                // the link is broken
                break 'occ;
            };
            let trace = neighbours_u
                .iter()
                .copied()
                .chain(std::iter::once((t, dis_t)))
                .map(|(v, dis_v)| (Reverse(dis_v), AlwaysEqual(v)))
                .flat_map(|item| {
                    let (Reverse(dis_u), AlwaysEqual(u)) = item;
                    let vertex_guard = index.read(u.0);
                    let Some(vertex_bytes) = vertex_guard.get(u.1) else {
                        // the link is broken
                        return None;
                    };
                    let vertex_tuple = VertexTuple::deserialize_ref(vertex_bytes);
                    let pointers_u = vertex_tuple.pointers().to_vec();
                    Some((Reverse(dis_u), AlwaysEqual((pointers_u, u))))
                })
                .flat_map(|item| {
                    let (Reverse(dis_u), AlwaysEqual((pointers_u, u))) = item;
                    let Ok((vector_u, _, _, _)) = crate::vectors::read::<R, O, _, _>(
                        by_read(index, pointers_u.iter().copied()),
                        CloneAccessor::<O::Vector>::default(),
                        copy_nothing,
                    ) else {
                        // the link is broken
                        return None;
                    };
                    Some(((Reverse(dis_u), AlwaysEqual((pointers_u, u))), vector_u))
                })
                .collect::<Vec<_>>();
            let outs = crate::prune::prune(
                |x, y| O::distance(x.as_borrowed(), y.as_borrowed()),
                (pointers_u.to_vec(), u),
                trace.into_iter(),
                m,
                max_alpha,
                |(_, u)| *u,
                O::DISTANCE == DistanceKind::L2S,
            );
            if update::<R, O>(
                (index, pointers_u),
                (version, neighbours_u),
                outs.iter()
                    .map(|&(Reverse(dis_u), AlwaysEqual((_, u)))| (u, dis_u)),
            ) != Ok(false)
            {
                break 'occ;
            }
        }
    }
}

fn append_vertex_tuple<'r, R: RelationRead + RelationWrite>(
    index: &'r R,
    first: u32,
    size: usize,
) -> R::WriteGuard<'r>
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, true);
            if write.freespace() as usize >= size {
                return write;
            }
            if write.get_opaque().next == u32::MAX {
                let link = index
                    .extend(
                        Opaque {
                            next: u32::MAX,
                            link: u32::MAX,
                        },
                        false,
                    )
                    .id();
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link,
                    },
                    true,
                );
                { write }.get_opaque_mut().next = extend.id();
                if extend.freespace() as usize >= size {
                    return extend;
                } else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
}

fn append_vector_tuple<'r, R: RelationRead + RelationWrite>(
    index: &'r R,
    first: u32,
    size: usize,
) -> R::WriteGuard<'r>
where
    R::Page: Page<Opaque = Opaque>,
{
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= size || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, false);
            if write.freespace() as usize >= size {
                return write;
            }
            if write.get_opaque().next == u32::MAX {
                let extend = index.extend(
                    Opaque {
                        next: u32::MAX,
                        link: u32::MAX,
                    },
                    false,
                );
                { write }.get_opaque_mut().next = extend.id();
                if extend.freespace() as usize >= size {
                    return extend;
                } else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
                }
            }
            current = write.get_opaque().next
        } else {
            current = read.get_opaque().next
        }
    }
}
