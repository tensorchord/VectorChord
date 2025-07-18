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

use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::by_next;
use crate::tuples::*;
use crate::vectors::{self};
use crate::{Opaque, Page, tape};
use algo::accessor::{FunctionalAccessor, LAccess};
use algo::prefetcher::{Prefetcher, PrefetcherHeapFamily};
use algo::{Bump, RelationRead, RelationWrite};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Extra<'b> = &'b mut (u32, u16, f32, &'b mut [u32]);

pub fn insert_vector<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    payload: NonZero<u64>,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> (Vec<u32>, u16)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let vectors_first = meta_tuple.vectors_first();

    drop(meta_guard);

    if !rerank_in_heap {
        vectors::append::<O, R>(index, vectors_first, vector, payload)
    } else {
        (Vec::new(), 0)
    }
}

pub fn insert<'r, 'b: 'r, R: RelationRead + RelationWrite, O: Operator>(
    index: &'r R,
    payload: NonZero<u64>,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    key: (Vec<u32>, u16),
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'r, R>,
    skip_freespaces: bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    let freepages_first = meta_tuple.freepages_first();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let epsilon = 1.9;

    type State = (Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>);
    let mut state: State = if !is_residual {
        let first = meta_tuple.first();
        // it's safe to leave it a fake value
        (
            Reverse(Distance::ZERO),
            AlwaysEqual(0.0),
            AlwaysEqual(first),
        )
    } else {
        let prefetch = bump.alloc_slice(meta_tuple.centroid_prefetch());
        let head = meta_tuple.centroid_head();
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        let distance = vectors::read_for_h1_tuple::<R, O, _>(
            prefetch.iter().map(|&id| index.read(id)),
            head,
            LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
        );
        (Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))
    };

    drop(meta_guard);
    let lut = (O::Vector::block_preprocess(vector),);

    let mut step = |state: State| {
        let mut results = LinkedVec::<(_, AlwaysEqual<Extra<'b>>)>::new();
        {
            let (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) = state;
            let process = |value, code, delta, lut| {
                O::block_process(value, code, lut, is_residual, dis_f.to_f32(), delta, norm)
            };
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first),
                || O::block_access(&lut.0, process),
                |(rough, err), head, norm, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((first, head, norm, bump.alloc_slice(prefetch)))),
                    ));
                },
            );
        }
        let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
        let mut cache = BinaryHeap::<(_, _, _)>::new();
        {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, head, norm, ..))), prefetch)) =
                heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                let distance = vectors::read_for_h1_tuple::<R, O, _>(
                    prefetch,
                    head,
                    LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                );
                cache.push((Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first)));
            }
            cache.pop()
        }
        .expect("invariant is violated: tree is not height-balanced")
    };

    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let (_, _, AlwaysEqual(first)) = state;

    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);

    let (code, delta) = O::build(
        vector,
        is_residual.then(|| {
            vectors::read_for_h1_tuple::<R, O, _>(
                jump_tuple
                    .centroid_prefetch()
                    .iter()
                    .map(|&id| index.read(id)),
                jump_tuple.centroid_head(),
                FunctionalAccessor::new(Vec::new(), Vec::extend_from_slice, O::Vector::pack),
            )
        }),
    );

    let (prefetch, head) = key;
    let serialized = AppendableTuple::serialize(&AppendableTuple {
        metadata: [
            code.0.dis_u_2,
            code.0.factor_cnt,
            code.0.factor_ip,
            code.0.factor_err,
        ],
        delta,
        payload: Some(payload),
        prefetch,
        head,
        elements: rabitq::bit::binary::pack_code(&code.1),
    });

    tape::append(
        index,
        jump_tuple.appendable_first(),
        &serialized,
        false,
        (!skip_freespaces).then_some(freepages_first),
    );
}
