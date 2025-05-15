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
use crate::{Bump, Page, Prefetcher, PrefetcherHeapFamily, RelationRead, RelationWrite, tape};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Extra<'b> = &'b mut (u32, u16, &'b mut [u32]);

pub fn insert_vector<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    payload: NonZero<u64>,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
) -> (Vec<u32>, u16) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let vectors_first = meta_tuple.vectors_first();

    drop(meta_guard);

    if !rerank_in_heap {
        vectors::append::<O>(index, vectors_first, vector, payload)
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
) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    let epsilon = 1.9;

    type State = (Reverse<Distance>, AlwaysEqual<u32>);
    let mut state: State = if !is_residual {
        let first = meta_tuple.first();
        // it's safe to leave it a fake value
        (Reverse(Distance::ZERO), AlwaysEqual(first))
    } else {
        let prefetch = bump.alloc_slice(meta_tuple.centroid_prefetch());
        let head = meta_tuple.centroid_head();
        let first = meta_tuple.first();
        let distance = vectors::read_for_h1_tuple::<R, O, _>(
            prefetch.iter().map(|&id| index.read(id)),
            head,
            LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
        );
        (Reverse(distance), AlwaysEqual(first))
    };

    drop(meta_guard);
    let lut = (O::Vector::block_preprocess(vector),);

    let mut step = |state: State| {
        let mut results = LinkedVec::<(Reverse<Distance>, AlwaysEqual<Extra<'b>>)>::new();
        {
            let (Reverse(dis_f), AlwaysEqual(first)) = state;
            let process = |value, code, delta, mut lut: (_, _, _, _)| {
                lut.0 = std::hint::select_unpredictable(!is_residual, lut.0, dis_f.to_f32());
                O::process(value, code, delta, lut)
            };
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first),
                || RAccess::new((&lut.0.1, lut.0.0), O::block_access(process)),
                |(rough, err), head, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((first, head, bump.alloc_slice(prefetch)))),
                    ));
                },
            );
        }
        let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
        let mut cache = BinaryHeap::<(_, _)>::new();
        {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, head, ..))), prefetch)) =
                heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                let distance = vectors::read_for_h1_tuple::<R, O, _>(
                    prefetch.into_iter(),
                    head,
                    LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                );
                cache.push((Reverse(distance), AlwaysEqual(first)));
            }
            cache.pop()
        }
        .expect("invariant is violated: tree is not height-balanced")
    };

    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let (_, AlwaysEqual(first)) = state;

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
        dis_u_2: code.dis_u_2,
        factor_cnt: code.factor_cnt,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        delta,
        payload: Some(payload),
        prefetch,
        head,
        elements: rabitq::pack_to_u64(&code.signs),
    });

    tape::append(index, jump_tuple.appendable_first(), &serialized, false);
}
