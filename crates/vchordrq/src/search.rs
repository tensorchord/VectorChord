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

use crate::closure_lifetime_binder::id_2;
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::{by_directory, by_next};
use crate::tuples::*;
use crate::{Opaque, Page, tape, vectors};
use algo::accessor::LAccess;
use algo::prefetcher::{Prefetcher, PrefetcherHeapFamily, PrefetcherSequenceFamily};
use algo::{Bump, RelationRead};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Extra<'b> = &'b mut (NonZero<u64>, u16, &'b mut [u32]);

pub fn default_search<'r, 'b: 'r, R: RelationRead, O: Operator>(
    index: &'r R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'r, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'r, R>,
) -> Vec<((Reverse<Distance>, AlwaysEqual<()>), AlwaysEqual<Extra<'b>>)>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }

    type State = Vec<(Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>)>;
    let mut state: State = if !is_residual {
        let first = meta_tuple.first();
        // it's safe to leave it a fake value
        vec![(
            Reverse(Distance::ZERO),
            AlwaysEqual(0.0),
            AlwaysEqual(first),
        )]
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
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    };

    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);

    let mut step = |state: State| {
        let mut results = LinkedVec::new();
        for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
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
        std::iter::from_fn(move || {
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
        })
    };

    for i in (1..height_of_root).rev() {
        state = step(state).take(probes[i as usize - 1] as _).collect();
    }

    let mut results = LinkedVec::new();
    for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
        let block_process = |value, code, delta, lut| {
            O::block_process(value, code, lut, is_residual, dis_f.to_f32(), delta, norm)
        };
        let binary_process = |value, code, delta, lut| {
            O::binary_process(value, code, lut, is_residual, dis_f.to_f32(), delta, norm)
        };
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_2(|(rough, err), head, payload, prefetch| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(())),
                AlwaysEqual(bump.alloc((payload, head, bump.alloc_slice(prefetch)))),
            ));
        });
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory),
                || O::block_access(&lut.0, block_process),
                &mut callback,
            );
        } else {
            tape::read_frozen_tape::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                || O::block_access(&lut.0, block_process),
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, binary_process),
            &mut callback,
        );
    }
    results.into_vec()
}

pub fn maxsim_search<'r, 'b: 'r, R: RelationRead, O: Operator>(
    index: &'r R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    mut threshold: u32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'r, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'r, R>,
) -> (
    Vec<(
        (Reverse<Distance>, AlwaysEqual<Distance>),
        AlwaysEqual<Extra<'b>>,
    )>,
    Distance,
)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.dims(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }

    type State = Vec<(Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>)>;
    let mut state: State = if !is_residual {
        let first = meta_tuple.first();
        // it's safe to leave it a fake value
        vec![(
            Reverse(Distance::ZERO),
            AlwaysEqual(0.0),
            AlwaysEqual(first),
        )]
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
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    };

    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);

    let mut step = |state: State| {
        let mut results = LinkedVec::new();
        for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
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
        std::iter::from_fn(move || {
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
        })
    };

    let mut it = None;
    for i in (1..height_of_root).rev() {
        let it = it.insert(step(state));
        state = it.take(probes[i as usize - 1] as _).collect();
    }

    let mut results = LinkedVec::new();
    for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
        let block_process = |value, code, delta, lut| {
            O::block_process(value, code, lut, is_residual, dis_f.to_f32(), delta, norm)
        };
        let binary_process = |value, code, delta, lut| {
            O::binary_process(value, code, lut, is_residual, dis_f.to_f32(), delta, norm)
        };
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_2(|(rough, err), head, payload, prefetch| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            let rough = Distance::from_f32(rough);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(rough)),
                AlwaysEqual(bump.alloc((payload, head, bump.alloc_slice(prefetch)))),
            ));
        });
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory),
                || O::block_access(&lut.0, block_process),
                &mut callback,
            );
        } else {
            tape::read_frozen_tape::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                || O::block_access(&lut.0, block_process),
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, binary_process),
            &mut callback,
        );
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
    }
    let mut estimation_by_threshold = Distance::NEG_INFINITY;
    for (Reverse(distance), AlwaysEqual(_), AlwaysEqual(first)) in it.into_iter().flatten() {
        if threshold == 0 {
            break;
        }
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
        estimation_by_threshold = distance;
    }
    (results.into_vec(), estimation_by_threshold)
}
