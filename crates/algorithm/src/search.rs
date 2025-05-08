use crate::closure_lifetime_binder::id_2;
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::prefetcher::Prefetcher;
use crate::tuples::*;
use crate::{Bump, Page, RelationRead, tape, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Item<'b> = (
    Reverse<Distance>,
    AlwaysEqual<&'b mut (u32, u16, &'b mut [u32])>,
);

type Extra<'b> = &'b mut (NonZero<u64>, u16, &'b mut [u32]);

pub fn default_search<'b, R: RelationRead, O: Operator, P: Prefetcher<R = R, Item = Item<'b>>>(
    index: R,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
    bump: &'b impl Bump,
    mut prefetch: impl FnMut(Vec<Item<'b>>) -> P,
) -> Vec<((Reverse<Distance>, AlwaysEqual<()>), AlwaysEqual<Extra<'b>>)> {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }
    let root_prefetch = meta_tuple.root_prefetch().to_vec();
    let root_head = meta_tuple.root_head();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::preprocess(vector.as_borrowed()))
    } else {
        None
    };

    type State<O> = Vec<(u32, Option<<O as Operator>::Vector>)>;
    let mut state: State<O> = vec![{
        if is_residual {
            let list = root_prefetch.into_iter().map(|id| index.read(id));
            let residual = vectors::read_for_h1_tuple::<R, O, _>(
                root_head,
                list,
                LAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual))
        } else {
            (root_first, None)
        }
    }];
    let mut step = |state: State<O>| {
        let mut results = LinkedVec::new();
        for (first, residual) in state {
            let block_lut = if let Some(residual) = residual {
                &O::Vector::block_preprocess(residual.as_borrowed())
            } else if let Some((block_lut, _)) = default_lut.as_ref() {
                block_lut
            } else {
                unreachable!()
            };
            tape::read_h1_tape(
                index.clone(),
                first,
                || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
                |(rough, err), head, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((first, head, bump.alloc_slice(prefetch)))),
                    ));
                },
                |_| (),
            );
        }
        let mut heap = (prefetch)(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        let vector = vector.as_borrowed();
        std::iter::from_fn(move || {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, head, ..))), list, _)) =
                heap.pop_if(|(d, ..)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                if is_residual {
                    let (distance, residual) = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(
                            O::Vector::unpack(vector),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(distance),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual)),
                    ));
                } else {
                    let distance = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                    );
                    cache.push((Reverse(distance), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean))
        })
    };
    for i in (1..height_of_root).rev() {
        state = step(state).take(probes[i as usize - 1] as _).collect();
    }

    let mut results = LinkedVec::new();
    for (first, residual) in state {
        let (block_lut, binary_lut) =
            if let Some(residual) = residual.as_ref().map(|x| x.as_borrowed()) {
                &O::Vector::preprocess(residual)
            } else if let Some(lut) = default_lut.as_ref() {
                lut
            } else {
                unreachable!()
            };
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_2(|(rough, err), mean, payload, prefetch| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(())),
                AlwaysEqual(bump.alloc((payload, mean, bump.alloc_slice(prefetch)))),
            ));
        });
        tape::read_frozen_tape(
            index.clone(),
            jump_tuple.frozen_first(),
            || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
            &mut callback,
            |_| (),
        );
        tape::read_appendable_tape(
            index.clone(),
            jump_tuple.appendable_first(),
            |code| O::binary_process(binary_lut, code),
            &mut callback,
            |_| (),
        );
    }
    results.into_vec()
}

pub fn maxsim_search<'b, R: RelationRead, O: Operator, P: Prefetcher<R = R, Item = Item<'b>>>(
    index: R,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
    mut threshold: u32,
    bump: &'b impl Bump,
    mut prefetch: impl FnMut(Vec<Item<'b>>) -> P,
) -> (
    Vec<(
        (Reverse<Distance>, AlwaysEqual<Distance>),
        AlwaysEqual<Extra<'b>>,
    )>,
    Distance,
) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }
    let root_prefetch = meta_tuple.root_prefetch().to_vec();
    let root_head = meta_tuple.root_head();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::preprocess(vector.as_borrowed()))
    } else {
        None
    };

    type State<O> = Vec<(u32, Option<<O as Operator>::Vector>)>;
    let mut state: State<O> = vec![{
        if is_residual {
            let list = root_prefetch.into_iter().map(|id| index.read(id));
            let residual = vectors::read_for_h1_tuple::<R, O, _>(
                root_head,
                list,
                LAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual))
        } else {
            (root_first, None)
        }
    }];
    let mut step = |state: State<O>| {
        let mut results = LinkedVec::new();
        for (first, residual) in state {
            let block_lut = if let Some(residual) = residual {
                &O::Vector::block_preprocess(residual.as_borrowed())
            } else if let Some((block_lut, _)) = default_lut.as_ref() {
                block_lut
            } else {
                unreachable!()
            };
            tape::read_h1_tape(
                index.clone(),
                first,
                || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
                |(rough, err), head, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((first, head, bump.alloc_slice(prefetch)))),
                    ));
                },
                |_| (),
            );
        }
        let mut heap = (prefetch)(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        let vector = vector.as_borrowed();
        std::iter::from_fn(move || {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, head, ..))), list, _)) =
                heap.pop_if(|(d, ..)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                if is_residual {
                    let (distance, residual) = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(
                            O::Vector::unpack(vector),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(distance),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual)),
                    ));
                } else {
                    let distance = vectors::read_for_h1_tuple::<R, O, _>(
                        head,
                        list.into_iter(),
                        LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                    );
                    cache.push((Reverse(distance), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (Reverse(distance), AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean, distance))
        })
    };
    let mut it = None;
    for i in (1..height_of_root).rev() {
        let it = it.insert(step(state)).map(|(first, mean, _)| (first, mean));
        state = it.take(probes[i as usize - 1] as _).collect();
    }

    let mut results = LinkedVec::new();
    for (first, residual) in state {
        let (block_lut, binary_lut) =
            if let Some(residual) = residual.as_ref().map(|x| x.as_borrowed()) {
                &O::Vector::preprocess(residual)
            } else if let Some(lut) = default_lut.as_ref() {
                lut
            } else {
                unreachable!()
            };
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_2(|(rough, err), mean, payload, prefetch| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            let rough = Distance::from_f32(rough);
            results.push((
                (Reverse(lowerbound), AlwaysEqual(rough)),
                AlwaysEqual(bump.alloc((payload, mean, bump.alloc_slice(prefetch)))),
            ));
        });
        tape::read_frozen_tape(
            index.clone(),
            jump_tuple.frozen_first(),
            || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
            &mut callback,
            |_| (),
        );
        tape::read_appendable_tape(
            index.clone(),
            jump_tuple.appendable_first(),
            |code| O::binary_process(binary_lut, code),
            &mut callback,
            |_| (),
        );
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
    }
    let mut estimation_by_threshold = Distance::NEG_INFINITY;
    for (first, _, distance) in it.into_iter().flatten() {
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
