use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tuples::*;
use crate::{IndexPointer, Page, RelationRead, tape, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Result<T> = (
    Reverse<Distance>,
    AlwaysEqual<T>,
    AlwaysEqual<NonZero<u64>>,
    AlwaysEqual<IndexPointer>,
);

pub fn default_search<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
) -> Vec<Result<()>> {
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
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::preprocess(vector.as_borrowed()))
    } else {
        None
    };

    type State<O> = Vec<(u32, Option<<O as Operator>::Vector>)>;
    let mut state: State<O> = vec![{
        let mean = root_mean;
        if is_residual {
            let residual_u = vectors::read_for_h1_tuple::<O, _>(
                index.clone(),
                mean,
                LAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual_u))
        } else {
            (root_first, None)
        }
    }];
    let step = |state: State<O>| {
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
                |(rough, err), mean, first| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((Reverse(lowerbound), AlwaysEqual(first), AlwaysEqual(mean)));
                },
                |_| (),
            );
        }
        let mut heap = BinaryHeap::from(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        let index = index.clone();
        let vector = vector.as_borrowed();
        std::iter::from_fn(move || {
            while let Some((Reverse(_), AlwaysEqual(first), AlwaysEqual(mean))) =
                pop_if(&mut heap, |(d, ..)| {
                    Some(*d) > cache.peek().map(|(d, ..)| *d)
                })
            {
                if is_residual {
                    let (dis_u, residual_u) = vectors::read_for_h1_tuple::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::unpack(vector),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(dis_u),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual_u)),
                    ));
                } else {
                    let dis_u = vectors::read_for_h1_tuple::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                    );
                    cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean))
        })
    };
    for i in (1..height_of_root).rev() {
        state = step(state).take(probes[i as usize - 1] as usize).collect();
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
        let mut callback = |(rough, err), mean, payload| {
            let lowerbound = Distance::from_f32(rough - err * 1.9);
            results.push((
                Reverse(lowerbound),
                AlwaysEqual(()),
                AlwaysEqual(payload),
                AlwaysEqual(mean),
            ));
        };
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

pub fn maxsim_search<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
    mut t: u32,
) -> (Vec<Result<Distance>>, Distance) {
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
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::preprocess(vector.as_borrowed()))
    } else {
        None
    };

    type State<O> = Vec<(u32, Option<<O as Operator>::Vector>)>;
    let mut state: State<O> = vec![{
        let mean = root_mean;
        if is_residual {
            let residual_u = vectors::read_for_h1_tuple::<O, _>(
                index.clone(),
                mean,
                LAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual_u))
        } else {
            (root_first, None)
        }
    }];
    let step = |state: State<O>| {
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
                |(rough, err), mean, first| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((Reverse(lowerbound), AlwaysEqual(first), AlwaysEqual(mean)));
                },
                |_| (),
            );
        }
        let mut heap = BinaryHeap::from(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        let index = index.clone();
        let vector = vector.as_borrowed();
        std::iter::from_fn(move || {
            while let Some((Reverse(_), AlwaysEqual(first), AlwaysEqual(mean))) =
                pop_if(&mut heap, |(d, ..)| {
                    Some(*d) > cache.peek().map(|(d, ..)| *d)
                })
            {
                if is_residual {
                    let (dis_u, residual_u) = vectors::read_for_h1_tuple::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::unpack(vector),
                            (
                                O::DistanceAccessor::default(),
                                O::ResidualAccessor::default(),
                            ),
                        ),
                    );
                    cache.push((
                        Reverse(dis_u),
                        AlwaysEqual(first),
                        AlwaysEqual(Some(residual_u)),
                    ));
                } else {
                    let dis_u = vectors::read_for_h1_tuple::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(O::Vector::unpack(vector), O::DistanceAccessor::default()),
                    );
                    cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (Reverse(distance), AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean, distance))
        })
    };
    for i in (2..height_of_root).rev() {
        state = step(state)
            .map(|(x, y, _)| (x, y))
            .take(probes[i as usize - 1] as usize)
            .collect();
    }
    let mut iter: Box<dyn Iterator<Item = (u32, Distance)>>;
    if height_of_root > 1 {
        let mut it = step(state);
        state = std::iter::from_fn(|| it.next())
            .map(|(x, y, _)| (x, y))
            .take(probes[0] as usize)
            .collect();
        iter = Box::new(it.map(|(x, _, z)| (x, z)));
    } else {
        iter = Box::new(std::iter::empty());
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
        let mut callback = |(rough, err), mean, payload| {
            let lowerbound = Distance::from_f32(rough - err * epsilon);
            let rough = Distance::from_f32(rough);
            results.push((
                Reverse(lowerbound),
                AlwaysEqual(rough),
                AlwaysEqual(payload),
                AlwaysEqual(mean),
            ));
        };
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
        t = t.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as u32);
    }
    let mut estimation = f32::NAN;
    loop {
        if t != 0 {
            if let Some((first, distance)) = iter.next() {
                let jump_guard = index.read(first);
                let jump_bytes = jump_guard.get(1).expect("data corruption");
                let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
                t = t.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as u32);
                estimation = distance.to_f32();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    (results.into_vec(), Distance::from_f32(estimation))
}

fn pop_if<T: Ord>(
    heap: &mut BinaryHeap<T>,
    mut predicate: impl FnMut(&mut T) -> bool,
) -> Option<T> {
    use std::collections::binary_heap::PeekMut;
    let mut peek = heap.peek_mut()?;
    if predicate(&mut peek) {
        Some(PeekMut::pop(peek))
    } else {
        None
    }
}
