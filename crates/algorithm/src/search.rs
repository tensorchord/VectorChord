use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::pipe::Pipe;
use crate::tape::{access_0, access_1};
use crate::tuples::*;
use crate::{Page, RelationRead, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::{VectorBorrowed, VectorOwned};

pub fn search<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    probes: Vec<u32>,
    epsilon: f32,
) -> (
    RerankMethod,
    Vec<(
        Reverse<Distance>,
        AlwaysEqual<IndexPointer>,
        AlwaysEqual<NonZeroU64>,
    )>,
) {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let default_lut = if !is_residual {
        Some(O::Vector::compute_lut(vector.as_borrowed()))
    } else {
        None
    };

    type State<O> = Vec<(u32, Option<<O as Operator>::Vector>)>;
    let mut state: State<O> = vec![{
        let mean = root_mean;
        if is_residual {
            let residual_u = vectors::access_1::<O, _>(
                index.clone(),
                mean,
                LAccess::new(
                    O::Vector::elements_and_metadata(vector.as_borrowed()),
                    O::ResidualAccessor::default(),
                ),
            );
            (root_first, Some(residual_u))
        } else {
            (root_first, None)
        }
    }];
    let step = |state: State<O>, probes| {
        let mut results = LinkedVec::new();
        for (first, residual) in state {
            let lut = if let Some(residual) = residual {
                &O::Vector::compute_lut_block(residual.as_borrowed())
            } else {
                default_lut.as_ref().map(|x| &x.0).unwrap()
            };
            access_1(
                index.clone(),
                first,
                || {
                    RAccess::new(
                        (&lut.4, (lut.0, lut.1, lut.2, lut.3, epsilon)),
                        O::Distance::block_accessor(),
                    )
                },
                |lowerbound, mean, first| {
                    results.push((Reverse(lowerbound), AlwaysEqual(mean), AlwaysEqual(first)));
                },
            );
        }
        let mut heap = BinaryHeap::from(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        std::iter::from_fn(|| {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                if is_residual {
                    let (dis_u, residual_u) = vectors::access_1::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
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
                    let dis_u = vectors::access_1::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::elements_and_metadata(vector.as_borrowed()),
                            O::DistanceAccessor::default(),
                        ),
                    );
                    cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean))
        })
        .take(probes as usize)
        .collect()
    };
    for i in (1..height_of_root).rev() {
        state = step(state, probes[i as usize - 1]);
    }

    let mut results = LinkedVec::new();
    for (first, residual) in state {
        let lut = if let Some(residual) = residual.as_ref().map(|x| x.as_borrowed()) {
            &O::Vector::compute_lut(residual)
        } else {
            default_lut.as_ref().unwrap()
        };
        let jump_guard = index.read(first);
        let jump_tuple = jump_guard
            .get(1)
            .expect("data corruption")
            .pipe(read_tuple::<JumpTuple>);
        let first = jump_tuple.first();
        access_0(
            index.clone(),
            first,
            || {
                RAccess::new(
                    (&lut.0.4, (lut.0.0, lut.0.1, lut.0.2, lut.0.3, epsilon)),
                    O::Distance::block_accessor(),
                )
            },
            |code| O::Distance::compute_lowerbound_binary(&lut.1, code, epsilon),
            |lowerbound, mean, payload| {
                results.push((Reverse(lowerbound), AlwaysEqual(mean), AlwaysEqual(payload)));
            },
        );
    }
    (
        if rerank_in_heap {
            RerankMethod::Heap
        } else {
            RerankMethod::Index
        },
        results.into_vec(),
    )
}
