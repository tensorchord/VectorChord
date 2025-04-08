use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::select_heap::SelectHeap;
use crate::tuples::*;
use crate::vectors::{self};
use crate::{IndexPointer, Page, RelationWrite, tape};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

pub fn insert<O: Operator>(index: impl RelationWrite, payload: NonZero<u64>, vector: O::Vector) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    let height_of_root = meta_tuple.height_of_root();
    assert_eq!(dims, vector.as_borrowed().dims(), "unmatched dimensions");
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);

    let default_block_lut = if !is_residual {
        Some(O::Vector::block_preprocess(vector.as_borrowed()))
    } else {
        None
    };

    let mean = if !rerank_in_heap {
        vectors::append::<O>(index.clone(), vectors_first, vector.as_borrowed(), payload)
    } else {
        IndexPointer::default()
    };

    type State<O> = (u32, Option<<O as Operator>::Vector>);
    let mut state: State<O> = {
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
    };
    let step = |state: State<O>| {
        let mut results = LinkedVec::new();
        {
            let (first, residual) = state;
            let block_lut = if let Some(residual) = residual {
                &O::Vector::block_preprocess(residual.as_borrowed())
            } else if let Some(block_lut) = default_block_lut.as_ref() {
                block_lut
            } else {
                unreachable!()
            };
            tape::read_h1_tape(
                index.clone(),
                first,
                || RAccess::new((&block_lut.1, block_lut.0), O::BlockAccessor::default()),
                |(rough, err), mean, first| {
                    let lowerbound = Distance::from_f32(rough - err * 1.9);
                    results.push((Reverse(lowerbound), AlwaysEqual(mean), AlwaysEqual(first)));
                },
                |_| (),
            );
        }
        let mut heap = SelectHeap::from_vec(results.into_vec());
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        {
            while let Some((Reverse(_), AlwaysEqual(mean), AlwaysEqual(first))) =
                pop_if(&mut heap, |(d, ..)| {
                    Some(*d) > cache.peek().map(|(d, ..)| *d)
                })
            {
                if is_residual {
                    let (dis_u, residual_u) = vectors::read_for_h1_tuple::<O, _>(
                        index.clone(),
                        mean,
                        LAccess::new(
                            O::Vector::unpack(vector.as_borrowed()),
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
                        LAccess::new(
                            O::Vector::unpack(vector.as_borrowed()),
                            O::DistanceAccessor::default(),
                        ),
                    );
                    cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(None)));
                }
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache
                .pop()
                .expect("invariant is violated: tree is not height-balanced");
            (first, mean)
        }
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }

    let (first, residual) = state;
    let code = if let Some(residual) = residual {
        O::Vector::code(residual.as_borrowed())
    } else {
        O::Vector::code(vector.as_borrowed())
    };
    let bytes = AppendableTuple::serialize(&AppendableTuple {
        mean,
        dis_u_2: code.dis_u_2,
        factor_ppc: code.factor_ppc,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        payload: Some(payload),
        elements: rabitq::pack_to_u64(&code.signs),
    });

    let jump_guard = index.read(first);
    let jump_bytes = jump_guard.get(1).expect("data corruption");
    let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);

    tape::append(index.clone(), jump_tuple.appendable_first(), &bytes, false);
}

fn pop_if<T: Ord>(heap: &mut SelectHeap<T>, mut predicate: impl FnMut(&T) -> bool) -> Option<T> {
    let peek = heap.peek()?;
    if predicate(peek) { heap.pop() } else { None }
}
