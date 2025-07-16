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

use crate::operator::{Operator, Vector};
use crate::tuples::*;
use algo::accessor::Accessor1;
use algo::{Page, RelationRead, RelationWrite};
use distance::Distance;
use std::collections::VecDeque;
use std::num::{NonZero, Wrapping};

pub fn by_prefetch<'r, R: RelationRead>(
    guards: impl ExactSizeIterator<Item = R::ReadGuard<'r>>,
    pointers_u: impl ExactSizeIterator<Item = Pointer>,
) -> impl ExactSizeIterator<Item = (R::ReadGuard<'r>, u16)> {
    assert!(guards.len() == pointers_u.len() && pointers_u.len() > 0);
    pointers_u
        .map(Pointer::into_inner)
        .zip(guards)
        .map(|(pointer, guard)| (guard, pointer.1))
}

pub fn by_read<'r, R: RelationRead>(
    index: &'r R,
    pointers_u: impl ExactSizeIterator<Item = Pointer>,
) -> impl ExactSizeIterator<Item = (R::ReadGuard<'r>, u16)> {
    assert!(pointers_u.len() > 0);
    pointers_u
        .map(Pointer::into_inner)
        .map(|pointer| (pointer, index.read(pointer.0)))
        .map(|(pointer, guard)| (guard, pointer.1))
}

pub fn copy_nothing(_: &[OptionNeighbour]) {}

pub fn copy_outs(x: &[OptionNeighbour]) -> VecDeque<(u32, u16)> {
    x.iter()
        .flat_map(|neighbour| neighbour.into_inner())
        .map(|(v, _)| v)
        .collect::<VecDeque<_>>()
}

pub fn copy_all(x: &[OptionNeighbour]) -> VecDeque<((u32, u16), Distance)> {
    x.iter()
        .flat_map(|neighbour| neighbour.into_inner())
        .collect::<VecDeque<_>>()
}

pub fn read<
    'r,
    R: RelationRead,
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
    Output,
>(
    mut iterator: impl ExactSizeIterator<Item = (R::ReadGuard<'r>, u16)>,
    accessor: A,
    copy: impl FnOnce(&[OptionNeighbour]) -> Output,
) -> Result<(A::Output, Output, Option<NonZero<u64>>, Wrapping<u32>), ()> {
    let m = strict_sub(iterator.len(), 1);
    let mut result = accessor;
    for index in 0..m {
        let (vector_guard, i) = iterator.next().expect("internal: bad size");
        let Some(vector_bytes) = vector_guard.get(i) else {
            // the link is broken
            return Err(());
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        let VectorTupleReader::_1(vector_tuple) = vector_tuple else {
            // the link is broken
            return Err(());
        };
        if vector_tuple.index() as usize != index {
            // the link is broken
            return Err(());
        }
        result.push(vector_tuple.elements());
    }
    let value_u;
    let payload_u;
    let neighbours_u;
    let version_u;
    {
        let (vector_guard, i) = iterator.next().expect("internal: bad size");
        let Some(vector_bytes) = vector_guard.get(i) else {
            // the link is broken
            return Err(());
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        let VectorTupleReader::_0(vector_tuple) = vector_tuple else {
            // the link is broken
            return Err(());
        };
        result.push(vector_tuple.elements());
        value_u = result.finish(*vector_tuple.metadata());
        neighbours_u = copy(vector_tuple.neighbours());
        payload_u = vector_tuple.payload();
        version_u = vector_tuple.version();
    }
    Ok((value_u, neighbours_u, payload_u, version_u))
}

pub fn read_without_accessor<R: RelationRead, O: Operator, Output>(
    (index, pointers_u): (&R, &[Pointer]),
    copy: impl FnOnce(&[OptionNeighbour]) -> Output,
) -> Result<(Output, Option<NonZero<u64>>, Wrapping<u32>), ()> {
    let payload_u;
    let neighbours_u;
    let version_u;
    {
        let (id, i) = pointers_u.last().expect("internal: bad size").into_inner();
        let vector_guard = index.read(id);
        let Some(vector_bytes) = vector_guard.get(i) else {
            // the link is broken
            return Err(());
        };
        let vector_tuple = VectorTuple::<O::Vector>::deserialize_ref(vector_bytes);
        let VectorTupleReader::_0(vector_tuple) = vector_tuple else {
            // the link is broken
            return Err(());
        };
        neighbours_u = copy(vector_tuple.neighbours());
        payload_u = vector_tuple.payload();
        version_u = vector_tuple.version();
    }
    Ok((neighbours_u, payload_u, version_u))
}

pub fn update<R: RelationWrite, O: Operator>(
    (index, pointers_u): (&R, &[Pointer]),
    (version, neighbours_u): (Wrapping<u32>, VecDeque<((u32, u16), Distance)>),
    outs: impl Iterator<Item = ((u32, u16), Distance)> + Clone,
) -> Result<bool, ()> {
    if outs.clone().eq(neighbours_u) {
        return Ok(true);
    }
    let mut vector_guard = index.write(pointers_u.last().unwrap().into_inner().0, false);
    let Some(vector_bytes) = vector_guard.get_mut(pointers_u.last().unwrap().into_inner().1) else {
        // the link is broken
        return Err(());
    };
    let vector_tuple = VectorTuple::<O::Vector>::deserialize_mut(vector_bytes);
    let VectorTupleWriter::_0(mut vector_tuple) = vector_tuple else {
        // the link is broken
        return Err(());
    };
    if *vector_tuple.version() != version {
        return Ok(false);
    } else {
        *vector_tuple.version() += 1;
    }
    let filling = outs
        .map(|(v, dis_v)| OptionNeighbour::some(v, dis_v))
        .chain(std::iter::repeat(OptionNeighbour::NONE));
    for (hole, fill) in std::iter::zip(vector_tuple.neighbours().iter_mut(), filling) {
        *hole = fill;
    }
    Ok(true)
}

// Emulate unstable library feature `strict_overflow_ops`.
// See https://github.com/rust-lang/rust/issues/118260.

#[inline]
const fn strict_sub(lhs: usize, rhs: usize) -> usize {
    let (a, b) = lhs.overflowing_sub(rhs);
    if b { panic!() } else { a }
}
