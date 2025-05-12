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

use crate::operator::*;
use crate::tuples::*;
use crate::{Page, PageGuard, RelationRead, RelationWrite, tape};
use std::num::NonZero;
use vector::VectorOwned;

pub fn read_for_h1_tuple<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    head: u16,
    mut list: impl Iterator<Item = R::ReadGuard<'a>>,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = list.next().expect("data corruption");
        let bytes = guard.get(head).expect("data corruption");
        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
        if tuple.payload().is_some() {
            panic!("data corruption");
        }
        result.push(tuple.elements());
        cursor = tuple.metadata_or_head();
    }
    if list.next().is_some() {
        panic!("data corruption");
    }
    result.finish(cursor.expect("data corruption"))
}

pub fn read_for_h0_tuple<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: TryAccessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    head: u16,
    mut list: impl Iterator<Item = R::ReadGuard<'a>>,
    payload: NonZero<u64>,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = list.next()?;
        let bytes = guard.get(head)?;
        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
        if tuple.payload().is_none() {
            panic!("data corruption");
        }
        if tuple.payload() != Some(payload) {
            return None;
        }
        result.push(tuple.elements())?;
        cursor = tuple.metadata_or_head();
    }
    if list.next().is_some() {
        return None;
    }
    result.finish(cursor.ok()?)
}

pub fn append<O: Operator>(
    index: &(impl RelationRead + RelationWrite),
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZero<u64>,
) -> (Vec<u32>, u16) {
    fn append(index: &(impl RelationRead + RelationWrite), first: u32, bytes: &[u8]) -> (u32, u16) {
        if let Some(mut write) = index.search(bytes.len()) {
            let i = write
                .alloc(bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            return (write.id(), i);
        }
        tape::append(index, first, bytes, true)
    }
    let (slices, metadata) = O::Vector::split(vector);
    let mut chain = Ok(metadata);
    let mut prefetch = Vec::new();
    for i in (0..slices.len()).rev() {
        let bytes = VectorTuple::<O::Vector>::serialize(&match chain {
            Ok(metadata) => VectorTuple::_0 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                metadata,
            },
            Err(head) => VectorTuple::_1 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                head,
            },
        });
        let (id, head) = append(index, vectors_first, &bytes);
        chain = Err(head);
        prefetch.push(id);
    }
    prefetch.reverse();
    (
        prefetch,
        chain.expect_err("internal error: 0-dimensional vector"),
    )
}
