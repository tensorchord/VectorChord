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
use crate::{Opaque, Page, PageGuard, tape};
use algo::accessor::TryAccessor1;
use algo::{RelationRead, RelationWrite};
use std::num::NonZero;
use vector::VectorOwned;

pub fn read<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: TryAccessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    mut prefetch: impl Iterator<Item = R::ReadGuard<'a>>,
    head: u16,
    payload: NonZero<u64>,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = prefetch.next()?;
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
    if prefetch.next().is_some() {
        return None;
    }
    result.finish(cursor.ok()?)
}

pub fn append<O: Operator, R: RelationRead + RelationWrite>(
    index: &R,
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZero<u64>,
    skip_search: bool,
) -> (Vec<u32>, u16)
where
    R::Page: Page<Opaque = Opaque>,
{
    fn append<R: RelationRead + RelationWrite>(
        index: &R,
        first: u32,
        bytes: &[u8],
        skip_search: bool,
    ) -> (u32, u16)
    where
        R::Page: Page<Opaque = Opaque>,
    {
        if !skip_search && let Some(mut write) = index.search(bytes.len()) {
            let i = write
                .alloc(bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            return (write.id(), i);
        }
        tape::append(index, first, bytes, true, None)
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
        let (id, head) = append(index, vectors_first, &bytes, skip_search);
        chain = Err(head);
        prefetch.push(id);
    }
    prefetch.reverse();
    (
        prefetch,
        chain.expect_err("internal error: 0-dimensional vector"),
    )
}
