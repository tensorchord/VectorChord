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

use crate::Page;
use crate::operator::*;
use crate::tuples::*;
use algo::RelationRead;
use algo::accessor::Accessor1;

pub fn read<
    'a,
    R: RelationRead + 'a,
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    mut prefetch: impl Iterator<Item = R::ReadGuard<'a>>,
    head: u16,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(head);
    let mut result = accessor;
    while let Err(head) = cursor {
        let guard = prefetch.next().expect("data corruption");
        let bytes = guard.get(head).expect("data corruption");
        let tuple = CentroidTuple::<O::Vector>::deserialize_ref(bytes);
        result.push(tuple.elements());
        cursor = tuple.metadata_or_head();
    }
    if prefetch.next().is_some() {
        panic!("data corruption");
    }
    result.finish(cursor.expect("data corruption"))
}
