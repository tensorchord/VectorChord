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

use crate::Opaque;
use crate::operator::Operator;
use crate::tuples::{MetaTuple, VertexTuple, WithReader, WithWriter};
use algo::{Page, RelationRead, RelationWrite};
use std::num::NonZero;

pub fn bulkdelete<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let start = meta_tuple.start();
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let Some(_) = start.into_inner() else {
        return;
    };
    let mut current = link;
    while current != u32::MAX {
        check();
        let read = index.read(current);
        let flag = 'flag: {
            for i in 1..=read.len() {
                if let Some(bytes) = read.get(i) {
                    let tuple = VertexTuple::deserialize_ref(bytes);
                    let p = tuple.payload();
                    if Some(true) == p.map(&callback) {
                        break 'flag true;
                    }
                }
            }
            false
        };
        if flag {
            drop(read);
            let mut write = index.write(current, false);
            for i in 1..=write.len() {
                if let Some(bytes) = write.get_mut(i) {
                    let mut tuple = VertexTuple::deserialize_mut(bytes);
                    let p = tuple.payload();
                    if Some(true) == p.map(&callback) {
                        *p = None;
                    }
                }
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
}
