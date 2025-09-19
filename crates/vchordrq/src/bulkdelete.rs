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

use crate::closure_lifetime_binder::{id_0, id_1};
use crate::operator::Operator;
use crate::tape::by_next;
use crate::tuples::*;
use crate::{Opaque, Page, tape};
use algo::accessor::FunctionalAccessor;
use algo::{RelationRead, RelationWrite};
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
    let height_of_root = meta_tuple.height_of_root();

    type State = Vec<u32>;
    let mut state: State = vec![meta_tuple.first()];

    drop(meta_guard);

    let step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|_| check()),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                |(), _, _, first, _| results.push(first),
            );
        }
        results
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }
    for first in state {
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut directory = tape::read_directory_tape::<R>(
            by_next(index, jump_tuple.directory_first()).inspect(|_| check()),
        );
        {
            let mut current = directory.next().unwrap_or(u32::MAX);
            while current != u32::MAX {
                check();
                let read = index.read(current);
                let flag = 'flag: {
                    for i in 1..=read.len() {
                        let bytes = read.get(i).expect("data corruption");
                        let tuple = FrozenTuple::deserialize_ref(bytes);
                        if let FrozenTupleReader::_0(tuple) = tuple {
                            for p in tuple.payload().iter() {
                                if Some(true) == p.map(&callback) {
                                    break 'flag true;
                                }
                            }
                        }
                    }
                    false
                };
                if flag {
                    drop(read);
                    let mut write = index.write(current, false);
                    for i in 1..=write.len() {
                        let bytes = write.get_mut(i).expect("data corruption");
                        let tuple = FrozenTuple::deserialize_mut(bytes);
                        if let FrozenTupleWriter::_0(mut tuple) = tuple {
                            for p in tuple.payload().iter_mut() {
                                if Some(true) == p.map(&callback) {
                                    *p = None;
                                }
                            }
                        }
                    }
                }
                current = directory.next().unwrap_or(u32::MAX);
            }
        }
        {
            let mut current = jump_tuple.appendable_first();
            while current != u32::MAX {
                check();
                let read = index.read(current);
                let flag = 'flag: {
                    for i in 1..=read.len() {
                        let bytes = read.get(i).expect("data corruption");
                        let tuple = AppendableTuple::deserialize_ref(bytes);
                        let p = tuple.payload();
                        if Some(true) == p.map(&callback) {
                            break 'flag true;
                        }
                    }
                    false
                };
                if flag {
                    drop(read);
                    let mut write = index.write(current, false);
                    for i in 1..=write.len() {
                        let bytes = write.get_mut(i).expect("data corruption");
                        let mut tuple = AppendableTuple::deserialize_mut(bytes);
                        let p = tuple.payload();
                        if Some(true) == p.map(&callback) {
                            *p = None;
                        }
                    }
                    current = write.get_opaque().next;
                } else {
                    current = read.get_opaque().next;
                }
            }
        }
    }
}

pub fn bulkdelete_vectors<R: RelationRead + RelationWrite, O: Operator>(
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let vectors_first = meta_tuple.vectors_first().to_vec();

    drop(meta_guard);

    for vectors_first in vectors_first {
        let mut current = vectors_first;
        while current != u32::MAX {
            check();
            let read = index.read(current);
            let flag = 'flag: {
                for i in 1..=read.len() {
                    if let Some(bytes) = read.get(i) {
                        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
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
                let mut write = index.write(current, true);
                for i in 1..=write.len() {
                    if let Some(bytes) = write.get(i) {
                        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                        let p = tuple.payload();
                        if Some(true) == p.map(&callback) {
                            write.free(i);
                        }
                    };
                }
                current = write.get_opaque().next;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}
