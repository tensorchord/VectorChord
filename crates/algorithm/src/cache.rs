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
use crate::operator::FunctionalAccessor;
use crate::tape::by_next;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, PageGuard, RelationRead, tape};

pub fn cache<R: RelationRead>(index: &R) -> Vec<u32> {
    let mut trace = vec![0_u32];
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);
    type State = Vec<u32>;
    let mut state: State = vec![root_first];
    let mut step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|guard| trace.push(guard.id())),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                |(), _, first, _| {
                    results.push(first);
                },
            );
        }
        results
    };
    for _ in (1..height_of_root).rev() {
        state = step(state);
    }
    for first in state {
        trace.push(first);
    }
    trace
}
