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

use crate::closure_lifetime_binder::{id_0, id_1, id_2};
use crate::operator::{FunctionalAccessor, Operator};
use crate::tape::{by_directory, by_next};
use crate::tuples::*;
use crate::{Page, PrefetcherSequenceFamily, RelationRead, tape, vectors};
use std::fmt::Write;

pub fn prewarm<'r, R: RelationRead, O: Operator>(
    index: &'r R,
    height: i32,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'r, R>,
) -> String {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let height_of_root = meta_tuple.height_of_root();
    let root_prefetch = meta_tuple.root_prefetch().to_vec();
    let root_head = meta_tuple.root_head();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let mut message = String::new();
    writeln!(message, "height of root: {height_of_root}").unwrap();
    let prewarm_max_height = if height < 0 { 0 } else { height as u32 };
    if prewarm_max_height > height_of_root {
        return message;
    }
    type State = Vec<u32>;
    let mut state: State = {
        let mut results = Vec::new();
        {
            let list = root_prefetch.into_iter().map(|id| index.read(id));
            vectors::read_for_h1_tuple::<R, O, _>(root_head, list, ());
            results.push(root_first);
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", 1).unwrap();
        results
    };
    let mut step = |state: State| {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for first in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first).inspect(|_| counter += 1),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                |(), head, first, prefetch| {
                    let list = prefetch.iter().map(|&id| index.read(id));
                    vectors::read_for_h1_tuple::<R, O, _>(head, list, ());
                    results.push(first);
                },
            );
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {counter}").unwrap();
        results
    };
    for _ in (std::cmp::max(1, prewarm_max_height)..height_of_root).rev() {
        state = step(state);
    }
    if prewarm_max_height == 0 {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for first in state {
            let jump_guard = index.read(first);
            let jump_bytes = jump_guard.get(1).expect("data corruption");
            let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory).inspect(|_| counter += 1),
                || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); 32])),
                id_2(|_, _, _, _| {
                    results.push(());
                }),
            );
            tape::read_appendable_tape::<R, _>(
                by_next(index, jump_tuple.appendable_first()).inspect(|_| counter += 1),
                |_| (),
                id_2(|_, _, _, _| {
                    results.push(());
                }),
            );
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {counter}").unwrap();
    }
    message
}
