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
use crate::tape::by_next;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Opaque, Page, PageGuard, tape};
use algo::RelationRead;
use algo::accessor::FunctionalAccessor;

pub fn cache<R: RelationRead>(index: &R, level: i32) -> Vec<u32>
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let height_of_root = meta_tuple.height_of_root();
    let centroids_first = meta_tuple.centroids_first();

    type State = Vec<u32>;
    let mut state: State = vec![meta_tuple.first()];

    drop(meta_guard);

    let mut trace = Vec::new();

    if level >= 0 {
        // meta tuple
        trace.push(0);
    }

    if level >= 1 {
        let mut step = |state: State| {
            let mut results = Vec::new();
            for first in state {
                tape::read_h1_tape::<R, _, _>(
                    by_next(index, first).inspect(|guard| trace.push(guard.id())),
                    || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                    |(), _, _, first, _| {
                        results.push(first);
                    },
                );
            }
            results
        };

        // h1 tuples
        for _ in (1..height_of_root).rev() {
            state = step(state);
        }

        // jump tuples
        for first in state {
            trace.push(first);
        }
    }

    if level >= 2 {
        // centroid tuples
        for guard in by_next(index, centroids_first) {
            trace.push(guard.id());
        }
    }

    trace
}
