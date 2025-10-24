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

use crate::tuples::*;
use crate::{Opaque, freepages};
use index::relation::{Page, PageGuard, RelationRead, RelationWrite};

pub fn consume<R: RelationRead + RelationWrite>(index: &R)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let freepages_first = meta_tuple.freepages_first();

    drop(meta_guard);

    let id = index
        .extend(
            Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            },
            false,
        )
        .id();

    freepages::free(index, freepages_first, id);
}
