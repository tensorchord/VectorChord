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
use crate::tuples::{MetaTuple, OptionPointer, Tuple};
use crate::types::{VchordgIndexOptions, VectorOptions};
use algo::{Page, PageGuard, RelationWrite};

pub fn build<R: RelationWrite, O: Operator>(
    vector_options: VectorOptions,
    index_options: VchordgIndexOptions,
    index: &R,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let mut meta_guard = index.extend(
        Opaque {
            next: u32::MAX,
            link: 1,
        },
        false,
    );
    assert_eq!(meta_guard.id(), 0);
    let vertex_guard = index.extend(
        Opaque {
            next: u32::MAX,
            link: 2,
        },
        true,
    );
    assert_eq!(vertex_guard.id(), 1);
    drop(vertex_guard);
    let vector_guard = index.extend(
        Opaque {
            next: u32::MAX,
            link: u32::MAX,
        },
        false,
    );
    assert_eq!(vector_guard.id(), 2);
    drop(vector_guard);
    let serialized = MetaTuple::serialize(&MetaTuple {
        dims: vector_options.dims,
        bits: index_options.bits,
        m: index_options.m,
        max_alpha: index_options.max_alpha,
        ef_construction: index_options.ef_construction,
        beam_construction: index_options.beam_construction,
        start: OptionPointer::NONE,
        skip: 1,
    });
    let i = meta_guard
        .alloc(&serialized)
        .expect("implementation: a free page cannot accommodate a single tuple");
    assert_eq!(i, 1);
}
