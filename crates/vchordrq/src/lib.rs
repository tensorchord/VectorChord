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

mod build;
mod bulkdelete;
mod cache;
mod closure_lifetime_binder;
mod cost;
mod fast_heap;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
mod prewarm;
mod rerank;
mod search;
mod tape;
mod tape_writer;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

use algo::{Page, PageGuard};
pub use build::build;
pub use bulkdelete::{bulkdelete, bulkdelete_vectors};
pub use cache::cache;
pub use cost::cost;
pub use fast_heap::FastHeap;
pub use insert::{insert, insert_vector};
pub use maintain::maintain;
pub use prewarm::prewarm;
pub use rerank::{how, rerank_heap, rerank_index};
pub use search::{default_search, maxsim_search};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

#[allow(unsafe_code)]
unsafe impl algo::Opaque for Opaque {}

pub(crate) struct Branch<T> {
    pub code: rabitq::bit::Code,
    pub delta: f32,
    pub prefetch: Vec<u32>,
    pub head: u16,
    pub norm: f32,
    pub extra: T,
}
