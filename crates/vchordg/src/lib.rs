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
mod candidates;
mod insert;
mod maintain;
mod prewarm;
mod prune;
mod results;
mod search;
mod tuples;
mod vectors;
mod visited;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::bulkdelete;
pub use insert::insert;
pub use maintain::maintain;
pub use prewarm::prewarm;
pub use search::search;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub link: u32,
}

#[allow(unsafe_code)]
unsafe impl algo::Opaque for Opaque {}

pub type Id = (u32, u16);

impl algo::Fetch1 for tuples::Pointer {
    fn fetch_1(&self) -> u32 {
        self.into_inner().0
    }
}
