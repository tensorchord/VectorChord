#![allow(clippy::type_complexity)]

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
mod visited;

pub mod operator;
pub mod types;

use algo::tuples::Padding;
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
    pub skip: u32,
    pub link: u32,
    pub _padding_0: [Padding; 4],
}

#[allow(unsafe_code)]
unsafe impl algo::Opaque for Opaque {}

pub type Id = (u32, u16);

impl algo::Fetch1 for tuples::Pointer {
    fn fetch_1(&self) -> u32 {
        self.into_inner().0
    }
}
