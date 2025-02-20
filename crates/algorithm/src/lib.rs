#![allow(clippy::collapsible_else_if)]
#![allow(clippy::type_complexity)]
#![allow(clippy::len_without_is_empty)]

mod build;
mod bulkdelete;
mod cache;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
mod pipe;
mod prewarm;
mod rerank;
mod search;
mod select_heap;
mod tape;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::bulkdelete;
pub use cache::cache;
pub use insert::insert;
pub use maintain::maintain;
pub use prewarm::prewarm;
pub use rerank::{rerank_heap, rerank_index};
pub use search::search;

use std::ops::{Deref, DerefMut};
use zerocopy_derive::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

#[allow(clippy::len_without_is_empty)]
pub trait Page: Sized {
    fn get_opaque(&self) -> &Opaque;
    fn get_opaque_mut(&mut self) -> &mut Opaque;
    fn len(&self) -> u16;
    fn get(&self, i: u16) -> Option<&[u8]>;
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]>;
    fn alloc(&mut self, data: &[u8]) -> Option<u16>;
    fn free(&mut self, i: u16);
    fn freespace(&self) -> u16;
    fn clear(&mut self);
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait RelationRead: Clone {
    type Page: Page;
    type ReadGuard<'a>: PageGuard + Deref<Target = Self::Page>
    where
        Self: 'a;
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWrite: RelationRead {
    type WriteGuard<'a>: PageGuard + DerefMut<Target = Self::Page>
    where
        Self: 'a;
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}
