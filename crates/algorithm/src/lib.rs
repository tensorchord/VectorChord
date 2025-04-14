mod build;
mod bulkdelete;
mod cache;
mod cost;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
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
pub use cost::cost;
pub use insert::insert;
pub use maintain::maintain;
pub use prewarm::prewarm;
pub use rerank::{how, rerank_heap, rerank_index};
pub use search::{default_search, maxsim_search};

use std::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

pub trait Page: Sized {
    #[must_use]
    fn get_opaque(&self) -> &Opaque;
    #[must_use]
    fn get_opaque_mut(&mut self) -> &mut Opaque;
    #[must_use]
    fn len(&self) -> u16;
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[must_use]
    fn get(&self, i: u16) -> Option<&[u8]>;
    #[must_use]
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]>;
    #[must_use]
    fn alloc(&mut self, data: &[u8]) -> Option<u16>;
    fn free(&mut self, i: u16);
    #[must_use]
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

pub(crate) struct Branch<T> {
    pub mean: IndexPointer,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
    pub extra: T,
}

#[repr(transparent)]
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable, KnownLayout,
)]
pub struct IndexPointer(pub u64);
