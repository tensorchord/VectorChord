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

#![allow(clippy::type_complexity)]

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
mod prefetcher;
mod prewarm;
mod rerank;
mod search;
mod tape;
mod tape_writer;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

use always_equal::AlwaysEqual;
pub use build::build;
pub use bulkdelete::bulkdelete;
pub use cache::cache;
pub use cost::cost;
pub use fast_heap::FastHeap;
pub use insert::insert;
pub use maintain::maintain;
pub use prefetcher::*;
pub use prewarm::prewarm;
pub use rerank::{how, rerank_heap, rerank_index};
pub use search::{default_search, maxsim_search};

use std::collections::BinaryHeap;
use std::iter::Peekable;
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

pub trait ReadStream<'s> {
    type Relation: RelationRead;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationRead>::ReadGuard<'s>>,
    )>;
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationRead>::ReadGuard<'s>>,
    )>;
    fn into_inner(self) -> Self::Inner;
}

pub trait WriteStream<'s> {
    type Relation: RelationWrite;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        Self::Item,
        Vec<<Self::Relation as RelationWrite>::WriteGuard<'s>>,
    )>;
    fn into_inner(self) -> Self::Inner;
}

pub trait Relation {
    type Page: Page;
}

pub trait RelationRead: Relation {
    type ReadGuard<'a>: PageGuard + Deref<Target = Self::Page>
    where
        Self: 'a;
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWrite: Relation {
    type WriteGuard<'a>: PageGuard + DerefMut<Target = Self::Page>
    where
        Self: 'a;
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

pub trait RelationPrefetch: Relation {
    fn prefetch(&self, id: u32);
}

#[derive(Debug, Default, Clone)]
pub struct Hints {
    pub full: bool,
}

impl Hints {
    #[allow(clippy::needless_update)]
    pub fn full(self, full: bool) -> Self {
        Self { full, ..self }
    }
}

pub trait RelationReadStream: RelationRead {
    type ReadStream<'s, I: Iterator>: ReadStream<'s, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch,
        Self: 's;
    fn read_stream<I: Iterator>(&self, iter: I, hints: Hints) -> Self::ReadStream<'_, I>
    where
        I::Item: Fetch;
}

pub trait RelationWriteStream: RelationWrite {
    type WriteStream<'s, I: Iterator>: WriteStream<'s, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch,
        Self: 's;
    fn write_stream<I: Iterator>(&self, iter: I, hints: Hints) -> Self::WriteStream<'_, I>
    where
        I::Item: Fetch;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub(crate) struct Branch<T> {
    pub head: u16,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
    pub prefetch: Vec<u32>,
    pub extra: T,
}

pub trait Bump: 'static {
    #[allow(clippy::mut_from_ref)]
    fn alloc<T>(&self, value: T) -> &mut T;
    #[allow(clippy::mut_from_ref)]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T];
}

pub trait Fetch {
    fn fetch(&self) -> &[u32];
}

impl Fetch for u32 {
    fn fetch(&self) -> &[u32] {
        std::slice::from_ref(self)
    }
}

impl<'b, T, A, B> Fetch for (T, AlwaysEqual<&'b mut (A, B, &'b mut [u32])>) {
    fn fetch(&self) -> &[u32] {
        let (_, AlwaysEqual((.., list))) = self;
        list
    }
}

pub trait Sequence {
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next(&mut self) -> Option<Self::Item>;
    fn next_if(&mut self, predicate: impl FnOnce(&Self::Item) -> bool) -> Option<Self::Item>;
    fn into_inner(self) -> Self::Inner;
}

impl<T: Ord> Sequence for BinaryHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    fn next_if(&mut self, predicate: impl FnOnce(&T) -> bool) -> Option<T> {
        let peek = self.peek()?;
        if predicate(peek) { self.pop() } else { None }
    }
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<I: Iterator> Sequence for Peekable<I> {
    type Item = I::Item;
    type Inner = Peekable<I>;
    fn next(&mut self) -> Option<I::Item> {
        Iterator::next(self)
    }
    fn next_if(&mut self, predicate: impl FnOnce(&I::Item) -> bool) -> Option<I::Item> {
        Peekable::next_if(self, predicate)
    }
    fn into_inner(self) -> Self::Inner {
        self
    }
}
