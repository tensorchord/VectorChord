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

pub mod accessor;
pub mod prefetcher;
pub mod tuples;

use always_equal::AlwaysEqual;
use dary_heap::DaryHeap;
use std::collections::{BinaryHeap, VecDeque};
use std::iter::Peekable;
use std::ops::{Deref, DerefMut};
use zerocopy::{FromBytes, IntoBytes};

pub trait Page: Sized + 'static {
    type Opaque: Opaque;

    #[must_use]
    fn get_opaque(&self) -> &Self::Opaque;
    #[must_use]
    fn get_opaque_mut(&mut self) -> &mut Self::Opaque;
    #[must_use]
    fn len(&self) -> u16;
    #[must_use]
    #[inline]
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
    fn clear(&mut self, opaque: Self::Opaque);
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait ReadStream<'b> {
    type Relation: RelationReadTypes;
    type Guards: ExactSizeIterator<Item = <Self::Relation as RelationReadTypes>::ReadGuard<'b>>;
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    fn next(&mut self) -> Option<(Self::Item, Self::Guards)>;
    fn next_if<P: FnOnce(&Self::Item) -> bool>(
        &mut self,
        predicate: P,
    ) -> Option<(Self::Item, Self::Guards)>;
    fn into_inner(self) -> Self::Inner;
}

pub trait Relation {
    type Page: Page;
}

pub trait RelationReadTypes: Relation {
    type ReadGuard<'b>: PageGuard + Deref<Target = Self::Page>;
}

pub trait RelationRead: RelationReadTypes {
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWriteTypes: Relation {
    type WriteGuard<'b>: PageGuard + DerefMut<Target = Self::Page>;
}

pub trait RelationWrite: RelationWriteTypes {
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_>;
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
    #[inline]
    pub fn full(self, full: bool) -> Self {
        Self { full, ..self }
    }
}

pub trait RelationReadStreamTypes: RelationReadTypes {
    type ReadStream<'b, I: Iterator>: ReadStream<'b, Item = I::Item, Relation = Self>
    where
        I::Item: Fetch<'b>;
}

pub trait RelationReadStream: RelationReadStreamTypes {
    fn read_stream<'b, I: Iterator>(&'b self, iter: I, hints: Hints) -> Self::ReadStream<'b, I>
    where
        I::Item: Fetch<'b>;
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub trait Bump: 'static {
    #[allow(clippy::mut_from_ref)]
    fn alloc<T: Copy>(&self, value: T) -> &mut T;
    #[allow(clippy::mut_from_ref)]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T];
}

impl Bump for bumpalo::Bump {
    #[inline]
    fn alloc<T: Copy>(&self, value: T) -> &mut T {
        self.alloc(value)
    }

    #[inline]
    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        self.alloc_slice_copy(slice)
    }
}

pub type BorrowedIter<'b> = small_iter::borrowed::Iter<'b, u32, 1>;

pub trait Fetch<'b> {
    type Iter: ExactSizeIterator<Item = u32> + 'b;
    #[must_use]
    fn fetch(&self) -> Self::Iter;
}

impl Fetch<'_> for u32 {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        std::iter::once(*self)
    }
}

impl<'b, T, A, B, W: 'b + PackedRefMut<T = (A, B, BorrowedIter<'b>)>> Fetch<'b>
    for (T, AlwaysEqual<W>)
{
    type Iter = BorrowedIter<'b>;
    #[inline(always)]
    fn fetch(&self) -> BorrowedIter<'b> {
        let (.., list) = self.1.0.get();
        *list
    }
}

impl<'b, T, A, B, C> Fetch<'b> for (T, AlwaysEqual<&mut (A, B, C, BorrowedIter<'b>)>) {
    type Iter = BorrowedIter<'b>;
    #[inline(always)]
    fn fetch(&self) -> BorrowedIter<'b> {
        let (_, AlwaysEqual((.., list))) = self;
        *list
    }
}

impl<T> Fetch<'_> for (T, AlwaysEqual<(u32, u16)>) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        let (_, AlwaysEqual((x, _))) = self;
        std::iter::once(*x)
    }
}

impl Fetch<'_> for (u32, u16) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        std::iter::once(self.0)
    }
}

impl<T> Fetch<'_> for (T, AlwaysEqual<((u32, u16), (u32, u16))>) {
    type Iter = std::iter::Once<u32>;
    #[inline(always)]
    fn fetch(&self) -> std::iter::Once<u32> {
        let (_, AlwaysEqual(((x, _), _))) = self;
        std::iter::once(*x)
    }
}

pub trait Fetch1 {
    fn fetch_1(&self) -> u32;
}

#[repr(transparent)]
pub struct Fetch1Iter<I> {
    iter: I,
}

impl<I: Iterator> Iterator for Fetch1Iter<I>
where
    I::Item: Fetch1,
{
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|x| x.fetch_1())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Fetch1Iter<I> where I::Item: Fetch1 {}

impl<'b, T, F: Fetch1 + Copy> Fetch<'b> for (T, AlwaysEqual<&'b [F]>) {
    type Iter = Fetch1Iter<std::iter::Copied<std::slice::Iter<'b, F>>>;
    #[inline(always)]
    fn fetch(&self) -> Self::Iter {
        Fetch1Iter {
            iter: self.1.0.iter().copied(),
        }
    }
}

impl<'b, T, U, F: Fetch1 + Copy> Fetch<'b> for (T, AlwaysEqual<(&'b [F], U)>) {
    type Iter = Fetch1Iter<std::iter::Copied<std::slice::Iter<'b, F>>>;
    #[inline(always)]
    fn fetch(&self) -> Self::Iter {
        Fetch1Iter {
            iter: self.1.0.0.iter().copied(),
        }
    }
}

pub trait Sequence {
    type Item;
    type Inner: Iterator<Item = Self::Item>;
    #[must_use]
    fn next(&mut self) -> Option<Self::Item>;
    #[must_use]
    fn peek(&mut self) -> Option<&Self::Item>;
    #[must_use]
    fn into_inner(self) -> Self::Inner;
}

impl<T: Ord> Sequence for BinaryHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<const N: usize, T: Ord> Sequence for DaryHeap<T, N> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        (self as &Self).peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_vec().into_iter()
    }
}

impl<I: Iterator> Sequence for Peekable<I> {
    type Item = I::Item;
    type Inner = Peekable<I>;
    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        Iterator::next(self)
    }
    #[inline]
    fn peek(&mut self) -> Option<&I::Item> {
        self.peek()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self
    }
}

impl<T> Sequence for VecDeque<T> {
    type Item = T;
    type Inner = std::collections::vec_deque::IntoIter<T>;
    #[inline]
    fn next(&mut self) -> Option<T> {
        self.pop_front()
    }
    #[inline]
    fn peek(&mut self) -> Option<&T> {
        self.front()
    }
    #[inline]
    fn into_inner(self) -> Self::Inner {
        self.into_iter()
    }
}

/// # Safety
///
/// * `Opaque` must aligned to 8 bytes.
#[allow(unsafe_code)]
pub unsafe trait Opaque: Copy + Send + Sync + FromBytes + IntoBytes + 'static {}

pub trait PackedRefMut {
    type T;
    fn get(&self) -> &Self::T;
    fn get_mut(&mut self) -> &mut Self::T;
}

impl<T> PackedRefMut for &mut T {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self
    }
}

#[repr(Rust, packed(4))]
pub struct PackedRefMut4<'b, T>(pub &'b mut T);

impl<'b, T> PackedRefMut for PackedRefMut4<'b, T> {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self.0
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self.0
    }
}

#[repr(Rust, packed(8))]
pub struct PackedRefMut8<'b, T>(pub &'b mut T);

impl<'a, T> PackedRefMut for PackedRefMut8<'a, T> {
    type T = T;
    #[inline(always)]
    fn get(&self) -> &T {
        self.0
    }
    #[inline(always)]
    fn get_mut(&mut self) -> &mut T {
        self.0
    }
}
