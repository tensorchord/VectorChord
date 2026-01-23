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

use crate::fetch::Fetch;
use crate::relation::{
    Hints, ReadStream, RelationPrefetch, RelationRead, RelationReadStream, RelationReadTypes,
};
use dary_heap::DaryHeap;
use std::collections::{BinaryHeap, VecDeque};

pub const WINDOW_SIZE: usize = 32;
const _: () = assert!(WINDOW_SIZE > 0);

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

impl<I: Iterator> Sequence for std::iter::Peekable<I> {
    type Item = I::Item;
    type Inner = std::iter::Peekable<I>;
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

pub trait Prefetcher<'b>: IntoIterator
where
    Self::Item: Fetch<'b>,
{
    type R: RelationRead + 'b;
    type Guards: ExactSizeIterator<Item = <Self::R as RelationReadTypes>::ReadGuard<'b>>;

    #[must_use]
    fn next(&mut self) -> Option<(Self::Item, Self::Guards)>;
    #[must_use]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(Self::Item, Self::Guards)>;
}

pub struct PlainPrefetcher<'b, R, S: Sequence> {
    relation: &'b R,
    sequence: S,
}

impl<'b, R, S: Sequence> PlainPrefetcher<'b, R, S> {
    #[inline]
    pub fn new(relation: &'b R, sequence: S) -> Self {
        Self { relation, sequence }
    }
}

impl<'b, R, S: Sequence> IntoIterator for PlainPrefetcher<'b, R, S> {
    type Item = S::Item;

    type IntoIter = S::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.sequence.into_inner()
    }
}

impl<'b, R: RelationRead, S: Sequence> Prefetcher<'b> for PlainPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;
    type Guards = PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>;

    #[inline]
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        S::Item,
        PlainPrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        if !predicate(self.sequence.peek()?) {
            return None;
        }
        let e = self.sequence.next()?;
        let list = e.fetch();
        Some((
            e,
            PlainPrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }
}

pub struct PlainPrefetcherGuards<'b, R, L> {
    relation: &'b R,
    list: L,
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> Iterator for PlainPrefetcherGuards<'b, R, L> {
    type Item = R::ReadGuard<'b>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> ExactSizeIterator
    for PlainPrefetcherGuards<'b, R, L>
{
}

pub struct SimplePrefetcher<'b, R, S: Sequence> {
    relation: &'b R,
    window: VecDeque<S::Item>,
    sequence: S,
}

impl<'b, R, S: Sequence> SimplePrefetcher<'b, R, S> {
    #[inline]
    pub fn new(relation: &'b R, sequence: S) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            sequence,
        }
    }
}

impl<'b, R, S: Sequence> IntoIterator for SimplePrefetcher<'b, R, S> {
    type Item = S::Item;

    type IntoIter = std::iter::Chain<std::collections::vec_deque::IntoIter<S::Item>, S::Inner>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.sequence.into_inner())
    }
}

impl<'b, R: RelationRead + RelationPrefetch, S: Sequence> Prefetcher<'b>
    for SimplePrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;
    type Guards = SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>;

    #[inline]
    fn next(
        &mut self,
    ) -> Option<(
        Self::Item,
        SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = self.window.pop_front()?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&S::Item) -> bool,
    ) -> Option<(
        S::Item,
        SimplePrefetcherGuards<'b, R, <S::Item as Fetch<'b>>::Iter>,
    )> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.sequence.next()
        {
            for id in e.fetch() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let e = self.window.pop_front_if(move |x| predicate(x))?;
        let list = e.fetch();
        Some((
            e,
            SimplePrefetcherGuards {
                relation: self.relation,
                list,
            },
        ))
    }
}

pub struct SimplePrefetcherGuards<'b, R, L> {
    relation: &'b R,
    list: L,
}

impl<'b, R: RelationRead, L: Iterator<Item = u32>> Iterator for SimplePrefetcherGuards<'b, R, L> {
    type Item = R::ReadGuard<'b>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.list.next()?;
        Some(self.relation.read(id))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<'b, R: RelationRead, L: ExactSizeIterator<Item = u32>> ExactSizeIterator
    for SimplePrefetcherGuards<'b, R, L>
{
}

pub struct StreamPrefetcherSequence<S>(S);

impl<S: Sequence> Iterator for StreamPrefetcherSequence<S> {
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

pub struct StreamPrefetcher<'b, R: RelationReadStream + 'b, S: Sequence>
where
    S::Item: Fetch<'b>,
{
    stream: R::ReadStream<'b, StreamPrefetcherSequence<S>>,
}

impl<'b, R: RelationReadStream, S: Sequence> StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    #[inline]
    pub fn new(relation: &'b R, sequence: S, hints: Hints) -> Self {
        let stream = relation.read_stream(StreamPrefetcherSequence(sequence), hints);
        Self { stream }
    }
}

impl<'b, R: RelationReadStream, S: Sequence> IntoIterator for StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type Item = S::Item;

    type IntoIter = <R::ReadStream<'b, StreamPrefetcherSequence<S>> as ReadStream<'b>>::Inner;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<'b, R: RelationRead + RelationReadStream, S: Sequence> Prefetcher<'b>
    for StreamPrefetcher<'b, R, S>
where
    S::Item: Fetch<'b>,
{
    type R = R;

    type Guards = <R::ReadStream<'b, StreamPrefetcherSequence<S>> as ReadStream<'b>>::Guards;

    #[inline]
    fn next(&mut self) -> Option<(S::Item, Self::Guards)> {
        self.stream.next()
    }

    #[inline]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(S::Item, Self::Guards)> {
        self.stream.next_if(predicate)
    }
}

pub trait PrefetcherSequenceFamily<'b, R> {
    type P<S: Sequence>: Prefetcher<'b, R = R, Item = S::Item>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>;

    fn is_not_plain(&self) -> bool;
}

pub trait PrefetcherHeapFamily<'b, R> {
    type P<T>: Prefetcher<'b, R = R, Item = T>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>;

    fn is_not_plain(&self) -> bool;
}
