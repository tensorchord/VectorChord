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

use algo::prefetcher::{Prefetcher, PrefetcherSequenceFamily};
use algo::{Fetch, RelationRead};
use std::collections::{BinaryHeap, VecDeque};
use std::marker::PhantomData;

pub struct Candidates<'a, P, F, R>
where
    P: Prefetcher<'a>,
    P::Item: Fetch,
{
    beam: usize,
    front: Option<P>,
    heap: BinaryHeap<P::Item>,
    prefetch: F,
    _phantom: PhantomData<fn(&'a R) -> &'a R>,
}

impl<'a, P, F, R> Candidates<'a, P, F, R>
where
    P: Prefetcher<'a, R = R>,
    P::Item: Fetch + Ord,
    F: PrefetcherSequenceFamily<'a, R, P<VecDeque<P::Item>> = P>,
    R: RelationRead,
{
    pub fn new(beam: usize, prefetch: F) -> Self {
        assert_ne!(beam, 0);
        Self {
            beam,
            front: None,
            heap: BinaryHeap::new(),
            prefetch,
            _phantom: PhantomData,
        }
    }
    pub fn pop(&mut self) -> Option<(P::Item, P::Guards)> {
        if let Some(front) = self.front.as_mut()
            && let Some(item) = front.next()
        {
            return Some(item);
        }
        self.front = Some(
            self.prefetch.prefetch(
                (0..self.beam)
                    .flat_map(|_| self.heap.pop())
                    .collect::<VecDeque<_>>(),
            ),
        );
        if let Some(front) = self.front.as_mut()
            && let Some(item) = front.next()
        {
            return Some(item);
        }
        None
    }
    pub fn push(&mut self, item: P::Item) {
        self.heap.push(item);
    }
}
