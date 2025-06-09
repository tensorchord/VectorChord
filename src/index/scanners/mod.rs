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

mod default;
mod maxsim;

use super::opclass::Opfamily;
use algorithm::{Bump, RelationPrefetch, RelationRead, RelationReadStream, Sequence};
use pgrx::pg_sys::Datum;
use std::cell::LazyCell;
use std::ops::DerefMut;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug, Clone, Copy)]
pub enum Io {
    Plain,
    Simple,
    #[cfg_attr(
        any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"),
        expect(dead_code)
    )]
    Stream,
}

#[derive(Debug)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub io_search: Io,
    pub io_rerank: Io,
    pub prefilter: bool,
}

pub trait SearchBuilder: 'static {
    fn new(opfamily: Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'a, R>(
        self,
        relation: &'a R,
        options: SearchOptions,
        fetcher: impl Fetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream;
}

pub trait Fetcher {
    type Tuple<'a>: Tuple
    where
        Self: 'a;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>>;
}

pub trait Tuple {
    fn build(&mut self) -> (&[Datum; 32], &[bool; 32]);
    fn filter(&mut self) -> bool;
}

impl<T: Fetcher, F: FnOnce() -> T> Fetcher for LazyCell<T, F> {
    type Tuple<'a>
        = T::Tuple<'a>
    where
        Self: 'a;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>> {
        self.deref_mut().fetch(key)
    }
}

pub struct Filter<S, P> {
    sequence: S,
    predicate: P,
}

impl<S, P> Sequence for Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    type Inner = S::Inner;

    fn next(&mut self) -> Option<Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.next()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.sequence.into_inner()
    }
}

pub fn filter<S, P>(sequence: S, predicate: P) -> Filter<S, P> {
    Filter {
        sequence,
        predicate,
    }
}
