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
use crate::index::lazy_cell::LazyCell;
use algorithm::{Bump, RelationPrefetch, RelationRead, RelationReadStream};
use pgrx::pg_sys::Datum;

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
}

pub trait SearchBuilder: 'static {
    fn new(opfamily: Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'a, R>(
        self,
        relation: &'a R,
        options: SearchOptions,
        fetcher: impl SearchFetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream;
}

pub trait SearchFetcher {
    fn fetch(&mut self, ctid: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])>;
    fn filter(&self, key: [u16; 3]) -> bool;
}

impl<T: SearchFetcher, F: FnOnce() -> T> SearchFetcher for LazyCell<T, F> {
    fn fetch(&mut self, key: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])> {
        LazyCell::force_mut(self).fetch(key)
    }
    fn filter(&self, key: [u16; 3]) -> bool {
        LazyCell::force(self).filter(key)
    }
}
