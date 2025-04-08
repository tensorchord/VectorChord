mod default;
mod maxsim;

use super::opclass::Opfamily;
use algorithm::RelationRead;
use pgrx::pg_sys::{Datum, ItemPointerData};
use std::cell::LazyCell;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub allows_skipping_rerank: bool,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub max_maxsim_tuples: Option<u32>,
    pub maxsim_threshold: u32,
}

pub trait SearchBuilder: 'static {
    fn new(opfamily: Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'a>(
        self,
        relation: impl RelationRead + 'a,
        options: SearchOptions,
        fetcher: impl SearchFetcher + 'a,
    ) -> Box<dyn Iterator<Item = (f32, ItemPointerData, bool)> + 'a>;
}

pub trait SearchFetcher {
    fn fetch(&mut self, ctid: ItemPointerData) -> Option<(&[Datum; 32], &[bool; 32])>;
}

impl<T: SearchFetcher, F: FnOnce() -> T> SearchFetcher for LazyCell<T, F> {
    fn fetch(&mut self, ctid: ItemPointerData) -> Option<(&[Datum; 32], &[bool; 32])> {
        LazyCell::force_mut(self).fetch(ctid)
    }
}
