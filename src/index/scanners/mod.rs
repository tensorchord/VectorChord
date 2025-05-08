mod default;
mod maxsim;

use super::opclass::Opfamily;
use crate::index::lazy_cell::LazyCell;
use algorithm::{Bump, RelationPrefetch, RelationReadStream};
use pgrx::pg_sys::Datum;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug, Clone, Copy)]
pub enum SearchIo {
    ReadBuffer,
    PrefetchBuffer,
    #[cfg_attr(
        any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"),
        expect(dead_code)
    )]
    ReadStream,
}

#[derive(Debug)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub io_rerank: SearchIo,
}

pub trait SearchBuilder: 'static {
    fn new(opfamily: Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'a>(
        self,
        relation: &'a (impl RelationPrefetch + RelationReadStream),
        options: SearchOptions,
        fetcher: impl SearchFetcher + 'a,
        bump: &'a impl Bump,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a>;
}

pub trait SearchFetcher {
    fn filter_fetch(&mut self, ctid: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])>;
    fn filter_only(&self, ctid: [u16; 3]) -> bool;
}

impl<T: SearchFetcher, F: FnOnce() -> T> SearchFetcher for LazyCell<T, F> {
    fn filter_fetch(&mut self, key: [u16; 3]) -> Option<(&[Datum; 32], &[bool; 32])> {
        LazyCell::force_mut(self).filter_fetch(key)
    }
    fn filter_only(&self, key: [u16; 3]) -> bool {
        LazyCell::force(self).filter_only(key)
    }
}
