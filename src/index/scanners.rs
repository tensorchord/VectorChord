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

use crate::index::collector::CollectorSender;
use crate::index::fetcher::Fetcher;
use algo::{Bump, Page, RelationPrefetch, RelationRead, RelationReadStream};
use pgrx::pg_sys::Datum;

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

pub trait SearchBuilder: 'static {
    type Options;

    type Opfamily;

    type Opaque: Copy;

    fn new(opfamily: Self::Opfamily) -> Self;

    unsafe fn add(&mut self, strategy: u16, datum: Option<Datum>);

    fn build<'b, R>(
        self,
        relation: &'b R,
        options: Self::Options,
        fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        sender: impl CollectorSender,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = Self::Opaque>;
}
