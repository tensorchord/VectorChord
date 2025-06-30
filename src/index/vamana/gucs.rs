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

use crate::index::vamana::scanners::Io;
use pgrx::PostgresGucEnum;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

#[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
#[derive(Debug, Clone, Copy, PostgresGucEnum)]
pub enum PostgresIo {
    #[name = c"read_buffer"]
    ReadBuffer,
    #[name = c"prefetch_buffer"]
    PrefetchBuffer,
}

#[cfg(any(feature = "pg17", feature = "pg18"))]
#[derive(Debug, Clone, Copy, PostgresGucEnum)]
pub enum PostgresIo {
    #[name = c"read_buffer"]
    ReadBuffer,
    #[name = c"prefetch_buffer"]
    PrefetchBuffer,
    #[name = c"read_stream"]
    ReadStream,
}

static EF_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(64);
static BEAM_SIZE: GucSetting<i32> = GucSetting::<i32>::new(1);
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static IO_SEARCH: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

static IO_RERANK: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

pub fn init() {
    GucRegistry::define_int_guc(
        c"vamana.ef_search",
        c"`ef_search` argument of vamana.",
        c"`ef_search` argument of vamana.",
        &EF_SEARCH,
        1,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vamana.beam_search",
        c"`beam_search` argument of vamana.",
        c"`beam_search` argument of vamana.",
        &BEAM_SIZE,
        1,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vamana.max_scan_tuples",
        c"`max_scan_tuples` argument of vamana.",
        c"`max_scan_tuples` argument of vamana.",
        &MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vamana.io_search",
        c"`io_search` argument of vamana.",
        c"`io_search` argument of vamana.",
        &IO_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vamana.io_rerank",
        c"`io_rerank` argument of vamana.",
        c"`io_rerank` argument of vamana.",
        &IO_RERANK,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vamana".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vamana".as_ptr());
    }
}

pub fn ef_search() -> u32 {
    EF_SEARCH.get() as u32
}

pub fn beam_search() -> u32 {
    BEAM_SIZE.get() as u32
}

pub fn max_scan_tuples() -> Option<u32> {
    let x = MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}

pub fn io_search() -> Io {
    match IO_RERANK.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}

pub fn io_rerank() -> Io {
    match IO_RERANK.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}
