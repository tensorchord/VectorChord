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

use crate::index::vchordrq::scanners::Io;
use pgrx::PostgresGucEnum;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CString;

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

static PROBES: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(Some(c""));
static EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static MAXSIM_REFINE: GucSetting<i32> = GucSetting::<i32>::new(0);
static MAXSIM_THRESHOLD: GucSetting<i32> = GucSetting::<i32>::new(0);

static PREFILTER: GucSetting<bool> = GucSetting::<bool>::new(false);

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
    GucRegistry::define_string_guc(
        c"vchordrq.probes",
        c"`probes` argument of vchordrq.",
        c"`probes` argument of vchordrq.",
        &PROBES,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"vchordrq.epsilon",
        c"`epsilon` argument of vchordrq.",
        c"`epsilon` argument of vchordrq.",
        &EPSILON,
        0.0,
        4.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.max_scan_tuples",
        c"`max_scan_tuples` argument of vchordrq.",
        c"`max_scan_tuples` argument of vchordrq.",
        &MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.maxsim_refine",
        c"`maxsim_refine` argument of vchordrq.",
        c"`maxsim_refine` argument of vchordrq.",
        &MAXSIM_REFINE,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.maxsim_threshold",
        c"`maxsim_threshold` argument of vchordrq.",
        c"`maxsim_threshold` argument of vchordrq.",
        &MAXSIM_THRESHOLD,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"vchordrq.prefilter",
        c"`prefilter` argument of vchordrq.",
        c"`prefilter` argument of vchordrq.",
        &PREFILTER,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordrq.io_search",
        c"`io_search` argument of vchordrq.",
        c"`io_search` argument of vchordrq.",
        &IO_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordrq.io_rerank",
        c"`io_rerank` argument of vchordrq.",
        c"`io_rerank` argument of vchordrq.",
        &IO_RERANK,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchordrq".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchordrq".as_ptr());
    }
}

pub fn probes() -> Vec<u32> {
    match PROBES.get() {
        None => Vec::new(),
        Some(probes) => {
            let mut result = Vec::new();
            let mut current = None;
            for &c in probes.to_bytes() {
                match c {
                    b' ' => continue,
                    b',' => result.push(current.take().expect("empty probes")),
                    b'0'..=b'9' => {
                        if let Some(x) = current.as_mut() {
                            *x = *x * 10 + (c - b'0') as u32;
                        } else {
                            current = Some((c - b'0') as u32);
                        }
                    }
                    c => pgrx::error!("unknown character in probes: ASCII = {c}"),
                }
            }
            if let Some(current) = current {
                result.push(current);
            }
            result
        }
    }
}

pub fn epsilon() -> f32 {
    EPSILON.get() as f32
}

pub fn max_scan_tuples() -> Option<u32> {
    let x = MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}

pub fn maxsim_refine() -> u32 {
    MAXSIM_REFINE.get() as u32
}

pub fn maxsim_threshold() -> u32 {
    MAXSIM_THRESHOLD.get() as u32
}

pub fn prefilter() -> bool {
    PREFILTER.get()
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
