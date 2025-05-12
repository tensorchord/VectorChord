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

use super::scanners::Io;
use pgrx::PostgresGucEnum;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CStr;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PostgresGucEnum)]
pub enum PostgresIo {
    read_buffer,
    prefetch_buffer,
    #[cfg(feature = "pg17")]
    read_stream,
}

static PREWARM_DIM: GucSetting<Option<&CStr>> =
    GucSetting::<Option<&CStr>>::new(Some(c"64,128,256,384,512,768,1024,1536"));

static PROBES: GucSetting<Option<&'static CStr>> = GucSetting::<Option<&CStr>>::new(Some(c""));
static EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static MAXSIM_REFINE: GucSetting<i32> = GucSetting::<i32>::new(0);
static MAXSIM_THRESHOLD: GucSetting<i32> = GucSetting::<i32>::new(0);

static PRERERANK_FILTERING: GucSetting<bool> = GucSetting::<bool>::new(false);

static IO_SEARCH: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::prefetch_buffer,
    #[cfg(feature = "pg17")]
    PostgresIo::read_stream,
);

static IO_RERANK: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::prefetch_buffer,
    #[cfg(feature = "pg17")]
    PostgresIo::read_stream,
);

pub fn init() {
    GucRegistry::define_string_guc(
        "vchordrq.probes",
        "`probes` argument of vchordrq.",
        "`probes` argument of vchordrq.",
        &PROBES,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        "vchordrq.epsilon",
        "`epsilon` argument of vchordrq.",
        "`epsilon` argument of vchordrq.",
        &EPSILON,
        0.0,
        4.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vchordrq.max_scan_tuples",
        "`max_scan_tuples` argument of vchordrq.",
        "`max_scan_tuples` argument of vchordrq.",
        &MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        "vchordrq.prewarm_dim",
        "prewarm_dim when the extension is loading.",
        "prewarm_dim when the extension is loading.",
        &PREWARM_DIM,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vchordrq.maxsim_refine",
        "`maxsim_refine` argument of vchordrq.",
        "`maxsim_refine` argument of vchordrq.",
        &MAXSIM_REFINE,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vchordrq.maxsim_threshold",
        "`maxsim_threshold` argument of vchordrq.",
        "`maxsim_threshold` argument of vchordrq.",
        &MAXSIM_THRESHOLD,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vchordrq.prererank_filtering",
        "`prererank_filtering` argument of vchordrq.",
        "`prererank_filtering` argument of vchordrq.",
        &PRERERANK_FILTERING,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        "vchordrq.io_search",
        "`io_search` argument of vchordrq.",
        "`io_search` argument of vchordrq.",
        &IO_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        "vchordrq.io_rerank",
        "`io_rerank` argument of vchordrq.",
        "`io_rerank` argument of vchordrq.",
        &IO_RERANK,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchordrq".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
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

pub fn prewarm_dim() -> Vec<u32> {
    if let Some(prewarm_dim) = PREWARM_DIM.get() {
        if let Ok(prewarm_dim) = prewarm_dim.to_str() {
            let mut result = Vec::new();
            for dim in prewarm_dim.split(',') {
                if let Ok(dim) = dim.trim().parse::<u32>() {
                    result.push(dim);
                } else {
                    pgrx::warning!("{dim:?} is not a valid integer");
                }
            }
            result
        } else {
            pgrx::warning!("vchordrq.prewarm_dim is not a valid UTF-8 string");
            Vec::new()
        }
    } else {
        Vec::new()
    }
}

pub fn prererank_filtering() -> bool {
    PRERERANK_FILTERING.get()
}

pub fn io_search() -> Io {
    match IO_RERANK.get() {
        PostgresIo::read_buffer => Io::Plain,
        PostgresIo::prefetch_buffer => Io::Simple,
        #[cfg(feature = "pg17")]
        PostgresIo::read_stream => Io::Stream,
    }
}

pub fn io_rerank() -> Io {
    match IO_RERANK.get() {
        PostgresIo::read_buffer => Io::Plain,
        PostgresIo::prefetch_buffer => Io::Simple,
        #[cfg(feature = "pg17")]
        PostgresIo::read_stream => Io::Stream,
    }
}
