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

use crate::index::scanners::Io;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting, PostgresGucEnum};
use std::ffi::CString;

#[derive(Debug, Clone, Copy, PostgresGucEnum)]
pub enum PostgresIo {
    #[name = c"read_buffer"]
    ReadBuffer,
    #[name = c"prefetch_buffer"]
    PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    #[name = c"read_stream"]
    ReadStream,
}

static VCHORDRQ_QUERY_SAMPLING_ENABLE: GucSetting<bool> = GucSetting::<bool>::new(false);

static VCHORDRQ_QUERY_SAMPLING_MAX_RECORDS: GucSetting<i32> = GucSetting::<i32>::new(0);

static VCHORDRQ_QUERY_SAMPLING_RATE: GucSetting<f64> = GucSetting::<f64>::new(0.0);

static VCHORDG_ENABLE_SCAN: GucSetting<bool> = GucSetting::<bool>::new(true);

static VCHORDG_EF_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(64);

static VCHORDG_BEAM_SEARCH: GucSetting<i32> = GucSetting::<i32>::new(1);

static VCHORDG_MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static VCHORDG_IO_SEARCH: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

static VCHORDG_IO_RERANK: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

static VCHORDRQ_ENABLE_SCAN: GucSetting<bool> = GucSetting::<bool>::new(true);

static VCHORDRQ_PROBES: GucSetting<Option<CString>> = GucSetting::<Option<CString>>::new(Some(c""));

static VCHORDRQ_EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);

static VCHORDRQ_MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static VCHORDRQ_MAXSIM_REFINE: GucSetting<i32> = GucSetting::<i32>::new(0);

static VCHORDRQ_MAXSIM_THRESHOLD: GucSetting<i32> = GucSetting::<i32>::new(0);

static VCHORDRQ_PREFILTER: GucSetting<bool> = GucSetting::<bool>::new(false);

static VCHORDRQ_IO_SEARCH: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

static VCHORDRQ_IO_RERANK: GucSetting<PostgresIo> = GucSetting::<PostgresIo>::new(
    #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16"))]
    PostgresIo::PrefetchBuffer,
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    PostgresIo::ReadStream,
);

pub fn init() {
    GucRegistry::define_bool_guc(
        c"vchordrq.enable_scan",
        c"`enable_scan` argument of vchordrq.",
        c"`enable_scan` argument of vchordrq.",
        &VCHORDRQ_ENABLE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_string_guc(
        c"vchordrq.probes",
        c"`probes` argument of vchordrq.",
        c"`probes` argument of vchordrq.",
        &VCHORDRQ_PROBES,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"vchordrq.epsilon",
        c"`epsilon` argument of vchordrq.",
        c"`epsilon` argument of vchordrq.",
        &VCHORDRQ_EPSILON,
        0.0,
        4.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.max_scan_tuples",
        c"`max_scan_tuples` argument of vchordrq.",
        c"`max_scan_tuples` argument of vchordrq.",
        &VCHORDRQ_MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.maxsim_refine",
        c"`maxsim_refine` argument of vchordrq.",
        c"`maxsim_refine` argument of vchordrq.",
        &VCHORDRQ_MAXSIM_REFINE,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.maxsim_threshold",
        c"`maxsim_threshold` argument of vchordrq.",
        c"`maxsim_threshold` argument of vchordrq.",
        &VCHORDRQ_MAXSIM_THRESHOLD,
        0,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"vchordrq.prefilter",
        c"`prefilter` argument of vchordrq.",
        c"`prefilter` argument of vchordrq.",
        &VCHORDRQ_PREFILTER,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordrq.io_search",
        c"`io_search` argument of vchordrq.",
        c"`io_search` argument of vchordrq.",
        &VCHORDRQ_IO_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordrq.io_rerank",
        c"`io_rerank` argument of vchordrq.",
        c"`io_rerank` argument of vchordrq.",
        &VCHORDRQ_IO_RERANK,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        c"vchordrq.query_sampling_enable",
        c"`query_sampling_enable` argument of vchordrq.",
        c"`query_sampling_enable` argument of vchordrq.",
        &VCHORDRQ_QUERY_SAMPLING_ENABLE,
        GucContext::Suset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordrq.query_sampling_max_records",
        c"`query_sampling_max_records` argument of vchordrq.",
        c"`query_sampling_max_records` argument of vchordrq.",
        &VCHORDRQ_QUERY_SAMPLING_MAX_RECORDS,
        0,
        10000,
        GucContext::Suset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        c"vchordrq.query_sampling_rate",
        c"`query_sampling_rate` argument of vchordrq.",
        c"`query_sampling_rate` argument of vchordrq.",
        &VCHORDRQ_QUERY_SAMPLING_RATE,
        0.0,
        1.0,
        GucContext::Suset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(feature = "pg14")]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchordrq".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchordrq".as_ptr());
    }
    GucRegistry::define_bool_guc(
        c"vchordg.enable_scan",
        c"`enable_scan` argument of vchordg.",
        c"`enable_scan` argument of vchordg.",
        &VCHORDG_ENABLE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordg.ef_search",
        c"`ef_search` argument of vchordg.",
        c"`ef_search` argument of vchordg.",
        &VCHORDG_EF_SEARCH,
        1,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordg.beam_search",
        c"`beam_search` argument of vchordg.",
        c"`beam_search` argument of vchordg.",
        &VCHORDG_BEAM_SEARCH,
        1,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        c"vchordg.max_scan_tuples",
        c"`max_scan_tuples` argument of vchordg.",
        c"`max_scan_tuples` argument of vchordg.",
        &VCHORDG_MAX_SCAN_TUPLES,
        -1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordg.io_search",
        c"`io_search` argument of vchordg.",
        c"`io_search` argument of vchordg.",
        &VCHORDG_IO_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_enum_guc(
        c"vchordg.io_rerank",
        c"`io_rerank` argument of vchordg.",
        c"`io_rerank` argument of vchordg.",
        &VCHORDG_IO_RERANK,
        GucContext::Userset,
        GucFlags::default(),
    );
    unsafe {
        #[cfg(feature = "pg14")]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchordg".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchordg".as_ptr());
    }
}

pub fn vchordg_enable_scan() -> bool {
    VCHORDG_ENABLE_SCAN.get()
}

pub fn vchordg_ef_search() -> u32 {
    VCHORDG_EF_SEARCH.get() as u32
}

pub fn vchordg_beam_search() -> u32 {
    VCHORDG_BEAM_SEARCH.get() as u32
}

pub fn vchordg_max_scan_tuples() -> Option<u32> {
    let x = VCHORDG_MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}

pub fn vchordg_io_search() -> Io {
    match VCHORDG_IO_SEARCH.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}

pub fn vchordg_io_rerank() -> Io {
    match VCHORDG_IO_RERANK.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}

pub fn vchordrq_enable_scan() -> bool {
    VCHORDRQ_ENABLE_SCAN.get()
}

pub fn vchordrq_probes() -> Vec<u32> {
    match VCHORDRQ_PROBES.get() {
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

pub fn vchordrq_epsilon() -> f32 {
    VCHORDRQ_EPSILON.get() as f32
}

pub fn vchordrq_max_scan_tuples() -> Option<u32> {
    let x = VCHORDRQ_MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}

pub fn vchordrq_maxsim_refine() -> u32 {
    VCHORDRQ_MAXSIM_REFINE.get() as u32
}

pub fn vchordrq_maxsim_threshold() -> u32 {
    VCHORDRQ_MAXSIM_THRESHOLD.get() as u32
}

pub fn vchordrq_prefilter() -> bool {
    VCHORDRQ_PREFILTER.get()
}

pub fn vchordrq_io_search() -> Io {
    match VCHORDRQ_IO_SEARCH.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}

pub fn vchordrq_io_rerank() -> Io {
    match VCHORDRQ_IO_RERANK.get() {
        PostgresIo::ReadBuffer => Io::Plain,
        PostgresIo::PrefetchBuffer => Io::Simple,
        #[cfg(any(feature = "pg17", feature = "pg18"))]
        PostgresIo::ReadStream => Io::Stream,
    }
}

pub fn vchordrq_query_sampling_enable() -> bool {
    VCHORDRQ_QUERY_SAMPLING_ENABLE.get()
}

pub fn vchordrq_query_sampling_max_records() -> u32 {
    VCHORDRQ_QUERY_SAMPLING_MAX_RECORDS.get() as u32
}

pub fn vchordrq_query_sampling_rate() -> f64 {
    VCHORDRQ_QUERY_SAMPLING_RATE.get()
}
