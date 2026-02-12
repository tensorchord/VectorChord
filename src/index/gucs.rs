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
use std::ffi::{CStr, CString};

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

static mut VCHORDG_EF_SEARCH_CONFIG: *mut pgrx::pg_sys::config_generic = core::ptr::null_mut();

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

static mut VCHORDRQ_PROBES_CONFIG: *mut pgrx::pg_sys::config_generic = core::ptr::null_mut();

static VCHORDRQ_EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);

static mut VCHORDRQ_EPSILON_CONFIG: *mut pgrx::pg_sys::config_generic = core::ptr::null_mut();

static VCHORDRQ_MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

static VCHORDRQ_MAXSIM_REFINE: GucSetting<i32> = GucSetting::<i32>::new(0);

static mut VCHORDRQ_MAXSIM_REFINE_CONFIG: *mut pgrx::pg_sys::config_generic = core::ptr::null_mut();

static VCHORDRQ_MAXSIM_THRESHOLD: GucSetting<i32> = GucSetting::<i32>::new(0);

static mut VCHORDRQ_MAXSIM_THRESHOLD_CONFIG: *mut pgrx::pg_sys::config_generic =
    core::ptr::null_mut();

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
    assert!(crate::is_main());
    let targets = vec![
        (c"vchordg.ef_search", &raw mut VCHORDG_EF_SEARCH_CONFIG),
        (c"vchordrq.epsilon", &raw mut VCHORDRQ_EPSILON_CONFIG),
        (
            c"vchordrq.maxsim_refine",
            &raw mut VCHORDRQ_MAXSIM_REFINE_CONFIG,
        ),
        (
            c"vchordrq.maxsim_threshold",
            &raw mut VCHORDRQ_MAXSIM_THRESHOLD_CONFIG,
        ),
        (c"vchordrq.probes", &raw mut VCHORDRQ_PROBES_CONFIG),
    ];
    #[cfg(any(feature = "pg14", feature = "pg15"))]
    unsafe {
        let len = pgrx::pg_sys::GetNumConfigOptions() as usize;
        let arr = pgrx::pg_sys::get_guc_variables();
        let mut sources = (0..len).map(|i| arr.add(i).read());
        debug_assert!(targets.is_sorted_by(|(a, _), (b, _)| guc_name_compare(a, b).is_le()));
        for (name, ptr) in targets {
            *ptr = loop {
                if let Some(source) = sources.next() {
                    if !(*source).name.is_null() && CStr::from_ptr((*source).name) == name {
                        break source;
                    } else {
                        continue;
                    }
                } else {
                    pgrx::error!("failed to find GUC {name:?}");
                }
            };
            assert!(check(*ptr, name), "failed to find GUC {name:?}");
        }
    }
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    unsafe {
        use pgrx::pg_sys::PGERROR;
        for (name, ptr) in targets {
            *ptr = pgrx::pg_sys::find_option(name.as_ptr(), false, false, PGERROR as _);
            assert!(check(*ptr, name), "failed to find GUC {name:?}");
        }
    }
}

unsafe fn check(p: *mut pgrx::pg_sys::config_generic, name: &CStr) -> bool {
    if p.is_null() {
        return false;
    }
    if unsafe { (*p).flags } & pgrx::pg_sys::GUC_CUSTOM_PLACEHOLDER as core::ffi::c_int != 0 {
        return false;
    }
    if unsafe { (*p).name }.is_null() {
        return false;
    }
    if unsafe { CStr::from_ptr((*p).name) != name } {
        return false;
    }
    true
}

pub fn vchordg_enable_scan() -> bool {
    VCHORDG_ENABLE_SCAN.get()
}

pub fn vchordg_ef_search(index: pgrx::pg_sys::Relation) -> u32 {
    fn parse(x: i32) -> u32 {
        x as u32
    }
    assert!(crate::is_main());
    const DEFAULT: i32 = 64;
    if unsafe { (*VCHORDG_EF_SEARCH_CONFIG).source } != pgrx::pg_sys::GucSource::PGC_S_DEFAULT {
        let value = VCHORDG_EF_SEARCH.get();
        parse(value)
    } else {
        use crate::index::vchordg::am::Reloption;
        let value = unsafe { Reloption::ef_search((*index).rd_options as _, DEFAULT) };
        parse(value)
    }
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

pub unsafe fn vchordrq_probes(index: pgrx::pg_sys::Relation) -> Vec<u32> {
    fn parse(value: &CStr) -> Vec<u32> {
        let mut result = Vec::new();
        let mut current = None;
        for &c in value.to_bytes() {
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
    assert!(crate::is_main());
    const DEFAULT: &CStr = c"";
    if unsafe { (*VCHORDRQ_PROBES_CONFIG).source } != pgrx::pg_sys::GucSource::PGC_S_DEFAULT {
        let value = VCHORDRQ_PROBES.get();
        parse(value.as_deref().unwrap_or(DEFAULT))
    } else {
        use crate::index::vchordrq::am::Reloption;
        let value = unsafe { Reloption::probes((*index).rd_options as _, DEFAULT) };
        parse(value)
    }
}

pub unsafe fn vchordrq_epsilon(index: pgrx::pg_sys::Relation) -> f32 {
    fn parse(x: f64) -> f32 {
        x as f32
    }
    assert!(crate::is_main());
    const DEFAULT: f64 = 1.9;
    if unsafe { (*VCHORDRQ_EPSILON_CONFIG).source } != pgrx::pg_sys::GucSource::PGC_S_DEFAULT {
        let value = VCHORDRQ_EPSILON.get();
        parse(value)
    } else {
        use crate::index::vchordrq::am::Reloption;
        let value = unsafe { Reloption::epsilon((*index).rd_options as _, DEFAULT) };
        parse(value)
    }
}

pub fn vchordrq_max_scan_tuples() -> Option<u32> {
    let x = VCHORDRQ_MAX_SCAN_TUPLES.get();
    if x < 0 { None } else { Some(x as u32) }
}

pub fn vchordrq_maxsim_refine(index: pgrx::pg_sys::Relation) -> u32 {
    fn parse(x: i32) -> u32 {
        x as u32
    }
    assert!(crate::is_main());
    const DEFAULT: i32 = 0;
    if unsafe { (*VCHORDRQ_MAXSIM_REFINE_CONFIG).source } != pgrx::pg_sys::GucSource::PGC_S_DEFAULT
    {
        let value = VCHORDRQ_MAXSIM_REFINE.get();
        parse(value)
    } else {
        use crate::index::vchordrq::am::Reloption;
        let value = unsafe { Reloption::maxsim_refine((*index).rd_options as _, DEFAULT) };
        parse(value)
    }
}

pub fn vchordrq_maxsim_threshold(index: pgrx::pg_sys::Relation) -> u32 {
    fn parse(x: i32) -> u32 {
        x as u32
    }
    assert!(crate::is_main());
    const DEFAULT: i32 = 0;
    if unsafe { (*VCHORDRQ_MAXSIM_THRESHOLD_CONFIG).source }
        != pgrx::pg_sys::GucSource::PGC_S_DEFAULT
    {
        let value = VCHORDRQ_MAXSIM_THRESHOLD.get();
        parse(value)
    } else {
        use crate::index::vchordrq::am::Reloption;
        let value = unsafe { Reloption::maxsim_threshold((*index).rd_options as _, DEFAULT) };
        parse(value)
    }
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

#[allow(dead_code)]
fn guc_name_compare(a: &CStr, b: &CStr) -> std::cmp::Ordering {
    let (a, b) = (a.to_bytes_with_nul(), b.to_bytes_with_nul());
    let mut i = 0;
    while a[i] != 0 && b[i] != 0 {
        let a = a[i].to_ascii_lowercase();
        let b = b[i].to_ascii_lowercase();
        if a != b {
            return Ord::cmp(&a, &b);
        }
        i += 1;
    }
    if b[i] != 0 {
        std::cmp::Ordering::Less
    } else if a[i] != 0 {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}
