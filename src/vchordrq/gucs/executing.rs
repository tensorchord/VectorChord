use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CStr;

static PROBES: GucSetting<Option<&'static CStr>> = GucSetting::<Option<&CStr>>::new(Some(c"10"));
static EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);
static MAX_SCAN_TUPLES: GucSetting<i32> = GucSetting::<i32>::new(-1);

pub unsafe fn init() {
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
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
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
            result.push(current.take().expect("empty probes"));
            result
        }
    }
}

pub fn epsilon() -> f32 {
    EPSILON.get() as f32
}

pub fn max_scan_tuples() -> Option<u32> {
    let x = MAX_SCAN_TUPLES.get();
    if x < 0 {
        None
    } else {
        Some(x as u32)
    }
}
