use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CStr;

static PREWARM_DIM: GucSetting<Option<&CStr>> =
    GucSetting::<Option<&CStr>>::new(Some(c"64,128,256,384,512,768,1024,1536"));

pub unsafe fn init() {
    GucRegistry::define_string_guc(
        "vchordrq.prewarm_dim",
        "prewarm_dim when the extension is loading.",
        "prewarm_dim when the extension is loading.",
        &PREWARM_DIM,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub fn prewarm() {
    if let Some(prewarm_dim) = PREWARM_DIM.get() {
        if let Ok(prewarm_dim) = prewarm_dim.to_str() {
            for dim in prewarm_dim.split(',') {
                if let Ok(dim) = dim.trim().parse::<usize>() {
                    crate::projection::prewarm(dim as _);
                } else {
                    pgrx::warning!("{dim:?} is not a valid integer");
                }
            }
        } else {
            pgrx::warning!("vchordrq.prewarm_dim is not a valid UTF-8 string");
        }
    }
}
