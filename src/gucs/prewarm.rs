use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};
use std::ffi::CStr;

static PREWARM_DIM: GucSetting<Option<&CStr>> = GucSetting::<Option<&CStr>>::new(None);

pub unsafe fn init() {
    GucRegistry::define_string_guc(
        "rabbithole.prewarm_dim",
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
                    crate::algorithm::rabitq::prewarm(dim as _);
                } else {
                    pgrx::warning!("{dim:?} is not a valid integer");
                }
            }
        } else {
            pgrx::warning!("rabbithole.prewarm_dim is not a valid UTF-8 string");
        }
    }
}
