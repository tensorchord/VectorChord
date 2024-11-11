use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

static PROBES: GucSetting<i32> = GucSetting::<i32>::new(10);
static EPSILON: GucSetting<f64> = GucSetting::<f64>::new(1.9);

pub unsafe fn init() {
    GucRegistry::define_int_guc(
        "rabbithole.probes",
        "`probes` argument of rabbithole.",
        "`probes` argument of rabbithole.",
        &PROBES,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        "rabbithole.epsilon",
        "`epsilon` argument of rabbithole.",
        "`epsilon` argument of rabbithole.",
        &EPSILON,
        1.0,
        4.0,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub fn probes() -> u32 {
    PROBES.get() as u32
}

pub fn epsilon() -> f32 {
    EPSILON.get() as f32
}
