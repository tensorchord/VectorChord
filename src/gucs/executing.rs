use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

static NPROBE_2: GucSetting<i32> = GucSetting::<i32>::new(1);
static NPROBE_1: GucSetting<i32> = GucSetting::<i32>::new(10);

pub unsafe fn init() {
    GucRegistry::define_int_guc(
        "rabbithole.nprobe_2",
        "`nprobe` argument of rabbithole.",
        "`nprobe` argument of rabbithole.",
        &NPROBE_2,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "rabbithole.nprobe_1",
        "`nprobe` argument of rabbithole.",
        "`nprobe` argument of rabbithole.",
        &NPROBE_1,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub fn nprobe_2() -> u32 {
    NPROBE_2.get() as u32
}

pub fn nprobe_1() -> u32 {
    NPROBE_1.get() as u32
}
