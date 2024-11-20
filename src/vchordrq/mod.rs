mod algorithm;
mod gucs;
mod index;

pub unsafe fn init() {
    unsafe {
        index::init();
        gucs::init();
    }
}
