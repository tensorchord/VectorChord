pub mod am;
pub mod am_options;
pub mod am_scan;
pub mod functions;
pub mod opclass;
pub mod utils;

pub unsafe fn init() {
    unsafe {
        am::init();
    }
}
