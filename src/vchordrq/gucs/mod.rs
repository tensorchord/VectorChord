pub mod executing;
pub mod prewarm;

pub unsafe fn init() {
    unsafe {
        executing::init();
        prewarm::init();
        prewarm::prewarm();
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchordrq".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchordrq".as_ptr());
    }
}
