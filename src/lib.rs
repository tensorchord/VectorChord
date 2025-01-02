#![allow(clippy::collapsible_else_if)]
#![allow(clippy::infallible_destructuring_match)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

mod datatype;
mod postgres;
mod projection;
mod upgrade;
mod utils;
mod vchordrq;
mod vchordrqfscan;

pgrx::pg_module_magic!();
pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[pgrx::pg_guard]
unsafe extern "C" fn _PG_init() {
    if unsafe { pgrx::pg_sys::IsUnderPostmaster } {
        pgrx::error!("vchord must be loaded via shared_preload_libraries.");
    }
    unsafe {
        vchordrq::init();
        vchordrqfscan::init();

        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchord".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchord".as_ptr());
    }
}

#[cfg(not(target_endian = "little"))]
compile_error!("Target architecture is not supported.");
