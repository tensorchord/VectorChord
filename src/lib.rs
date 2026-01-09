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

#![allow(unsafe_code)]
#![deny(ffi_unwind_calls)]

mod datatype;
mod index;
mod recorder;
mod upgrade;

pgrx::pg_module_magic!(
    name = c"vchord",
    version = {
        const RAW: &str = env!("VCHORD_VERSION");
        const BUFFER: [u8; RAW.len() + 1] = {
            let mut buffer = [0u8; RAW.len() + 1];
            let mut i = 0_usize;
            while i < RAW.len() {
                buffer[i] = RAW.as_bytes()[i];
                i += 1;
            }
            buffer
        };
        const STR: &::core::ffi::CStr =
            if let Ok(s) = ::core::ffi::CStr::from_bytes_with_nul(&BUFFER) {
                s
            } else {
                panic!("there are null characters in VCHORD_VERSION")
            };
        const { STR }
    }
);
pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[pgrx::pg_guard]
#[unsafe(export_name = "_PG_init")]
unsafe extern "C-unwind" fn _pg_init() {
    if !unsafe { pgrx::pg_sys::process_shared_preload_libraries_in_progress } {
        pgrx::error!("vchord must be loaded via shared_preload_libraries.");
    }
    IS_MAIN.set(true);
    index::init();
    recorder::init();
    unsafe {
        #[cfg(feature = "pg14")]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchord".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchord".as_ptr());
    }
}

std::thread_local! {
    static IS_MAIN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[must_use]
fn is_main() -> bool {
    IS_MAIN.get()
}

#[cfg(not(panic = "unwind"))]
compile_error!("This crate must be compiled with `-Cpanic=unwind`.");

#[cfg(not(miri))]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;
