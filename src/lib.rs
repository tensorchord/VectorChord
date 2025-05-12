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

mod datatype;
mod index;
mod upgrade;

pgrx::pg_module_magic!();
pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[pgrx::pg_guard]
extern "C-unwind" fn _PG_init() {
    if unsafe { pgrx::pg_sys::IsUnderPostmaster } {
        pgrx::error!("vchord must be loaded via shared_preload_libraries.");
    }
    index::init();
    unsafe {
        #[cfg(any(feature = "pg13", feature = "pg14"))]
        pgrx::pg_sys::EmitWarningsOnPlaceholders(c"vchord".as_ptr());
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::MarkGUCPrefixReserved(c"vchord".as_ptr());
    }
}

#[cfg(not(target_endian = "little"))]
compile_error!("Target architecture is not supported.");

#[cfg(not(miri))]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL_ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;
