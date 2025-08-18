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

use crate::collector::QueryCollectorMaster;
use crate::collector::types::{
    BGWORKER_LATCH, BGWORKER_LATCH_SIZE, BGWORKER_TO_MAIN_MQ, COMMAND_MQ, MAX_QUERY_LEN,
    VCHORD_MAGIC,
};

#[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
static mut PREV_SHMEM_REQUEST_HOOK: pgrx::pg_sys::shmem_request_hook_type = None;

static mut PREV_SHMEM_STARTUP_HOOK: pgrx::pg_sys::shmem_startup_hook_type = None;

static mut PREV_OBJECT_ACCESS_HOOK: pgrx::pg_sys::object_access_hook_type = None;

unsafe fn shm_estimate_size() -> pgrx::pg_sys::Size {
    let mut estimator = pgrx::pg_sys::shm_toc_estimator {
        space_for_chunks: MAX_QUERY_LEN * 2 + BGWORKER_LATCH_SIZE,
        number_of_keys: 3,
    };
    unsafe { pgrx::pg_sys::shm_toc_estimate(&mut estimator) }
}

#[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn collector_shmem_request() {
    unsafe {
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_shmem_request) = PREV_SHMEM_REQUEST_HOOK {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| prev_shmem_request());
        }
        pgrx::pg_sys::RequestAddinShmemSpace(shm_estimate_size());
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn collector_shmem_startup() {
    unsafe {
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_shmem_startup) = PREV_SHMEM_STARTUP_HOOK {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| prev_shmem_startup());
        }
        let mut found = false;
        let pgws = pgrx::pg_sys::ShmemInitStruct(
            c"vchord_query_collector".as_ptr(),
            shm_estimate_size(),
            &mut found,
        );
        if !found {
            let toc = pgrx::pg_sys::shm_toc_create(VCHORD_MAGIC, pgws, shm_estimate_size());
            COMMAND_MQ =
                pgrx::pg_sys::shm_toc_allocate(toc, MAX_QUERY_LEN).cast::<pgrx::pg_sys::shm_mq>();
            pgrx::pg_sys::shm_toc_insert(toc, 0, COMMAND_MQ.cast());
            BGWORKER_TO_MAIN_MQ =
                pgrx::pg_sys::shm_toc_allocate(toc, MAX_QUERY_LEN).cast::<pgrx::pg_sys::shm_mq>();
            pgrx::pg_sys::shm_toc_insert(toc, 1, BGWORKER_TO_MAIN_MQ.cast());
            BGWORKER_LATCH = pgrx::pg_sys::shm_toc_allocate(toc, BGWORKER_LATCH_SIZE)
                .cast::<pgrx::pg_sys::Latch>();
            pgrx::pg_sys::shm_toc_insert(toc, 2, BGWORKER_LATCH.cast());
            pgrx::pg_sys::InitSharedLatch(BGWORKER_LATCH);
        } else {
            let toc = pgrx::pg_sys::shm_toc_attach(VCHORD_MAGIC, pgws);
            COMMAND_MQ = pgrx::pg_sys::shm_toc_lookup(toc, 0, false) as *mut _;
            BGWORKER_TO_MAIN_MQ = pgrx::pg_sys::shm_toc_lookup(toc, 1, false) as *mut _;
            BGWORKER_LATCH = pgrx::pg_sys::shm_toc_lookup(toc, 2, false) as *mut _;
        }
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn collector_object_access(
    access: pgrx::pg_sys::ObjectAccessType::Type,
    class_id: pgrx::pg_sys::Oid,
    object_id: pgrx::pg_sys::Oid,
    sub_id: ::std::os::raw::c_int,
    arg: *mut ::std::os::raw::c_void,
) {
    unsafe {
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_object_access_hook) = PREV_OBJECT_ACCESS_HOOK {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| {
                prev_object_access_hook(access, class_id, object_id, sub_id, arg)
            });
        }
        if access == pgrx::pg_sys::ObjectAccessType::OAT_DROP
            && class_id == pgrx::pg_sys::RelationRelationId
        {
            QueryCollectorMaster::drop(pgrx::pg_sys::MyDatabaseId.to_u32(), object_id.to_u32());
        }
    }
}

pub fn init() {
    assert!(crate::is_main());
    unsafe {
        #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
        {
            PREV_SHMEM_REQUEST_HOOK = pgrx::pg_sys::shmem_request_hook;
            pgrx::pg_sys::shmem_request_hook = Some(collector_shmem_request);
        }
        PREV_OBJECT_ACCESS_HOOK = pgrx::pg_sys::object_access_hook;
        PREV_SHMEM_STARTUP_HOOK = pgrx::pg_sys::shmem_startup_hook;
        pgrx::pg_sys::object_access_hook = Some(collector_object_access);
        pgrx::pg_sys::shmem_startup_hook = Some(collector_shmem_startup);
    }
}
