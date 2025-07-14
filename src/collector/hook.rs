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

static mut PREV_OBJECT_ACCESS_HOOK: pgrx::pg_sys::object_access_hook_type = None;

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
        PREV_OBJECT_ACCESS_HOOK = pgrx::pg_sys::object_access_hook;
        pgrx::pg_sys::object_access_hook = Some(collector_object_access);
    }
}
