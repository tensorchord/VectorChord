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

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn rewrite_plan_state(
    node: *mut pgrx::pg_sys::PlanState,
    context: *mut core::ffi::c_void,
) -> bool {
    unsafe fn dirty_check_vchordg(index_relation: *mut pgrx::pg_sys::RelationData) -> Option<bool> {
        type FnPtr = unsafe extern "C-unwind" fn(
            *mut pgrx::pg_sys::RelationData,
            i32,
            i32,
        ) -> *mut pgrx::pg_sys::IndexScanDescData;
        unsafe {
            let index_relation = index_relation.as_ref()?;
            let indam = index_relation.rd_indam.as_ref()?;
            let ambeginscan = indam.ambeginscan.as_ref()?;
            Some(core::ptr::fn_addr_eq::<FnPtr, FnPtr>(
                *ambeginscan,
                crate::index::vchordg::am::ambeginscan,
            ))
        }
    }

    unsafe fn dirty_check_vchordrq(
        index_relation: *mut pgrx::pg_sys::RelationData,
    ) -> Option<bool> {
        type FnPtr = unsafe extern "C-unwind" fn(
            *mut pgrx::pg_sys::RelationData,
            i32,
            i32,
        ) -> *mut pgrx::pg_sys::IndexScanDescData;
        unsafe {
            let index_relation = index_relation.as_ref()?;
            let indam = index_relation.rd_indam.as_ref()?;
            let ambeginscan = indam.ambeginscan.as_ref()?;
            Some(core::ptr::fn_addr_eq::<FnPtr, FnPtr>(
                *ambeginscan,
                crate::index::vchordrq::am::ambeginscan,
            ))
        }
    }

    unsafe {
        if (*node).type_ == pgrx::pg_sys::NodeTag::T_IndexScanState {
            let node = node as *mut pgrx::pg_sys::IndexScanState;
            let index_relation = (*node).iss_RelationDesc;
            if (*node).iss_ScanDesc.is_null() {
                if Some(true) == dirty_check_vchordg(index_relation) {
                    use crate::index::vchordg::am::Scanner;

                    (*node).iss_ScanDesc = pgrx::pg_sys::index_beginscan(
                        (*node).ss.ss_currentRelation,
                        (*node).iss_RelationDesc,
                        (*(*node).ss.ps.state).es_snapshot,
                        #[cfg(feature = "pg18")]
                        std::ptr::null_mut(),
                        (*node).iss_NumScanKeys,
                        (*node).iss_NumOrderByKeys,
                    );

                    let scanner = &mut *((*(*node).iss_ScanDesc).opaque as *mut Scanner);
                    scanner.hack = std::ptr::NonNull::new(node);

                    if (*node).iss_NumRuntimeKeys == 0 || (*node).iss_RuntimeKeysReady {
                        pgrx::pg_sys::index_rescan(
                            (*node).iss_ScanDesc,
                            (*node).iss_ScanKeys,
                            (*node).iss_NumScanKeys,
                            (*node).iss_OrderByKeys,
                            (*node).iss_NumOrderByKeys,
                        );
                    }
                }
                if Some(true) == dirty_check_vchordrq(index_relation) {
                    use crate::index::vchordrq::am::Scanner;

                    (*node).iss_ScanDesc = pgrx::pg_sys::index_beginscan(
                        (*node).ss.ss_currentRelation,
                        (*node).iss_RelationDesc,
                        (*(*node).ss.ps.state).es_snapshot,
                        #[cfg(feature = "pg18")]
                        std::ptr::null_mut(),
                        (*node).iss_NumScanKeys,
                        (*node).iss_NumOrderByKeys,
                    );

                    let scanner = &mut *((*(*node).iss_ScanDesc).opaque as *mut Scanner);
                    scanner.hack = std::ptr::NonNull::new(node);

                    if (*node).iss_NumRuntimeKeys == 0 || (*node).iss_RuntimeKeysReady {
                        pgrx::pg_sys::index_rescan(
                            (*node).iss_ScanDesc,
                            (*node).iss_ScanKeys,
                            (*node).iss_NumScanKeys,
                            (*node).iss_OrderByKeys,
                            (*node).iss_NumOrderByKeys,
                        );
                    }
                }
            }
        }
        pgrx::pg_sys::planstate_tree_walker(node, Some(rewrite_plan_state), context)
    }
}

static mut PREV_EXECUTOR_START: pgrx::pg_sys::ExecutorStart_hook_type = None;

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn executor_start(
    query_desc: *mut pgrx::pg_sys::QueryDesc,
    eflags: core::ffi::c_int,
) {
    unsafe {
        use core::ptr::null_mut;
        use pgrx::pg_sys::submodules::ffi::pg_guard_ffi_boundary;
        if let Some(prev_executor_start) = PREV_EXECUTOR_START {
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| prev_executor_start(query_desc, eflags))
        } else {
            pgrx::pg_sys::standard_ExecutorStart(query_desc, eflags)
        }
        pg_guard_ffi_boundary(|| rewrite_plan_state((*query_desc).planstate, null_mut()));
    }
}

pub fn init() {
    assert!(crate::is_main());
    unsafe {
        PREV_EXECUTOR_START = pgrx::pg_sys::ExecutorStart_hook;
        pgrx::pg_sys::ExecutorStart_hook = Some(executor_start);
    }
}
