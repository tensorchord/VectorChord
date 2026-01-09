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

use crate::index::vchordrq::am::PostgresRelation;
use crate::index::vchordrq::opclass::opfamily;
use vchordrq::MaintainChooser;

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let index_relation = unsafe { (*info).index };
    unsafe {
        sequential_vacuumcleanup(index_relation, || (), || ());
    }
    stats
}

unsafe fn sequential_vacuumcleanup(
    index_relation: pgrx::pg_sys::Relation,
    sync_0: impl FnOnce(),
    sync_1: impl FnOnce(),
) {
    struct ChooseAll;
    impl MaintainChooser for ChooseAll {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
    }

    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let check = || unsafe {
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        pgrx::pg_sys::vacuum_delay_point();
        #[cfg(feature = "pg18")]
        pgrx::pg_sys::vacuum_delay_point(false);
    };

    sync_0();

    let mut chooser = ChooseAll;
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_1();
}
