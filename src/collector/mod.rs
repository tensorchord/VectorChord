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

use crate::collector::types::BGWORKER_LATCH;
use std::sync::atomic::Ordering;
pub use types::{Operator, Query};
pub use worker::QueryCollectorMaster;
use worker::QueryCollectorWorker;

mod hook;
mod mqueue;
mod types;
mod worker;

pub unsafe fn init() {
    use pgrx::bgworkers::{BackgroundWorkerBuilder, BgWorkerStartTime};
    use std::time::Duration;
    BackgroundWorkerBuilder::new("vchord_collector")
        .set_library("vchord")
        .set_function("_query_collector_main")
        .set_argument(None)
        .enable_shmem_access(None)
        .set_start_time(BgWorkerStartTime::PostmasterStart)
        .set_restart_time(Some(Duration::from_secs(15)))
        .load();
    hook::init();
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn _query_collector_main(_arg: pgrx::pg_sys::Datum) {
    use core::mem::transmute;
    use pgrx::pg_sys;
    unsafe {
        pg_sys::OwnLatch(BGWORKER_LATCH);
        pg_sys::pqsignal(
            pg_sys::SIGUSR1 as _,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::procsignal_sigusr1_handler as _),
        );
        pg_sys::pqsignal(
            pg_sys::SIGHUP as _,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::SignalHandlerForConfigReload as _),
        );
        pg_sys::pqsignal(pg_sys::SIGTERM as _, Some(handle_shutdown));
        pg_sys::pqsignal(pg_sys::SIGQUIT as _, Some(handle_shutdown));
        pg_sys::BackgroundWorkerUnblockSignals();
        let mut worker = QueryCollectorWorker::new();
        worker.run();
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn handle_shutdown(_postgres_signal_arg: i32) {
    worker::SHUTDOWN_REQUEST.store(true, Ordering::Relaxed);
}
