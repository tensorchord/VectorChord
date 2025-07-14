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

pub use types::Query;
pub use worker::QueryCollectorMaster;
use worker::{QueryCollectorWorker, set_command};

mod hook;
mod mqueue;
mod types;
mod worker;

pub unsafe fn init() {
    use pgrx::bgworkers::{BackgroundWorkerBuilder, BgWorkerStartTime};
    use std::time::Duration;
    BackgroundWorkerBuilder::new("vchord_collector")
        .set_library("vchord")
        .set_function("_query_logger_main")
        .set_argument(None)
        .enable_shmem_access(None)
        .set_start_time(BgWorkerStartTime::PostmasterStart)
        .set_restart_time(Some(Duration::from_secs(15)))
        .load();
    hook::init()
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn _query_logger_main(_arg: pgrx::pg_sys::Datum) {
    use core::mem::transmute;
    use pgrx::pg_sys;
    unsafe {
        pg_sys::pqsignal(pg_sys::SIGTERM as i32, Some(handle_sigterm));
        pg_sys::pqsignal(
            pg_sys::SIGHUP as i32,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::SignalHandlerForConfigReload as _),
        );
        pg_sys::pqsignal(
            pg_sys::SIGUSR1 as i32,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::procsignal_sigusr1_handler as _),
        );
        pg_sys::BackgroundWorkerUnblockSignals();
        let mut worker = QueryCollectorWorker::new();
        worker.run();
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn handle_sigterm(_postgres_signal_arg: i32) {
    pgrx::warning!("Collector worker: Received SIGTERM, shutting down background worker");
    unsafe { set_command(types::Command::Shutdown) };
}
