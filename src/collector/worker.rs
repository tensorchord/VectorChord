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

use crate::collector::mqueue::{MessageQueueError, MessageQueueReceiver, MessageQueueSender};
use crate::collector::types::{
    BGWORKER_LATCH, BGWORKER_TO_MAIN_MQ, BgWorkerLockGuard, COMMAND_MQ, CollectorState, Command,
    MAX_FLOATS_PER_INDEX, Query, VCHORD_MAGIC,
};
use crate::index::gucs;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHUTDOWN_REQUEST: AtomicBool = AtomicBool::new(false);
pub const MULTI_ACCESS_LOCK: u32 = 0;
pub const BACKGROUND_WORKER_LOCK: u32 = 1;

pub struct QueryCollectorMaster {}

impl QueryCollectorMaster {
    pub unsafe fn push(query: Query) {
        let multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, false);
        let collector_idle = {
            let collector_idle = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, false);
            collector_idle.is_success()
        };
        if !multi_access_lock.is_success() || !collector_idle {
            return;
        }
        unsafe { set_command(Command::Push(query)) };
    }
    pub unsafe fn drop(database_oid: u32, index_oid: u32) {
        let _multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, true);
        let _collector_idle = {
            let collector_idle = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
            collector_idle.is_success()
        };
        unsafe { set_command(Command::Drop(database_oid, index_oid)) };
    }
    pub unsafe fn load_all(database_oid: u32, index_oid: u32) -> Vec<Query> {
        let _multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, true);
        let _collector_idle = {
            let collector_idle = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
            collector_idle.is_success()
        };
        unsafe {
            let receiver = MessageQueueReceiver::new(BGWORKER_TO_MAIN_MQ, true);
            set_command(Command::Length(database_oid, index_oid));
            let length: u32 = match receiver.recv() {
                Ok(l) => l,
                Err(e) => {
                    pgrx::warning!("Collector load: Error receiving length: {:?}", e);
                    0
                }
            };
            let mut queries = Vec::with_capacity(length as usize);
            for i in 0..length {
                let _collector_idle = {
                    let collector_idle = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, false);
                    collector_idle.is_success()
                };
                set_command(Command::Load(database_oid, index_oid, i));
                match receiver.recv() {
                    Ok(query) => queries.push(query),
                    Err(e) => {
                        pgrx::warning!("Collector load: Error receiving query: {:?}", e);
                        break;
                    }
                };
            }
            queries
        }
    }
}

pub struct QueryCollectorWorker {
    db_index_query: HashMap<u32, HashMap<u32, VecDeque<Query>>>,
}

impl QueryCollectorWorker {
    pub unsafe fn new() -> Self {
        if let Some(restored) = CollectorState::load() {
            Self {
                db_index_query: restored.data.clone(),
            }
        } else {
            Self {
                db_index_query: HashMap::new(),
            }
        }
    }
    fn queue_length(&self, database_oid: u32, index_oid: u32) -> usize {
        match self.db_index_query.get(&database_oid) {
            Some(index_query) => index_query.get(&index_oid).map_or(0, |vec| vec.len()),
            None => 0,
        }
    }
    pub unsafe fn run(&mut self) {
        loop {
            pgrx::check_for_interrupts!();
            if SHUTDOWN_REQUEST.load(Ordering::Relaxed) {
                let sender = unsafe { MessageQueueSender::new(BGWORKER_TO_MAIN_MQ, false) };
                let receiver = unsafe { MessageQueueReceiver::new(COMMAND_MQ, false) };
                let saved = CollectorState {
                    data: self.db_index_query.clone(),
                    magic: VCHORD_MAGIC,
                };
                if let Err(e) = saved.save() {
                    pgrx::warning!("Collector worker: Error saving state: {:?}", e);
                }
                unsafe {
                    sender.shutdown();
                    receiver.shutdown();
                }
                break;
            }
            unsafe {
                if pgrx::pg_sys::ConfigReloadPending != 0 {
                    pgrx::pg_sys::ConfigReloadPending = 0;
                    pgrx::pg_sys::ProcessConfigFile(pgrx::pg_sys::GucContext::PGC_SIGHUP);
                }
            }
            match unsafe { wait_command() } {
                Command::Push(query) => {
                    let _bgworker_busy = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                    let max_length = max_length(&query);
                    let database_oid = query.database_oid;
                    let index_oid = query.index_oid;
                    if max_length == 0 {
                        self.db_index_query.entry(database_oid).and_modify(|inner| {
                            inner.remove(&index_oid);
                        });
                    }
                    while self.queue_length(database_oid, index_oid) >= max_length as usize
                        && max_length > 0
                    {
                        self.db_index_query.entry(database_oid).and_modify(|inner| {
                            inner.entry(index_oid).and_modify(|e| {
                                e.pop_back();
                            });
                        });
                    }
                    self.db_index_query
                        .entry(database_oid)
                        .or_default()
                        .entry(index_oid)
                        .or_default()
                        .push_front(query);
                }
                Command::Length(database_oid, index_oid) => {
                    let _worker_busy = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                    let sender = unsafe { MessageQueueSender::new(BGWORKER_TO_MAIN_MQ, false) };
                    let index_query = match self.db_index_query.get(&database_oid) {
                        Some(inner) => inner,
                        None => &HashMap::new(),
                    };
                    let length = index_query
                        .get(&index_oid)
                        .map(|inner| inner.len())
                        .unwrap_or(0) as u32;
                    if let Err(e) = sender.send(length) {
                        pgrx::warning!("Collector worker: Error sending length: {:?}", e);
                    }
                }
                Command::Load(database_oid, index_oid, offset) => {
                    let _worker_busy = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                    let sender = unsafe { MessageQueueSender::new(BGWORKER_TO_MAIN_MQ, false) };
                    let index_query = match self.db_index_query.get(&database_oid) {
                        Some(inner) => inner,
                        None => {
                            pgrx::warning!(
                                "Collector worker: No queries found for database {}",
                                database_oid
                            );
                            continue;
                        }
                    };
                    let query = if let Some(q) = index_query
                        .get(&index_oid)
                        .and_then(|inner| inner.get(offset as usize))
                    {
                        q
                    } else {
                        pgrx::warning!("Collector worker: No query found at index {}", offset);
                        continue;
                    };
                    if let Err(e) = sender.send(query) {
                        pgrx::warning!("Collector worker: Error sending query: {:?}", e);
                    }
                }
                Command::Drop(database_oid, index_oid) => {
                    let _worker_busy = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                    if let Some(inner) = self.db_index_query.get_mut(&database_oid)
                        && let Some(queries) = inner.get_mut(&index_oid)
                    {
                        queries.clear();
                    }
                }
                Command::None => {}
            }
        }
    }
}

unsafe fn set_command(command: Command) {
    if command == Command::None {
        return;
    }
    unsafe {
        let sender = MessageQueueSender::new(COMMAND_MQ, true);
        if let Err(e) = sender.send(command) {
            pgrx::warning!("Collector worker: Error sending command: {:?}", e);
        }
        pgrx::pg_sys::SetLatch(BGWORKER_LATCH);
    }
}

unsafe fn wait_command() -> Command {
    unsafe {
        let latch = pgrx::pg_sys::WaitLatch(
            BGWORKER_LATCH,
            (pgrx::pg_sys::WL_LATCH_SET
                | pgrx::pg_sys::WL_TIMEOUT
                | pgrx::pg_sys::WL_POSTMASTER_DEATH
                | pgrx::pg_sys::WL_EXIT_ON_PM_DEATH) as _,
            1000,
            pgrx::pg_sys::WaitEventTimeout::WAIT_EVENT_PG_SLEEP,
        ) as u32;
        pgrx::pg_sys::ResetLatch(BGWORKER_LATCH);
        match latch {
            pgrx::pg_sys::WL_LATCH_SET => {}
            pgrx::pg_sys::WL_TIMEOUT => return Command::None,
            pgrx::pg_sys::WL_EXIT_ON_PM_DEATH | pgrx::pg_sys::WL_POSTMASTER_DEATH => {
                SHUTDOWN_REQUEST.store(true, Ordering::SeqCst);
                return Command::None;
            }
            _ => {
                pgrx::warning!("Collector worker: Unexpected latch event: {}", latch);
                return Command::None;
            }
        };
        let receiver = MessageQueueReceiver::new(COMMAND_MQ, false);
        let command: Command = match receiver.recv() {
            Ok(c) => c,
            Err(MessageQueueError::WouldBlock) => Command::None,
            Err(e) => {
                pgrx::warning!("Collector load: Error receiving command: {:?}", e);
                Command::None
            }
        };
        command
    }
}

fn max_length(query: &Query) -> u32 {
    let global_max_length = gucs::vchordrq_max_logged_queries_per_index();
    let local_max_length = (MAX_FLOATS_PER_INDEX / query.vector.len()) as u32;
    if local_max_length <= global_max_length {
        pgrx::warning!(
            "Collector worker: current vchordrq.max_logged_queries_per_index={} might be too high, better to set it to {}",
            global_max_length,
            local_max_length,
        );
    }
    std::cmp::min(local_max_length, global_max_length)
}
