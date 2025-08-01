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

use crate::collector::mqueue::{MessageQueueReceiver, MessageQueueSender};
use crate::collector::types::{
    BgWorkerLockGuard, COMMAND_REQUEST, Command, QUERY_LOAD_MQ, QUERY_PUSH_MQ, Query, VCHORD_MAGIC,
    WorkerState,
};
use crate::index::gucs;
use std::collections::{HashMap, VecDeque};

pub const MULTI_ACCESS_LOCK: u32 = 0;
pub const BACKGROUND_WORKER_LOCK: u32 = 1;

pub struct QueryCollectorMaster {}

impl QueryCollectorMaster {
    pub unsafe fn push(query: Query) {
        unsafe {
            let multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, false);
            let worker_ok = {
                let worker_ok = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, false);
                worker_ok.is_success()
            };
            if !multi_access_lock.is_success() || !worker_ok {
                return;
            }
            let sender = MessageQueueSender::new(QUERY_PUSH_MQ, true);
            if let Err(e) = sender.send(&query) {
                pgrx::warning!("Collector push: Error sending query: {:?}", e);
                return;
            }
            set_command(Command::Push);
        }
    }
    pub unsafe fn drop(database_oid: u32, index_oid: u32) {
        unsafe {
            let _multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, true);
            let _worker_ok = {
                let worker_ok = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                worker_ok.is_success()
            };
            set_command(Command::Drop(database_oid, index_oid));
        }
    }
    pub unsafe fn load_all(database_oid: u32, index_oid: u32) -> Vec<Query> {
        unsafe {
            let _multi_access_lock = BgWorkerLockGuard::new(MULTI_ACCESS_LOCK, true);
            let _worker_ok = {
                let worker_ok = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                worker_ok.is_success()
            };
            let receiver = MessageQueueReceiver::new(QUERY_LOAD_MQ, true);
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
                let _worker_ok = {
                    let worker_ok = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, false);
                    worker_ok.is_success()
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
        if let Some(restored) = WorkerState::load() {
            pgrx::warning!(
                "Collector worker: Restored state with {} databases",
                restored.data.len()
            );
            Self {
                db_index_query: restored.data.clone(),
            }
        } else {
            pgrx::warning!("Collector worker: No saved state found, starting fresh");
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
            unsafe {
                if pgrx::pg_sys::ConfigReloadPending != 0 {
                    pgrx::pg_sys::ConfigReloadPending = 0;
                    pgrx::pg_sys::ProcessConfigFile(pgrx::pg_sys::GucContext::PGC_SIGHUP);
                }
            }
            let command = unsafe { wait_command() };
            match command {
                Command::Push => {
                    let _bgworker_busy = BgWorkerLockGuard::new(BACKGROUND_WORKER_LOCK, true);
                    let receiver = unsafe { MessageQueueReceiver::new(QUERY_PUSH_MQ, false) };
                    let query: Query = match receiver.recv() {
                        Ok(query) => query,
                        Err(e) => {
                            pgrx::warning!("Collector load: Error receiving query: {:?}", e);
                            continue;
                        }
                    };
                    let max_length = gucs::vchordrq_max_logged_queries_per_index();
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
                        pgrx::warning!(
                            "Collector worker: Query queue is full, removing oldest query: {:?} > {}",
                            self.queue_length(database_oid, index_oid),
                            max_length
                        );
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
                    let sender = unsafe { MessageQueueSender::new(QUERY_LOAD_MQ, false) };
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
                    let sender = unsafe { MessageQueueSender::new(QUERY_LOAD_MQ, false) };
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
                Command::Shutdown => {
                    let sender = unsafe { MessageQueueSender::new(QUERY_LOAD_MQ, false) };
                    let receiver = unsafe { MessageQueueReceiver::new(QUERY_PUSH_MQ, false) };
                    let saved = WorkerState {
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
                Command::None => {}
            }
        }
    }
}

pub unsafe fn set_command(command: Command) {
    let is_none = command == Command::None;
    unsafe {
        *COMMAND_REQUEST = command;
        if !pgrx::pg_sys::MyProc.is_null() && !is_none {
            pgrx::pg_sys::SetLatch(pgrx::pg_sys::MyLatch);
        }
    }
}

pub unsafe fn wait_command() -> Command {
    unsafe {
        pgrx::pg_sys::WaitLatch(
            pgrx::pg_sys::MyLatch,
            (pgrx::pg_sys::WL_LATCH_SET
                | pgrx::pg_sys::WL_TIMEOUT
                | pgrx::pg_sys::WL_EXIT_ON_PM_DEATH) as _,
            1000,
            pgrx::pg_sys::WaitEventTimeout::WAIT_EVENT_PG_SLEEP,
        );
        pgrx::pg_sys::ResetLatch(pgrx::pg_sys::MyLatch);
        let command = COMMAND_REQUEST.as_ref().cloned().unwrap_or(Command::None);
        set_command(Command::None);
        command
    }
}
