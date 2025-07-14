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
use crate::collector::types::{
    BgWorkerLockGuard, Command, MAX_FLOATS_PER_INDEX, Query, VCHORD_MAGIC, WorkerState,
};
use crate::collector::unix::{accept, connect};
use crate::index::gucs;
use std::collections::{HashMap, VecDeque};

pub const MULTI_CONNECT_LOCK: u32 = 0;

pub struct QueryCollectorMaster {}

impl QueryCollectorMaster {
    pub unsafe fn push(query: Query) {
        let mut socket = {
            let multi_connect_lock = BgWorkerLockGuard::new(MULTI_CONNECT_LOCK, false);
            if !multi_connect_lock.is_success() {
                return;
            }
            connect()
        };
        let command = Command::Push(query.clone());
        if let Err(e) = socket.send(command) {
            pgrx::warning!("Collector push: Error send request: {}", e);
        }
    }
    pub unsafe fn drop(database_oid: u32, index_oid: u32) {
        let mut socket = {
            let _multi_connect_lock = BgWorkerLockGuard::new(MULTI_CONNECT_LOCK, true);
            connect()
        };
        let command = Command::Drop(database_oid, index_oid);
        if let Err(e) = socket.send(command) {
            pgrx::warning!("Collector drop: Error send request: {}", e);
        }
    }
    pub unsafe fn load_all(database_oid: u32, index_oid: u32) -> Vec<Query> {
        let mut socket = {
            let _multi_connect_lock = BgWorkerLockGuard::new(MULTI_CONNECT_LOCK, true);
            connect()
        };
        let command = Command::Load(database_oid, index_oid);
        if let Err(e) = socket.send(command) {
            pgrx::warning!("Collector load: Error send request: {}", e);
            return vec![];
        }
        match socket.recv::<Vec<Query>>() {
            Ok(q) => q,
            Err(e) => {
                pgrx::warning!("Collector load: Error load query: {}", e);
                vec![]
            }
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
            let mut socket = accept();
            let command = socket.recv::<Command>().unwrap_or(Command::None);
            match command {
                Command::Push(query) => {
                    let global_max_length = gucs::vchordrq_max_logged_queries_per_index();
                    let local_max_length = (MAX_FLOATS_PER_INDEX / query.vector.len()) as u32;
                    if local_max_length <= global_max_length {
                        pgrx::warning!(
                            "Collector worker: current vchordrq.max_logged_queries_per_index={} might be too high, better to set it to {}",
                            global_max_length,
                            local_max_length,
                        );
                    }
                    let max_length = std::cmp::min(local_max_length, global_max_length);
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
                Command::Load(database_oid, index_oid) => {
                    let result: Vec<Query> = self
                        .db_index_query
                        .get(&database_oid)
                        .and_then(|inner_map| inner_map.get(&index_oid))
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .collect();
                    if let Err(e) = socket.send(result) {
                        pgrx::warning!("Collector worker: Error sending query: {}", e);
                    }
                }
                Command::Drop(database_oid, index_oid) => {
                    if let Some(inner) = self.db_index_query.get_mut(&database_oid)
                        && let Some(queries) = inner.get_mut(&index_oid)
                    {
                        queries.clear();
                    }
                }
                Command::Shutdown => {
                    let saved = WorkerState {
                        data: self.db_index_query.clone(),
                        magic: VCHORD_MAGIC,
                    };
                    if let Err(e) = saved.save() {
                        pgrx::warning!("Collector worker: Error saving state: {:?}", e);
                    }
                    break;
                }
                Command::ReloadConfig => {
                    unsafe {
                        pgrx::pg_sys::ProcessConfigFile(pgrx::pg_sys::GucContext::PGC_SIGHUP)
                    };
                }
                Command::None => {}
            }
        }
    }
}
