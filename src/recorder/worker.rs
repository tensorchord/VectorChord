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
use crate::recorder::types::PgRefCell;
use std::cell::RefMut;
use std::fs;
use std::path::Path;

// Safety: The directory name must start with "pgsql_tmp" to be excluded by pg_basebackup
const RECORDER_DIR: &str = "pgsql_tmp_vchord_sampling";
const RECORDER_VERSION: u32 = 1;

static CONNECTION: PgRefCell<Option<rusqlite::Connection>> =
    PgRefCell::<Option<rusqlite::Connection>>::new(None);

fn get<'a>() -> Option<RefMut<'a, rusqlite::Connection>> {
    if unsafe { !pgrx::pg_sys::IsBackendPid(pgrx::pg_sys::MyProcPid) } {
        return None;
    }
    let database_oid = unsafe { pgrx::pg_sys::MyDatabaseId.to_u32() };
    if database_oid == 0 {
        return None;
    }
    let mut connection = CONNECTION.borrow_mut();
    if connection.is_none()
        && let Err(err) = || -> rusqlite::Result<()> {
            if !Path::new(RECORDER_DIR).exists() {
                let _ = fs::create_dir_all(RECORDER_DIR);
            }
            let p = format!("{RECORDER_DIR}/database_{database_oid}.sqlite");
            let mut conn = rusqlite::Connection::open(&p)?;
            conn.pragma_update(Some("main"), "journal_mode", "WAL")?;
            conn.pragma_update(Some("main"), "synchronous", "NORMAL")?;
            let tx = conn.transaction()?;
            let version: u32 = tx
                .pragma_query_value(Some("main"), "user_version", |row| row.get(0))
                .unwrap_or(RECORDER_VERSION);
            if version != RECORDER_VERSION && version != 0 {
                let mut statement = tx.prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'index_%';",
                )?;
                let tables = statement.query_map((), |row| row.get::<usize, String>(0))?;
                for name in tables.into_iter().flatten() {
                    let drop_statement = format!("DROP TABLE IF EXISTS {name}");
                    tx.execute(&drop_statement, ())?;
                }
            }
            tx.pragma_update(Some("main"), "user_version", RECORDER_VERSION)?;
            tx.commit()?;
            let _ = connection.insert(conn);
            Ok(())
        }()
    {
        if err.sqlite_error_code() == Some(rusqlite::ErrorCode::DatabaseCorrupt) {
            delete_database(database_oid);
        }
        pgrx::debug1!("Recorder: Error initializing database: {}", err);
        return None;
    }
    RefMut::filter_map(connection, |c| c.as_mut()).ok()
}

pub fn push(index: u32, sample: &str, max_records: u32) {
    let mut connection = match get() {
        Some(c) => c,
        None => return,
    };
    let init_statement = format!(
        "
        CREATE TABLE IF NOT EXISTS index_{index} (sample TEXT, create_at REAL);
        CREATE INDEX IF NOT EXISTS i ON index_{index} (create_at);
        "
    );
    let insert_statement =
        format!("INSERT INTO index_{index} (sample, create_at) VALUES (?1, unixepoch('subsec'))");
    let count_statement = format!("SELECT COUNT(create_at) FROM index_{index}");
    let maintain_statement = format!(
        "DELETE FROM index_{index} WHERE rowid = (
        SELECT rowid FROM index_{index} ORDER BY create_at ASC LIMIT ?1);"
    );
    if let Err(err) = || -> rusqlite::Result<()> {
        let tx = connection.transaction()?;
        tx.execute_batch(&init_statement)?;
        tx.prepare_cached(&insert_statement)?.execute((sample,))?;
        let records = tx.query_one(&count_statement, (), |row| row.get::<usize, u32>(0))?;
        if records > max_records {
            tx.execute(&maintain_statement, (records - max_records,))?;
        }
        tx.commit()?;
        Ok(())
    }() {
        pgrx::debug1!("Recorder: Error pushing sample: {}", err);
    }
}

pub fn delete_index(index: u32) {
    let connection = match get() {
        Some(c) => c,
        None => return,
    };
    let drop_statement = format!("DROP TABLE IF EXISTS index_{index}");
    if let Err(e) = connection.execute(&drop_statement, ()) {
        pgrx::debug1!("Recorder: Error deleting index table: {}", e);
    };
}

pub fn delete_database(database_oid: u32) {
    let _ = fs::remove_file(format!("{RECORDER_DIR}/database_{database_oid}.sqlite"));
    let _ = fs::remove_file(format!("{RECORDER_DIR}/database_{database_oid}.sqlite-shm"));
    let _ = fs::remove_file(format!("{RECORDER_DIR}/database_{database_oid}.sqlite-wal"));
}

pub fn dump(index: u32) -> Vec<String> {
    let connection = match get() {
        Some(c) => c,
        None => return Vec::new(),
    };
    let load_statement = format!("SELECT sample FROM index_{index} ORDER BY create_at DESC");
    match || -> rusqlite::Result<Vec<String>> {
        let mut stmt = connection.prepare(&load_statement)?;
        let mut rows = stmt.query(())?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            if let Ok(sample) = row.get::<usize, String>(0) {
                result.push(sample);
            }
        }
        Ok(result)
    }() {
        Ok(v) => v,
        Err(e) => {
            pgrx::debug1!("Recorder: Error loading samples: {}", e);
            Vec::new()
        }
    }
}
