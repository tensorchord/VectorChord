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
use crate::collector::types::{PGLockGuard, Result};
use std::cell::OnceCell;
use std::ffi::{CStr, CString};
use std::fs;
use std::path::Path;

const SQLITE_DATABASE: Option<&CStr> = Some(c"main");
const COLLECTOR_DIR: &str = "pg_vchord_sample";
const COLLECTOR_VERSION_PATH: &str = "pg_vchord_sample/VERSION";
const COLLECTOR_VERSION: u32 = 1;

const MULTI_ACCESS_LOCK: u32 = 0;

thread_local! {
    static GLOBAL_INIT: OnceCell<bool> = const { OnceCell::new() };
}

fn init() -> Result<bool> {
    let path = Path::new(COLLECTOR_VERSION_PATH);
    if path.exists() && path.is_file() {
        let content = fs::read_to_string(path)?;
        if content.trim().parse::<u32>()? == COLLECTOR_VERSION {
            return Ok(true);
        }
    }
    match fs::remove_dir_all(COLLECTOR_DIR) {
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(e) => return Err(e.into()),
    };
    fs::create_dir_all(COLLECTOR_DIR)?;
    fs::write(path, format!("{COLLECTOR_VERSION}"))?;
    Ok(true)
}

fn sqlite_connect(
    create: bool,
    database_oid: u32,
    index_oid: u32,
) -> Result<Option<rusqlite::Connection>> {
    if !GLOBAL_INIT.with(|c| *c.get_or_init(|| init().unwrap_or(false))) {
        return Ok(None);
    }
    let p = format!("pg_vchord_sample/database_{database_oid}.sqlite");
    let path = Path::new(&p);
    if !create && !path.exists() {
        return Ok(None);
    }
    let connection = rusqlite::Connection::open(path)?;
    let table_name = CString::new(format!("index_{index_oid}")).unwrap();
    let table_exists = connection
        .table_exists(SQLITE_DATABASE, &table_name)
        .unwrap_or(false);
    if table_exists {
        return Ok(Some(connection));
    }
    if !create {
        return Ok(None);
    }
    let init_statement = format!(
        "CREATE TABLE IF NOT EXISTS index_{index_oid} (
            sample TEXT,
            create_at REAL
        )"
    );
    connection.execute(&init_statement, ())?;
    Ok(Some(connection))
}

pub fn push(database_oid: u32, index_oid: u32, sample: &str, max_length: u32) -> Result<()> {
    let multi_access_lock = PGLockGuard::new(MULTI_ACCESS_LOCK, false);
    if !multi_access_lock.is_success() {
        return Ok(());
    }
    let mut conn = match sqlite_connect(true, database_oid, index_oid)? {
        Some(c) => c,
        None => return Ok(()),
    };
    let maintain_statement = format!("DELETE FROM index_{index_oid}
                WHERE rowid NOT IN (SELECT rowid FROM index_{index_oid} ORDER BY create_at DESC LIMIT ?1)");
    let insert_statement = format!(
        "INSERT INTO index_{index_oid} (sample, create_at) VALUES (?1, unixepoch('subsec'))"
    );
    let tx = conn.transaction()?;
    tx.execute(&insert_statement, (sample,))?;
    tx.execute(&maintain_statement, (max_length,))?;
    tx.commit()?;
    Ok(())
}

pub fn delete_index(database_oid: u32, index_oid: u32) -> Result<()> {
    let _multi_access_lock = PGLockGuard::new(MULTI_ACCESS_LOCK, true);
    let conn = match sqlite_connect(false, database_oid, index_oid)? {
        Some(c) => c,
        None => return Ok(()),
    };
    let drop_statement = format!("DROP TABLE IF EXISTS index_{index_oid}");
    conn.execute(&drop_statement, ())?;
    Ok(())
}

pub fn delete_database(database_oid: u32) -> Result<()> {
    let _multi_access_lock = PGLockGuard::new(MULTI_ACCESS_LOCK, true);
    match fs::remove_file(format!("pg_vchord_sample/database_{database_oid}.sqlite")) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

pub fn load_all(database_oid: u32, index_oid: u32) -> Result<Vec<String>> {
    let _multi_access_lock = PGLockGuard::new(MULTI_ACCESS_LOCK, true);
    let conn = match sqlite_connect(false, database_oid, index_oid)? {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    let load_statement = format!("SELECT sample FROM index_{index_oid} ORDER BY create_at DESC");
    let mut stmt = conn.prepare_cached(&load_statement)?;
    let mut rows = stmt.query(())?;
    let mut result = Vec::new();
    while let Some(row) = rows.next().unwrap_or(None) {
        let sample = row.get::<usize, String>(0)?;
        result.push(sample);
    }
    Ok(result)
}
