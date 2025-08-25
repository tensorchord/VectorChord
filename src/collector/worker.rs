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
use crate::collector::Query;
use rusqlite::Connection;
use std::ffi::CStr;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

const SQLITE_DB_PATH: &str = "vectorchord/collector.db";
const SQLITE_DATABASE: Option<&CStr> = Some(c"main");
const SQLITE_TABLE: &CStr = c"collector";

pub static COLLECTOR: Mutex<Option<Connection>> = Mutex::new(None);

pub struct QueryCollector {}

impl QueryCollector {
    pub fn init() {
        let init_statement = "
            CREATE TABLE IF NOT EXISTS collector (
                database_oid INTEGER,
                table_oid INTEGER,
                index_oid INTEGER,
                operator TEXT,
                vector TEXT,
                create_at REAL
            )";
        let path = Path::new(SQLITE_DB_PATH);
        if let Some(parent_dir) = path.parent() {
            let _ = fs::create_dir_all(parent_dir);
        }
        let connection = match Connection::open(SQLITE_DB_PATH) {
            Ok(conn) => conn,
            Err(e) => {
                pgrx::warning!("Collector: Error opening database: {}", e);
                return;
            }
        };
        verify_or_destroy(&connection);
        match connection.execute(init_statement, ()) {
            Ok(_) => COLLECTOR.lock().unwrap().replace(connection),
            Err(e) => {
                pgrx::warning!("Collector: Error initializing database: {}", e);
                None
            }
        };
    }
    pub fn push(query: Query, max_length: u32) {
        let maintain_statement = "DELETE FROM collector
            WHERE rowid NOT IN (SELECT rowid FROM collector WHERE database_oid = ?1 AND index_oid = ?2
            ORDER BY create_at DESC LIMIT ?3) AND database_oid = ?1 AND index_oid = ?2";
        let insert_statement = "INSERT INTO collector
            (database_oid, table_oid, index_oid, operator, vector, create_at) VALUES (?1, ?2, ?3, ?4, ?5, unixepoch('subsec'))";
        let mut lock = match COLLECTOR.try_lock() {
            Ok(lock) => lock,
            Err(e) => {
                pgrx::warning!(
                    "Collector: Failed to acquire lock on collector connection: {}",
                    e
                );
                return;
            }
        };
        let connection = match lock.as_mut() {
            Some(conn) => conn,
            None => {
                pgrx::warning!("Collector: No collector connection available");
                return;
            }
        };
        let tx = match connection.transaction() {
            Ok(t) => t,
            Err(e) => {
                pgrx::warning!("Collector: Error starting transaction: {}", e);
                return;
            }
        };
        let insert = tx.execute(
            insert_statement,
            (
                query.database_oid,
                query.table_oid,
                query.index_oid,
                query.operator.to_string(),
                query.vector_text,
            ),
        );
        if let Err(e) = insert {
            pgrx::warning!("Collector: Error inserting query: {}", e);
            return;
        }
        let maintain = tx.execute(
            maintain_statement,
            (query.database_oid, query.index_oid, max_length),
        );
        if let Err(e) = maintain {
            pgrx::warning!("Collector: Error maintaining queries: {}", e);
            return;
        }
        if let Err(e) = tx.commit() {
            pgrx::warning!("Collector: Error committing transaction: {}", e);
        }
    }
    pub fn delete(database_oid: u32, index_oid: u32) {
        let lock = COLLECTOR.lock().unwrap();
        let connection = match lock.as_ref() {
            Some(conn) => conn,
            None => {
                pgrx::warning!("Collector: No collector connection available");
                return;
            }
        };
        let drop_statement = "DELETE FROM collector WHERE database_oid = ?1 AND index_oid = ?2";
        let _ = connection
            .execute(drop_statement, (database_oid, index_oid))
            .map_err(|e| {
                pgrx::warning!("Collector: Error dropping queries: {}", e);
                e
            });
    }
    pub fn load_all(database_oid: u32, index_oid: u32) -> Vec<Query> {
        let lock = COLLECTOR.lock().unwrap();
        let connection = match lock.as_ref() {
            Some(conn) => conn,
            None => {
                pgrx::warning!("Collector: No collector connection available");
                return Vec::new();
            }
        };
        let load_statement =
            "SELECT database_oid, table_oid, index_oid, operator, vector FROM collector
            WHERE database_oid = ?1 AND index_oid = ?2";
        let mut stmt = match connection.prepare_cached(load_statement) {
            Ok(s) => s,
            Err(e) => {
                pgrx::warning!("Collector: Error preparing statement: {}", e);
                return Vec::new();
            }
        };
        let mut rows = match stmt.query([database_oid, index_oid]) {
            Ok(r) => r,
            Err(e) => {
                pgrx::warning!("Collector: Error executing query: {}", e);
                return Vec::new();
            }
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next().unwrap_or(None) {
            let database_oid = match row.get(0) {
                Ok(oid) => oid,
                Err(e) => {
                    pgrx::warning!("Collector: Error getting database_oid: {}", e);
                    continue;
                }
            };
            let table_oid = match row.get::<usize, u32>(1) {
                Ok(oid) => oid,
                Err(e) => {
                    pgrx::warning!("Collector: Error getting table_oid: {}", e);
                    continue;
                }
            };
            let index_oid = match row.get::<usize, u32>(2) {
                Ok(oid) => oid,
                Err(e) => {
                    pgrx::warning!("Collector: Error getting index_oid: {}", e);
                    continue;
                }
            };
            let operator = match row.get::<usize, String>(3).map(|op| op.as_str().try_into()) {
                Ok(Ok(op)) => op,
                Ok(Err(e)) => {
                    pgrx::warning!("Collector: Error converting operator: {}", e);
                    continue;
                }
                Err(e) => {
                    pgrx::warning!("Collector: Error getting index_oid: {}", e);
                    continue;
                }
            };
            let vector_text = match row.get::<usize, String>(4) {
                Ok(text) => text,
                Err(e) => {
                    pgrx::warning!("Collector: Error getting vector_text: {}", e);
                    continue;
                }
            };
            result.push(Query {
                database_oid,
                table_oid,
                index_oid,
                operator,
                vector_text,
            });
        }
        result
    }
}

fn verify_or_destroy(connection: &Connection) {
    if connection.table_exists(SQLITE_DATABASE, SQLITE_TABLE) != Ok(true) {
        return;
    }
    struct ColumnSpec<'a> {
        name: &'a CStr,
        expected_type: &'a CStr,
    }

    const EXPECTED_COLUMNS: &[ColumnSpec] = &[
        ColumnSpec {
            name: c"database_oid",
            expected_type: c"INTEGER",
        },
        ColumnSpec {
            name: c"table_oid",
            expected_type: c"INTEGER",
        },
        ColumnSpec {
            name: c"index_oid",
            expected_type: c"INTEGER",
        },
        ColumnSpec {
            name: c"operator",
            expected_type: c"TEXT",
        },
        ColumnSpec {
            name: c"vector",
            expected_type: c"TEXT",
        },
        ColumnSpec {
            name: c"create_at",
            expected_type: c"REAL",
        },
    ];

    const COMMON_METADATA_SUFFIX: (Option<&CStr>, bool, bool, bool) =
        (Some(c"BINARY"), false, false, false);

    for column_spec in EXPECTED_COLUMNS {
        let expected = Ok((
            Some(column_spec.expected_type),
            COMMON_METADATA_SUFFIX.0,
            COMMON_METADATA_SUFFIX.1,
            COMMON_METADATA_SUFFIX.2,
            COMMON_METADATA_SUFFIX.3,
        ));
        if connection.column_metadata(SQLITE_DATABASE, SQLITE_TABLE, column_spec.name) != expected {
            pgrx::warning!("Collector: Invalid collector database schema, destroying the database");
            let drop_statement = "DROP TABLE collector";
            let _ = connection.execute(drop_statement, ()).map_err(|e| {
                pgrx::warning!("Collector: Error dropping table: {}", e);
                e
            });
            return;
        }
    }
}
