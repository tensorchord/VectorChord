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

use crate::collector::QueryCollectorMaster;
use crate::index::storage::PostgresRelation;
use pgrx::iter::TableIterator;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgAm, PgClass, PgClassRelkind};
use std::collections::HashMap;

#[pgrx::pg_extern(sql = "")]
fn _vchordg_prewarm(indexrelid: Oid) -> String {
    let pg_am = PgAm::search_amname(c"vchordg").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("vchord is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("the relation does not exist");
    };
    if pg_class.relkind() != PgClassRelkind::Index {
        pgrx::error!("the relation {:?} is not an index", pg_class.relname());
    }
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("the index {:?} is not a vchordg index", pg_class.relname());
    }
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let opfamily = unsafe { crate::index::vchordg::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::vchordg::algo::prewarm(opfamily, &index)
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_prewarm(indexrelid: Oid, height: i32) -> String {
    let pg_am = PgAm::search_amname(c"vchordrq").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("vchord is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("the relation does not exist");
    };
    if pg_class.relkind() != PgClassRelkind::Index {
        pgrx::error!("the relation {:?} is not an index", pg_class.relname());
    }
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("the index {:?} is not a vchordrq index", pg_class.relname());
    }
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let opfamily = unsafe { crate::index::vchordrq::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::vchordrq::algo::prewarm(opfamily, &index, height)
}

struct Index {
    raw: *mut pgrx::pg_sys::RelationData,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl Index {
    fn open(indexrelid: Oid, lockmode: pgrx::pg_sys::LOCKMASK) -> Self {
        Self {
            raw: unsafe { pgrx::pg_sys::index_open(indexrelid, lockmode) },
            lockmode,
        }
    }
    fn raw(&self) -> *mut pgrx::pg_sys::RelationData {
        self.raw
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::index_close(self.raw, self.lockmode);
        }
    }
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_logged_queries(
    indexrelid: Oid,
) -> TableIterator<
    'static,
    (
        pgrx::name!(schema_name, String),
        pgrx::name!(table_name, String),
        pgrx::name!(column_name, Option<String>),
        pgrx::name!(operator, String),
        pgrx::name!(vector_text, String),
        pgrx::name!(simplified_query, Option<String>),
    ),
> {
    let pg_am = PgAm::search_amname(c"vchordrq").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("vchord is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("the relation does not exist");
    };
    if pg_class.relkind() != PgClassRelkind::Index {
        pgrx::error!("the relation {:?} is not an index", pg_class.relname());
    }
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("the index {:?} is not a vchordrq index", pg_class.relname());
    }
    // The user must have access to the index, if not, raise an error from Postgres.
    let _relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);

    let queries = unsafe {
        let database_oid = pgrx::pg_sys::MyDatabaseId.to_u32();
        QueryCollectorMaster::load_all(database_oid, indexrelid.to_u32())
    };
    let mut table_cache: HashMap<u32, (String, String)> = HashMap::new();
    let mut index_cache: HashMap<u32, Option<String>> = HashMap::new();
    let mut dumps = Vec::with_capacity(queries.len());
    for query in queries {
        let (namespace, table_name) = match table_cache.get(&query.table_oid).cloned() {
            Some((namespace, table_name)) => (namespace, table_name),
            None => match query.search_table() {
                Some((namespace, table_name)) => {
                    table_cache.insert(query.table_oid, (namespace.clone(), table_name.clone()));
                    (namespace, table_name)
                }
                None => continue,
            },
        };
        let column_name = match index_cache.get(&query.index_oid).cloned() {
            Some(column) => column,
            None => {
                let column = query.search_column();
                index_cache.insert(query.index_oid, column.clone());
                column
            }
        };
        dumps.push(query.dump(namespace, table_name, column_name));
    }
    TableIterator::new(dumps.into_iter().map(|dump| {
        (
            dump.namespace,
            dump.table_name,
            dump.column_name,
            dump.operator,
            dump.vector_text,
            dump.simplified_query,
        )
    }))
}
