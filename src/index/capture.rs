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

use crate::index::gucs;
use crate::index::vchordrq::opclass::Opfamily;
use pgrx::IntoDatum;
use pgrx::pg_sys::Oid;
use pgrx::pg_sys::panic::ErrorReportable;
use pgrx_catalog::{PgClass, PgIndex, PgNamespace};
use std::ffi::CString;
use std::fmt;
use std::sync::{LazyLock, RwLock};
use vchordrq::types::OwnedVector;

const INTERNAL_TABLE_NAME: &str = "_internal_vchord_query_storage";
static INTERNAL_TABLE_SCHEMA: LazyLock<RwLock<String>> =
    LazyLock::new(|| RwLock::new(String::from("public")));

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    L2,
    Cosine,
    Dot,
}

impl TryFrom<&str> for Operator {
    type Error = &'static str;

    fn try_from(text: &str) -> Result<Self, Self::Error> {
        match text {
            "<->" => Ok(Operator::L2),
            "<=>" => Ok(Operator::Cosine),
            "<#>" => Ok(Operator::Dot),
            _ => Err("Unknown operator text"),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::L2 => write!(f, "<->"),
            Operator::Cosine => write!(f, "<=>"),
            Operator::Dot => write!(f, "<#>"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Query {
    pub table_oid: u32,
    pub index_oid: u32,
    pub operator: Operator,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct QueryDump {
    pub namespace: String,
    pub table_name: String,
    pub column_name: String,
    pub operator: String,
    pub vector_text: String,
    pub simplified_query: String,
}

impl Query {
    pub fn new(
        table_oid: u32,
        index_oid: u32,
        opfamily: Opfamily,
        vector: OwnedVector,
    ) -> Option<Self> {
        let operator = match opfamily {
            Opfamily::HalfvecCosine | Opfamily::VectorCosine => Operator::Cosine,
            Opfamily::HalfvecIp | Opfamily::VectorIp => Operator::Dot,
            Opfamily::HalfvecL2 | Opfamily::VectorL2 => Operator::L2,
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim => return None,
        };
        let vector = match vector {
            OwnedVector::Vecf32(v) => v.into_vec(),
            OwnedVector::Vecf16(v) => v.into_vec().into_iter().map(|f| f.to_f32()).collect(),
        };
        Some(Self {
            table_oid,
            index_oid,
            operator,
            vector,
        })
    }

    pub fn dump(&self) -> Option<QueryDump> {
        let (table_name, namespace_oid) = {
            let tuple = PgClass::search_reloid(self.table_oid.into());
            if let Some(tuple) = tuple {
                let pg_class = tuple.get().unwrap();
                let name = pg_class.relname().to_str().map(|s| s.to_owned()).unwrap();
                let namespace_oid_datum = pg_class.relnamespace();
                (name, namespace_oid_datum)
            } else {
                pgrx::warning!(
                    "Attribute with relid {} not found in syscache",
                    self.table_oid,
                );
                return None;
            }
        };
        let namespace = {
            let tuple = PgNamespace::search_namespaceoid(namespace_oid);
            if let Some(tuple) = tuple {
                let pg_class = tuple.get().unwrap();
                pg_class.nspname().to_str().map(|s| s.to_owned()).unwrap()
            } else {
                pgrx::warning!("Namespace with oid {} not found in syscache", namespace_oid,);
                return None;
            }
        };
        let column_attnum = {
            let tuple = PgIndex::search_indexrelid(self.index_oid.into());
            if let Some(tuple) = tuple {
                let pg_index = tuple.get().unwrap();
                *pg_index
                    .indkey()
                    .first()
                    .expect("Index should have at least one column")
            } else {
                pgrx::warning!("Index with oid {} not found in syscache", self.index_oid,);
                return None;
            }
        };
        let column_name = unsafe {
            let table_oid_datum = Oid::from_u32(self.table_oid).into_datum().unwrap();
            let column_attnum_datum = column_attnum.into_datum().unwrap();
            let tuple = pgrx::pg_sys::SearchSysCache2(
                pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                table_oid_datum,
                column_attnum_datum,
            );
            if tuple.is_null() {
                pgrx::warning!(
                    "Attribute with relid {} and attno {} not found in syscache",
                    self.table_oid,
                    column_attnum
                );
                return None;
            }
            #[cfg(not(any(feature = "pg13", feature = "pg14", feature = "pg15")))]
            let inner = {
                let mut is_null = false;
                let datum = pgrx::pg_sys::SysCacheGetAttr(
                    pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                    tuple,
                    pgrx::pg_sys::Anum_pg_attribute_attname as i16,
                    &mut is_null,
                );
                pgrx::pg_sys::DatumGetName(datum).as_ref().unwrap()
            };
            #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
            let inner = &(*pgrx::pg_sys::GETSTRUCT(tuple)
                .cast::<pgrx::pg_sys::FormData_pg_attribute>())
            .attname;

            let result = pgrx::pg_sys::name_data_to_str(inner).to_string();
            pgrx::pg_sys::ReleaseSysCache(tuple);
            result
        };
        let operator = self.operator.to_string();
        let vector_format: Vec<String> = self.vector.iter().map(|f| format!("{f:.2}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text = format!("'[{joined_elements}]'");
        let simplified_query = format!(
            "SELECT ctid from {namespace}.{table_name} ORDER BY {column_name} {operator} {vector_text}"
        );
        Some(QueryDump {
            namespace,
            table_name,
            column_name,
            operator,
            vector_text,
            simplified_query,
        })
    }
}

pub struct QueryLoggerMaster {}

impl QueryLoggerMaster {
    pub fn init() {
        pgrx::spi::Spi::connect_mut(|client| {
            let namespace_query = "SELECT n.nspname::TEXT
                FROM pg_catalog.pg_extension e
                LEFT JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                WHERE e.extname = 'vchord';";
            let vchord_namespace: String = client
                .select(namespace_query, None, &[])
                .unwrap_or_report()
                .first()
                .get_by_name("nspname")
                .expect("external build: cannot get namespace of vchord")
                .expect("external build: cannot get namespace of vchord");
            let mut namespace_guard = INTERNAL_TABLE_SCHEMA.write().unwrap();
            *namespace_guard = vchord_namespace.clone();

            let namespace_oid = {
                let c_namespace = CString::new(vchord_namespace.as_str()).unwrap();
                let tuple = PgNamespace::search_namespacename(&c_namespace);
                if let Some(tuple) = tuple {
                    let pg_namespace = tuple.get().unwrap();
                    pg_namespace.oid()
                } else {
                    pgrx::warning!(
                        "Namespace with namespace {:?} not found in syscache",
                        c_namespace
                    );
                    return;
                }
            };
            let exist = {
                let c_table = CString::new(INTERNAL_TABLE_NAME).unwrap();
                let tuple = PgClass::search_relnamensp(&c_table, namespace_oid);
                if let Some(inner) = tuple
                    && inner.get().is_some()
                {
                    true
                } else {
                    false
                }
            };

            if !exist {
                let create_sql = format!(
                    "CREATE TABLE IF NOT EXISTS {vchord_namespace}.{INTERNAL_TABLE_NAME} (
                     id BIGSERIAL PRIMARY KEY, table_oid OID, index_oid OID,
                     operator TEXT, data TEXT)",
                );
                let _ = client.update(&create_sql, None, &[]);
            }
        });
    }

    pub fn push(query: Query) {
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;
        let Query {
            table_oid,
            index_oid,
            operator,
            vector,
        } = query;
        let ops = operator.to_string();
        let vector_format: Vec<String> = vector.iter().map(|f| format!("{f:.2}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text_rep = format!("[{joined_elements}]");
        pgrx::spi::Spi::connect_mut(|client| {
            let insert_sql = format!(
                "INSERT INTO {namespace}.{INTERNAL_TABLE_NAME} (table_oid, index_oid, operator, data) VALUES
                     ({table_oid}, {index_oid}, '{ops}', '{vector_text_rep}')",
            );
            let _ = client.update(&insert_sql, None, &[]);
        });
    }

    pub fn maintain() {
        Self::init();
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;
        let limit = gucs::vchordrq_log_queries_size();
        pgrx::spi::Spi::connect_mut(|client| {
            let delete_sql = format!(
                "DELETE FROM {namespace}.{INTERNAL_TABLE_NAME} WHERE id NOT IN (
                    SELECT id FROM {namespace}.{INTERNAL_TABLE_NAME} ORDER BY id DESC LIMIT {limit})"
            );
            let _ = client.update(&delete_sql, None, &[]);
        });
    }

    pub fn load_all() -> Vec<Query> {
        let limit = gucs::vchordrq_log_queries_size();
        let mut queries = Vec::new();
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;

        let query_sql = format!(
            "SELECT table_oid, index_oid, operator, data
             FROM {namespace}.{INTERNAL_TABLE_NAME} ORDER BY id DESC LIMIT {limit}",
        );

        pgrx::spi::Spi::connect(|client| {
            let rows = client
                .select(&query_sql, Some(limit as i64), &[])
                .unwrap_or_report();
            'r: for row in rows {
                let table_oid: Oid = if let Some(e) = row.get_by_name("table_oid").unwrap() {
                    e
                } else {
                    pgrx::warning!("table_oid is null in logged query");
                    continue 'r;
                };
                let index_oid: Oid = if let Some(e) = row.get_by_name("index_oid").unwrap() {
                    e
                } else {
                    pgrx::warning!("index_oid is null in logged query");
                    continue 'r;
                };
                let operator: &str = if let Some(e) = row.get_by_name("operator").unwrap() {
                    e
                } else {
                    pgrx::warning!("operator is null in logged query");
                    continue 'r;
                };
                let vector_text: &str = if let Some(e) = row.get_by_name("data").unwrap() {
                    e
                } else {
                    continue 'r;
                };
                let vector_text = vector_text.trim();
                let inner_str = &vector_text[1..vector_text.len() - 1];
                if inner_str.trim().is_empty() {
                    pgrx::warning!("Empty vector text, skipping");
                    continue 'r;
                }
                let parts = inner_str.split(',');

                let mut vector = Vec::new();
                for part in parts {
                    let trimmed_part = part.trim();
                    if trimmed_part.is_empty() {
                        pgrx::warning!("Empty part in vector text, skipping");
                        continue 'r;
                    }
                    match trimmed_part.parse::<f32>() {
                        Ok(value) => vector.push(value),
                        Err(e) => {
                            pgrx::warning!("Failed to parse vector part '{}': {}", trimmed_part, e);
                            continue 'r;
                        }
                    }
                }

                queries.push(Query {
                    table_oid: table_oid.into(),
                    index_oid: index_oid.into(),
                    operator: Operator::try_from(operator).expect("Failed to parse operator"),
                    vector,
                });
            }
            queries
        })
    }
}
