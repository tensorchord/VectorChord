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

use crate::collector::QueryCollector;
use pgrx::IntoDatum;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgClass, PgIndex, PgNamespace};
use rand::Rng;
use std::fmt;

pub trait CollectorSender {
    fn send_vchordrq(
        &self,
        opfamily: crate::index::vchordrq::opclass::Opfamily,
        vector: vchordrq::types::OwnedVector,
    );
    fn send_vchordg(
        &self,
        opfamily: crate::index::vchordg::opclass::Opfamily,
        vector: vchordg::types::OwnedVector,
    );
}

#[derive(Debug)]
pub struct DefaultSender {
    pub send_prob: Option<f64>,
    pub max_records: u32,
    pub database_oid: u32,
    pub table_oid: u32,
    pub index_oid: u32,
}

impl CollectorSender for DefaultSender {
    fn send_vchordrq(
        &self,
        opfamily: crate::index::vchordrq::opclass::Opfamily,
        vector: vchordrq::types::OwnedVector,
    ) {
        use crate::index::vchordrq::opclass::Opfamily;
        if let Some(rate) = self.send_prob {
            let mut rng = rand::rng();
            if rng.random_bool(rate) {
                let operator = match opfamily {
                    Opfamily::VectorL2 | Opfamily::HalfvecL2 => Some(Operator::L2),
                    Opfamily::VectorCosine | Opfamily::HalfvecCosine => Some(Operator::Cosine),
                    Opfamily::VectorIp | Opfamily::HalfvecIp => Some(Operator::Dot),
                    _ => return,
                };
                let vector = match vector {
                    vchordrq::types::OwnedVector::Vecf32(v) => v.into_vec(),
                    vchordrq::types::OwnedVector::Vecf16(v) => {
                        v.into_vec().into_iter().map(|f| f.to_f32()).collect()
                    }
                };
                let query = Query::new(
                    self.database_oid,
                    self.table_oid,
                    self.index_oid,
                    operator,
                    vector,
                );
                if let Some(q) = query {
                    QueryCollector::push(q, self.max_records);
                }
            }
        }
    }

    fn send_vchordg(
        &self,
        opfamily: crate::index::vchordg::opclass::Opfamily,
        vector: vchordg::types::OwnedVector,
    ) {
        use crate::index::vchordg::opclass::Opfamily;
        if let Some(rate) = self.send_prob {
            let mut rng = rand::rng();
            if rng.random_bool(rate) {
                let operator = match opfamily {
                    Opfamily::VectorL2 | Opfamily::HalfvecL2 => Some(Operator::L2),
                    Opfamily::VectorCosine | Opfamily::HalfvecCosine => Some(Operator::Cosine),
                    Opfamily::VectorIp | Opfamily::HalfvecIp => Some(Operator::Dot),
                };
                let vector = match vector {
                    vchordg::types::OwnedVector::Vecf32(v) => v.into_vec(),
                    vchordg::types::OwnedVector::Vecf16(v) => {
                        v.into_vec().into_iter().map(|f| f.to_f32()).collect()
                    }
                };
                let query = Query::new(
                    self.database_oid,
                    self.table_oid,
                    self.index_oid,
                    operator,
                    vector,
                );
                if let Some(q) = query {
                    QueryCollector::push(q, self.max_records);
                }
            }
        }
    }
}

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
pub struct QueryDump {
    pub namespace: String,
    pub table_name: String,
    pub column_name: Option<String>,
    pub operator: String,
    pub vector_text: String,
    pub simplified_query: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub database_oid: u32,
    pub table_oid: u32,
    pub index_oid: u32,
    pub operator: Operator,
    pub vector_text: String,
}

impl Query {
    pub fn new(
        database_oid: u32,
        table_oid: u32,
        index_oid: u32,
        operator: Option<Operator>,
        vector: Vec<f32>,
    ) -> Option<Self> {
        let operator = operator?;
        let vector_format: Vec<String> = vector.iter().map(|f| format!("{f:.4}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text = format!("'[{joined_elements}]'");
        Some(Self {
            database_oid,
            table_oid,
            index_oid,
            operator,
            vector_text,
        })
    }
    pub fn search_table(&self) -> Option<(String, String)> {
        let (table_name, namespace_oid) = {
            let tuple = PgClass::search_reloid(self.table_oid.into());
            if let Some(tuple) = tuple {
                let pg_class = tuple.get()?;
                let name = pg_class.relname().to_str().map(|s| s.to_owned()).ok()?;
                let namespace_oid_datum = pg_class.relnamespace();
                (name, namespace_oid_datum)
            } else {
                pgrx::warning!(
                    "Collector load: Attribute with relid {} not found in syscache",
                    self.table_oid,
                );
                return None;
            }
        };
        let namespace = {
            let tuple = PgNamespace::search_namespaceoid(namespace_oid);
            if let Some(tuple) = tuple {
                let pg_class = tuple.get()?;
                pg_class.nspname().to_str().map(|s| s.to_owned()).ok()?
            } else {
                pgrx::warning!(
                    "Collector load: Namespace with oid {} not found in syscache",
                    namespace_oid,
                );
                return None;
            }
        };
        Some((namespace, table_name))
    }
    pub fn search_column(&self) -> Option<String> {
        let column_attnum = {
            let tuple = PgIndex::search_indexrelid(self.index_oid.into());
            if let Some(tuple) = tuple {
                let pg_index = tuple.get()?;
                *pg_index.indkey().first()?
            } else {
                pgrx::warning!(
                    "Collector load: Index with oid {} not found in syscache",
                    self.index_oid,
                );
                return None;
            }
        };
        let column_name = unsafe {
            let table_oid_datum = Oid::from_u32(self.table_oid).into_datum()?;
            let column_attnum_datum = column_attnum.into_datum()?;
            let tuple = pgrx::pg_sys::SearchSysCache2(
                pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                table_oid_datum,
                column_attnum_datum,
            );
            if tuple.is_null() {
                // An index on expression has no corresponding column in pg_attribute.
                return None;
            }
            #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
            let inner = {
                let mut is_null = false;
                let datum = pgrx::pg_sys::SysCacheGetAttr(
                    pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                    tuple,
                    pgrx::pg_sys::Anum_pg_attribute_attname as i16,
                    &mut is_null,
                );
                pgrx::pg_sys::DatumGetName(datum).as_ref()?
            };
            #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
            let inner = &(*pgrx::pg_sys::GETSTRUCT(tuple)
                .cast::<pgrx::pg_sys::FormData_pg_attribute>())
            .attname;

            let result = pgrx::pg_sys::name_data_to_str(inner).to_string();
            pgrx::pg_sys::ReleaseSysCache(tuple);
            result
        };
        Some(column_name)
    }
    pub fn dump(
        self,
        namespace: String,
        table_name: String,
        column_name: Option<String>,
    ) -> QueryDump {
        let operator = self.operator.to_string();
        let vector_text = self.vector_text.clone();
        let simplified_query = column_name.as_ref().map(|column|
            format!("SELECT ctid from {namespace}.{table_name} ORDER BY {column} {operator} {vector_text}")
        );
        QueryDump {
            namespace,
            table_name,
            column_name,
            operator,
            vector_text,
            simplified_query,
        }
    }
}
