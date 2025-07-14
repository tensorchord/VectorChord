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

use crate::index::vchordrq::opclass::Opfamily;
use bincode::{Decode, Encode, config, decode_from_reader, encode_into_std_write};
use pgrx::IntoDatum;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgClass, PgIndex, PgNamespace};
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::path::Path;
use std::{fmt, fs, io};
use vchordrq::types::OwnedVector;

pub const MAX_FLOATS_PER_INDEX: usize = 1000 * 10000;
pub const VCHORD_MAGIC: u64 = 0x5643484f;
pub const BINCODE_CONFIG: config::Configuration = config::standard();
const WORKER_STATE_SAVED_PATH: &str = "vectorchord/collector.bincode";

#[derive(Debug, Clone, Default, Encode, Decode)]
pub struct WorkerState {
    pub data: HashMap<u32, HashMap<u32, VecDeque<Query>>>,
    pub magic: u64,
}

impl WorkerState {
    pub fn load() -> Option<Self> {
        let path = Path::new(WORKER_STATE_SAVED_PATH);
        path.try_exists().ok().and_then(|exists| {
            if !exists {
                return None;
            }
            let file = fs::File::open(path).ok()?;
            let mut reader = io::BufReader::new(file);
            let decode_result =
                decode_from_reader::<WorkerState, _, _>(&mut reader, BINCODE_CONFIG);
            let data = decode_result.ok();
            if let Some(ref restored) = data
                && restored.magic != VCHORD_MAGIC
            {
                return None;
            }
            data
        })
    }
    pub fn save(&self) -> Result<(), Box<dyn Error>> {
        let path = Path::new(WORKER_STATE_SAVED_PATH);
        if let Some(parent_dir) = path.parent() {
            fs::create_dir_all(parent_dir)?;
        }
        let file = fs::File::create(WORKER_STATE_SAVED_PATH).unwrap();
        let mut writer = io::BufWriter::new(file);
        encode_into_std_write(self, &mut writer, BINCODE_CONFIG).map(|_| Ok(()))?
    }
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum Command {
    None,
    Shutdown,
    ReloadConfig,
    Push(Query),
    Load(u32, u32),
    Drop(u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
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

#[derive(Debug, Clone, Encode, Decode, PartialEq)]
pub struct Query {
    pub database_oid: u32,
    pub table_oid: u32,
    pub index_oid: u32,
    pub operator: Operator,
    pub vector: Vec<f32>,
}

impl Query {
    pub fn new(
        database_oid: u32,
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
            database_oid,
            table_oid,
            index_oid,
            operator,
            vector,
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
            #[cfg(not(any(feature = "pg13", feature = "pg14", feature = "pg15")))]
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
        &self,
        namespace: String,
        table_name: String,
        column_name: Option<String>,
    ) -> QueryDump {
        let operator = self.operator.to_string();
        let vector_format: Vec<String> = self.vector.iter().map(|f| format!("{f:.4}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text = format!("'[{joined_elements}]'");
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

pub struct BgWorkerLockGuard {
    lock_tag: pgrx::pg_sys::LOCKTAG,
    success: bool,
}

impl BgWorkerLockGuard {
    pub fn new(lock_id: u32, wait: bool) -> Self {
        let lock_tag = pgrx::pg_sys::LOCKTAG {
            locktag_type: pgrx::pg_sys::LockTagType::LOCKTAG_ADVISORY as u8,
            locktag_lockmethodid: pgrx::pg_sys::USER_LOCKMETHOD as u8,
            locktag_field1: VCHORD_MAGIC as u32,
            locktag_field2: lock_id,
            locktag_field3: 0,
            locktag_field4: 0,
        };
        let status = unsafe {
            pgrx::pg_sys::LockAcquire(
                &lock_tag as *const _ as *mut _,
                pgrx::pg_sys::ExclusiveLock as _,
                false,
                !wait,
            )
        };
        let success = status == pgrx::pg_sys::LockAcquireResult::LOCKACQUIRE_OK;
        Self { lock_tag, success }
    }
    pub fn is_success(&self) -> bool {
        self.success
    }
}

impl Drop for BgWorkerLockGuard {
    fn drop(&mut self) {
        unsafe {
            if self.success {
                pgrx::pg_sys::LockRelease(
                    &self.lock_tag as *const _ as *mut _,
                    pgrx::pg_sys::ExclusiveLock as _,
                    false,
                );
            }
        }
    }
}
