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

use crate::collector::worker::push;
use rand::Rng;
use std::error::Error;
use std::num::ParseIntError;
use std::{io, result};

#[derive(Debug)]
pub enum CollectorError {
    Sqlite(rusqlite::Error),
    Io(io::Error),
    ParseIntError(ParseIntError),
}

impl std::fmt::Display for CollectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectorError::Sqlite(e) => write!(f, "SQLite error: {e}"),
            CollectorError::Io(e) => write!(f, "IO error: {e}"),
            CollectorError::ParseIntError(e) => write!(f, "Parse int error: {e}"),
        }
    }
}

impl Error for CollectorError {}

impl From<rusqlite::Error> for CollectorError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sqlite(value)
    }
}

impl From<io::Error> for CollectorError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ParseIntError> for CollectorError {
    fn from(value: ParseIntError) -> Self {
        Self::ParseIntError(value)
    }
}

pub type Result<T> = result::Result<T, CollectorError>;

pub struct PGLockGuard {
    lock_tag: pgrx::pg_sys::LOCKTAG,
    success: bool,
}

const VCHORD_MAGIC: u32 = 0x5643484f;

impl PGLockGuard {
    pub fn new(lock_id: u32, block: bool) -> Self {
        let lock_tag = pgrx::pg_sys::LOCKTAG {
            locktag_type: pgrx::pg_sys::LockTagType::LOCKTAG_ADVISORY as u8,
            locktag_lockmethodid: pgrx::pg_sys::USER_LOCKMETHOD as u8,
            locktag_field1: VCHORD_MAGIC,
            locktag_field2: lock_id,
            locktag_field3: 0,
            locktag_field4: 0,
        };
        let status = unsafe {
            pgrx::pg_sys::LockAcquire(
                &lock_tag as *const _ as *mut _,
                pgrx::pg_sys::ExclusiveLock as _,
                false,
                !block,
            )
        };
        let success = status == pgrx::pg_sys::LockAcquireResult::LOCKACQUIRE_OK;
        Self { lock_tag, success }
    }
    pub fn is_success(&self) -> bool {
        self.success
    }
}

impl Drop for PGLockGuard {
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

pub enum SendVector {
    Vchordrq(vchordrq::types::OwnedVector),
    Vchordg(vchordg::types::OwnedVector),
}

impl std::fmt::Display for SendVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendVector::Vchordrq(vchordrq::types::OwnedVector::Vecf32(e))
            | SendVector::Vchordg(vchordg::types::OwnedVector::Vecf32(e)) => {
                let vector_format: Vec<String> = e
                    .clone()
                    .into_vec()
                    .iter()
                    .map(|f| format!("{f:.4}"))
                    .collect();
                let joined_elements = vector_format.join(", ");
                write!(f, "'[{joined_elements}]'")
            }
            SendVector::Vchordrq(vchordrq::types::OwnedVector::Vecf16(e))
            | SendVector::Vchordg(vchordg::types::OwnedVector::Vecf16(e)) => {
                let vector_format: Vec<String> = e
                    .clone()
                    .into_vec()
                    .iter()
                    .map(|f| format!("{:.4}", f.to_f32()))
                    .collect();
                let joined_elements = vector_format.join(", ");
                write!(f, "'[{joined_elements}]'")
            }
        }
    }
}

pub trait CollectorSender {
    fn send(&self, sample: &str);
}

#[derive(Debug)]
pub struct DefaultSender {
    pub enable: bool,
    pub send_prob: Option<f64>,
    pub max_records: u32,
    pub database_oid: u32,
    pub index_oid: u32,
}

impl CollectorSender for DefaultSender {
    fn send(&self, sample: &str) {
        if !self.enable {
            return;
        }
        if let Some(rate) = self.send_prob {
            let mut rng = rand::rng();
            if rng.random_bool(rate)
                && let Err(e) = push(self.database_oid, self.index_oid, sample, self.max_records)
            {
                pgrx::warning!("Collector: Error pushing sample: {}", e);
            }
        }
    }
}
