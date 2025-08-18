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

use crate::collector::types::{BINCODE_CONFIG, MAX_QUERY_LEN};
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode, decode_from_slice, encode_to_vec};
use pgrx::pg_sys;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ptr::NonNull;

#[derive(Debug)]
pub enum MessageQueueError {
    Detached,
    WouldBlock,
    EncodeFailed(EncodeError),
    DecodeFailed(DecodeError),
    Unknown(pg_sys::shm_mq_result::Type),
}

impl Display for MessageQueueError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageQueueError::Detached => write!(f, "queue is detached"),
            MessageQueueError::WouldBlock => write!(f, "queue is full"),
            MessageQueueError::EncodeFailed(e) => write!(f, "failed to encode data: {e}"),
            MessageQueueError::DecodeFailed(e) => write!(f, "failed to decode data: {e}"),
            MessageQueueError::Unknown(other) => write!(f, "unknown error code: {other}"),
        }
    }
}

impl Error for MessageQueueError {}

impl From<pg_sys::shm_mq_result::Type> for MessageQueueError {
    fn from(value: pg_sys::shm_mq_result::Type) -> Self {
        match value {
            pg_sys::shm_mq_result::SHM_MQ_WOULD_BLOCK => Self::WouldBlock,
            pg_sys::shm_mq_result::SHM_MQ_DETACHED => Self::Detached,
            other => Self::Unknown(other),
        }
    }
}

pub struct MessageQueueSender {
    handle: NonNull<pg_sys::shm_mq_handle>,
    creator: bool,
}

impl Drop for MessageQueueSender {
    fn drop(&mut self) {
        unsafe {
            if !self.creator {
                pg_sys::pfree(self.handle.as_ptr() as _);
            }
        }
    }
}

impl MessageQueueSender {
    pub unsafe fn new(mq: *mut pg_sys::shm_mq, creator: bool) -> Self {
        unsafe {
            let send_mq = if creator {
                pg_sys::shm_mq_create(mq.cast(), MAX_QUERY_LEN)
            } else {
                mq
            };
            if pg_sys::shm_mq_get_sender(send_mq).is_null() {
                pg_sys::shm_mq_set_sender(send_mq, pg_sys::MyProc);
            }
            let handle = pg_sys::shm_mq_attach(send_mq, std::ptr::null_mut(), std::ptr::null_mut());
            Self {
                handle: NonNull::new_unchecked(handle),
                creator,
            }
        }
    }
    pub unsafe fn shutdown(&self) {
        unsafe {
            pg_sys::shm_mq_detach(self.handle.as_ptr());
        }
    }
    pub fn send<E: Encode>(&self, data: E) -> Result<(), MessageQueueError> {
        let msg = match encode_to_vec(data, BINCODE_CONFIG) {
            Ok(b) => b,
            Err(e) => return Err(MessageQueueError::EncodeFailed(e)),
        };
        unsafe {
            let msg: &[u8] = msg.as_ref();
            #[cfg(any(feature = "pg13", feature = "pg14"))]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                false,
            );

            #[cfg(any(feature = "pg15", feature = "pg16", feature = "pg17", feature = "pg18"))]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                false,
                true,
            );
            match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => Ok(()),
                other => Err(MessageQueueError::from(other)),
            }
        }
    }
}

pub struct MessageQueueReceiver {
    handle: NonNull<pg_sys::shm_mq_handle>,
    creator: bool,
}

impl Drop for MessageQueueReceiver {
    fn drop(&mut self) {
        unsafe {
            if !self.creator {
                pg_sys::pfree(self.handle.as_ptr() as _);
            }
        }
    }
}

impl MessageQueueReceiver {
    pub unsafe fn new(mq: *mut pg_sys::shm_mq, creator: bool) -> Self {
        unsafe {
            let recv_mq = if creator {
                pg_sys::shm_mq_create(mq.cast(), MAX_QUERY_LEN)
            } else {
                mq
            };
            if pg_sys::shm_mq_get_receiver(recv_mq).is_null() {
                pg_sys::shm_mq_set_receiver(recv_mq, pg_sys::MyProc);
            }
            let handle = pg_sys::shm_mq_attach(recv_mq, std::ptr::null_mut(), std::ptr::null_mut());
            Self {
                handle: NonNull::new_unchecked(handle),
                creator,
            }
        }
    }
    pub unsafe fn shutdown(&self) {
        unsafe {
            pg_sys::shm_mq_detach(self.handle.as_ptr());
        }
    }
    pub fn recv<D: Decode<()>>(&self) -> Result<D, MessageQueueError> {
        unsafe {
            let mut len = 0usize;
            let mut msg = std::ptr::null_mut();
            let result = pg_sys::shm_mq_receive(self.handle.as_ptr(), &mut len, &mut msg, false);
            let msg_bytes = match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => {
                    std::slice::from_raw_parts(msg as *mut u8, len).to_vec()
                }
                other => return Err(MessageQueueError::from(other)),
            };
            let (data, _) = decode_from_slice::<D, _>(&msg_bytes, BINCODE_CONFIG)
                .map_err(MessageQueueError::DecodeFailed)?;
            Ok(data)
        }
    }
}
