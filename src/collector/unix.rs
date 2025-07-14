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

use crate::collector::types::BINCODE_CONFIG;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode, decode_from_slice, encode_to_vec};
use byteorder::{ReadBytesExt, WriteBytesExt};
use send_fd::SendFd;
use std::fmt::{Display, Formatter};
use std::io::{Read, Write};
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::sync::OnceLock;

#[derive(Debug)]
pub enum ConnectionError {
    ClosedConnection,
    PacketTooLarge,
    EncodeFailed(EncodeError),
    DecodeFailed(DecodeError),
}

impl Display for ConnectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionError::ClosedConnection => write!(f, "socket is closed"),
            ConnectionError::PacketTooLarge => write!(f, "packet is too large"),
            ConnectionError::EncodeFailed(e) => write!(f, "failed to encode data: {e}"),
            ConnectionError::DecodeFailed(e) => write!(f, "failed to decode data: {e}"),
        }
    }
}

static CHANNEL: OnceLock<SendFd> = OnceLock::new();

pub fn init() {
    CHANNEL.set(SendFd::new().unwrap()).ok().unwrap();
}

pub fn accept() -> Socket {
    let fd = CHANNEL.get().unwrap().recv().unwrap();
    let stream = UnixStream::from(fd);
    Socket { stream }
}

pub fn connect() -> Socket {
    let (other, stream) = UnixStream::pair().unwrap();
    CHANNEL.get().unwrap().send(other.as_fd()).unwrap();
    Socket { stream }
}

pub struct Socket {
    stream: UnixStream,
}

macro_rules! resolve_closed {
    ($t: expr) => {
        match $t {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(ConnectionError::ClosedConnection)
            }
            Err(e) => panic!("{}", e),
            Ok(e) => e,
        }
    };
}

impl Socket {
    pub fn send<E: Encode>(&mut self, data: E) -> Result<(), ConnectionError> {
        use byteorder::NativeEndian as N;
        let packet = match encode_to_vec(data, BINCODE_CONFIG) {
            Ok(b) => b,
            Err(e) => return Err(ConnectionError::EncodeFailed(e)),
        };
        let len = u32::try_from(packet.len()).map_err(|_| ConnectionError::PacketTooLarge)?;
        resolve_closed!(self.stream.write_u32::<N>(len));
        resolve_closed!(self.stream.write_all(&packet));
        Ok(())
    }
    pub fn recv<D: Decode<()>>(&mut self) -> Result<D, ConnectionError> {
        use byteorder::NativeEndian as N;
        let len = resolve_closed!(self.stream.read_u32::<N>());
        let mut packet = vec![0u8; len as usize];
        resolve_closed!(self.stream.read_exact(&mut packet));
        let (data, _) = decode_from_slice::<D, _>(&packet, BINCODE_CONFIG)
            .map_err(ConnectionError::DecodeFailed)?;
        Ok(data)
    }
}
