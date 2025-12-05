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

use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use pgrx::datum::Internal;
use pgrx::pg_sys::Oid;
use vector::VectorBorrowed;
use vector::rabitq4::Rabitq4Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_send(vector: Rabitq4Input<'_>) -> Vec<u8> {
    let vector = vector.as_borrowed();
    let mut stream = Vec::<u8>::new();
    stream.extend(vector.dim().to_be_bytes());
    stream.extend(vector.sum_of_x2().to_be_bytes());
    stream.extend(vector.norm_of_lattice().to_be_bytes());
    stream.extend(vector.sum_of_code().to_be_bytes());
    stream.extend(vector.sum_of_abs_x().to_be_bytes());
    for &c in vector.packed_code() {
        stream.extend(c.to_be_bytes());
    }
    stream
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_recv(mut internal: Internal, oid: Oid, typmod: i32) -> Rabitq4Output {
    let _ = (oid, typmod);
    let buf = unsafe { internal.get_mut::<pgrx::pg_sys::StringInfoData>().unwrap() };

    let dim = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        u32::from_be_bytes(raw)
    };
    let sum_of_x2 = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let norm_of_lattice = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let sum_of_code = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let sum_of_abs_x = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let packed_code = {
        let mut result = Vec::new();
        for _ in 0..dim.div_ceil(2) {
            result.push({
                assert!(buf.cursor < i32::MAX - 1 && buf.cursor + 1 <= buf.len);
                let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 1]>().read() };
                buf.cursor += 1;
                u8::from_be_bytes(raw)
            });
        }
        result
    };

    if let Some(x) = Rabitq4Borrowed::new_checked(
        dim,
        sum_of_x2,
        norm_of_lattice,
        sum_of_code,
        sum_of_abs_x,
        &packed_code,
    ) {
        Rabitq4Output::new(x)
    } else {
        pgrx::error!("detect data corruption");
    }
}
