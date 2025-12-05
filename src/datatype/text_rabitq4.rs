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
use pgrx::pg_sys::Oid;
use std::ffi::{CStr, CString};
use vector::rabitq4::Rabitq4Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_in(input: &CStr, oid: Oid, typmod: i32) -> Rabitq4Output {
    let _ = (oid, typmod);
    let mut input = input.to_bytes().iter();
    let mut p0 = Vec::<f32>::new();
    let mut p1 = Vec::<u8>::new();
    {
        loop {
            let Some(c) = input.next().copied() else {
                pgrx::error!("incorrect vector")
            };
            match c {
                b' ' => (),
                b'(' => break,
                _ => pgrx::error!("incorrect vector"),
            }
        }
    }
    {
        let mut s = Option::<String>::None;
        loop {
            let Some(c) = input.next().copied() else {
                pgrx::error!("incorrect vector")
            };
            s = match (s, c) {
                (s, b' ') => s,
                (None, c @ (b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'.' | b'+' | b'-')) => {
                    Some(String::from(c as char))
                }
                (Some(s), c @ (b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'.' | b'+' | b'-')) => {
                    let mut x = s;
                    x.push(c as char);
                    Some(x)
                }
                (Some(s), b',') => {
                    p0.push(s.parse().expect("failed to parse number"));
                    None
                }
                (None, b',') => {
                    pgrx::error!("incorrect vector")
                }
                (Some(s), b')') => {
                    p0.push(s.parse().expect("failed to parse number"));
                    break;
                }
                (None, b')') => break,
                _ => pgrx::error!("incorrect vector"),
            };
        }
    }
    {
        loop {
            let Some(c) = input.next().copied() else {
                pgrx::error!("incorrect vector")
            };
            match c {
                b' ' => (),
                b'[' => break,
                _ => pgrx::error!("incorrect vector"),
            }
        }
    }
    {
        let mut s = Option::<String>::None;
        loop {
            let Some(c) = input.next().copied() else {
                pgrx::error!("incorrect vector")
            };
            s = match (s, c) {
                (s, b' ') => s,
                (None, c @ (b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'.' | b'+' | b'-')) => {
                    Some(String::from(c as char))
                }
                (Some(s), c @ (b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'.' | b'+' | b'-')) => {
                    let mut x = s;
                    x.push(c as char);
                    Some(x)
                }
                (Some(s), b',') => {
                    p1.push(s.parse().expect("failed to parse number"));
                    None
                }
                (None, b',') => {
                    pgrx::error!("incorrect vector")
                }
                (Some(s), b']') => {
                    p1.push(s.parse().expect("failed to parse number"));
                    break;
                }
                (None, b']') => break,
                _ => pgrx::error!("incorrect vector"),
            };
        }
    }
    if p0.len() != 4 {
        pgrx::error!("incorrect vector");
    }
    if p1.is_empty() {
        pgrx::error!("vector must have at least 1 dimension");
    }
    let sum_of_x2 = p0[0];
    let norm_of_lattice = p0[1];
    let sum_of_code = p0[2];
    let sum_of_abs_x = p0[3];
    let unpacked_code = p1;
    let packed_code = rabitq::halfbyte::pack_code(&unpacked_code);
    if let Some(x) = Rabitq4Borrowed::new_checked(
        unpacked_code.len() as _,
        sum_of_x2,
        norm_of_lattice,
        sum_of_code,
        sum_of_abs_x,
        &packed_code,
    ) {
        Rabitq4Output::new(x)
    } else {
        pgrx::error!("incorrect vector");
    }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_out(vector: Rabitq4Input<'_>) -> CString {
    let vector = vector.as_borrowed();
    let mut buffer = String::new();
    buffer.push('(');
    buffer.push_str(format!("{}", vector.sum_of_x2()).as_str());
    buffer.push_str(format!(", {}", vector.norm_of_lattice()).as_str());
    buffer.push_str(format!(", {}", vector.sum_of_code()).as_str());
    buffer.push_str(format!(", {}", vector.sum_of_abs_x()).as_str());
    buffer.push(')');
    buffer.push('[');
    let mut unpacked_code = vector.unpacked_code();
    if let Some(x) = unpacked_code.next() {
        buffer.push_str(format!("{x}").as_str());
    }
    for x in unpacked_code {
        buffer.push_str(format!(", {x}").as_str());
    }
    buffer.push(']');
    CString::new(buffer).unwrap()
}
