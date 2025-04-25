use super::memory_scalar8::Scalar8Output;
use crate::datatype::memory_scalar8::Scalar8Input;
use pgrx::pg_sys::Oid;
use std::ffi::{CStr, CString};
use vector::scalar8::Scalar8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_in(input: &CStr, oid: Oid, typmod: i32) -> Scalar8Output {
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
    let k = p0[1];
    let b = p0[2];
    let sum_of_code = p0[3];
    let code = p1;
    if let Some(x) = Scalar8Borrowed::new_checked(sum_of_x2, k, b, sum_of_code, &code) {
        Scalar8Output::new(x)
    } else {
        pgrx::error!("incorrect vector");
    }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_out(vector: Scalar8Input<'_>) -> CString {
    let vector = vector.as_borrowed();
    let mut buffer = String::new();
    buffer.push('(');
    buffer.push_str(format!("{}", vector.sum_of_x2()).as_str());
    buffer.push_str(format!(", {}", vector.k()).as_str());
    buffer.push_str(format!(", {}", vector.b()).as_str());
    buffer.push_str(format!(", {}", vector.sum_of_code()).as_str());
    buffer.push(')');
    buffer.push('[');
    if let Some(&x) = vector.code().first() {
        buffer.push_str(format!("{x}").as_str());
    }
    for &x in vector.code().iter().skip(1) {
        buffer.push_str(format!(", {x}").as_str());
    }
    buffer.push(']');
    CString::new(buffer).unwrap()
}
