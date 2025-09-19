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

use std::ffi::{CStr, CString};
use std::num::NonZero;

#[derive(Debug, Clone, Copy)]
pub enum Typmod {
    Any,
    Dims(NonZero<u32>),
}

impl Typmod {
    pub fn parse_from_i32(x: i32) -> Option<Self> {
        use Typmod::*;
        if x == -1 {
            Some(Any)
        } else if x >= 1 {
            Some(Dims(NonZero::new(x as u32).unwrap()))
        } else {
            None
        }
    }
    pub fn into_option_string(self) -> Option<String> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(x) => Some(x.get().to_string()),
        }
    }
    pub fn into_i32(self) -> i32 {
        use Typmod::*;
        match self {
            Any => -1,
            Dims(x) => x.get() as i32,
        }
    }
    pub fn dims(self) -> Option<NonZero<u32>> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(dims) => Some(dims),
        }
    }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_typmod_in_65535(list: pgrx::datum::Array<&CStr>) -> i32 {
    if list.is_empty() {
        -1
    } else if list.len() == 1 {
        let s = list.get(0).unwrap().unwrap().to_str().unwrap();
        let d = s.parse::<u32>().ok();
        if let Some(d @ 1..=65535) = d {
            let typmod = Typmod::Dims(NonZero::new(d).unwrap());
            typmod.into_i32()
        } else {
            pgrx::error!("Modifier of the type is invalid.")
        }
    } else {
        pgrx::error!("Modifier of the type is invalid.")
    }
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_typmod_out(typmod: i32) -> CString {
    let typmod = Typmod::parse_from_i32(typmod).unwrap();
    match typmod.into_option_string() {
        Some(s) => CString::new(format!("({s})")).unwrap(),
        None => CString::new("()").unwrap(),
    }
}
