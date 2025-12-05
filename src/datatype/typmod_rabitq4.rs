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

use std::ffi::CStr;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_typmod_in(list: pgrx::datum::Array<&CStr>) -> i32 {
    if list.is_empty() {
        -1
    } else if list.len() == 1 {
        let s = list.get(0).unwrap().unwrap().to_str().unwrap();
        if let Ok(d) = s.parse::<i32>() {
            if d < 1 {
                pgrx::error!("dimensions for type rabitq4 must be at least 1");
            }
            if d > 65535 {
                pgrx::error!("dimensions for type rabitq4 cannot exceed 65535");
            }
            d
        } else {
            pgrx::error!("invalid type modifier")
        }
    } else {
        pgrx::error!("invalid type modifier")
    }
}
