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

use std::num::NonZero;

#[derive(Debug, Clone, Copy)]
pub enum Typmod {
    Any,
    Dims(NonZero<u32>),
}

impl Typmod {
    pub fn new(x: i32) -> Option<Self> {
        use Typmod::*;
        if x == -1 {
            Some(Any)
        } else if x >= 1 {
            Some(Dims(NonZero::new(x as u32).unwrap()))
        } else {
            None
        }
    }
    pub fn dim(self) -> Option<NonZero<u32>> {
        use Typmod::*;
        match self {
            Any => None,
            Dims(dim) => Some(dim),
        }
    }
}
