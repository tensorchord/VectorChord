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

use crate::Id;
use std::collections::HashSet;

pub struct Visited {
    inner: HashSet<Id>,
}

impl Visited {
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }
    pub fn insert(&mut self, x: Id) {
        self.inner.insert(x);
    }
    pub fn contains(&mut self, x: Id) -> bool {
        self.inner.contains(&x)
    }
}
