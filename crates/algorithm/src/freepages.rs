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

use crate::tuples::*;
use crate::*;
use std::cmp::Reverse;

pub fn mark(index: &impl RelationWrite, freepage_first: u32, values: &[u32]) {
    let mut values = {
        let mut values = values.to_vec();
        values.sort_by_key(|x| Reverse(*x));
        values.dedup();
        values
    };
    let (mut current, mut offset) = (freepage_first, 0_u32);
    loop {
        let mut freespace_guard = index.write(current, false);
        if freespace_guard.len() == 0 {
            freespace_guard
                .alloc(&FreepageTuple::serialize(&FreepageTuple {}))
                .expect("implementation: a clear page cannot accommodate a single tuple");
        }
        let freespace_bytes = freespace_guard.get_mut(1).expect("data corruption");
        let mut freespace_tuple = FreepageTuple::deserialize_mut(freespace_bytes);
        while let Some(target) = values.pop_if(|&mut x| x < offset + 32768) {
            freespace_tuple.mark((target - offset) as usize);
        }
        if values.is_empty() {
            return;
        }
        if freespace_guard.get_opaque().next == u32::MAX {
            let extend = index.extend(false);
            freespace_guard.get_opaque_mut().next = extend.id();
        }
        (current, offset) = (freespace_guard.get_opaque().next, offset + 32768);
    }
}

pub fn fetch(index: &impl RelationWrite, freepage_first: u32) -> Option<u32> {
    let (mut current, mut offset) = (freepage_first, 0_u32);
    loop {
        let mut freespace_guard = index.write(current, false);
        if freespace_guard.len() == 0 {
            freespace_guard
                .alloc(&FreepageTuple::serialize(&FreepageTuple {}))
                .expect("implementation: a clear page cannot accommodate a single tuple");
        }
        let freespace_bytes = freespace_guard.get_mut(1).expect("data corruption");
        let mut freespace_tuple = FreepageTuple::deserialize_mut(freespace_bytes);
        if let Some(x) = freespace_tuple.fetch() {
            return Some(x as u32 + offset);
        }
        if freespace_guard.get_opaque().next == u32::MAX {
            return None;
        }
        (current, offset) = (freespace_guard.get_opaque().next, offset + 32768);
    }
}
