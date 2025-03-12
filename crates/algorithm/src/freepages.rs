use crate::tuples::*;
use crate::*;
use std::cmp::Reverse;

pub fn mark(index: impl RelationWrite, freepage_first: u32, pages: &[u32]) {
    let mut pages = pages.to_vec();
    pages.sort_by_key(|x| Reverse(*x));
    pages.dedup();
    let (mut current, mut offset) = (freepage_first, 0_u32);
    while !pages.is_empty() {
        let locals = {
            let mut local = Vec::new();
            while let Some(target) = pages.pop_if(|x| (offset..offset + 32768).contains(x)) {
                local.push(target - offset);
            }
            local
        };
        let mut freespace_guard = index.write(current, false);
        if freespace_guard.len() == 0 {
            freespace_guard.alloc(&FreepageTuple::serialize(&FreepageTuple {}));
        }
        let freespace_bytes = freespace_guard.get_mut(1).expect("data corruption");
        let mut freespace_tuple = FreepageTuple::deserialize_mut(freespace_bytes);
        for local in locals {
            freespace_tuple.mark(local as _);
        }
        if freespace_guard.get_opaque().next == u32::MAX {
            let extend = index.extend(false);
            freespace_guard.get_opaque_mut().next = extend.id();
        }
        (current, offset) = (freespace_guard.get_opaque().next, offset + 32768);
    }
}

pub fn fetch(index: impl RelationWrite, freepage_first: u32) -> Option<u32> {
    let (mut current, mut offset) = (freepage_first, 0_u32);
    loop {
        let mut freespace_guard = index.write(current, false);
        if freespace_guard.len() == 0 {
            return None;
        }
        let freespace_bytes = freespace_guard.get_mut(1).expect("data corruption");
        let mut freespace_tuple = FreepageTuple::deserialize_mut(freespace_bytes);
        if let Some(local) = freespace_tuple.fetch() {
            return Some(local as u32 + offset);
        }
        if freespace_guard.get_opaque().next == u32::MAX {
            return None;
        }
        (current, offset) = (freespace_guard.get_opaque().next, offset + 32768);
    }
}
