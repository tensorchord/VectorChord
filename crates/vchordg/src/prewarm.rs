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

use crate::Opaque;
use crate::operator::Operator;
use algo::{Page, RelationRead};

pub fn prewarm<R: RelationRead, O: Operator>(index: &R) -> String
where
    R::Page: Page<Opaque = Opaque>,
{
    use std::fmt::Write;
    let meta_guard = index.read(0);
    let link = meta_guard.get_opaque().link;
    drop(meta_guard);
    let mut number_of_vertex_pages = 0_u64;
    let mut number_of_vertices = 0_u64;
    {
        let mut current = link;
        while current != u32::MAX {
            let vertex_guard = index.read(current);
            number_of_vertex_pages += 1;
            for i in 1..=vertex_guard.len() {
                if vertex_guard.get(i).is_some() {
                    number_of_vertices += 1;
                }
            }
            current = vertex_guard.get_opaque().next;
        }
    }
    let mut message = String::new();
    writeln!(message, "number of vertex pages: {number_of_vertex_pages}").unwrap();
    writeln!(message, "number of vertices: {number_of_vertices}").unwrap();
    message
}
