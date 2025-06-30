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

use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::fmt::Debug;

pub fn robust_prune<T: Copy + Ord + Debug, V>(
    mut d: impl FnMut(&V, &V) -> Distance,
    u: T,
    outs: impl Iterator<Item = ((Reverse<Distance>, AlwaysEqual<T>), V)>,
    m: u32,
    alpha: f32,
) -> Vec<(Reverse<Distance>, AlwaysEqual<T>)> {
    // V ← (V ∪ Nout(p)) \ {p}
    let mut trace = outs.collect::<Vec<_>>();
    trace.sort_by_key(|((_, AlwaysEqual(v)), _)| *v);
    trace.dedup_by_key(|((_, AlwaysEqual(v)), _)| *v);
    trace.retain(|((_, AlwaysEqual(v)), _)| *v != u);
    trace.sort_by_key(|&((Reverse(d), _), _)| d);
    // Nout(p) ← ∅
    let mut result = Vec::new();
    let mut pruned = Vec::new();
    for ((Reverse(dis_u), AlwaysEqual(u)), vector_u) in trace {
        if result.len() == m as usize {
            break;
        }
        let check = result
            .iter()
            .map(|(_, vector_v)| d(&vector_u, vector_v))
            .all(|dis| dis_u.to_f32() < alpha * dis.to_f32());
        if check {
            result.push(((Reverse(dis_u), AlwaysEqual(u)), vector_u));
        } else {
            pruned.push(((Reverse(dis_u), AlwaysEqual(u)), vector_u));
        }
    }
    result
        .into_iter()
        .chain(pruned)
        .map(|(x, _)| x)
        .take(m as _)
        .collect()
}
