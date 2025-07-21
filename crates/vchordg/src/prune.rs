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

pub fn prune<T, V, K: Ord>(
    mut d: impl FnMut(&V, &V) -> Distance,
    u: T,
    outs: impl Iterator<Item = ((Reverse<Distance>, AlwaysEqual<T>), V)>,
    m: u32,
    alpha: &[f32],
    key: impl Fn(&T) -> K,
    is_l2s: bool,
) -> Vec<(Reverse<Distance>, AlwaysEqual<T>)> {
    // V ← (V ∪ Nout(p)) \ {p}
    let mut trace = outs.collect::<Vec<_>>();
    trace.sort_by_key(|((_, AlwaysEqual(v)), _)| key(v));
    trace.dedup_by_key(|((_, AlwaysEqual(v)), _)| key(v));
    trace.retain(|((_, AlwaysEqual(v)), _)| key(v) != key(&u));
    trace.sort_by_key(|&((Reverse(d), _), _)| d);
    // Nout(p) ← ∅
    let mut result = Vec::new();
    for &alpha in if is_l2s { alpha } else { &[1.0] } {
        if result.len() == m as usize {
            break;
        }
        trace = robust_prune(&mut d, m, alpha, &mut result, trace);
    }
    if !(result.len() == m as usize) {
        result.extend(trace);
        result.truncate(m as usize);
    }
    result.into_iter().map(|(x, _)| x).collect()
}

fn robust_prune<T, V>(
    mut d: impl FnMut(&V, &V) -> Distance,
    m: u32,
    alpha: f32,
    result: &mut Vec<((Reverse<Distance>, AlwaysEqual<T>), V)>,
    trace: Vec<((Reverse<Distance>, AlwaysEqual<T>), V)>,
) -> Vec<((Reverse<Distance>, AlwaysEqual<T>), V)> {
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
    pruned
}
