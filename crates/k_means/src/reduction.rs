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

use crate::square::Square;
use simd::Floating;
use std::collections::BinaryHeap;

pub fn k_means_lookup(vector: &[f32], centroids: &Square) -> usize {
    assert_ne!(centroids.len(), 0);
    let mut result = (f32::INFINITY, 0);
    for i in 0..centroids.len() {
        let dis = f32::reduce_sum_of_d2(vector, &centroids[i]);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

pub fn k_means_centroids_inplace(centroids: &mut [f32], count: u32, is_spherical: bool) {
    assert!(!centroids.is_empty());
    assert!(count > 0);
    let dim = centroids.len();
    let c = 1.0 / count as f32;
    for d in 0..dim {
        centroids[d] *= c;
    }
    if is_spherical {
        let l = f32::reduce_sum_of_x2(centroids).sqrt();
        f32::vector_mul_scalar_inplace(centroids, 1.0 / l);
    }
}

/// Allocate clusters to different parts according to the given proportions
/// by successive quotients method.
///
/// See: https://en.wikipedia.org/wiki/Sainte-Lagu%C3%AB_method
pub fn successive_quotients_allocate(all_clusters: u32, proportion: Vec<u32>) -> Vec<u32> {
    let mut alloc_lists = vec![1u32; proportion.len()];
    let mut diff = all_clusters as i32 - proportion.len() as i32;
    if diff < 0 {
        panic!(
            "build.lists is too large: requested {}, but only {} are available.",
            all_clusters,
            proportion.len()
        );
    }
    let mut priorities: BinaryHeap<PriorityItem> = proportion
        .iter()
        .enumerate()
        .map(|(i, x)| PriorityItem {
            index: i,
            priority: *x as f64 / (alloc_lists[0] as f64),
        })
        .collect();
    while diff > 0 {
        let top = priorities.pop().unwrap();
        alloc_lists[top.index] += 1;
        priorities.push(PriorityItem {
            index: top.index,
            priority: proportion[top.index] as f64 / (alloc_lists[top.index] as f64),
        });
        diff -= 1;
    }
    alloc_lists
}

#[derive(Debug, PartialEq)]
struct PriorityItem {
    index: usize,
    priority: f64,
}

impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.priority.is_nan() {
            std::cmp::Ordering::Less
        } else if other.priority.is_nan() {
            std::cmp::Ordering::Greater
        } else {
            match self.priority.partial_cmp(&other.priority) {
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal,
            }
        }
    }
}
