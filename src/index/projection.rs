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

use random_orthogonal_matrix::random_orthogonal_matrix;
use std::sync::OnceLock;

fn matrix(n: usize) -> Option<&'static Vec<Vec<f32>>> {
    static MATRIXS: [OnceLock<Vec<Vec<f32>>>; 1 + 60000] = [const { OnceLock::new() }; 1 + 60000];
    MATRIXS
        .get(n)
        .map(|x| x.get_or_init(|| random_orthogonal_matrix(n)))
}

pub fn prewarm(n: usize) {
    let _ = matrix(n);
}

pub fn project(vector: &[f32]) -> Vec<f32> {
    use simd::Floating;
    let n = vector.len();
    let matrix = matrix(n).expect("dimension too large");
    (0..n)
        .map(|i| f32::reduce_sum_of_xy(vector, &matrix[i]))
        .collect()
}
