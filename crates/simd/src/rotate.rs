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

pub mod givens {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn givens(lhs: &mut [f32], rhs: &mut [f32]) {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let scale = 1.0 / (2.0_f32).sqrt();
        for i in 0..n {
            (lhs[i], rhs[i]) = ((lhs[i] + rhs[i]) * scale, (lhs[i] - rhs[i]) * scale);
        }
    }
}

pub fn givens(lhs: &mut [f32], rhs: &mut [f32]) {
    givens::givens(lhs, rhs)
}

pub mod flip {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn flip(bits: &[u64; 1024], result: &mut [f32]) {
        use std::hint::select_unpredictable;
        let result: &mut [u32] = zerocopy::transmute_mut!(result);
        let (arrays, remainder) = result.as_chunks_mut::<64>();
        let n = arrays.len();
        assert!(n <= 1024);
        for i in 0..n {
            for j in 0..64 {
                arrays[i][j] ^= select_unpredictable((bits[i] & (1 << j)) != 0, 0x80000000, 0);
            }
        }
        for j in 0..remainder.len() {
            remainder[j] ^= select_unpredictable((bits[n] & (1 << j)) != 0, 0x80000000, 0);
        }
    }
}

pub fn flip(bits: &[u64; 1024], result: &mut [f32]) {
    flip::flip(bits, result)
}
