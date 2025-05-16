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

use simd::Floating;
use std::ops::Add;

const BITS: usize = 6;
pub const STEP: usize = 65535_usize / ((1_usize << (2 + BITS)) - 1);
pub type BlockLut = ((f32, f32, f32, f32), Vec<[u8; 16]>);
pub type BlockCode<'a> = (
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [[u8; 16]],
);

pub fn preprocess(vector: &[f32]) -> BlockLut {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = simd::quantize::quantize(vector, ((1 << BITS) - 1) as f32);
    let qvector_sum = if vector.len() <= (65535_usize / ((1 << BITS) - 1)) {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x_as_u32(&qvector) as f32
    };
    ((dis_v_2, b, k, qvector_sum), compress(&qvector))
}

pub fn process_l2(
    lut: &BlockLut,
    (dis_u_2, factor_cnt, factor_ip, factor_err, t): BlockCode<'_>,
) -> [(f32, f32); 32] {
    use std::iter::zip;
    let &((dis_v_2, b, k, qvector_sum), ref s) = lut;
    let mut sum = [0_u32; 32];
    for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
        let delta = simd::fast_scan::scan(t, s);
        simd::fast_scan::accu(&mut sum, &delta);
    }
    std::array::from_fn(|i| {
        let e = k * ((2.0 * sum[i] as f32) - qvector_sum) + b * factor_cnt[i];
        let rough = dis_u_2[i] + dis_v_2 - 2.0 * e * factor_ip[i];
        let err = 2.0 * factor_err[i] * dis_v_2.sqrt();
        (rough, err)
    })
}

pub fn process_dot(
    lut: &BlockLut,
    (_, factor_cnt, factor_ip, factor_err, t): BlockCode<'_>,
) -> [(f32, f32); 32] {
    use std::iter::zip;
    let &((dis_v_2, b, k, qvector_sum), ref s) = lut;
    let mut sum = [0_u32; 32];
    for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
        let delta = simd::fast_scan::scan(t, s);
        simd::fast_scan::accu(&mut sum, &delta);
    }
    std::array::from_fn(|i| {
        let e = k * ((2.0 * sum[i] as f32) - qvector_sum) + b * factor_cnt[i];
        let rough = -e * factor_ip[i];
        let err = factor_err[i] * dis_v_2.sqrt();
        (rough, err)
    })
}

pub(crate) fn compress<T: Copy + Default + Add<T, Output = T>>(vector: &[T]) -> Vec<[T; 16]> {
    let f = |[t_0, t_1, t_2, t_3]: [T; 4]| {
        [
            T::default(),
            t_0,
            t_1,
            t_1 + t_0,
            t_2,
            t_2 + t_0,
            t_2 + t_1,
            t_2 + t_1 + t_0,
            t_3,
            t_3 + t_0,
            t_3 + t_1,
            t_3 + t_1 + t_0,
            t_3 + t_2,
            t_3 + t_2 + t_0,
            t_3 + t_2 + t_1,
            t_3 + t_2 + t_1 + t_0,
        ]
    };

    let (arrays, reminder) = vector.as_chunks::<4>();
    let mut result = arrays.iter().copied().map(f).collect::<Vec<_>>();
    if !reminder.is_empty() {
        let mut array = [T::default(); 4];
        array[..reminder.len()].copy_from_slice(reminder);
        result.push(f(array));
    }
    result
}
