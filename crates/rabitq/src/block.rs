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

use crate::CodeMetadata;

const BITS: usize = 8;
pub const STEP: usize = 65535_usize / ((1_usize << BITS) - 1);

#[derive(Debug, Clone, Copy)]
pub struct BlockLutMetadata {
    dis_v_2: f32,
    k: f32,
    c: f32,
}

pub type BlockLut = (BlockLutMetadata, Vec<[u8; 16]>);
pub type BlockCode<'a> = (
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [f32; 32],
    &'a [[u8; 16]],
);

pub fn preprocess(vector: &[f32]) -> BlockLut {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    preprocess_with_distance(vector, dis_v_2)
}

pub(crate) fn preprocess_with_distance(vector: &[f32], dis_v_2: f32) -> BlockLut {
    let cvector = compress(vector);
    let (k, b, cqvector) =
        simd::quantize::quantize(cvector.as_flattened(), ((1 << BITS) - 1) as f32);
    (
        BlockLutMetadata {
            dis_v_2,
            k,
            c: b * vector.len().div_ceil(4) as f32,
        },
        cqvector.as_chunks::<16>().0.to_vec(),
    )
}

pub fn full_process_l2(
    lut: &BlockLut,
    (dis_u_2, _, factor_ip, factor_err, t): BlockCode<'_>,
) -> [(f32, f32); 32] {
    use std::iter::zip;
    let &(BlockLutMetadata { dis_v_2, k, c }, ref s) = lut;
    let mut sum = [0_u32; 32];
    for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
        let delta = simd::fast_scan::scan(t, s);
        simd::fast_scan::accu(&mut sum, &delta);
    }
    std::array::from_fn(|i| {
        let e = k * (sum[i] as f32) + c;
        let rough = dis_u_2[i] + dis_v_2 - 2.0 * e * factor_ip[i];
        let err = 2.0 * factor_err[i] * dis_v_2.sqrt();
        (rough, err)
    })
}

pub fn full_process_dot(
    lut: &BlockLut,
    (_, _, factor_ip, factor_err, t): BlockCode<'_>,
) -> [(f32, f32); 32] {
    use std::iter::zip;
    let &(BlockLutMetadata { dis_v_2, k, c }, ref s) = lut;
    let mut sum = [0_u32; 32];
    for (t, s) in zip(t.chunks(STEP), s.chunks(STEP)) {
        let delta = simd::fast_scan::scan(t, s);
        simd::fast_scan::accu(&mut sum, &delta);
    }
    std::array::from_fn(|i| {
        let e = k * (sum[i] as f32) + c;
        let rough = -e * factor_ip[i];
        let err = factor_err[i] * dis_v_2.sqrt();
        (rough, err)
    })
}

pub fn half_process_l2(
    sum: u32,
    CodeMetadata {
        dis_u_2,
        factor_cnt: _,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
) -> (f32, f32) {
    let e = k * (sum as f32) + c;
    let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
    let err = 2.0 * factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn half_process_l2_residual(
    sum: u32,
    CodeMetadata {
        dis_u_2,
        factor_cnt: _,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BlockLutMetadata { dis_v_2: _, k, c }: BlockLutMetadata,
    dis_f: f32,
    delta: f32,
) -> (f32, f32) {
    let e = k * (sum as f32) + c;
    let rough = dis_u_2 + dis_f - 2.0 * e * factor_ip + delta;
    let err = 2.0 * factor_err * dis_f.sqrt();
    (rough, err)
}

pub fn half_process_dot(
    sum: u32,
    CodeMetadata {
        dis_u_2: _,
        factor_cnt: _,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
) -> (f32, f32) {
    let e = k * (sum as f32) + c;
    let rough = -e * factor_ip;
    let err = factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn half_process_dot_residual(
    sum: u32,
    CodeMetadata {
        dis_u_2: _,
        factor_cnt: _,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BlockLutMetadata { dis_v_2, k, c }: BlockLutMetadata,
    dis_f: f32,
    delta: f32,
    norm: f32,
) -> (f32, f32) {
    let e = k * (sum as f32) + c;
    let rough = -e * factor_ip + dis_f + delta;
    let err = factor_err * (dis_v_2 + norm * norm + 2.0 * dis_f).sqrt();
    (rough, err)
}

fn compress(vector: &[f32]) -> Vec<[f32; 16]> {
    let f = |[t_0, t_1, t_2, t_3]: [f32; 4]| {
        [
            0.0 - t_3 - t_2 - t_1 - t_0,
            0.0 - t_3 - t_2 - t_1 + t_0,
            0.0 - t_3 - t_2 + t_1 - t_0,
            0.0 - t_3 - t_2 + t_1 + t_0,
            0.0 - t_3 + t_2 - t_1 - t_0,
            0.0 - t_3 + t_2 - t_1 + t_0,
            0.0 - t_3 + t_2 + t_1 - t_0,
            0.0 - t_3 + t_2 + t_1 + t_0,
            0.0 + t_3 - t_2 - t_1 - t_0,
            0.0 + t_3 - t_2 - t_1 + t_0,
            0.0 + t_3 - t_2 + t_1 - t_0,
            0.0 + t_3 - t_2 + t_1 + t_0,
            0.0 + t_3 + t_2 - t_1 - t_0,
            0.0 + t_3 + t_2 - t_1 + t_0,
            0.0 + t_3 + t_2 + t_1 - t_0,
            0.0 + t_3 + t_2 + t_1 + t_0,
        ]
    };

    let (arrays, remainder) = vector.as_chunks::<4>();
    let mut result = arrays.iter().copied().map(f).collect::<Vec<_>>();
    if !remainder.is_empty() {
        let mut array = [0.0; 4];
        array[..remainder.len()].copy_from_slice(remainder);
        result.push(f(array));
    }
    result
}
