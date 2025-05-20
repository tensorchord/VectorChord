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

const BITS: usize = 6;

#[derive(Debug, Clone, Copy)]
pub struct BinaryLutMetadata {
    dis_v_2: f32,
    b: f32,
    k: f32,
    qvector_sum: f32,
}

pub type BinaryLut = (BinaryLutMetadata, [Vec<u64>; BITS]);
pub type BinaryCode<'a> = ((f32, f32, f32, f32), &'a [u64]);

pub fn preprocess(vector: &[f32]) -> BinaryLut {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    preprocess_with_distance(vector, dis_v_2)
}

pub(crate) fn preprocess_with_distance(vector: &[f32], dis_v_2: f32) -> BinaryLut {
    let (k, b, qvector) = simd::quantize::quantize(vector, ((1 << BITS) - 1) as f32);
    let qvector_sum = if vector.len() <= (65535_usize / ((1 << BITS) - 1)) {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x_as_u32(&qvector) as f32
    };
    (
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        },
        binarize(&qvector),
    )
}

pub fn full_process_l2(
    lut: &BinaryLut,
    ((dis_u_2, factor_cnt, factor_ip, factor_err), t): BinaryCode<'_>,
) -> (f32, f32) {
    let &(
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        },
        ref s,
    ) = lut;
    let sum = asymmetric_binary_dot_product(t, s);
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
    let err = 2.0 * factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn full_process_dot(
    lut: &BinaryLut,
    ((_, factor_cnt, factor_ip, factor_err), t): BinaryCode<'_>,
) -> (f32, f32) {
    let &(
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        },
        ref s,
    ) = lut;
    let sum = asymmetric_binary_dot_product(t, s);
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = -e * factor_ip;
    let err = factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn half_process_l2(
    sum: u32,
    CodeMetadata {
        dis_u_2,
        factor_cnt,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BinaryLutMetadata {
        dis_v_2,
        b,
        k,
        qvector_sum,
    }: BinaryLutMetadata,
) -> (f32, f32) {
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
    let err = 2.0 * factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn half_process_l2_residual(
    sum: u32,
    CodeMetadata {
        dis_u_2,
        factor_cnt,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BinaryLutMetadata {
        dis_v_2: _,
        b,
        k,
        qvector_sum,
    }: BinaryLutMetadata,
    dis_f: f32,
    delta: f32,
) -> (f32, f32) {
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = dis_u_2 + dis_f - 2.0 * e * factor_ip + delta;
    let err = 2.0 * factor_err * dis_f.sqrt();
    (rough, err)
}

pub fn half_process_dot(
    sum: u32,
    CodeMetadata {
        dis_u_2: _,
        factor_cnt,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BinaryLutMetadata {
        dis_v_2,
        b,
        k,
        qvector_sum,
    }: BinaryLutMetadata,
) -> (f32, f32) {
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = -e * factor_ip;
    let err = factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn half_process_dot_residual(
    sum: u32,
    CodeMetadata {
        dis_u_2: _,
        factor_cnt,
        factor_ip,
        factor_err,
    }: CodeMetadata,
    BinaryLutMetadata {
        dis_v_2,
        b,
        k,
        qvector_sum,
    }: BinaryLutMetadata,
    dis_f: f32,
    delta: f32,
    norm: f32,
) -> (f32, f32) {
    let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
    let rough = -e * factor_ip + dis_f + delta;
    let err = factor_err * (dis_v_2 + norm * norm + 2.0 * dis_f).sqrt();
    (rough, err)
}

pub(crate) fn binarize(vector: &[u8]) -> [Vec<u64>; BITS] {
    let n = vector.len();
    let mut t: [_; BITS] = std::array::from_fn(|_| vec![0_u64; n.div_ceil(64)]);
    for i in 0..BITS {
        for j in 0..n {
            let bit = (vector[j] >> i) & 1;
            t[i][j / 64] |= (bit as u64) << (j % 64);
        }
    }
    t
}

pub fn asymmetric_binary_dot_product(x: &[u64], y: &[Vec<u64>; BITS]) -> u32 {
    let mut result = 0_u32;
    for i in 0..BITS {
        result += simd::bit::reduce_sum_of_and(x, y[i].as_slice()) << i;
    }
    result
}
