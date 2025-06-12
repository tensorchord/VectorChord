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

use binary::BinaryLut;
use block::BlockLut;
use simd::Floating;

#[derive(Debug, Clone, Copy)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub factor_cnt: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
}

impl CodeMetadata {
    pub fn into_tuple(self) -> (f32, f32, f32, f32) {
        (
            self.dis_u_2,
            self.factor_cnt,
            self.factor_ip,
            self.factor_err,
        )
    }
    pub fn into_array(self) -> [f32; 4] {
        [
            self.dis_u_2,
            self.factor_cnt,
            self.factor_ip,
            self.factor_err,
        ]
    }
    pub fn from_tuple((dis_u_2, factor_cnt, factor_ip, factor_err): (f32, f32, f32, f32)) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }
    }
    pub fn from_array([dis_u_2, factor_cnt, factor_ip, factor_err]: [f32; 4]) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
            factor_err,
        }
    }
}

pub type Code = (CodeMetadata, Vec<bool>);

pub fn code(vector: &[f32]) -> Code {
    let n = vector.len();
    let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
    let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
    (
        CodeMetadata {
            dis_u_2: sum_of_x_2,
            factor_cnt: {
                let cnt_pos = vector
                    .iter()
                    .map(|x| x.is_sign_positive() as i32)
                    .sum::<i32>();
                let cnt_neg = vector
                    .iter()
                    .map(|x| x.is_sign_negative() as i32)
                    .sum::<i32>();
                (cnt_pos - cnt_neg) as f32
            },
            factor_ip: sum_of_x_2 / sum_of_abs_x,
            factor_err: {
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (n as f32).sqrt();
                dis_u * (1.0 / (x_0 * x_0) - 1.0).sqrt() / (n as f32 - 1.0).sqrt()
            },
        },
        {
            let mut signs = Vec::new();
            for i in 0..n {
                signs.push(vector[i].is_sign_positive());
            }
            signs
        },
    )
}

pub fn preprocess(vector: &[f32]) -> (BlockLut, BinaryLut) {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    (
        block::preprocess_with_distance(vector, dis_v_2),
        binary::preprocess_with_distance(vector, dis_v_2),
    )
}

pub mod binary {
    pub fn pack_code(input: &[bool]) -> Vec<u64> {
        let f = |t: &[bool; 64]| {
            let mut result = 0_u64;
            for i in 0..64 {
                result |= (t[i] as u64) << i;
            }
            result
        };
        let (arrays, remainder) = input.as_chunks::<64>();
        let mut buffer = [false; 64];
        let tailing = if !remainder.is_empty() {
            buffer[..remainder.len()].copy_from_slice(remainder);
            Some(&buffer)
        } else {
            None
        };
        arrays.iter().chain(tailing).map(f).collect()
    }

    use super::CodeMetadata;
    use simd::Floating;

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

    pub fn accumulate(x: &[u64], y: &[impl AsRef<[u64]>; BITS]) -> u32 {
        let mut result = 0_u32;
        for i in 0..BITS {
            result += simd::bit::reduce_sum_of_and(x, y[i].as_ref()) << i;
        }
        result
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
}

pub mod block {
    use super::CodeMetadata;
    use simd::Floating;

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
        (dis_u_2, _, factor_ip, factor_err, t): BlockCode<'_>,
        lut: &BlockLut,
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
        (_, _, factor_ip, factor_err, t): BlockCode<'_>,
        lut: &BlockLut,
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
        let f = |&[t_0, t_1, t_2, t_3]: &[f32; 4]| {
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
        let mut buffer = [0.0f32; 4];
        let tailing = if !remainder.is_empty() {
            buffer[..remainder.len()].copy_from_slice(remainder);
            Some(&buffer)
        } else {
            None
        };
        arrays.iter().chain(tailing).map(f).collect()
    }
}
