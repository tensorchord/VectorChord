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

#[derive(Debug, Clone, Copy)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub factor_cnt: f32,
    pub factor_ip: f32,
}

impl CodeMetadata {
    pub fn into_tuple(self) -> (f32, f32, f32) {
        (self.dis_u_2, self.factor_cnt, self.factor_ip)
    }
    pub fn into_array(self) -> [f32; 3] {
        [self.dis_u_2, self.factor_cnt, self.factor_ip]
    }
    pub fn from_tuple((dis_u_2, factor_cnt, factor_ip): (f32, f32, f32)) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
        }
    }
    pub fn from_array([dis_u_2, factor_cnt, factor_ip]: [f32; 3]) -> Self {
        Self {
            dis_u_2,
            factor_cnt,
            factor_ip,
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
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
    ) -> (f32,) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = dis_u_2 + dis_v_2 - 2.0 * e * factor_ip;
        (rough,)
    }

    pub fn half_process_dot(
        sum: u32,
        CodeMetadata {
            dis_u_2: _,
            factor_cnt,
            factor_ip,
        }: CodeMetadata,
        BinaryLutMetadata {
            dis_v_2: _,
            b,
            k,
            qvector_sum,
        }: BinaryLutMetadata,
    ) -> (f32,) {
        let e = k * ((2.0 * sum as f32) - qvector_sum) + b * factor_cnt;
        let rough = -e * factor_ip;
        (rough,)
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
