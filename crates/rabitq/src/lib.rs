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

pub mod binary;
pub mod block;
pub mod packing;

use binary::BinaryLut;
use block::BlockLut;
use simd::Floating;

#[derive(Debug, Clone)]
pub struct Code {
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
}

pub fn pack_to_u4(signs: &[bool]) -> Vec<u8> {
    fn f(x: [bool; 4]) -> u8 {
        x[0] as u8 | (x[1] as u8) << 1 | (x[2] as u8) << 2 | (x[3] as u8) << 3
    }
    let mut result = Vec::with_capacity(signs.len().div_ceil(4));
    for i in 0..signs.len().div_ceil(4) {
        let x = std::array::from_fn(|j| signs.get(i * 4 + j).copied().unwrap_or_default());
        result.push(f(x));
    }
    result
}

pub fn pack_to_u64(signs: &[bool]) -> Vec<u64> {
    fn f(x: [bool; 64]) -> u64 {
        let mut result = 0_u64;
        for i in 0..64 {
            result |= (x[i] as u64) << i;
        }
        result
    }
    let mut result = Vec::with_capacity(signs.len().div_ceil(64));
    for i in 0..signs.len().div_ceil(64) {
        let x = std::array::from_fn(|j| signs.get(i * 64 + j).copied().unwrap_or_default());
        result.push(f(x));
    }
    result
}

pub fn code(dims: u32, vector: &[f32]) -> Code {
    let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
    let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
    let dis_u = sum_of_x_2.sqrt();
    let x0 = sum_of_abs_x / (sum_of_x_2 * (dims as f32)).sqrt();
    let x_x0 = dis_u / x0;
    let fac_norm = (dims as f32).sqrt();
    let max_x1 = 1.0f32 / (dims as f32 - 1.0).sqrt();
    let factor_err = 2.0f32 * max_x1 * (x_x0 * x_x0 - dis_u * dis_u).sqrt();
    let factor_ip = -2.0f32 / fac_norm * x_x0;
    let cnt_pos = vector
        .iter()
        .map(|x| x.is_sign_positive() as i32)
        .sum::<i32>();
    let cnt_neg = vector
        .iter()
        .map(|x| x.is_sign_negative() as i32)
        .sum::<i32>();
    let factor_ppc = factor_ip * (cnt_pos - cnt_neg) as f32;
    let mut signs = Vec::new();
    for i in 0..dims {
        signs.push(vector[i as usize].is_sign_positive());
    }
    Code {
        dis_u_2: sum_of_x_2,
        factor_ppc,
        factor_ip,
        factor_err,
        signs,
    }
}

pub fn preprocess(vector: &[f32]) -> (BlockLut, BinaryLut) {
    use simd::Floating;
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = simd::quantize::quantize(vector, 15.0);
    let qvector_sum = if vector.len() <= 4369 {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x_as_u32(&qvector) as f32
    };
    let binary = binary::binarize(&qvector);
    let block = block::compress(qvector);
    (
        ((dis_v_2, b, k, qvector_sum), block),
        ((dis_v_2, b, k, qvector_sum), binary),
    )
}
