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
pub mod rotate;

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

pub type Code = (CodeMetadata, Vec<bool>);

pub fn code(dims: u32, vector: &[f32]) -> Code {
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
                let x_0 = sum_of_abs_x / dis_u / (dims as f32).sqrt();
                dis_u * (1.0 / (x_0 * x_0) - 1.0).sqrt() / (dims as f32 - 1.0).sqrt()
            },
        },
        {
            let mut signs = Vec::new();
            for i in 0..dims {
                signs.push(vector[i as usize].is_sign_positive());
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
