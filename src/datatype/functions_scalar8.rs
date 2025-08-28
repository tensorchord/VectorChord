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

use crate::datatype::memory_halfvec::HalfvecInput;
use crate::datatype::memory_scalar8::Scalar8Output;
use crate::datatype::memory_vector::VectorInput;
use simd::{Floating, f16};
use vector::scalar8::Scalar8Borrowed;

#[pgrx::pg_extern(sql = "")]
fn _vchord_vector_quantize_to_scalar8(vector: VectorInput) -> Scalar8Output {
    let vector = vector.as_borrowed();
    let sum_of_x2 = f32::reduce_sum_of_x2(vector.slice());
    let (k, b, code) =
        simd::quantize::quantize(f32::vector_to_f32_borrowed(vector.slice()).as_ref(), 255.0);
    let sum_of_code = simd::u8::reduce_sum_of_x_as_u32(&code) as f32;
    Scalar8Output::new(Scalar8Borrowed::new(sum_of_x2, k, b, sum_of_code, &code))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_halfvec_quantize_to_scalar8(vector: HalfvecInput) -> Scalar8Output {
    let vector = vector.as_borrowed();
    let sum_of_x2 = f16::reduce_sum_of_x2(vector.slice());
    let (k, b, code) =
        simd::quantize::quantize(f16::vector_to_f32_borrowed(vector.slice()).as_ref(), 255.0);
    let sum_of_code = simd::u8::reduce_sum_of_x_as_u32(&code) as f32;
    Scalar8Output::new(Scalar8Borrowed::new(sum_of_x2, k, b, sum_of_code, &code))
}
