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

pub use crate::extended::{Code, CodeMetadata};

#[deprecated]
pub fn code(vector: &[f32]) -> Code {
    crate::extended::code::<4>(vector)
}

pub fn ugly_code(vector: &[f32]) -> Code {
    crate::extended::ugly_code::<4>(vector)
}

pub fn pack_code(input: &[u8]) -> Vec<u8> {
    let f = |t: &[u8; 2]| t[0] | t[1] << 4;
    let (arrays, remainder) = input.as_chunks::<2>();
    let mut buffer = [0u8; 2];
    let tailing = if !remainder.is_empty() {
        buffer[..remainder.len()].copy_from_slice(remainder);
        Some(&buffer)
    } else {
        None
    };
    arrays.iter().chain(tailing).map(f).collect()
}

pub mod binary {
    use crate::extended::CodeMetadata;

    const BITS: usize = 4;

    pub type BinaryLutMetadata = CodeMetadata;
    pub type BinaryLut = (BinaryLutMetadata, Vec<u8>);
    pub type BinaryCode<'a> = ((f32, f32, f32, f32), &'a [u8]);

    #[deprecated]
    pub fn preprocess(vector: &[f32]) -> BinaryLut {
        let (metadata, elements) = crate::extended::code::<BITS>(vector);
        (metadata, super::pack_code(&elements))
    }

    pub fn ugly_preprocess(vector: &[f32]) -> BinaryLut {
        let (metadata, elements) = crate::extended::ugly_code::<BITS>(vector);
        (metadata, super::pack_code(&elements))
    }

    pub fn accumulate(x: &[u8], y: &[u8]) -> u32 {
        simd::halfbyte::reduce_sum_of_x_as_u32_y_as_u32(x, y)
    }

    pub fn half_process_dot(
        n: u32,
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_dot::<4, BITS>(n, value, code, lut);
        (rough,)
    }

    pub fn half_process_l2(
        n: u32,
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_l2::<4, BITS>(n, value, code, lut);
        (rough,)
    }

    pub fn half_process_cos(
        n: u32,
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
    ) -> (f32,) {
        let rough = crate::extended::half_process_cos::<4, BITS>(n, value, code, lut);
        (rough,)
    }
}
