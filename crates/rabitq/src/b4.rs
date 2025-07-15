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

pub fn code(vector: &[f32]) -> Code {
    crate::extended::code::<4>(vector)
}

pub mod binary {
    pub fn pack_code(input: &[u8]) -> Vec<u64> {
        crate::extended::pack_code::<4>(input)
            .into_iter()
            .flatten()
            .collect()
    }

    use crate::extended::CodeMetadata;

    const BITS: usize = 4;

    pub type BinaryLutMetadata = CodeMetadata;
    pub type BinaryLut = (BinaryLutMetadata, [Vec<u64>; BITS]);
    pub type BinaryCode<'a> = ((f32, f32, f32, f32), &'a [u8]);

    pub fn preprocess(vector: &[f32]) -> BinaryLut {
        let (metadata, elements) = crate::extended::code::<BITS>(vector);
        (metadata, crate::extended::pack_code::<4>(&elements))
    }

    pub fn accumulate(lhs: &[u64], rhs: &[Vec<u64>; 4]) -> u32 {
        crate::extended::accumulate::<4, 4>(lhs, rhs)
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
}
