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

use simd::f16;
use vector::vect::VectBorrowed;

pub fn vector_out(vector: VectBorrowed<'_, f32>) -> String {
    let mut result = String::from("[");
    for x in vector.slice() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}

pub fn halfvec_out(vector: VectBorrowed<'_, f16>) -> String {
    let mut result = String::from("[");
    for x in vector.slice() {
        if !result.ends_with('[') {
            result.push(',');
        }
        result.push_str(&x.to_string());
    }
    result.push(']');
    result
}
