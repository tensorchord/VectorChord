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
use crate::datatype::memory_rabitq4::Rabitq4Output;
use crate::datatype::memory_vector::VectorInput;
use simd::{Floating, f16};
use vector::VectorBorrowed;
use vector::rabitq4::Rabitq4Borrowed;

#[pgrx::pg_extern(sql = "")]
fn _vchord_vector_quantize_to_rabitq4(vector: VectorInput) -> Rabitq4Output {
    let vector = vector.as_borrowed();
    let dim = vector.dim();
    let mut vector = vector.slice().to_vec();
    rabitq::rotate::rotate_inplace(&mut vector);
    let (metadata, elements) = rabitq::halfbyte::ugly_code(&vector);
    let elements = rabitq::halfbyte::pack_code(&elements);
    Rabitq4Output::new(Rabitq4Borrowed::new(
        dim,
        metadata.dis_u_2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        f32::reduce_sum_of_abs_x(&vector),
        &elements,
    ))
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_halfvec_quantize_to_rabitq4(vector: HalfvecInput) -> Rabitq4Output {
    let vector = vector.as_borrowed();
    let dim = vector.dim();
    let mut vector = f16::vector_to_f32(vector.slice());
    rabitq::rotate::rotate_inplace(&mut vector);
    let (metadata, elements) = rabitq::halfbyte::ugly_code(&vector);
    let elements = rabitq::halfbyte::pack_code(&elements);
    Rabitq4Output::new(Rabitq4Borrowed::new(
        dim,
        metadata.dis_u_2,
        metadata.norm_of_lattice,
        metadata.sum_of_code,
        f32::reduce_sum_of_abs_x(&vector),
        &elements,
    ))
}
