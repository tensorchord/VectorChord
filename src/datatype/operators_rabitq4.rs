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

use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use pgrx::datum::Array;
use std::num::NonZero;
use vector::VectorBorrowed;
use vector::rabitq4::Rabitq4Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_operator_l2(lhs: Rabitq4Input<'_>, rhs: Rabitq4Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq4Borrowed::operator_l2s(lhs, rhs).to_f32().sqrt()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_operator_ip(lhs: Rabitq4Input<'_>, rhs: Rabitq4Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq4Borrowed::operator_dot(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_operator_cosine(lhs: Rabitq4Input<'_>, rhs: Rabitq4Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dim() != rhs.dim() {
        pgrx::error!("dimension is not matched");
    }
    Rabitq4Borrowed::operator_cos(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_sphere_l2_in(
    lhs: Rabitq4Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq4"),
) -> bool {
    let center: Rabitq4Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq4Borrowed::operator_l2s(lhs, center).to_f32().sqrt();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_sphere_ip_in(
    lhs: Rabitq4Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq4"),
) -> bool {
    let center: Rabitq4Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq4Borrowed::operator_dot(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_rabitq4_sphere_cosine_in(
    lhs: Rabitq4Input<'_>,
    rhs: pgrx::composite_type!("sphere_rabitq4"),
) -> bool {
    let center: Rabitq4Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let lhs = lhs.as_borrowed();
    let center = center.as_borrowed();
    if lhs.dim() != center.dim() {
        pgrx::error!("dimension is not matched");
    }
    let d = Rabitq4Borrowed::operator_cos(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(sql = "")]
fn _vchord_rabitq4_operator_maxsim(
    lhs: Array<'_, Rabitq4Input<'_>>,
    rhs: Array<'_, Rabitq4Input<'_>>,
) -> f32 {
    let mut maxsim = 0.0f32;
    for rhs in rhs.iter().flatten() {
        let mut d = f32::INFINITY;
        for lhs in lhs.iter().flatten() {
            let lhs = lhs.as_borrowed();
            let rhs = rhs.as_borrowed();
            d = d.min(Rabitq4Borrowed::operator_dot(lhs, rhs).to_f32());
        }
        maxsim += d;
    }
    maxsim
}
