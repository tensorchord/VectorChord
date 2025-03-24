use crate::datatype::memory_halfvec::{HalfvecInput, HalfvecOutput};
use pgrx::Array;
use std::num::NonZero;
use vector::VectorBorrowed;
use vector::vect::VectBorrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_halfvec_sphere_l2_in(
    lhs: HalfvecInput<'_>,
    rhs: pgrx::composite_type!("sphere_halfvec"),
) -> bool {
    let center: HalfvecOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let d = VectBorrowed::operator_l2(lhs, center).to_f32().sqrt();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_halfvec_sphere_ip_in(
    lhs: HalfvecInput<'_>,
    rhs: pgrx::composite_type!("sphere_halfvec"),
) -> bool {
    let center: HalfvecOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let d = VectBorrowed::operator_dot(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_halfvec_sphere_cosine_in(
    lhs: HalfvecInput<'_>,
    rhs: pgrx::composite_type!("sphere_halfvec"),
) -> bool {
    let center: HalfvecOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let d = VectBorrowed::operator_cos(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_halfvec_operator_maxsim(
    lhs: Array<'_, HalfvecInput<'_>>,
    rhs: Array<'_, HalfvecInput<'_>>,
) -> f32 {
    let mut maxsim = 0.0f32;
    for rhs in rhs.iter().flatten() {
        let mut d = f32::INFINITY;
        for lhs in lhs.iter().flatten() {
            let lhs = lhs.as_borrowed();
            let rhs = rhs.as_borrowed();
            d = d.min(VectBorrowed::operator_dot(lhs, rhs).to_f32());
        }
        maxsim += d;
    }
    maxsim
}
