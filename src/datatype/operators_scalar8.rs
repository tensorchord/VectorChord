use crate::datatype::memory_scalar8::{Scalar8Input, Scalar8Output};
use std::num::NonZero;
use vector::VectorBorrowed;
use vector::scalar8::Scalar8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_operator_ip(lhs: Scalar8Input<'_>, rhs: Scalar8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dims() != rhs.dims() {
        pgrx::error!("dimension is not matched");
    }
    Scalar8Borrowed::operator_dot(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_operator_l2(lhs: Scalar8Input<'_>, rhs: Scalar8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dims() != rhs.dims() {
        pgrx::error!("dimension is not matched");
    }
    Scalar8Borrowed::operator_l2(lhs, rhs).to_f32().sqrt()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_operator_cosine(lhs: Scalar8Input<'_>, rhs: Scalar8Input<'_>) -> f32 {
    let lhs = lhs.as_borrowed();
    let rhs = rhs.as_borrowed();
    if lhs.dims() != rhs.dims() {
        pgrx::error!("dimension is not matched");
    }
    Scalar8Borrowed::operator_cos(lhs, rhs).to_f32()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_sphere_ip_in(
    lhs: Scalar8Input<'_>,
    rhs: pgrx::composite_type!("sphere_scalar8"),
) -> bool {
    let center: Scalar8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    let d = Scalar8Borrowed::operator_dot(lhs, center).to_f32();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_sphere_l2_in(
    lhs: Scalar8Input<'_>,
    rhs: pgrx::composite_type!("sphere_scalar8"),
) -> bool {
    let center: Scalar8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    let d = Scalar8Borrowed::operator_l2(lhs, center).to_f32().sqrt();
    d < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_sphere_cosine_in(
    lhs: Scalar8Input<'_>,
    rhs: pgrx::composite_type!("sphere_scalar8"),
) -> bool {
    let center: Scalar8Output = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
    let d = Scalar8Borrowed::operator_cos(lhs, center).to_f32();
    d < radius
}
