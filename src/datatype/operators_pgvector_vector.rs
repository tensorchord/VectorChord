use crate::datatype::memory_pgvector_vector::{PgvectorVectorInput, PgvectorVectorOutput};
use std::num::NonZero;
use vector::VectorBorrowed;
use vector::vect::VectBorrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_vector_sphere_l2_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
fn _vchord_vector_sphere_ip_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
fn _vchord_vector_sphere_cosine_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
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
