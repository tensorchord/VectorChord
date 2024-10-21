use crate::datatype::memory_pgvector_vector::*;
use base::scalar::ScalarLike;
use std::num::NonZero;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_pgvector_vector_sphere_l2_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    f32::reduce_sum_of_d2(lhs.slice(), center.slice()).to_f32() < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_pgvector_vector_sphere_dot_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    -f32::reduce_sum_of_xy(lhs.slice(), center.slice()) < radius
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_pgvector_vector_sphere_cos_in(
    lhs: PgvectorVectorInput<'_>,
    rhs: pgrx::composite_type!("sphere_vector"),
) -> bool {
    let center: PgvectorVectorOutput = match rhs.get_by_index(NonZero::new(1).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty center at sphere"),
        Err(_) => unreachable!(),
    };
    if lhs.dims() != center.dims() {
        pgrx::error!("dimension is not matched");
    }
    let radius: f32 = match rhs.get_by_index(NonZero::new(2).unwrap()) {
        Ok(Some(s)) => s,
        Ok(None) => pgrx::error!("Bad input: empty radius at sphere"),
        Err(_) => unreachable!(),
    };
    let xy = f32::reduce_sum_of_xy(lhs.slice(), center.slice());
    let x2 = f32::reduce_sum_of_x2(lhs.slice());
    let y2 = f32::reduce_sum_of_x2(center.slice());
    1.0 - xy / (x2 * y2).sqrt() < radius
}
