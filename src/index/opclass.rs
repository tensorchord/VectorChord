use crate::datatype::memory_halfvec::{HalfvecInput, HalfvecOutput};
use crate::datatype::memory_vector::{VectorInput, VectorOutput};
use algorithm::types::*;
use distance::Distance;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use pgrx::pg_sys::Datum;
use std::num::NonZero;
use vector::VectorBorrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_l2_ops() -> String {
    "vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_ip_ops() -> String {
    "vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_cosine_ops() -> String {
    "vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_l2_ops() -> String {
    "halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_ip_ops() -> String {
    "halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_cosine_ops() -> String {
    "halfvec_cosine_ops".to_string()
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PostgresDistanceKind {
    L2,
    Ip,
    Cosine,
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Opfamily {
    vector: VectorKind,
    postgres_distance: PostgresDistanceKind,
}

impl Opfamily {
    fn input(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use {BorrowedVector as B, OwnedVector as O, PostgresDistanceKind as D};
        match (vector, self.postgres_distance) {
            (B::Vecf32(x), D::L2) => O::Vecf32(x.own()),
            (B::Vecf32(x), D::Ip) => O::Vecf32(x.own()),
            (B::Vecf32(x), D::Cosine) => O::Vecf32(x.function_normalize()),
            (B::Vecf16(x), D::L2) => O::Vecf16(x.own()),
            (B::Vecf16(x), D::Ip) => O::Vecf16(x.own()),
            (B::Vecf16(x), D::Cosine) => O::Vecf16(x.function_normalize()),
        }
    }
    pub unsafe fn input_vector(self, datum: Datum, is_null: bool) -> Option<OwnedVector> {
        if is_null || datum.is_null() {
            return None;
        }
        let vector = match self.vector {
            VectorKind::Vecf32 => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            VectorKind::Vecf16 => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
        };
        Some(vector)
    }
    pub unsafe fn input_sphere(self, datum: Datum, is_null: bool) -> Option<Sphere<OwnedVector>> {
        if is_null || datum.is_null() {
            return None;
        }
        let attno_1 = NonZero::new(1_usize).unwrap();
        let attno_2 = NonZero::new(2_usize).unwrap();
        let tuple = unsafe { PgHeapTuple::from_composite_datum(datum) };
        let center = match self.vector {
            VectorKind::Vecf32 => {
                let vector = tuple.get_by_index::<VectorOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            VectorKind::Vecf16 => {
                let vector = tuple.get_by_index::<HalfvecOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
        };
        let radius = tuple.get_by_index::<f32>(attno_2).unwrap()?;
        Some(Sphere { center, radius })
    }
    pub fn output(self, x: Distance) -> f32 {
        match self.postgres_distance {
            PostgresDistanceKind::Cosine => x.to_f32() + 1.0f32,
            PostgresDistanceKind::L2 => x.to_f32().sqrt(),
            PostgresDistanceKind::Ip => x.to_f32(),
        }
    }
    pub const fn distance_kind(self) -> DistanceKind {
        match self.postgres_distance {
            PostgresDistanceKind::L2 => DistanceKind::L2,
            PostgresDistanceKind::Ip | PostgresDistanceKind::Cosine => DistanceKind::Dot,
        }
    }
    pub const fn vector_kind(self) -> VectorKind {
        self.vector
    }
}

pub unsafe fn opfamily(index_relation: pgrx::pg_sys::Relation) -> Opfamily {
    use pgrx::pg_sys::Oid;

    let proc = unsafe { pgrx::pg_sys::index_getprocid(index_relation, 1, 1) };

    if proc == Oid::INVALID {
        pgrx::error!("support function 1 is not found");
    }

    let mut flinfo = pgrx::pg_sys::FmgrInfo::default();

    unsafe {
        pgrx::pg_sys::fmgr_info(proc, &mut flinfo);
    }

    let fn_addr = flinfo.fn_addr.expect("null function pointer");

    let mut fcinfo = unsafe { std::mem::zeroed::<pgrx::pg_sys::FunctionCallInfoBaseData>() };
    fcinfo.flinfo = &mut flinfo;
    fcinfo.fncollation = pgrx::pg_sys::DEFAULT_COLLATION_OID;
    fcinfo.context = std::ptr::null_mut();
    fcinfo.resultinfo = std::ptr::null_mut();
    fcinfo.isnull = true;
    fcinfo.nargs = 0;

    let result_datum = unsafe { pgrx::pg_sys::ffi::pg_guard_ffi_boundary(|| fn_addr(&mut fcinfo)) };

    let result_option = unsafe { String::from_datum(result_datum, fcinfo.isnull) };

    let result_string = result_option.expect("null return value");

    let (vector, postgres_distance) = match result_string.as_str() {
        "vector_l2_ops" => (VectorKind::Vecf32, PostgresDistanceKind::L2),
        "vector_ip_ops" => (VectorKind::Vecf32, PostgresDistanceKind::Ip),
        "vector_cosine_ops" => (VectorKind::Vecf32, PostgresDistanceKind::Cosine),
        "halfvec_l2_ops" => (VectorKind::Vecf16, PostgresDistanceKind::L2),
        "halfvec_ip_ops" => (VectorKind::Vecf16, PostgresDistanceKind::Ip),
        "halfvec_cosine_ops" => (VectorKind::Vecf16, PostgresDistanceKind::Cosine),
        _ => pgrx::error!("unknown operator class"),
    };

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    Opfamily {
        vector,
        postgres_distance,
    }
}
