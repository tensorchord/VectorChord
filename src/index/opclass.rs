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

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_l2_ops() -> String {
    "vector_maxsim_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_ip_ops() -> String {
    "vector_maxsim_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_cosine_ops() -> String {
    "vector_maxsim_cosine_ops".to_string()
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Opfamily {
    VectorL2,
    VectorIp,
    VectorCosine,
    HalfvecL2,
    HalfvecIp,
    HalfvecCosine,
    VectorMaxsimL2,
    VectorMaxsimIp,
    VectorMaxsimCosine,
}

impl Opfamily {
    fn input(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use {BorrowedVector as B, OwnedVector as O};
        match (vector, self) {
            (B::Vecf32(x), Self::VectorL2 | Self::VectorMaxsimL2) => O::Vecf32(x.own()),
            (B::Vecf32(x), Self::VectorIp | Self::VectorMaxsimIp) => O::Vecf32(x.own()),
            (B::Vecf32(x), Self::VectorCosine | Self::VectorMaxsimCosine) => {
                O::Vecf32(x.function_normalize())
            }
            (B::Vecf32(_), _) => unreachable!(),
            (B::Vecf16(x), Self::HalfvecL2) => O::Vecf16(x.own()),
            (B::Vecf16(x), Self::HalfvecIp) => O::Vecf16(x.own()),
            (B::Vecf16(x), Self::HalfvecCosine) => O::Vecf16(x.function_normalize()),
            (B::Vecf16(_), _) => unreachable!(),
        }
    }
    pub unsafe fn input_data(self, datum: Datum, is_null: bool) -> Option<Vec<OwnedVector>> {
        if is_null || datum.is_null() {
            return None;
        }
        let vector = match self {
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                vec![self.input(BorrowedVector::Vecf32(vector.as_borrowed()))]
            }
            Self::VectorMaxsimL2 | Self::VectorMaxsimIp | Self::VectorMaxsimCosine => {
                let vector =
                    unsafe { pgrx::Array::<VectorInput>::from_datum(datum, false).unwrap() };
                vector
                    .iter()
                    .flatten()
                    .map(|x| self.input(BorrowedVector::Vecf32(x.as_borrowed())))
                    .collect()
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                vec![self.input(BorrowedVector::Vecf16(vector.as_borrowed()))]
            }
        };
        Some(vector)
    }
    pub unsafe fn input_query(self, datum: Datum, is_null: bool) -> Option<OwnedVector> {
        if is_null || datum.is_null() {
            return None;
        }
        let vector = match self {
            Self::VectorL2
            | Self::VectorIp
            | Self::VectorCosine
            | Self::VectorMaxsimL2
            | Self::VectorMaxsimIp
            | Self::VectorMaxsimCosine => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine => {
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
        let center = match self {
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine => {
                let vector = tuple.get_by_index::<VectorOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine => {
                let vector = tuple.get_by_index::<HalfvecOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
            Self::VectorMaxsimL2 | Self::VectorMaxsimIp | Self::VectorMaxsimCosine => {
                unreachable!()
            }
        };
        let radius = tuple.get_by_index::<f32>(attno_2).unwrap()?;
        Some(Sphere { center, radius })
    }
    pub fn output(self, x: Distance) -> f32 {
        match self {
            Self::VectorCosine | Self::HalfvecCosine | Self::VectorMaxsimCosine => {
                x.to_f32() + 1.0f32
            }
            Self::VectorL2 | Self::HalfvecL2 | Self::VectorMaxsimL2 => x.to_f32().sqrt(),
            Self::VectorIp | Self::HalfvecIp | Self::VectorMaxsimIp => x.to_f32(),
        }
    }
    pub const fn distance_kind(self) -> DistanceKind {
        match self {
            Self::VectorL2 | Self::HalfvecL2 | Self::VectorMaxsimL2 => DistanceKind::L2,
            Self::VectorIp
            | Self::HalfvecIp
            | Self::VectorCosine
            | Self::HalfvecCosine
            | Self::VectorMaxsimIp
            | Self::VectorMaxsimCosine => DistanceKind::Dot,
        }
    }
    pub const fn vector_kind(self) -> VectorKind {
        match self {
            Self::VectorL2
            | Self::VectorIp
            | Self::VectorCosine
            | Self::VectorMaxsimL2
            | Self::VectorMaxsimIp
            | Self::VectorMaxsimCosine => VectorKind::Vecf32,
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine => VectorKind::Vecf16,
        }
    }
    pub const fn needs_distinct(self) -> bool {
        match self {
            Self::VectorL2
            | Self::VectorIp
            | Self::VectorCosine
            | Self::HalfvecL2
            | Self::HalfvecIp
            | Self::HalfvecCosine => false,
            Self::VectorMaxsimL2 | Self::VectorMaxsimIp | Self::VectorMaxsimCosine => true,
        }
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

    let result = match result_string.as_str() {
        "vector_l2_ops" => Opfamily::VectorL2,
        "vector_ip_ops" => Opfamily::VectorIp,
        "vector_cosine_ops" => Opfamily::VectorCosine,
        "halfvec_l2_ops" => Opfamily::HalfvecL2,
        "halfvec_ip_ops" => Opfamily::HalfvecIp,
        "halfvec_cosine_ops" => Opfamily::HalfvecCosine,
        "vector_maxsim_l2_ops" => Opfamily::VectorMaxsimL2,
        "vector_maxsim_ip_ops" => Opfamily::VectorMaxsimIp,
        "vector_maxsim_cosine_ops" => Opfamily::VectorMaxsimCosine,
        _ => pgrx::error!("unknown operator class"),
    };

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    result
}
