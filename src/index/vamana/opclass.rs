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

use crate::datatype::memory_halfvec::{HalfvecInput, HalfvecOutput};
use crate::datatype::memory_vector::{VectorInput, VectorOutput};
use distance::Distance;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use pgrx::pg_sys::Datum;
use std::num::NonZero;
use vamana::types::*;
use vector::VectorBorrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_vector_l2_ops() -> String {
    "vamana_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_vector_cosine_ops() -> String {
    "vamana_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_vector_ip_ops() -> String {
    "vamana_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_halfvec_l2_ops() -> String {
    "vamana_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_halfvec_cosine_ops() -> String {
    "vamana_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vamana_support_halfvec_ip_ops() -> String {
    "vamana_halfvec_ip_ops".to_string()
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Opfamily {
    VectorL2,
    VectorCosine,
    VectorIp,
    HalfvecL2,
    HalfvecCosine,
    HalfvecIp,
}

impl Opfamily {
    fn input(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use {BorrowedVector as B, OwnedVector as O};
        match (vector, self) {
            (B::Vecf32(x), Self::VectorL2) => O::Vecf32(x.own()),
            (B::Vecf32(x), Self::VectorCosine) => O::Vecf32(x.function_normalize()),
            (B::Vecf32(_), _) => unreachable!(),
            (B::Vecf16(x), Self::HalfvecL2) => O::Vecf16(x.own()),
            (B::Vecf16(x), Self::HalfvecCosine) => O::Vecf16(x.function_normalize()),
            (B::Vecf16(_), _) => unreachable!(),
        }
    }
    pub unsafe fn store(self, datum: Datum) -> Option<Vec<(OwnedVector, u16)>> {
        if datum.is_null() {
            return None;
        }
        let store = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Vecf32(vector.as_borrowed())), 0)]
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Vecf16(vector.as_borrowed())), 0)]
            }
        };
        Some(store)
    }
    pub unsafe fn input_sphere(self, datum: Datum) -> Option<Sphere<OwnedVector>> {
        if datum.is_null() {
            return None;
        }
        let attno_1 = NonZero::new(1_usize).unwrap();
        let attno_2 = NonZero::new(2_usize).unwrap();
        let tuple = unsafe { PgHeapTuple::from_composite_datum(datum) };
        let center = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = tuple.get_by_index::<VectorOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = tuple.get_by_index::<HalfvecOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
        };
        let radius = tuple.get_by_index::<f32>(attno_2).unwrap()?;
        Some(Sphere { center, radius })
    }
    pub unsafe fn input_vector(self, datum: Datum) -> Option<OwnedVector> {
        if datum.is_null() {
            return None;
        }
        let vector = match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
        };
        Some(vector)
    }
    pub fn output(self, x: Distance) -> f32 {
        match self {
            Self::VectorCosine | Self::HalfvecCosine => x.to_f32() * 0.5,
            Self::VectorL2 | Self::HalfvecL2 => x.to_f32().sqrt(),
            Self::VectorIp | Self::HalfvecIp => x.to_f32(),
        }
    }
    pub const fn distance_kind(self) -> DistanceKind {
        match self {
            Self::VectorL2 | Self::HalfvecL2 => DistanceKind::L2S,
            Self::VectorCosine | Self::HalfvecCosine => DistanceKind::L2S,
            Self::VectorIp | Self::HalfvecIp => DistanceKind::Dot,
        }
    }
    pub const fn vector_kind(self) -> VectorKind {
        match self {
            Self::VectorL2 | Self::VectorCosine | Self::VectorIp => VectorKind::Vecf32,
            Self::HalfvecL2 | Self::HalfvecCosine | Self::HalfvecIp => VectorKind::Vecf16,
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
        "vamana_vector_l2_ops" => Opfamily::VectorL2,
        "vamana_vector_cosine_ops" => Opfamily::VectorCosine,
        "vamana_halfvec_l2_ops" => Opfamily::HalfvecL2,
        "vamana_halfvec_cosine_ops" => Opfamily::HalfvecCosine,
        "vamana_vector_ip_ops" => Opfamily::VectorIp,
        "vamana_halfvec_ip_ops" => Opfamily::HalfvecIp,
        _ => pgrx::error!("unknown operator class"),
    };

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    result
}
