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
use crate::datatype::memory_rabitq4::{Rabitq4Input, Rabitq4Output};
use crate::datatype::memory_rabitq8::{Rabitq8Input, Rabitq8Output};
use crate::datatype::memory_vector::{VectorInput, VectorOutput};
use crate::index::opclass::Sphere;
use distance::Distance;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use pgrx::pg_sys::Datum;
use std::num::NonZero;
use vchordrq::types::*;
use vector::VectorBorrowed;

#[derive(Debug, Clone, Copy)]
pub enum Opfamily {
    VectorL2,
    VectorIp,
    VectorCosine,
    HalfvecL2,
    HalfvecIp,
    HalfvecCosine,
    Rabitq8L2,
    Rabitq8Ip,
    Rabitq8Cosine,
    Rabitq4L2,
    Rabitq4Ip,
    Rabitq4Cosine,
    VectorMaxsim,
    HalfvecMaxsim,
    Rabitq8Maxsim,
    Rabitq4Maxsim,
}

impl Opfamily {
    fn input(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use {BorrowedVector as B, OwnedVector as O};
        match (vector, self) {
            (B::Vecf32(x), Self::VectorL2) => O::Vecf32(x.own()),
            (B::Vecf32(x), Self::VectorIp | Self::VectorMaxsim) => O::Vecf32(x.own()),
            (B::Vecf32(x), Self::VectorCosine) => O::Vecf32(x.function_normalize()),
            (B::Vecf32(_), _) => unreachable!(),
            (B::Vecf16(x), Self::HalfvecL2) => O::Vecf16(x.own()),
            (B::Vecf16(x), Self::HalfvecIp | Self::HalfvecMaxsim) => O::Vecf16(x.own()),
            (B::Vecf16(x), Self::HalfvecCosine) => O::Vecf16(x.function_normalize()),
            (B::Vecf16(_), _) => unreachable!(),
            (B::Rabitq8(x), Self::Rabitq8L2) => O::Rabitq8(x.own()),
            (B::Rabitq8(x), Self::Rabitq8Ip | Self::Rabitq8Maxsim) => O::Rabitq8(x.own()),
            (B::Rabitq8(x), Self::Rabitq8Cosine) => O::Rabitq8(x.function_normalize()),
            (B::Rabitq8(_), _) => unreachable!(),
            (B::Rabitq4(x), Self::Rabitq4L2) => O::Rabitq4(x.own()),
            (B::Rabitq4(x), Self::Rabitq4Ip | Self::Rabitq4Maxsim) => O::Rabitq4(x.own()),
            (B::Rabitq4(x), Self::Rabitq4Cosine) => O::Rabitq4(x.function_normalize()),
            (B::Rabitq4(_), _) => unreachable!(),
        }
    }
    pub unsafe fn store(self, datum: Datum) -> Option<Vec<(OwnedVector, u16)>> {
        if datum.is_null() {
            return None;
        }
        let store = match self {
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Vecf32(vector.as_borrowed())), 0)]
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Vecf16(vector.as_borrowed())), 0)]
            }
            Self::Rabitq8L2 | Self::Rabitq8Ip | Self::Rabitq8Cosine => {
                let vector = unsafe { Rabitq8Input::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Rabitq8(vector.as_borrowed())), 0)]
            }
            Self::Rabitq4L2 | Self::Rabitq4Ip | Self::Rabitq4Cosine => {
                let vector = unsafe { Rabitq4Input::from_datum(datum, false).unwrap() };
                vec![(self.input(BorrowedVector::Rabitq4(vector.as_borrowed())), 0)]
            }
            Self::VectorMaxsim => {
                let vectors =
                    unsafe { pgrx::datum::Array::<VectorInput>::from_datum(datum, false).unwrap() };
                let mut result = Vec::with_capacity(vectors.len());
                for (i, vector) in vectors.iter_deny_null().enumerate() {
                    result.push((
                        self.input(BorrowedVector::Vecf32(vector.as_borrowed())),
                        i as u16,
                    ));
                }
                result
            }
            Self::HalfvecMaxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<HalfvecInput>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for (i, vector) in vectors.iter_deny_null().enumerate() {
                    result.push((
                        self.input(BorrowedVector::Vecf16(vector.as_borrowed())),
                        i as u16,
                    ));
                }
                result
            }
            Self::Rabitq8Maxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<Rabitq8Input>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for (i, vector) in vectors.iter_deny_null().enumerate() {
                    result.push((
                        self.input(BorrowedVector::Rabitq8(vector.as_borrowed())),
                        i as u16,
                    ));
                }
                result
            }
            Self::Rabitq4Maxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<Rabitq4Input>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for (i, vector) in vectors.iter_deny_null().enumerate() {
                    result.push((
                        self.input(BorrowedVector::Rabitq4(vector.as_borrowed())),
                        i as u16,
                    ));
                }
                result
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
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine | Self::VectorMaxsim => {
                let vector = tuple.get_by_index::<VectorOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine | Self::HalfvecMaxsim => {
                let vector = tuple.get_by_index::<HalfvecOutput>(attno_1).unwrap()?;
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
            Self::Rabitq8L2 | Self::Rabitq8Ip | Self::Rabitq8Cosine | Self::Rabitq8Maxsim => {
                let vector = tuple.get_by_index::<Rabitq8Output>(attno_1).unwrap()?;
                self.input(BorrowedVector::Rabitq8(vector.as_borrowed()))
            }
            Self::Rabitq4L2 | Self::Rabitq4Ip | Self::Rabitq4Cosine | Self::Rabitq4Maxsim => {
                let vector = tuple.get_by_index::<Rabitq4Output>(attno_1).unwrap()?;
                self.input(BorrowedVector::Rabitq4(vector.as_borrowed()))
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
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine | Self::VectorMaxsim => {
                let vector = unsafe { VectorInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine | Self::HalfvecMaxsim => {
                let vector = unsafe { HalfvecInput::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Vecf16(vector.as_borrowed()))
            }
            Self::Rabitq8L2 | Self::Rabitq8Ip | Self::Rabitq8Cosine | Self::Rabitq8Maxsim => {
                let vector = unsafe { Rabitq8Input::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Rabitq8(vector.as_borrowed()))
            }
            Self::Rabitq4L2 | Self::Rabitq4Ip | Self::Rabitq4Cosine | Self::Rabitq4Maxsim => {
                let vector = unsafe { Rabitq4Input::from_datum(datum, false).unwrap() };
                self.input(BorrowedVector::Rabitq4(vector.as_borrowed()))
            }
        };
        Some(vector)
    }
    pub unsafe fn input_vectors(self, datum: Datum) -> Option<Vec<OwnedVector>> {
        if datum.is_null() {
            return None;
        }
        let vectors = match self {
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine | Self::VectorMaxsim => {
                let vectors =
                    unsafe { pgrx::datum::Array::<VectorInput>::from_datum(datum, false).unwrap() };
                let mut result = Vec::with_capacity(vectors.len());
                for vector in vectors.iter_deny_null() {
                    result.push(self.input(BorrowedVector::Vecf32(vector.as_borrowed())));
                }
                result
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine | Self::HalfvecMaxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<HalfvecInput>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for vector in vectors.iter_deny_null() {
                    result.push(self.input(BorrowedVector::Vecf16(vector.as_borrowed())));
                }
                result
            }
            Self::Rabitq8L2 | Self::Rabitq8Ip | Self::Rabitq8Cosine | Self::Rabitq8Maxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<Rabitq8Input>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for vector in vectors.iter_deny_null() {
                    result.push(self.input(BorrowedVector::Rabitq8(vector.as_borrowed())));
                }
                result
            }
            Self::Rabitq4L2 | Self::Rabitq4Ip | Self::Rabitq4Cosine | Self::Rabitq4Maxsim => {
                let vectors = unsafe {
                    pgrx::datum::Array::<Rabitq4Input>::from_datum(datum, false).unwrap()
                };
                let mut result = Vec::with_capacity(vectors.len());
                for vector in vectors.iter_deny_null() {
                    result.push(self.input(BorrowedVector::Rabitq4(vector.as_borrowed())));
                }
                result
            }
        };
        Some(vectors)
    }
    pub fn output(self, x: Distance) -> f32 {
        match self {
            Self::VectorCosine
            | Self::HalfvecCosine
            | Self::Rabitq8Cosine
            | Self::Rabitq4Cosine => x.to_f32() + 1.0f32,
            Self::VectorL2 | Self::HalfvecL2 | Self::Rabitq8L2 | Self::Rabitq4L2 => {
                x.to_f32().sqrt()
            }
            Self::VectorIp
            | Self::HalfvecIp
            | Self::Rabitq8Ip
            | Self::Rabitq4Ip
            | Self::VectorMaxsim
            | Self::HalfvecMaxsim
            | Self::Rabitq8Maxsim
            | Self::Rabitq4Maxsim => x.to_f32(),
        }
    }
    pub const fn distance_kind(self) -> DistanceKind {
        match self {
            Self::VectorL2 | Self::HalfvecL2 | Self::Rabitq8L2 | Self::Rabitq4L2 => {
                DistanceKind::L2S
            }
            Self::VectorIp
            | Self::HalfvecIp
            | Self::Rabitq8Ip
            | Self::Rabitq4Ip
            | Self::VectorCosine
            | Self::HalfvecCosine
            | Self::Rabitq8Cosine
            | Self::Rabitq4Cosine
            | Self::VectorMaxsim
            | Self::HalfvecMaxsim
            | Self::Rabitq8Maxsim
            | Self::Rabitq4Maxsim => DistanceKind::Dot,
        }
    }
    pub const fn vector_kind(self) -> VectorKind {
        match self {
            Self::VectorL2 | Self::VectorIp | Self::VectorCosine | Self::VectorMaxsim => {
                VectorKind::Vecf32
            }
            Self::HalfvecL2 | Self::HalfvecIp | Self::HalfvecCosine | Self::HalfvecMaxsim => {
                VectorKind::Vecf16
            }
            Self::Rabitq8L2 | Self::Rabitq8Ip | Self::Rabitq8Cosine | Self::Rabitq8Maxsim => {
                VectorKind::Rabitq8
            }
            Self::Rabitq4L2 | Self::Rabitq4Ip | Self::Rabitq4Cosine | Self::Rabitq4Maxsim => {
                VectorKind::Rabitq4
            }
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

    let result_datum = unsafe {
        #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
        pgrx::pg_sys::ffi::pg_guard_ffi_boundary(|| fn_addr(&mut fcinfo))
    };

    let result_option = unsafe { String::from_datum(result_datum, fcinfo.isnull) };

    let result_string = result_option.expect("null return value");

    let result = match result_string.as_str() {
        "vchordrq_vector_l2_ops" => Opfamily::VectorL2,
        "vchordrq_vector_ip_ops" => Opfamily::VectorIp,
        "vchordrq_vector_cosine_ops" => Opfamily::VectorCosine,
        "vchordrq_halfvec_l2_ops" => Opfamily::HalfvecL2,
        "vchordrq_halfvec_ip_ops" => Opfamily::HalfvecIp,
        "vchordrq_halfvec_cosine_ops" => Opfamily::HalfvecCosine,
        "vchordrq_rabitq8_l2_ops" => Opfamily::Rabitq8L2,
        "vchordrq_rabitq8_ip_ops" => Opfamily::Rabitq8Ip,
        "vchordrq_rabitq8_cosine_ops" => Opfamily::Rabitq8Cosine,
        "vchordrq_rabitq4_l2_ops" => Opfamily::Rabitq4L2,
        "vchordrq_rabitq4_ip_ops" => Opfamily::Rabitq4Ip,
        "vchordrq_rabitq4_cosine_ops" => Opfamily::Rabitq4Cosine,
        "vchordrq_vector_maxsim_ops" => Opfamily::VectorMaxsim,
        "vchordrq_halfvec_maxsim_ops" => Opfamily::HalfvecMaxsim,
        "vchordrq_rabitq8_maxsim_ops" => Opfamily::Rabitq8Maxsim,
        "vchordrq_rabitq4_maxsim_ops" => Opfamily::Rabitq4Maxsim,
        _ => pgrx::error!("unknown operator class"),
    };

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    result
}
