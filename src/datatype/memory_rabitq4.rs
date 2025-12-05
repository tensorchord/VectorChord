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

use pgrx::datum::{FromDatum, IntoDatum};
use pgrx::pg_sys::{Datum, Oid};
use pgrx::pgrx_sql_entity_graph::metadata::*;
use std::marker::PhantomData;
use std::ptr::NonNull;
use vector::VectorBorrowed;
use vector::rabitq4::Rabitq4Borrowed;

#[repr(C)]
struct Rabitq4Header {
    varlena: u32,
    dim: u16,
    unused: u16,
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    elements: [u8; 0],
}

impl Rabitq4Header {
    fn size_of_by_dim(dim: usize) -> usize {
        Self::size_of_by_len(dim.div_ceil(2))
    }
    fn size_of_by_len(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        size_of::<Self>() + size_of::<u8>() * len
    }
    unsafe fn as_borrowed<'a>(this: NonNull<Self>) -> Rabitq4Borrowed<'a> {
        unsafe {
            let this = this.as_ptr();
            Rabitq4Borrowed::new(
                (&raw const (*this).dim).read() as u32,
                (&raw const (*this).sum_of_x2).read(),
                (&raw const (*this).norm_of_lattice).read(),
                (&raw const (*this).sum_of_code).read(),
                (&raw const (*this).sum_of_abs_x).read(),
                std::slice::from_raw_parts(
                    (&raw const (*this).elements).cast(),
                    (&raw const (*this).dim).read().div_ceil(2) as usize,
                ),
            )
        }
    }
}

pub struct Rabitq4Input<'a>(NonNull<Rabitq4Header>, PhantomData<&'a ()>, bool);

impl Rabitq4Input<'_> {
    unsafe fn from_ptr(p: NonNull<Rabitq4Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.as_ptr().cast()).cast()).unwrap()
        };
        unsafe {
            let varlena = q.cast::<u32>().read();
            #[cfg(target_endian = "big")]
            let size = varlena as usize;
            #[cfg(target_endian = "little")]
            let size = varlena as usize >> 2;
            let dim = q.byte_add(4).cast::<u16>().read();
            assert_eq!(Rabitq4Header::size_of_by_dim(dim as _), size);
            let unused = q.byte_add(6).cast::<u16>().read();
            assert_eq!(unused, 0);
        }
        Rabitq4Input(q, PhantomData, p != q)
    }
    pub fn as_borrowed(&self) -> Rabitq4Borrowed<'_> {
        unsafe { Rabitq4Header::as_borrowed(self.0) }
    }
}

impl Drop for Rabitq4Input<'_> {
    fn drop(&mut self) {
        if self.2 {
            unsafe {
                pgrx::pg_sys::pfree(self.0.as_ptr().cast());
            }
        }
    }
}

pub struct Rabitq4Output(NonNull<Rabitq4Header>);

impl Rabitq4Output {
    unsafe fn from_ptr(p: NonNull<Rabitq4Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum_copy(p.as_ptr().cast()).cast()).unwrap()
        };
        unsafe {
            let varlena = q.cast::<u32>().read();
            #[cfg(target_endian = "big")]
            let size = varlena as usize;
            #[cfg(target_endian = "little")]
            let size = varlena as usize >> 2;
            let dim = q.byte_add(4).cast::<u16>().read();
            assert_eq!(Rabitq4Header::size_of_by_dim(dim as _), size);
            let unused = q.byte_add(6).cast::<u16>().read();
            assert_eq!(unused, 0);
        }
        Self(q)
    }
    pub fn new(vector: Rabitq4Borrowed<'_>) -> Self {
        unsafe {
            let packed_code = vector.packed_code();
            let size = Rabitq4Header::size_of_by_len(packed_code.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut Rabitq4Header;
            // SET_VARSIZE_4B
            #[cfg(target_endian = "big")]
            (&raw mut (*ptr).varlena).write((size as u32) & 0x3FFFFFFF);
            #[cfg(target_endian = "little")]
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dim).write(vector.dim() as _);
            (&raw mut (*ptr).unused).write(0);
            (&raw mut (*ptr).sum_of_x2).write(vector.sum_of_x2());
            (&raw mut (*ptr).norm_of_lattice).write(vector.norm_of_lattice());
            (&raw mut (*ptr).sum_of_code).write(vector.sum_of_code());
            (&raw mut (*ptr).sum_of_abs_x).write(vector.sum_of_abs_x());
            std::ptr::copy_nonoverlapping(
                packed_code.as_ptr(),
                (&raw mut (*ptr).elements).cast(),
                packed_code.len(),
            );
            Self(NonNull::new(ptr).unwrap())
        }
    }
    pub fn as_borrowed(&self) -> Rabitq4Borrowed<'_> {
        unsafe { Rabitq4Header::as_borrowed(self.0) }
    }
    fn into_raw(self) -> *mut Rabitq4Header {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Drop for Rabitq4Output {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr().cast());
        }
    }
}

// FromDatum

impl FromDatum for Rabitq4Input<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
            unsafe { Some(Self::from_ptr(ptr)) }
        }
    }
}

impl FromDatum for Rabitq4Output {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
            unsafe { Some(Self::from_ptr(ptr)) }
        }
    }
}

// IntoDatum

impl IntoDatum for Rabitq4Output {
    fn into_datum(self) -> Option<Datum> {
        Some(Datum::from(self.into_raw()))
    }

    fn type_oid() -> Oid {
        Oid::INVALID
    }

    fn is_compatible_with(_: Oid) -> bool {
        true
    }
}

// UnboxDatum

unsafe impl<'a> pgrx::datum::UnboxDatum for Rabitq4Input<'a> {
    type As<'src>
        = Rabitq4Input<'src>
    where
        'a: 'src;
    #[inline]
    unsafe fn unbox<'src>(datum: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let datum = datum.sans_lifetime();
        let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
        unsafe { Self::from_ptr(ptr) }
    }
}

unsafe impl pgrx::datum::UnboxDatum for Rabitq4Output {
    type As<'src> = Rabitq4Output;
    #[inline]
    unsafe fn unbox<'src>(datum: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let datum = datum.sans_lifetime();
        let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
        unsafe { Self::from_ptr(ptr) }
    }
}

// SqlTranslatable

unsafe impl SqlTranslatable for Rabitq4Input<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("rabitq4")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("rabitq4"))))
    }
}

unsafe impl SqlTranslatable for Rabitq4Output {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("rabitq4")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("rabitq4"))))
    }
}

// ArgAbi

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for Rabitq4Input<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        let index = arg.index();
        unsafe {
            arg.unbox_arg_using_from_datum()
                .unwrap_or_else(|| panic!("argument {index} must not be null"))
        }
    }
}

// BoxRet

unsafe impl pgrx::callconv::BoxRet for Rabitq4Output {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        match self.into_datum() {
            Some(datum) => unsafe { fcinfo.return_raw_datum(datum) },
            None => fcinfo.return_null(),
        }
    }
}
