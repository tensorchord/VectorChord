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
use vector::scalar8::Scalar8Borrowed;

#[repr(C)]
struct Scalar8Header {
    varlena: u32,
    dims: u16,
    unused: u16,
    sum_of_x2: f32,
    k: f32,
    b: f32,
    sum_of_code: f32,
    elements: [u8; 0],
}

impl Scalar8Header {
    fn size_of(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        (size_of::<Self>() + size_of::<u8>() * len).next_multiple_of(8)
    }
    unsafe fn as_borrowed<'a>(this: NonNull<Self>) -> Scalar8Borrowed<'a> {
        unsafe {
            let this = this.as_ptr();
            Scalar8Borrowed::new(
                (&raw const (*this).sum_of_x2).read(),
                (&raw const (*this).k).read(),
                (&raw const (*this).b).read(),
                (&raw const (*this).sum_of_code).read(),
                std::slice::from_raw_parts(
                    (&raw const (*this).elements).cast(),
                    (&raw const (*this).dims).read() as usize,
                ),
            )
        }
    }
}

pub struct Scalar8Input<'a>(NonNull<Scalar8Header>, PhantomData<&'a ()>, bool);

impl Scalar8Input<'_> {
    unsafe fn from_ptr(p: NonNull<Scalar8Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.as_ptr().cast()).cast()).unwrap()
        };
        unsafe {
            let varlena = q.cast::<u32>().read();
            #[cfg(target_endian = "big")]
            let size = varlena as usize;
            #[cfg(target_endian = "little")]
            let size = varlena as usize >> 2;
            let dims = q.byte_add(4).cast::<u16>().read();
            assert_eq!(Scalar8Header::size_of(dims as _), size);
            let unused = q.byte_add(6).cast::<u16>().read();
            assert_eq!(unused, 0);
        }
        Scalar8Input(q, PhantomData, p != q)
    }
    pub fn as_borrowed(&self) -> Scalar8Borrowed<'_> {
        unsafe { Scalar8Header::as_borrowed(self.0) }
    }
}

impl Drop for Scalar8Input<'_> {
    fn drop(&mut self) {
        if self.2 {
            unsafe {
                pgrx::pg_sys::pfree(self.0.as_ptr().cast());
            }
        }
    }
}

pub struct Scalar8Output(NonNull<Scalar8Header>);

impl Scalar8Output {
    unsafe fn from_ptr(p: NonNull<Scalar8Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum_copy(p.as_ptr().cast()).cast()).unwrap()
        };
        unsafe {
            let varlena = q.cast::<u32>().read();
            #[cfg(target_endian = "big")]
            let size = varlena as usize;
            #[cfg(target_endian = "little")]
            let size = varlena as usize >> 2;
            let dims = q.byte_add(4).cast::<u16>().read();
            assert_eq!(Scalar8Header::size_of(dims as _), size);
            let unused = q.byte_add(6).cast::<u16>().read();
            assert_eq!(unused, 0);
        }
        Self(q)
    }
    pub fn new(vector: Scalar8Borrowed<'_>) -> Self {
        unsafe {
            let code = vector.code();
            let size = Scalar8Header::size_of(code.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut Scalar8Header;
            // SET_VARSIZE_4B
            #[cfg(target_endian = "big")]
            (&raw mut (*ptr).varlena).write((size as u32) & 0x3FFFFFFF);
            #[cfg(target_endian = "little")]
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dims).write(vector.dims() as _);
            (&raw mut (*ptr).unused).write(0);
            (&raw mut (*ptr).sum_of_x2).write(vector.sum_of_x2());
            (&raw mut (*ptr).k).write(vector.k());
            (&raw mut (*ptr).b).write(vector.b());
            (&raw mut (*ptr).sum_of_code).write(vector.sum_of_code());
            std::ptr::copy_nonoverlapping(
                code.as_ptr(),
                (&raw mut (*ptr).elements).cast(),
                code.len(),
            );
            Self(NonNull::new(ptr).unwrap())
        }
    }
    pub fn as_borrowed(&self) -> Scalar8Borrowed<'_> {
        unsafe { Scalar8Header::as_borrowed(self.0) }
    }
    fn into_raw(self) -> *mut Scalar8Header {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Drop for Scalar8Output {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr().cast());
        }
    }
}

// FromDatum

impl FromDatum for Scalar8Input<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
            unsafe { Some(Self::from_ptr(ptr)) }
        }
    }
}

impl FromDatum for Scalar8Output {
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

impl IntoDatum for Scalar8Output {
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

unsafe impl pgrx::datum::UnboxDatum for Scalar8Output {
    type As<'src> = Scalar8Output;
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

unsafe impl SqlTranslatable for Scalar8Input<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("scalar8")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("scalar8"))))
    }
}

unsafe impl SqlTranslatable for Scalar8Output {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("scalar8")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("scalar8"))))
    }
}

// ArgAbi

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for Scalar8Input<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        let index = arg.index();
        unsafe {
            arg.unbox_arg_using_from_datum()
                .unwrap_or_else(|| panic!("argument {index} must not be null"))
        }
    }
}

// BoxRet

unsafe impl pgrx::callconv::BoxRet for Scalar8Output {
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
