use half::f16;
use pgrx::datum::FromDatum;
use pgrx::datum::IntoDatum;
use pgrx::pg_sys::Datum;
use pgrx::pg_sys::Oid;
use pgrx::pgrx_sql_entity_graph::metadata::ArgumentError;
use pgrx::pgrx_sql_entity_graph::metadata::Returns;
use pgrx::pgrx_sql_entity_graph::metadata::ReturnsError;
use pgrx::pgrx_sql_entity_graph::metadata::SqlMapping;
use pgrx::pgrx_sql_entity_graph::metadata::SqlTranslatable;
use std::ops::Deref;
use std::ptr::NonNull;
use vector::VectorBorrowed;
use vector::vect::VectBorrowed;

#[repr(C, align(8))]
pub struct PgvectorHalfvecHeader {
    varlena: u32,
    dims: u16,
    unused: u16,
    phantom: [f16; 0],
}

impl PgvectorHalfvecHeader {
    fn size_of(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        (size_of::<Self>() + size_of::<f16>() * len).next_multiple_of(8)
    }
    pub fn as_borrowed(&self) -> VectBorrowed<'_, f16> {
        unsafe {
            VectBorrowed::new_unchecked(std::slice::from_raw_parts(
                self.phantom.as_ptr(),
                self.dims as usize,
            ))
        }
    }
}

pub enum PgvectorHalfvecInput<'a> {
    Owned(PgvectorHalfvecOutput),
    Borrowed(&'a PgvectorHalfvecHeader),
}

impl PgvectorHalfvecInput<'_> {
    unsafe fn new(p: NonNull<PgvectorHalfvecHeader>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            PgvectorHalfvecInput::Owned(PgvectorHalfvecOutput(q))
        } else {
            unsafe { PgvectorHalfvecInput::Borrowed(p.as_ref()) }
        }
    }
}

impl Deref for PgvectorHalfvecInput<'_> {
    type Target = PgvectorHalfvecHeader;

    fn deref(&self) -> &Self::Target {
        match self {
            PgvectorHalfvecInput::Owned(x) => x,
            PgvectorHalfvecInput::Borrowed(x) => x,
        }
    }
}

pub struct PgvectorHalfvecOutput(NonNull<PgvectorHalfvecHeader>);

impl PgvectorHalfvecOutput {
    pub fn new(vector: VectBorrowed<'_, f16>) -> PgvectorHalfvecOutput {
        unsafe {
            let slice = vector.slice();
            let size = PgvectorHalfvecHeader::size_of(slice.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut PgvectorHalfvecHeader;
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dims).write(vector.dims() as _);
            (&raw mut (*ptr).unused).write(0);
            std::ptr::copy_nonoverlapping(slice.as_ptr(), (*ptr).phantom.as_mut_ptr(), slice.len());
            PgvectorHalfvecOutput(NonNull::new(ptr).unwrap())
        }
    }
    pub fn into_raw(self) -> *mut PgvectorHalfvecHeader {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Deref for PgvectorHalfvecOutput {
    type Target = PgvectorHalfvecHeader;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl Drop for PgvectorHalfvecOutput {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr() as _);
        }
    }
}

impl FromDatum for PgvectorHalfvecInput<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr::<PgvectorHalfvecHeader>()).unwrap();
            unsafe { Some(PgvectorHalfvecInput::new(ptr)) }
        }
    }
}

impl IntoDatum for PgvectorHalfvecOutput {
    fn into_datum(self) -> Option<Datum> {
        Some(Datum::from(self.into_raw() as *mut ()))
    }

    fn type_oid() -> Oid {
        Oid::INVALID
    }

    fn is_compatible_with(_: Oid) -> bool {
        true
    }
}

impl FromDatum for PgvectorHalfvecOutput {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let p = NonNull::new(datum.cast_mut_ptr::<PgvectorHalfvecHeader>())?;
            let q =
                unsafe { NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast())? };
            if p != q {
                Some(PgvectorHalfvecOutput(q))
            } else {
                let header = p.as_ptr();
                let vector = unsafe { (*header).as_borrowed() };
                Some(PgvectorHalfvecOutput::new(vector))
            }
        }
    }
}

unsafe impl pgrx::datum::UnboxDatum for PgvectorHalfvecOutput {
    type As<'src> = PgvectorHalfvecOutput;
    #[inline]
    unsafe fn unbox<'src>(d: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let p = NonNull::new(d.sans_lifetime().cast_mut_ptr::<PgvectorHalfvecHeader>()).unwrap();
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            PgvectorHalfvecOutput(q)
        } else {
            let header = p.as_ptr();
            let vector = unsafe { (*header).as_borrowed() };
            PgvectorHalfvecOutput::new(vector)
        }
    }
}

unsafe impl SqlTranslatable for PgvectorHalfvecInput<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("halfvec")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("halfvec"))))
    }
}

unsafe impl SqlTranslatable for PgvectorHalfvecOutput {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("halfvec")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("halfvec"))))
    }
}

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for PgvectorHalfvecInput<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        unsafe { arg.unbox_arg_using_from_datum().unwrap() }
    }
}

unsafe impl pgrx::callconv::BoxRet for PgvectorHalfvecOutput {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        unsafe { fcinfo.return_raw_datum(Datum::from(self.into_raw() as *mut ())) }
    }
}
