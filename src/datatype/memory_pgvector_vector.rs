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
pub struct PgvectorVectorHeader {
    varlena: u32,
    dims: u16,
    unused: u16,
    phantom: [f32; 0],
}

impl PgvectorVectorHeader {
    fn size_of(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        (size_of::<Self>() + size_of::<f32>() * len).next_multiple_of(8)
    }
    pub fn as_borrowed(&self) -> VectBorrowed<'_, f32> {
        unsafe {
            VectBorrowed::new_unchecked(std::slice::from_raw_parts(
                self.phantom.as_ptr(),
                self.dims as usize,
            ))
        }
    }
}

pub enum PgvectorVectorInput<'a> {
    Owned(PgvectorVectorOutput),
    Borrowed(&'a PgvectorVectorHeader),
}

impl PgvectorVectorInput<'_> {
    unsafe fn new(p: NonNull<PgvectorVectorHeader>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            PgvectorVectorInput::Owned(PgvectorVectorOutput(q))
        } else {
            unsafe { PgvectorVectorInput::Borrowed(p.as_ref()) }
        }
    }
}

impl Deref for PgvectorVectorInput<'_> {
    type Target = PgvectorVectorHeader;

    fn deref(&self) -> &Self::Target {
        match self {
            PgvectorVectorInput::Owned(x) => x,
            PgvectorVectorInput::Borrowed(x) => x,
        }
    }
}

pub struct PgvectorVectorOutput(NonNull<PgvectorVectorHeader>);

impl PgvectorVectorOutput {
    pub fn new(vector: VectBorrowed<'_, f32>) -> PgvectorVectorOutput {
        unsafe {
            let slice = vector.slice();
            let size = PgvectorVectorHeader::size_of(slice.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut PgvectorVectorHeader;
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dims).write(vector.dims() as _);
            (&raw mut (*ptr).unused).write(0);
            std::ptr::copy_nonoverlapping(slice.as_ptr(), (*ptr).phantom.as_mut_ptr(), slice.len());
            PgvectorVectorOutput(NonNull::new(ptr).unwrap())
        }
    }
    pub fn into_raw(self) -> *mut PgvectorVectorHeader {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Deref for PgvectorVectorOutput {
    type Target = PgvectorVectorHeader;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl Drop for PgvectorVectorOutput {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr() as _);
        }
    }
}

impl FromDatum for PgvectorVectorInput<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr::<PgvectorVectorHeader>()).unwrap();
            unsafe { Some(PgvectorVectorInput::new(ptr)) }
        }
    }
}

impl IntoDatum for PgvectorVectorOutput {
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

impl FromDatum for PgvectorVectorOutput {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let p = NonNull::new(datum.cast_mut_ptr::<PgvectorVectorHeader>())?;
            let q =
                unsafe { NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast())? };
            if p != q {
                Some(PgvectorVectorOutput(q))
            } else {
                let header = p.as_ptr();
                let vector = unsafe { (*header).as_borrowed() };
                Some(PgvectorVectorOutput::new(vector))
            }
        }
    }
}

unsafe impl pgrx::datum::UnboxDatum for PgvectorVectorOutput {
    type As<'src> = PgvectorVectorOutput;
    #[inline]
    unsafe fn unbox<'src>(d: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let p = NonNull::new(d.sans_lifetime().cast_mut_ptr::<PgvectorVectorHeader>()).unwrap();
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            PgvectorVectorOutput(q)
        } else {
            let header = p.as_ptr();
            let vector = unsafe { (*header).as_borrowed() };
            PgvectorVectorOutput::new(vector)
        }
    }
}

unsafe impl SqlTranslatable for PgvectorVectorInput<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("vector")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("vector"))))
    }
}

unsafe impl SqlTranslatable for PgvectorVectorOutput {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("vector")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("vector"))))
    }
}

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for PgvectorVectorInput<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        unsafe { arg.unbox_arg_using_from_datum().unwrap() }
    }
}

unsafe impl pgrx::callconv::BoxRet for PgvectorVectorOutput {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        unsafe { fcinfo.return_raw_datum(Datum::from(self.into_raw() as *mut ())) }
    }
}
