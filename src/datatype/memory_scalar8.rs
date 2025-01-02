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
use vector::scalar8::Scalar8Borrowed;

#[repr(C, align(8))]
pub struct Scalar8Header {
    varlena: u32,
    dims: u16,
    unused: u16,
    sum_of_x2: f32,
    k: f32,
    b: f32,
    sum_of_code: f32,
    phantom: [u8; 0],
}

impl Scalar8Header {
    fn size_of(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        (size_of::<Self>() + size_of::<u8>() * len).next_multiple_of(8)
    }
    pub fn as_borrowed(&self) -> Scalar8Borrowed<'_> {
        unsafe {
            Scalar8Borrowed::new_unchecked(
                self.sum_of_x2,
                self.k,
                self.b,
                self.sum_of_code,
                std::slice::from_raw_parts(self.phantom.as_ptr(), self.dims as usize),
            )
        }
    }
}

pub enum Scalar8Input<'a> {
    Owned(Scalar8Output),
    Borrowed(&'a Scalar8Header),
}

impl Scalar8Input<'_> {
    unsafe fn new(p: NonNull<Scalar8Header>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            Scalar8Input::Owned(Scalar8Output(q))
        } else {
            unsafe { Scalar8Input::Borrowed(p.as_ref()) }
        }
    }
}

impl Deref for Scalar8Input<'_> {
    type Target = Scalar8Header;

    fn deref(&self) -> &Self::Target {
        match self {
            Scalar8Input::Owned(x) => x,
            Scalar8Input::Borrowed(x) => x,
        }
    }
}

pub struct Scalar8Output(NonNull<Scalar8Header>);

impl Scalar8Output {
    pub fn new(vector: Scalar8Borrowed<'_>) -> Scalar8Output {
        unsafe {
            let code = vector.code();
            let size = Scalar8Header::size_of(code.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut Scalar8Header;
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dims).write(vector.dims() as _);
            (&raw mut (*ptr).unused).write(0);
            (&raw mut (*ptr).sum_of_x2).write(vector.sum_of_x2());
            (&raw mut (*ptr).k).write(vector.k());
            (&raw mut (*ptr).b).write(vector.b());
            (&raw mut (*ptr).sum_of_code).write(vector.sum_of_code());
            std::ptr::copy_nonoverlapping(code.as_ptr(), (*ptr).phantom.as_mut_ptr(), code.len());
            Scalar8Output(NonNull::new(ptr).unwrap())
        }
    }
    pub fn into_raw(self) -> *mut Scalar8Header {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Deref for Scalar8Output {
    type Target = Scalar8Header;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Scalar8Output {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr() as _);
        }
    }
}

impl FromDatum for Scalar8Input<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr::<Scalar8Header>()).unwrap();
            unsafe { Some(Scalar8Input::new(ptr)) }
        }
    }
}

impl IntoDatum for Scalar8Output {
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

impl FromDatum for Scalar8Output {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let p = NonNull::new(datum.cast_mut_ptr::<Scalar8Header>())?;
            let q =
                unsafe { NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast())? };
            if p != q {
                Some(Scalar8Output(q))
            } else {
                let header = p.as_ptr();
                let vector = unsafe { (*header).as_borrowed() };
                Some(Scalar8Output::new(vector))
            }
        }
    }
}

unsafe impl pgrx::datum::UnboxDatum for Scalar8Output {
    type As<'src> = Scalar8Output;
    #[inline]
    unsafe fn unbox<'src>(d: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let p = NonNull::new(d.sans_lifetime().cast_mut_ptr::<Scalar8Header>()).unwrap();
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.cast().as_ptr()).cast()).unwrap()
        };
        if p != q {
            Scalar8Output(q)
        } else {
            let header = p.as_ptr();
            let vector = unsafe { (*header).as_borrowed() };
            Scalar8Output::new(vector)
        }
    }
}

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

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for Scalar8Input<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        unsafe { arg.unbox_arg_using_from_datum().unwrap() }
    }
}

unsafe impl pgrx::callconv::BoxRet for Scalar8Output {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        unsafe { fcinfo.return_raw_datum(Datum::from(self.into_raw() as *mut ())) }
    }
}
