use crate::datatype::memory_pgvector_vector::PgvectorVectorInput;
use crate::datatype::memory_pgvector_vector::PgvectorVectorOutput;
use crate::datatype::typmod::Typmod;
use crate::vchordrqfscan::types::{BorrowedVector, OwnedVector};
use crate::vchordrqfscan::types::{DistanceKind, VectorKind};
use crate::vchordrqfscan::types::{VchordrqfscanIndexingOptions, VectorOptions};
use distance::Distance;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use serde::Deserialize;
use std::ffi::CStr;
use std::num::NonZero;
use vector::VectorBorrowed;

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Reloption {
    vl_len_: i32,
    pub options: i32,
}

impl Reloption {
    pub const TAB: &'static [pgrx::pg_sys::relopt_parse_elt] = &[pgrx::pg_sys::relopt_parse_elt {
        optname: c"options".as_ptr(),
        opttype: pgrx::pg_sys::relopt_type::RELOPT_TYPE_STRING,
        offset: std::mem::offset_of!(Reloption, options) as i32,
    }];
    unsafe fn options(&self) -> &CStr {
        unsafe {
            let ptr = (&raw const *self)
                .cast::<std::ffi::c_char>()
                .offset(self.options as _);
            CStr::from_ptr(ptr)
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PgDistanceKind {
    L2,
    Dot,
    Cos,
}

impl PgDistanceKind {
    pub fn to_distance(self) -> DistanceKind {
        match self {
            PgDistanceKind::L2 => DistanceKind::L2,
            PgDistanceKind::Dot | PgDistanceKind::Cos => DistanceKind::Dot,
        }
    }
}

fn convert_name_to_vd(name: &str) -> Option<(VectorKind, PgDistanceKind)> {
    match name.strip_suffix("_ops") {
        Some("vector_l2") => Some((VectorKind::Vecf32, PgDistanceKind::L2)),
        Some("vector_ip") => Some((VectorKind::Vecf32, PgDistanceKind::Dot)),
        Some("vector_cosine") => Some((VectorKind::Vecf32, PgDistanceKind::Cos)),
        _ => None,
    }
}

unsafe fn convert_reloptions_to_options(
    reloptions: *const pgrx::pg_sys::varlena,
) -> VchordrqfscanIndexingOptions {
    #[derive(Debug, Clone, Deserialize, Default)]
    #[serde(deny_unknown_fields)]
    struct Parsed {
        #[serde(flatten)]
        rabitq: VchordrqfscanIndexingOptions,
    }
    let reloption = reloptions as *const Reloption;
    if reloption.is_null() || unsafe { (*reloption).options == 0 } {
        return Default::default();
    }
    let s = unsafe { (*reloption).options() }.to_string_lossy();
    match toml::from_str::<Parsed>(&s) {
        Ok(p) => p.rabitq,
        Err(e) => pgrx::error!("failed to parse options: {}", e),
    }
}

pub unsafe fn options(
    index: pgrx::pg_sys::Relation,
) -> (VectorOptions, VchordrqfscanIndexingOptions) {
    let att = unsafe { &mut *(*index).rd_att };
    let atts = unsafe { att.attrs.as_slice(att.natts as _) };
    if atts.is_empty() {
        pgrx::error!("indexing on no columns is not supported");
    }
    if atts.len() != 1 {
        pgrx::error!("multicolumn index is not supported");
    }
    // get dims
    let typmod = Typmod::parse_from_i32(atts[0].type_mod()).unwrap();
    let dims = if let Some(dims) = typmod.dims() {
        dims.get()
    } else {
        pgrx::error!(
            "Dimensions type modifier of a vector column is needed for building the index."
        );
    };
    // get v, d
    let opfamily = unsafe { opfamily(index) };
    let vector = VectorOptions {
        dims,
        v: opfamily.vector,
        d: opfamily.distance_kind(),
    };
    // get indexing, segment, optimizing
    let rabitq = unsafe { convert_reloptions_to_options((*index).rd_options) };
    (vector, rabitq)
}

#[derive(Debug, Clone, Copy)]
pub struct Opfamily {
    vector: VectorKind,
    pg_distance: PgDistanceKind,
}

impl Opfamily {
    pub unsafe fn datum_to_vector(
        self,
        datum: pgrx::pg_sys::Datum,
        is_null: bool,
    ) -> Option<OwnedVector> {
        if is_null || datum.is_null() {
            return None;
        }
        let vector = match self.vector {
            VectorKind::Vecf32 => {
                let vector = unsafe { PgvectorVectorInput::from_datum(datum, false).unwrap() };
                self.preprocess(BorrowedVector::Vecf32(vector.as_borrowed()))
            }
        };
        Some(vector)
    }
    pub unsafe fn datum_to_sphere(
        self,
        datum: pgrx::pg_sys::Datum,
        is_null: bool,
    ) -> (Option<OwnedVector>, Option<f32>) {
        if is_null || datum.is_null() {
            return (None, None);
        }
        let tuple = unsafe { PgHeapTuple::from_composite_datum(datum) };
        let center = match self.vector {
            VectorKind::Vecf32 => tuple
                .get_by_index::<PgvectorVectorOutput>(NonZero::new(1).unwrap())
                .unwrap()
                .map(|vector| self.preprocess(BorrowedVector::Vecf32(vector.as_borrowed()))),
        };
        let radius = tuple.get_by_index::<f32>(NonZero::new(2).unwrap()).unwrap();
        (center, radius)
    }
    pub fn preprocess(self, vector: BorrowedVector<'_>) -> OwnedVector {
        use BorrowedVector as B;
        use OwnedVector as O;
        match (vector, self.pg_distance) {
            (B::Vecf32(x), PgDistanceKind::L2) => O::Vecf32(x.own()),
            (B::Vecf32(x), PgDistanceKind::Dot) => O::Vecf32(x.own()),
            (B::Vecf32(x), PgDistanceKind::Cos) => O::Vecf32(x.function_normalize()),
        }
    }
    pub fn process(self, x: Distance) -> f32 {
        match self.pg_distance {
            PgDistanceKind::Cos => f32::from(x) + 1.0f32,
            PgDistanceKind::L2 => f32::from(x).sqrt(),
            _ => f32::from(x),
        }
    }
    pub fn distance_kind(self) -> DistanceKind {
        self.pg_distance.to_distance()
    }
}

pub unsafe fn opfamily(index: pgrx::pg_sys::Relation) -> Opfamily {
    use pgrx::pg_sys::Oid;

    let proc = unsafe { pgrx::pg_sys::index_getprocid(index, 1, 1) };

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

    let result_string = result_option.expect("null string");

    let (vector, pg_distance) = convert_name_to_vd(&result_string).unwrap();

    unsafe {
        pgrx::pg_sys::pfree(result_datum.cast_mut_ptr());
    }

    Opfamily {
        vector,
        pg_distance,
    }
}
