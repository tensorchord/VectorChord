use crate::datatype::memory_pgvector_vector::PgvectorVectorInput;
use crate::datatype::memory_pgvector_vector::PgvectorVectorOutput;
use crate::datatype::typmod::Typmod;
use crate::types::RabbitholeIndexingOptions;
use base::distance::*;
use base::index::*;
use base::vector::*;
use pgrx::datum::FromDatum;
use pgrx::heap_tuple::PgHeapTuple;
use serde::Deserialize;
use std::ffi::CStr;
use std::num::NonZero;

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
            let ptr = std::ptr::addr_of!(*self)
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

pub fn convert_opclass_to_vd(
    opclass_oid: pgrx::pg_sys::Oid,
) -> Option<(VectorKind, PgDistanceKind)> {
    let namespace = pgrx::pg_catalog::PgNamespace::search_namespacename(c"rabbithole").unwrap();
    let namespace = namespace.get().expect("rabbithole is not installed.");
    let opclass = pgrx::pg_catalog::PgOpclass::search_claoid(opclass_oid).unwrap();
    let opclass = opclass.get().expect("rabbithole is not installed.");
    if opclass.opcnamespace() == namespace.oid() {
        if let Ok(name) = opclass.opcname().to_str() {
            if let Some(p) = convert_name_to_vd(name) {
                return Some(p);
            }
        }
    }
    None
}

pub fn convert_opfamily_to_vd(
    opfamily_oid: pgrx::pg_sys::Oid,
) -> Option<(VectorKind, PgDistanceKind)> {
    let namespace = pgrx::pg_catalog::PgNamespace::search_namespacename(c"rabbithole").unwrap();
    let namespace = namespace.get().expect("rabbithole is not installed.");
    let opfamily = pgrx::pg_catalog::PgOpfamily::search_opfamilyoid(opfamily_oid).unwrap();
    let opfamily = opfamily.get().expect("rabbithole is not installed.");
    if opfamily.opfnamespace() == namespace.oid() {
        if let Ok(name) = opfamily.opfname().to_str() {
            if let Some(p) = convert_name_to_vd(name) {
                return Some(p);
            }
        }
    }
    None
}

fn convert_name_to_vd(name: &str) -> Option<(VectorKind, PgDistanceKind)> {
    match name.strip_suffix("_ops") {
        Some("vector_l2") => Some((VectorKind::Vecf32, PgDistanceKind::L2)),
        Some("vector_dot") => Some((VectorKind::Vecf32, PgDistanceKind::Dot)),
        Some("vector_cos") => Some((VectorKind::Vecf32, PgDistanceKind::Cos)),
        _ => None,
    }
}

unsafe fn convert_reloptions_to_options(
    reloptions: *const pgrx::pg_sys::varlena,
) -> RabbitholeIndexingOptions {
    #[derive(Debug, Clone, Deserialize, Default)]
    #[serde(deny_unknown_fields)]
    struct Parsed {
        #[serde(flatten)]
        rabitq: RabbitholeIndexingOptions,
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
) -> (VectorOptions, RabbitholeIndexingOptions, PgDistanceKind) {
    let opfamily = unsafe { (*index).rd_opfamily.read() };
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
    let (v, pg_d) = convert_opfamily_to_vd(opfamily).unwrap();
    let vector = VectorOptions {
        dims,
        v,
        d: pg_d.to_distance(),
    };
    // get indexing, segment, optimizing
    let rabitq = unsafe { convert_reloptions_to_options((*index).rd_options) };
    (vector, rabitq, pg_d)
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
            _ => unreachable!(),
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
            _ => unreachable!(),
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
            (B::Vecf16(x), _) => O::Vecf16(x.own()),
            (B::SVecf32(x), _) => O::SVecf32(x.own()),
            (B::BVector(x), _) => O::BVector(x.own()),
        }
    }
    pub fn process(self, x: Distance) -> f32 {
        match self.pg_distance {
            PgDistanceKind::Cos => f32::from(x) + 1.0f32,
            _ => f32::from(x),
        }
    }
    pub fn distance_kind(self) -> DistanceKind {
        self.pg_distance.to_distance()
    }
}

pub unsafe fn opfamily(index: pgrx::pg_sys::Relation) -> Opfamily {
    let opfamily = unsafe { (*index).rd_opfamily.read() };
    let (vector, pg_distance) = convert_opfamily_to_vd(opfamily).unwrap();
    Opfamily {
        vector,
        pg_distance,
    }
}
