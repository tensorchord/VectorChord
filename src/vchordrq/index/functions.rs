use super::am_options;
use crate::postgres::PostgresRelation;
use crate::vchordrq::algorithm::prewarm::prewarm;
use crate::vchordrq::types::VectorKind;
use half::f16;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgAm, PgClass};
use vector::vect::VectOwned;

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_prewarm(indexrelid: Oid, height: i32) -> String {
    let pg_am = PgAm::search_amname(c"vchordrq").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("vchord is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("there is no such index");
    };
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("{:?} is not a vchordrq index", pg_class.relname());
    }
    let index = unsafe { pgrx::pg_sys::index_open(indexrelid, pgrx::pg_sys::ShareLock as _) };
    let relation = unsafe { PostgresRelation::new(index) };
    let opfamily = unsafe { am_options::opfamily(index) };
    let message = match opfamily.vector_kind() {
        VectorKind::Vecf32 => prewarm::<VectOwned<f32>>(relation, height),
        VectorKind::Vecf16 => prewarm::<VectOwned<f16>>(relation, height),
    };
    unsafe {
        pgrx::pg_sys::index_close(index, pgrx::pg_sys::ShareLock as _);
    }
    message
}
