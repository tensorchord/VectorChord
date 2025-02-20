use crate::index::storage::PostgresRelation;
use algorithm::operator::{Dot, L2, Op};
use algorithm::types::*;
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
    let index = unsafe { pgrx::pg_sys::index_open(indexrelid, pgrx::pg_sys::AccessShareLock as _) };
    let relation = unsafe { PostgresRelation::new(index) };
    let opfamily = unsafe { crate::index::opclass::opfamily(index) };
    let message = match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::prewarm::<Op<VectOwned<f32>, L2>>(relation, height, || {
                pgrx::check_for_interrupts!();
            })
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::prewarm::<Op<VectOwned<f32>, Dot>>(relation, height, || {
                pgrx::check_for_interrupts!();
            })
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::prewarm::<Op<VectOwned<f16>, L2>>(relation, height, || {
                pgrx::check_for_interrupts!();
            })
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::prewarm::<Op<VectOwned<f16>, Dot>>(relation, height, || {
                pgrx::check_for_interrupts!();
            })
        }
    };
    unsafe {
        pgrx::pg_sys::index_close(index, pgrx::pg_sys::AccessShareLock as _);
    }
    message
}
