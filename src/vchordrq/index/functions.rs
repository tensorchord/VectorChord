use crate::postgres::Relation;
use crate::vchordrq::algorithm::prewarm::prewarm;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgAm, PgClass};

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
    let relation = unsafe { Relation::new(index) };
    let message = prewarm(relation, height);
    unsafe {
        pgrx::pg_sys::index_close(index, pgrx::pg_sys::ShareLock as _);
    }
    message
}
