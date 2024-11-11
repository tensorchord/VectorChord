use crate::algorithm::prewarm::prewarm;
use crate::postgres::Relation;
use pgrx::pg_extern;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgAm, PgClass};

#[pg_extern]
fn rabbithole_prewarm(indexrelid: Oid) -> String {
    let pg_am = PgAm::search_amname(c"rabbithole").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("rabbithole is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("there is no such index");
    };
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("{:?} is not a rabbithole index", pg_class.relname());
    }
    let index = unsafe { pgrx::pg_sys::index_open(indexrelid, pgrx::pg_sys::ShareLock as _) };
    let relation = unsafe { Relation::new(index) };
    let message = prewarm(relation);
    unsafe {
        pgrx::pg_sys::index_close(index, pgrx::pg_sys::ShareLock as _);
    }
    message
}
