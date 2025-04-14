use crate::index::storage::PostgresRelation;
use pgrx::pg_sys::Oid;
use pgrx_catalog::{PgAm, PgClass, PgClassRelkind};

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_prewarm(indexrelid: Oid, height: i32) -> String {
    let pg_am = PgAm::search_amname(c"vchordrq").unwrap();
    let Some(pg_am) = pg_am.get() else {
        pgrx::error!("vchord is not installed");
    };
    let pg_class = PgClass::search_reloid(indexrelid).unwrap();
    let Some(pg_class) = pg_class.get() else {
        pgrx::error!("the relation does not exist");
    };
    if pg_class.relkind() != PgClassRelkind::Index {
        pgrx::error!("the relation {:?} is not an index", pg_class.relname());
    }
    if pg_class.relam() != pg_am.oid() {
        pgrx::error!("the index {:?} is not a vchordrq index", pg_class.relname());
    }
    let relation = Index::open(indexrelid, pgrx::pg_sys::AccessShareLock as _);
    let opfamily = unsafe { crate::index::opclass::opfamily(relation.raw()) };
    let index = unsafe { PostgresRelation::new(relation.raw()) };
    crate::index::algorithm::prewarm(opfamily, index, height, || {
        pgrx::check_for_interrupts!();
    })
}

struct Index {
    raw: *mut pgrx::pg_sys::RelationData,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl Index {
    fn open(indexrelid: Oid, lockmode: pgrx::pg_sys::LOCKMASK) -> Self {
        Self {
            raw: unsafe { pgrx::pg_sys::index_open(indexrelid, lockmode) },
            lockmode,
        }
    }
    fn raw(&self) -> *mut pgrx::pg_sys::RelationData {
        self.raw
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::index_close(self.raw, self.lockmode);
        }
    }
}
