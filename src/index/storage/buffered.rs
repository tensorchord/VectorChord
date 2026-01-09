// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use crate::index::storage::{
    PostgresBufferReadGuard, PostgresBufferWriteGuard, PostgresPage, PostgresRelation,
};
use index::relation::{
    Opaque, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};

#[derive(Debug)]
pub struct BufferedPostgresRelation<Opaque> {
    postgres: PostgresRelation<Opaque>,
    list: std::cell::RefCell<arrayvec::ArrayVec<i32, 16>>,
}

impl<Opaque> BufferedPostgresRelation<Opaque> {
    pub unsafe fn new(raw: pgrx::pg_sys::Relation) -> Self {
        Self {
            postgres: unsafe { PostgresRelation::new(raw) },
            list: std::cell::RefCell::new(arrayvec::ArrayVec::new()),
        }
    }
    pub fn release(&self) -> bool {
        let list = self.list.borrow();
        !list.is_empty()
    }
}

impl<Opaque> Drop for BufferedPostgresRelation<Opaque> {
    fn drop(&mut self) {
        let free = self.list.get_mut();
        #[cfg(debug_assertions)]
        assert!(free.is_empty());
        for &mut buf in free {
            unsafe {
                pgrx::pg_sys::ReleaseBuffer(buf);
            }
        }
    }
}

impl<O: Opaque> Relation for BufferedPostgresRelation<O> {
    type Page = PostgresPage<O>;
}

impl<O: Opaque> RelationReadTypes for BufferedPostgresRelation<O> {
    type ReadGuard<'a> = PostgresBufferReadGuard<O>;
}

impl<O: Opaque> RelationWriteTypes for BufferedPostgresRelation<O> {
    type WriteGuard<'a> = PostgresBufferWriteGuard<O>;
}

impl<O: Opaque> RelationRead for BufferedPostgresRelation<O> {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.postgres.read(id)
    }
}

impl<O: Opaque> RelationWrite for BufferedPostgresRelation<O> {
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.postgres.write(id, tracking_freespace)
    }
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.postgres.search(freespace)
    }
    #[cfg(any(feature = "pg14", feature = "pg15"))]
    fn extend(&self, opaque: O, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.postgres.extend(opaque, tracking_freespace)
    }
    #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
    fn extend(&self, opaque: O, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, GENERIC_XLOG_FULL_IMAGE, GenericXLogRegisterBuffer,
                GenericXLogStart, LockBuffer,
            };
            use std::mem::MaybeUninit;
            use std::ptr::NonNull;
            let buf = {
                let mut list = self.list.borrow_mut();
                if list.is_empty() {
                    use pgrx::pg_sys::{BufferManagerRelation, ExtendBufferedRelBy, ForkNumber};
                    let bmr = BufferManagerRelation {
                        rel: self.postgres.raw,
                        smgr: std::ptr::null_mut(),
                        relpersistence: 0,
                    };
                    let mut len = 0_u32;
                    _ = ExtendBufferedRelBy(
                        bmr,
                        ForkNumber::MAIN_FORKNUM,
                        std::ptr::null_mut(),
                        0,
                        list.capacity() as _,
                        list.as_mut_ptr(),
                        &raw mut len,
                    );
                    assert!(1 <= len && len <= list.capacity() as _);
                    list.set_len(len as _);
                    list.reverse();
                }
                list.pop().expect("number of allocated pages is zero")
            };
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.postgres.raw);
            let mut page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage<O>>>(),
            )
            .expect("failed to get page");
            crate::index::storage::page_init(page.as_mut().as_mut_ptr(), opaque);
            PostgresBufferWriteGuard {
                raw: self.postgres.raw,
                buf,
                page: page.cast(),
                state,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
                tracking_freespace,
            }
        }
    }
}
