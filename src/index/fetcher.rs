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

use pgrx::pg_sys::{BlockIdData, Datum, ItemPointerData};
use std::cell::LazyCell;
use std::num::NonZero;
use std::ops::DerefMut;
use std::ptr::NonNull;

pub trait Fetcher {
    type Tuple<'a>: Tuple
    where
        Self: 'a;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>>;
}

pub trait Tuple {
    fn build(&mut self) -> (&[Datum; 32], &[bool; 32]);
    fn filter(&mut self) -> bool;
}

impl<T: Fetcher, F: FnOnce() -> T> Fetcher for LazyCell<T, F> {
    type Tuple<'a>
        = T::Tuple<'a>
    where
        Self: 'a;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>> {
        self.deref_mut().fetch(key)
    }
}

pub struct HeapFetcher {
    index_info: *mut pgrx::pg_sys::IndexInfo,
    estate: *mut pgrx::pg_sys::EState,
    econtext: *mut pgrx::pg_sys::ExprContext,
    heap_relation: pgrx::pg_sys::Relation,
    snapshot: pgrx::pg_sys::Snapshot,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    values: [Datum; 32],
    is_nulls: [bool; 32],
    hack: *mut pgrx::pg_sys::IndexScanState,
}

impl HeapFetcher {
    pub unsafe fn new(
        index_relation: pgrx::pg_sys::Relation,
        heap_relation: pgrx::pg_sys::Relation,
        snapshot: pgrx::pg_sys::Snapshot,
        hack: *mut pgrx::pg_sys::IndexScanState,
    ) -> Self {
        unsafe {
            let index_info = pgrx::pg_sys::BuildIndexInfo(index_relation);
            let estate = pgrx::pg_sys::CreateExecutorState();
            let econtext = pgrx::pg_sys::MakePerTupleExprContext(estate);
            Self {
                index_info,
                estate,
                econtext,
                heap_relation,
                snapshot,
                slot: pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut()),
                values: [Datum::null(); 32],
                is_nulls: [true; 32],
                hack,
            }
        }
    }
}

impl Drop for HeapFetcher {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            // free common resources
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            pgrx::pg_sys::FreeExecutorState(self.estate);
        }
    }
}

impl Fetcher for HeapFetcher {
    type Tuple<'a> = HeapTuple<'a>;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>> {
        unsafe {
            let mut ctid = key_to_ctid(key);
            let table_am = (*self.heap_relation).rd_tableam;
            let fetch_row_version = (*table_am)
                .tuple_fetch_row_version
                .expect("unsupported heap access method");
            if !fetch_row_version(self.heap_relation, &mut ctid, self.snapshot, self.slot) {
                return None;
            }
            Some(HeapTuple { this: self })
        }
    }
}

pub struct HeapTuple<'a> {
    this: &'a mut HeapFetcher,
}

impl Tuple for HeapTuple<'_> {
    fn build(&mut self) -> (&[Datum; 32], &[bool; 32]) {
        unsafe {
            let this = &mut self.this;
            (*this.econtext).ecxt_scantuple = this.slot;
            pgrx::pg_sys::MemoryContextReset((*this.econtext).ecxt_per_tuple_memory);
            pgrx::pg_sys::FormIndexDatum(
                this.index_info,
                this.slot,
                this.estate,
                this.values.as_mut_ptr(),
                this.is_nulls.as_mut_ptr(),
            );
            (&this.values, &this.is_nulls)
        }
    }

    #[allow(clippy::collapsible_if)]
    fn filter(&mut self) -> bool {
        unsafe {
            let this = &mut self.this;
            if !this.hack.is_null() {
                if let Some(qual) = NonNull::new((*this.hack).ss.ps.qual) {
                    use pgrx::datum::FromDatum;
                    use pgrx::memcxt::PgMemoryContexts;
                    assert!(qual.as_ref().flags & pgrx::pg_sys::EEO_FLAG_IS_QUAL as u8 != 0);
                    let evalfunc = qual.as_ref().evalfunc.expect("no evalfunc for qual");
                    if !(*this.hack).ss.ps.ps_ExprContext.is_null() {
                        let econtext = (*this.hack).ss.ps.ps_ExprContext;
                        (*econtext).ecxt_scantuple = this.slot;
                        pgrx::pg_sys::MemoryContextReset((*econtext).ecxt_per_tuple_memory);
                        let result = PgMemoryContexts::For((*econtext).ecxt_per_tuple_memory)
                            .switch_to(|_| {
                                let mut is_null = true;
                                let datum = evalfunc(qual.as_ptr(), econtext, &mut is_null);
                                bool::from_datum(datum, is_null)
                            });
                        if result != Some(true) {
                            return false;
                        }
                    }
                }
            }
            true
        }
    }
}

pub const fn ctid_to_key(
    ItemPointerData {
        ip_blkid: BlockIdData { bi_hi, bi_lo },
        ip_posid,
    }: ItemPointerData,
) -> [u16; 3] {
    [bi_hi, bi_lo, ip_posid]
}

pub const fn key_to_ctid([bi_hi, bi_lo, ip_posid]: [u16; 3]) -> ItemPointerData {
    ItemPointerData {
        ip_blkid: BlockIdData { bi_hi, bi_lo },
        ip_posid,
    }
}

pub const fn pointer_to_kv(pointer: NonZero<u64>) -> ([u16; 3], u16) {
    let value = pointer.get();
    let bi_hi = ((value >> 48) & 0xffff) as u16;
    let bi_lo = ((value >> 32) & 0xffff) as u16;
    let ip_posid = ((value >> 16) & 0xffff) as u16;
    let extra = value as u16;
    ([bi_hi, bi_lo, ip_posid], extra)
}

pub const fn kv_to_pointer((key, value): ([u16; 3], u16)) -> NonZero<u64> {
    let x = (key[0] as u64) << 48 | (key[1] as u64) << 32 | (key[2] as u64) << 16 | value as u64;
    NonZero::new(x).expect("invalid key")
}
