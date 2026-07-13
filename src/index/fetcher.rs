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
// Copyright (c) 2025-2026 TensorChord Inc.

use pgrx::pg_sys::{BlockIdData, Datum, ItemPointerData};
use std::cell::LazyCell;
use std::num::NonZero;
use std::ops::DerefMut;
use std::ptr::NonNull;

pub trait FilterableTuple: Tuple {
    fn filter(&mut self) -> bool;
}

pub trait Tuple {
    fn build(&mut self) -> (&[Datum; 32], &[bool; 32]);

    /// Read one user attribute from the already fetched heap slot.
    ///
    /// Callers that may expose the value outside PostgreSQL must call
    /// [`FilterableTuple::filter`] first.  Attribute numbers are PostgreSQL
    /// one-based `attnum` values, not zero-based Rust indexes.
    #[allow(dead_code, reason = "reserved for the optional Phase 3C heap source")]
    fn attribute(&mut self, attnum: i16) -> Option<TupleAttribute>;
}

#[derive(Clone, Copy)]
#[allow(dead_code, reason = "reserved for the optional Phase 3C heap source")]
pub struct TupleAttribute {
    pub datum: Datum,
    pub is_null: bool,
}

pub trait Fetcher {
    type Tuple<'a>: FilterableTuple
    where
        Self: 'a;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>>;
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
    heapfetch: *mut pgrx::pg_sys::IndexFetchTableData,
    owns_heapfetch: bool,
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
        heapfetch: *mut pgrx::pg_sys::IndexFetchTableData,
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
                heapfetch,
                owns_heapfetch: false,
                slot: pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut()),
                values: [Datum::null(); 32],
                is_nulls: [true; 32],
                hack,
            }
        }
    }

    /// Create a heap fetch state that is not owned by an `IndexScanDesc`.
    ///
    /// This is used by the restricted external MaxSim executor to resolve the
    /// root TIDs stored in the index through HOT chains before SQL-visible
    /// descriptor projection.  The table AM owns the rules for doing that;
    /// looking up `ctid` directly does not follow a HOT chain.
    pub unsafe fn new_standalone(
        index_relation: pgrx::pg_sys::Relation,
        heap_relation: pgrx::pg_sys::Relation,
        snapshot: pgrx::pg_sys::Snapshot,
    ) -> Self {
        use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;

        unsafe {
            let table_am = (*heap_relation).rd_tableam;
            if table_am.is_null() {
                panic!("unknown heap access method");
            }
            let index_fetch_begin = (*table_am)
                .index_fetch_begin
                .expect("unsupported heap access method");
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            let heapfetch = pg_guard_ffi_boundary(|| index_fetch_begin(heap_relation));
            if heapfetch.is_null() {
                panic!("heap access method returned a null index fetch state");
            }
            let mut fetcher = Self::new(
                index_relation,
                heap_relation,
                snapshot,
                heapfetch,
                std::ptr::null_mut(),
            );
            fetcher.owns_heapfetch = true;
            fetcher
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
            if self.owns_heapfetch {
                use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;

                let table_am = (*self.heap_relation).rd_tableam;
                let index_fetch_end = (*table_am)
                    .index_fetch_end
                    .expect("unsupported heap access method");
                #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                pg_guard_ffi_boundary(|| index_fetch_end(self.heapfetch));
            }
        }
    }
}

impl Fetcher for HeapFetcher {
    type Tuple<'a> = HeapTuple<'a>;

    fn fetch(&mut self, key: [u16; 3]) -> Option<Self::Tuple<'_>> {
        unsafe {
            use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;
            let mut ctid = key_to_ctid(key);
            let table_am = (*self.heap_relation).rd_tableam;
            if table_am.is_null() {
                panic!("unknown heap access method");
            }
            let index_fetch_tuple = (*table_am)
                .index_fetch_tuple
                .expect("unsupported heap access method");
            let found = 'a: {
                let mut call_again = false;
                let mut all_dead = false;
                #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                let found = pg_guard_ffi_boundary(|| {
                    index_fetch_tuple(
                        self.heapfetch,
                        &mut ctid,
                        self.snapshot,
                        self.slot,
                        &mut call_again,
                        &mut all_dead,
                    )
                });
                if found {
                    break 'a true;
                }
                while call_again {
                    #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
                    let found = pg_guard_ffi_boundary(|| {
                        index_fetch_tuple(
                            self.heapfetch,
                            &mut ctid,
                            self.snapshot,
                            self.slot,
                            &mut call_again,
                            &mut all_dead,
                        )
                    });
                    if found {
                        break 'a true;
                    }
                }
                false
            };
            if found {
                // The heap table AM rewrites the requested root TID to the
                // snapshot-visible HOT-chain member.  The slot itself keeps
                // the rewritten index TID as well, but carrying it explicitly
                // avoids depending on slot representation details.
                Some(HeapTuple {
                    this: self,
                    current_ctid: ctid,
                })
            } else {
                None
            }
        }
    }
}

pub struct HeapTuple<'a> {
    this: &'a mut HeapFetcher,
    current_ctid: ItemPointerData,
}

impl HeapTuple<'_> {
    /// Return the physical TID of the tuple version materialized in the slot.
    /// This may differ from the root TID supplied to `Fetcher::fetch` after a
    /// HOT update.
    pub fn ctid(&self) -> ItemPointerData {
        self.current_ctid
    }
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

    fn attribute(&mut self, attnum: i16) -> Option<TupleAttribute> {
        unsafe {
            use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;

            let slot = self.this.slot;
            let tuple_descriptor = (*slot).tts_tupleDescriptor;
            if attnum <= 0
                || tuple_descriptor.is_null()
                || i32::from(attnum) > (*tuple_descriptor).natts
            {
                return None;
            }
            #[allow(ffi_unwind_calls, reason = "protected by pg_guard_ffi_boundary")]
            pg_guard_ffi_boundary(|| {
                pgrx::pg_sys::slot_getsomeattrs_int(slot, i32::from(attnum));
            });
            let offset = usize::try_from(attnum - 1).ok()?;
            Some(TupleAttribute {
                datum: *(*slot).tts_values.add(offset),
                is_null: *(*slot).tts_isnull.add(offset),
            })
        }
    }
}

impl FilterableTuple for HeapTuple<'_> {
    fn filter(&mut self) -> bool {
        unsafe {
            use pgrx::pg_sys::ffi::pg_guard_ffi_boundary;
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
                                #[allow(
                                    ffi_unwind_calls,
                                    reason = "protected by pg_guard_ffi_boundary"
                                )]
                                let datum = pg_guard_ffi_boundary(|| {
                                    evalfunc(qual.as_ptr(), econtext, &mut is_null)
                                });
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
