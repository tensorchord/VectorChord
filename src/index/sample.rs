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

use pgrx::pg_sys::{Datum, ItemPointerData};
use std::ptr::NonNull;

pub trait Tuple {
    #[expect(dead_code)]
    fn id(&mut self) -> ItemPointerData;
    fn build(&mut self) -> (&[Datum; 32], &[bool; 32]);
}

pub trait Sample {
    type Tuple<'a>: Tuple
    where
        Self: 'a;

    fn next(&mut self) -> Option<Self::Tuple<'_>>;
}

pub trait Sampler {
    type Sample: Sample;

    fn sample(&self) -> Self::Sample;
}

pub struct HeapSampler {
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
    snapshot: pgrx::pg_sys::Snapshot,
}

impl HeapSampler {
    pub unsafe fn new(
        index_relation: pgrx::pg_sys::Relation,
        heap_relation: pgrx::pg_sys::Relation,
        snapshot: pgrx::pg_sys::Snapshot,
    ) -> Self {
        Self {
            heap_relation,
            index_relation,
            snapshot,
        }
    }
}

impl Drop for HeapSampler {
    fn drop(&mut self) {}
}

impl Sampler for HeapSampler {
    type Sample = HeapSample;

    fn sample(&self) -> Self::Sample {
        unsafe {
            let state = NonNull::new_unchecked(Box::into_raw(Box::new(State {
                blocks: None,
                tuples: None,
            })));
            let sample_scan_state =
                NonNull::new_unchecked(Box::into_raw(Box::new(pgrx::pg_sys::SampleScanState {
                    tsmroutine: (&raw const TSM.0).cast_mut().cast(),
                    tsm_state: state.as_ptr().cast(),
                    ..core::mem::zeroed()
                })));
            let table_scan_desc = pgrx::pg_sys::table_beginscan_sampling(
                self.heap_relation,
                self.snapshot,
                0,
                std::ptr::null_mut(),
                true,
                false,
                true,
            );
            let index_info = pgrx::pg_sys::BuildIndexInfo(self.index_relation);
            let estate = pgrx::pg_sys::CreateExecutorState();
            let econtext = pgrx::pg_sys::MakePerTupleExprContext(estate);
            HeapSample {
                index_info,
                estate,
                econtext,
                slot: pgrx::pg_sys::table_slot_create(self.heap_relation, std::ptr::null_mut()),
                values: [Datum::null(); 32],
                is_nulls: [true; 32],
                state,
                sample_scan_state,
                table_scan_desc,
                done: false,
                have_block: false,
            }
        }
    }
}

pub struct HeapSample {
    index_info: *mut pgrx::pg_sys::IndexInfo,
    estate: *mut pgrx::pg_sys::EState,
    econtext: *mut pgrx::pg_sys::ExprContext,
    slot: *mut pgrx::pg_sys::TupleTableSlot,
    values: [Datum; 32],
    is_nulls: [bool; 32],
    state: NonNull<State>,
    sample_scan_state: NonNull<pgrx::pg_sys::SampleScanState>,
    table_scan_desc: pgrx::pg_sys::TableScanDesc,
    done: bool,
    have_block: bool,
}

impl Drop for HeapSample {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            // free common resources
            pgrx::pg_sys::table_endscan(self.table_scan_desc);
            let _ = Box::from_raw(self.sample_scan_state.as_ptr());
            let _ = Box::from_raw(self.state.as_ptr());
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            pgrx::pg_sys::FreeExecutorState(self.estate);
        }
    }
}

impl Sample for HeapSample {
    type Tuple<'a> = HeapTuple<'a>;

    fn next(&mut self) -> Option<Self::Tuple<'_>> {
        unsafe {
            use pgrx::pg_sys::{table_scan_sample_next_block, table_scan_sample_next_tuple};

            if self.done {
                return None;
            }

            loop {
                if !self.have_block {
                    if !table_scan_sample_next_block(
                        self.table_scan_desc,
                        self.sample_scan_state.as_ptr(),
                    ) {
                        self.have_block = false;
                        self.done = true;
                        return None;
                    }

                    self.have_block = true;
                }

                if !table_scan_sample_next_tuple(
                    self.table_scan_desc,
                    self.sample_scan_state.as_ptr(),
                    self.slot,
                ) {
                    self.have_block = false;
                    continue;
                }

                break;
            }

            Some(HeapTuple { this: self })
        }
    }
}

pub struct HeapTuple<'a> {
    this: &'a mut HeapSample,
}

impl Tuple for HeapTuple<'_> {
    fn id(&mut self) -> ItemPointerData {
        unsafe {
            let this = &mut self.this;
            (*this.slot).tts_tid
        }
    }
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
}

fn sample(n: u32) -> Box<dyn Iterator<Item = u32>> {
    let width = (n.ilog2() + 1).next_multiple_of(2);
    let key_0 = rand::Rng::random(&mut rand::rng());
    let key_1 = rand::Rng::random(&mut rand::rng());
    let secret = move |round: u32, x: u32| {
        let buffer = [round.to_le_bytes(), x.to_le_bytes(), key_0, key_1];
        wyhash::wyhash(buffer.as_flattened(), 0) as u32
    };
    let permutation = (0..1 << width)
        .map(move |i| feistel::feistel(width, i, 8, secret))
        .filter(move |&x| x < n);
    Box::new(permutation)
}

pub struct State {
    blocks: Option<Box<dyn Iterator<Item = u32>>>,
    tuples: Option<std::ops::RangeInclusive<u16>>,
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn feistel_rows_nextsampleblock(
    node: *mut pgrx::pg_sys::SampleScanState,
    nblocks: pgrx::pg_sys::BlockNumber,
) -> pgrx::pg_sys::BlockNumber {
    let state: &mut State = unsafe { &mut *(*node).tsm_state.cast() };
    let iter = state.blocks.get_or_insert_with(|| sample(nblocks));
    if let Some(number) = iter.next() {
        number
    } else {
        pgrx::pg_sys::InvalidBlockNumber
    }
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn feistel_rows_nextsampletuple(
    node: *mut pgrx::pg_sys::SampleScanState,
    _blockno: pgrx::pg_sys::BlockNumber,
    maxoffset: pgrx::pg_sys::OffsetNumber,
) -> pgrx::pg_sys::OffsetNumber {
    let state: &mut State = unsafe { &mut *(*node).tsm_state.cast() };
    let iter = state.tuples.get_or_insert(1..=maxoffset);
    if let Some(number) = iter.next() {
        number
    } else {
        state.tuples = None;
        pgrx::pg_sys::InvalidOffsetNumber
    }
}

struct AssertSync<T>(T);

unsafe impl<T> Sync for AssertSync<T> {}

static TSM: AssertSync<sys::TsmRoutine> = AssertSync(sys::TsmRoutine {
    type_: pgrx::pg_sys::NodeTag::T_TsmRoutine,
    NextSampleBlock: Some(feistel_rows_nextsampleblock),
    NextSampleTuple: Some(feistel_rows_nextsampletuple),
    ..unsafe { core::mem::zeroed() }
});

#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
mod sys {
    #[cfg(not(feature = "pg14"))]
    #[cfg(not(feature = "pg15"))]
    #[cfg(not(feature = "pg16"))]
    #[cfg(not(feature = "pg17"))]
    #[cfg(not(feature = "pg18"))]
    compile_error!("bindings are not checked");

    use core::ffi::c_int;
    use pgrx::pg_sys::{
        BlockNumber, Datum, List, NodeTag, OffsetNumber, PlannerInfo, RelOptInfo, SampleScanState,
    };

    pub type SampleScanGetSampleSize_function = Option<
        unsafe extern "C-unwind" fn(
            root: *mut PlannerInfo,
            baserel: *mut RelOptInfo,
            paramexprs: *mut List,
            pages: *mut BlockNumber,
            tuples: *mut f64,
        ),
    >;

    pub type InitSampleScan_function =
        Option<unsafe extern "C-unwind" fn(node: *mut SampleScanState, eflags: c_int)>;

    pub type BeginSampleScan_function = Option<
        unsafe extern "C-unwind" fn(
            node: *mut SampleScanState,
            params: *mut Datum,
            nparams: c_int,
            seed: u32,
        ),
    >;

    pub type NextSampleBlock_function = Option<
        unsafe extern "C-unwind" fn(
            node: *mut SampleScanState,
            nblocks: BlockNumber,
        ) -> BlockNumber,
    >;

    pub type NextSampleTuple_function = Option<
        unsafe extern "C-unwind" fn(
            node: *mut SampleScanState,
            blockno: BlockNumber,
            maxoffset: OffsetNumber,
        ) -> OffsetNumber,
    >;

    pub type EndSampleScan_function =
        Option<unsafe extern "C-unwind" fn(node: *mut SampleScanState)>;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct TsmRoutine {
        pub type_: NodeTag,
        pub parameterTypes: *mut List,
        pub repeatable_across_queries: bool,
        pub repeatable_across_scans: bool,
        pub SampleScanGetSampleSize: SampleScanGetSampleSize_function,
        pub InitSampleScan: InitSampleScan_function,
        pub BeginSampleScan: BeginSampleScan_function,
        pub NextSampleBlock: NextSampleBlock_function,
        pub NextSampleTuple: NextSampleTuple_function,
        pub EndSampleScan: EndSampleScan_function,
    }
}
