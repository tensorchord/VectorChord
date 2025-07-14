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

use crate::datatype::typmod::Typmod;
use crate::index::fetcher::*;
use crate::index::storage::{PostgresPage, PostgresRelation};
use crate::index::vchordrq::am::Reloption;
use crate::index::vchordrq::opclass::{Opfamily, opfamily};
use crate::index::vchordrq::types::*;
use algo::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use half::f16;
use pgrx::pg_sys::{Datum, ItemPointerData};
use rand::Rng;
use simd::Floating;
use std::ffi::CStr;
use std::ops::Deref;
use vchordrq::types::*;
use vector::VectorOwned;
use vector::vect::VectOwned;

#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum BuildPhaseCode {
    Initializing = 0,
    InternalBuild = 1,
    ExternalBuild = 2,
    Build = 3,
    Inserting = 4,
    Compacting = 5,
}

pub struct BuildPhase(BuildPhaseCode, u16);

impl BuildPhase {
    pub const fn new(code: BuildPhaseCode, k: u16) -> Option<Self> {
        match (code, k) {
            (BuildPhaseCode::Initializing, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::InternalBuild, 0..102) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::ExternalBuild, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::Build, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::Inserting, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::Compacting, 0) => Some(BuildPhase(code, k)),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static CStr {
        match self {
            BuildPhase(BuildPhaseCode::Initializing, k) => {
                static RAW: [&CStr; 1] = [c"initializing"];
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::InternalBuild, k) => {
                static RAWS: [&[&CStr]; 2] = [
                    &[c"initializing index, by internal build"],
                    seq_macro::seq!(
                        N in 0..=100 {
                            &[#(
                                const {
                                    let bytes = concat!("initializing index, by internal build (", N, " %)\0").as_bytes();
                                    if let Ok(cstr) = CStr::from_bytes_with_nul(bytes) {
                                        cstr
                                    } else {
                                        unreachable!()
                                    }
                                },
                            )*]
                        }
                    ),
                ];
                static RAW: [&CStr; 102] = {
                    let mut result = [c""; 102];
                    let mut offset = 0_usize;
                    let mut i = 0_usize;
                    while i < RAWS.len() {
                        let mut j = 0_usize;
                        while j < RAWS[i].len() {
                            {
                                result[offset] = RAWS[i][j];
                                offset += 1;
                            }
                            j += 1;
                        }
                        i += 1;
                    }
                    assert!(offset == result.len());
                    result
                };
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::ExternalBuild, k) => {
                static RAW: [&CStr; 1] = [c"initializing index, by external build"];
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::Build, k) => {
                static RAW: [&CStr; 1] = [c"initializing index"];
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::Inserting, k) => {
                static RAW: [&CStr; 1] = [c"inserting tuples from table to index"];
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::Compacting, k) => {
                static RAW: [&CStr; 1] = [c"compacting tuples in index"];
                RAW[k as usize]
            }
        }
    }
    pub const fn from_code(code: BuildPhaseCode) -> Self {
        Self(code, 0)
    }
    pub const fn from_value(value: u32) -> Option<Self> {
        const INITIALIZING: u16 = BuildPhaseCode::Initializing as _;
        const INTERNAL_BUILD: u16 = BuildPhaseCode::InternalBuild as _;
        const EXTERNAL_BUILD: u16 = BuildPhaseCode::ExternalBuild as _;
        const BUILD: u16 = BuildPhaseCode::Build as _;
        const INSERTING: u16 = BuildPhaseCode::Inserting as _;
        const COMPACTING: u16 = BuildPhaseCode::Compacting as _;
        let k = value as u16;
        match (value >> 16) as u16 {
            INITIALIZING => Self::new(BuildPhaseCode::Initializing, k),
            INTERNAL_BUILD => Self::new(BuildPhaseCode::InternalBuild, k),
            EXTERNAL_BUILD => Self::new(BuildPhaseCode::ExternalBuild, k),
            BUILD => Self::new(BuildPhaseCode::Build, k),
            INSERTING => Self::new(BuildPhaseCode::Inserting, k),
            COMPACTING => Self::new(BuildPhaseCode::Compacting, k),
            _ => None,
        }
    }
    pub const fn into_value(self) -> u32 {
        (self.0 as u32) << 16 | (self.1 as u32)
    }
}

#[pgrx::pg_guard]
pub extern "C-unwind" fn ambuildphasename(x: i64) -> *mut core::ffi::c_char {
    if let Ok(x) = u32::try_from(x.wrapping_sub(1)) {
        if let Some(x) = BuildPhase::from_value(x) {
            x.name().as_ptr().cast_mut()
        } else {
            std::ptr::null_mut()
        }
    } else {
        std::ptr::null_mut()
    }
}

#[derive(Debug, Clone)]
struct Heap {
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    opfamily: Opfamily,
    scan: *mut pgrx::pg_sys::TableScanDescData,
}

impl Heap {
    fn traverse<F: FnMut((ItemPointerData, Vec<(OwnedVector, u16)>))>(
        &self,
        progress: bool,
        callback: F,
    ) {
        pub struct State<'a, F> {
            pub this: &'a Heap,
            pub callback: F,
        }
        #[pgrx::pg_guard]
        unsafe extern "C-unwind" fn call<F>(
            _index_relation: pgrx::pg_sys::Relation,
            ctid: pgrx::pg_sys::ItemPointer,
            values: *mut Datum,
            is_null: *mut bool,
            _tuple_is_alive: bool,
            state: *mut core::ffi::c_void,
        ) where
            F: FnMut((ItemPointerData, Vec<(OwnedVector, u16)>)),
        {
            let state = unsafe { &mut *state.cast::<State<F>>() };
            let opfamily = state.this.opfamily;
            let datum = unsafe { (!is_null.add(0).read()).then_some(values.add(0).read()) };
            let ctid = unsafe { ctid.read() };
            if let Some(store) = unsafe { datum.and_then(|x| opfamily.store(x)) } {
                (state.callback)((ctid, store));
            }
        }
        let table_am = unsafe { &*(*self.heap_relation).rd_tableam };
        let mut state = State {
            this: self,
            callback,
        };
        unsafe {
            table_am.index_build_range_scan.unwrap()(
                self.heap_relation,
                self.index_relation,
                self.index_info,
                true,
                false,
                progress,
                0,
                pgrx::pg_sys::InvalidBlockNumber,
                Some(call::<F>),
                (&mut state) as *mut State<F> as *mut _,
                self.scan,
            );
        }
    }
}

#[derive(Debug, Clone)]
struct PostgresReporter {}

impl PostgresReporter {
    fn phase(&mut self, phase: BuildPhase) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_SUBPHASE as _,
                (phase.into_value() as i64) + 1,
            );
        }
    }
    fn tuples_total(&mut self, tuples_total: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_TOTAL as _,
                tuples_total as _,
            );
        }
    }
    fn tuples_done(&mut self, tuples_done: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_DONE as _,
                tuples_done as _,
            );
        }
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap_relation: pgrx::pg_sys::Relation,
    index_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
) -> *mut pgrx::pg_sys::IndexBuildResult {
    use validator::Validate;
    crate::index::capture::QueryLoggerMaster::init();
    let (vector_options, vchordrq_options) = unsafe { options(index_relation) };
    if let Err(errors) = Validate::validate(&vector_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    if let Err(errors) = Validate::validate(&vchordrq_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    let opfamily = unsafe { opfamily(index_relation) };
    let index = unsafe { PostgresRelation::new(index_relation) };
    let heap = Heap {
        heap_relation,
        index_relation,
        index_info,
        opfamily,
        scan: std::ptr::null_mut(),
    };
    let mut reporter = PostgresReporter {};
    let structures = match vchordrq_options.build.source.clone() {
        VchordrqBuildSourceOptions::External(external_build) => {
            reporter.phase(BuildPhase::from_code(BuildPhaseCode::ExternalBuild));
            let reltuples = unsafe { (*(*index_relation).rd_rel).reltuples };
            reporter.tuples_total(reltuples as u64);
            make_external_build(vector_options.clone(), opfamily, external_build.clone())
        }
        VchordrqBuildSourceOptions::Internal(internal_build) => {
            reporter.phase(BuildPhase::from_code(BuildPhaseCode::InternalBuild));
            let mut tuples_total = 0_u64;
            let samples = 'a: {
                let mut rand = rand::rng();
                let Some(max_number_of_samples) = internal_build
                    .lists
                    .last()
                    .map(|x| x.saturating_mul(internal_build.sampling_factor))
                else {
                    break 'a Vec::new();
                };
                let mut samples = Vec::new();
                let mut number_of_samples = 0_u32;
                heap.traverse(false, |(_, store)| {
                    for (vector, _) in store {
                        let x = match vector {
                            OwnedVector::Vecf32(x) => VectOwned::build_to_vecf32(x.as_borrowed()),
                            OwnedVector::Vecf16(x) => VectOwned::build_to_vecf32(x.as_borrowed()),
                        };
                        assert_eq!(
                            vector_options.dims,
                            x.len() as u32,
                            "invalid vector dimensions"
                        );
                        if number_of_samples < max_number_of_samples {
                            samples.push(x);
                            number_of_samples += 1;
                        } else {
                            let index = rand.random_range(0..max_number_of_samples) as usize;
                            samples[index] = x;
                        }
                    }
                    tuples_total += 1;
                });
                samples
            };
            reporter.tuples_total(tuples_total);
            make_internal_build(
                vector_options.clone(),
                internal_build.clone(),
                samples,
                &mut reporter,
            )
        }
    };
    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Build));
    crate::index::vchordrq::algo::build(vector_options, vchordrq_options.index, &index, structures);
    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Inserting));
    let cache = if vchordrq_options.build.pin {
        let mut trace = vchordrq::cache(&index);
        trace.sort();
        trace.dedup();
        if let Some(max) = trace.last().copied() {
            let mut mapping = vec![u32::MAX; 1 + max as usize];
            let mut pages = Vec::<Box<PostgresPage<vchordrq::Opaque>>>::with_capacity(trace.len());
            for id in trace {
                mapping[id as usize] = pages.len() as u32;
                pages.push(index.read(id).clone_into_boxed());
            }
            vchordrq_cached::VchordrqCached::_1 { mapping, pages }
        } else {
            vchordrq_cached::VchordrqCached::_0 {}
        }
    } else {
        vchordrq_cached::VchordrqCached::_0 {}
    };
    if let Some(leader) = unsafe {
        VchordrqLeader::enter(
            heap_relation,
            index_relation,
            (*index_info).ii_Concurrent,
            cache,
        )
    } {
        unsafe {
            parallel_build(
                index_relation,
                heap_relation,
                index_info,
                leader.tablescandesc,
                leader.vchordrqshared,
                leader.vchordrqcached,
                |indtuples| {
                    reporter.tuples_done(indtuples);
                },
            );
            leader.wait();
            let nparticipants = leader.nparticipants;
            loop {
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*leader.vchordrqshared).mutex);
                if (*leader.vchordrqshared).nparticipantsdone == nparticipants {
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordrqshared).mutex);
                    break;
                }
                pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordrqshared).mutex);
                pgrx::pg_sys::ConditionVariableSleep(
                    &raw mut (*leader.vchordrqshared).workersdonecv,
                    pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                );
            }
            reporter.tuples_done((*leader.vchordrqshared).indtuples);
            reporter.tuples_total((*leader.vchordrqshared).indtuples);
            pgrx::pg_sys::ConditionVariableCancelSleep();
        }
    } else {
        let mut indtuples = 0;
        reporter.tuples_done(indtuples);
        heap.traverse(true, |(ctid, store)| {
            for (vector, extra) in store {
                let key = ctid_to_key(ctid);
                let payload = kv_to_pointer((key, extra));
                crate::index::vchordrq::algo::insert(opfamily, &index, payload, vector, true);
            }
            indtuples += 1;
            reporter.tuples_done(indtuples);
        });
        reporter.tuples_total(indtuples);
    }
    let check = || {
        pgrx::check_for_interrupts!();
    };
    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Compacting));
    crate::index::vchordrq::algo::maintain(opfamily, &index, check);
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

struct VchordrqShared {
    /* Immutable state */
    heaprelid: pgrx::pg_sys::Oid,
    indexrelid: pgrx::pg_sys::Oid,
    isconcurrent: bool,

    /* Worker progress */
    workersdonecv: pgrx::pg_sys::ConditionVariable,

    /* Mutex for mutable state */
    mutex: pgrx::pg_sys::slock_t,

    /* Mutable state */
    nparticipantsdone: i32,
    indtuples: u64,
}

mod vchordrq_cached {
    pub const ALIGN: usize = 8;
    pub type Tag = u64;

    use crate::index::storage::PostgresPage;
    use algo::tuples::RefChecker;
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct VchordrqCachedHeader0 {}

    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct VchordrqCachedHeader1 {
        mapping_s: usize,
        mapping_e: usize,
        pages_s: usize,
        pages_e: usize,
    }

    pub enum VchordrqCached {
        _0 {},
        _1 {
            mapping: Vec<u32>,
            pages: Vec<Box<PostgresPage<vchordrq::Opaque>>>,
        },
    }

    impl VchordrqCached {
        pub fn serialize(&self) -> Vec<u8> {
            let mut buffer = Vec::new();
            match self {
                VchordrqCached::_0 {} => {
                    buffer.extend((0 as Tag).to_ne_bytes());
                    buffer.extend(std::iter::repeat_n(0, size_of::<VchordrqCachedHeader0>()));
                    buffer[size_of::<Tag>()..][..size_of::<VchordrqCachedHeader0>()]
                        .copy_from_slice(VchordrqCachedHeader0 {}.as_bytes());
                }
                VchordrqCached::_1 { mapping, pages } => {
                    buffer.extend((1 as Tag).to_ne_bytes());
                    buffer.extend(std::iter::repeat_n(0, size_of::<VchordrqCachedHeader1>()));
                    let mapping_s = buffer.len();
                    buffer.extend(mapping.as_bytes());
                    let mapping_e = buffer.len();
                    while buffer.len() % ALIGN != 0 {
                        buffer.push(0u8);
                    }
                    let pages_s = buffer.len();
                    buffer.extend(pages.iter().flat_map(|x| unsafe {
                        std::mem::transmute::<&PostgresPage<vchordrq::Opaque>, &[u8; 8192]>(
                            x.as_ref(),
                        )
                    }));
                    let pages_e = buffer.len();
                    while buffer.len() % ALIGN != 0 {
                        buffer.push(0u8);
                    }
                    buffer[size_of::<Tag>()..][..size_of::<VchordrqCachedHeader1>()]
                        .copy_from_slice(
                            VchordrqCachedHeader1 {
                                mapping_s,
                                mapping_e,
                                pages_s,
                                pages_e,
                            }
                            .as_bytes(),
                        );
                }
            }
            buffer
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum VchordrqCachedReader<'a> {
        #[allow(dead_code)]
        _0(VchordrqCachedReader0<'a>),
        _1(VchordrqCachedReader1<'a>),
    }

    #[derive(Debug, Clone, Copy)]
    pub struct VchordrqCachedReader0<'a> {
        #[allow(dead_code)]
        header: &'a VchordrqCachedHeader0,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct VchordrqCachedReader1<'a> {
        #[allow(dead_code)]
        header: &'a VchordrqCachedHeader1,
        mapping: &'a [u32],
        pages: &'a [PostgresPage<vchordrq::Opaque>],
    }

    impl<'a> VchordrqCachedReader1<'a> {
        pub fn get(&self, id: u32) -> Option<&'a PostgresPage<vchordrq::Opaque>> {
            let index = *self.mapping.get(id as usize)?;
            if index == u32::MAX {
                return None;
            }
            Some(&self.pages[index as usize])
        }
    }

    impl<'a> VchordrqCachedReader<'a> {
        pub fn deserialize_ref(source: &'a [u8]) -> Self {
            let tag = u64::from_ne_bytes(std::array::from_fn(|i| source[i]));
            match tag {
                0 => {
                    let checker = RefChecker::new(source);
                    let header: &VchordrqCachedHeader0 = checker.prefix(size_of::<Tag>());
                    Self::_0(VchordrqCachedReader0 { header })
                }
                1 => {
                    let checker = RefChecker::new(source);
                    let header: &VchordrqCachedHeader1 = checker.prefix(size_of::<Tag>());
                    let mapping = checker.bytes(header.mapping_s, header.mapping_e);
                    let pages =
                        unsafe { checker.bytes_slice_unchecked(header.pages_s, header.pages_e) };
                    Self::_1(VchordrqCachedReader1 {
                        header,
                        mapping,
                        pages,
                    })
                }
                _ => panic!("bad bytes"),
            }
        }
    }
}

fn is_mvcc_snapshot(snapshot: *mut pgrx::pg_sys::SnapshotData) -> bool {
    matches!(
        unsafe { (*snapshot).snapshot_type },
        pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
            | pgrx::pg_sys::SnapshotType::SNAPSHOT_HISTORIC_MVCC
    )
}

struct VchordrqLeader {
    pcxt: *mut pgrx::pg_sys::ParallelContext,
    nparticipants: i32,
    snapshot: pgrx::pg_sys::Snapshot,
    vchordrqshared: *mut VchordrqShared,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    vchordrqcached: *const u8,
}

impl VchordrqLeader {
    pub unsafe fn enter(
        heap_relation: pgrx::pg_sys::Relation,
        index_relation: pgrx::pg_sys::Relation,
        isconcurrent: bool,
        cache: vchordrq_cached::VchordrqCached,
    ) -> Option<Self> {
        let _cache = cache.serialize();
        drop(cache);
        let cache = _cache;

        unsafe fn compute_parallel_workers(
            heap_relation: pgrx::pg_sys::Relation,
            index_relation: pgrx::pg_sys::Relation,
        ) -> i32 {
            unsafe {
                if pgrx::pg_sys::plan_create_index_workers(
                    (*heap_relation).rd_id,
                    (*index_relation).rd_id,
                ) == 0
                {
                    return 0;
                }
                if !(*heap_relation).rd_options.is_null() {
                    let std_options = (*heap_relation)
                        .rd_options
                        .cast::<pgrx::pg_sys::StdRdOptions>();
                    std::cmp::min(
                        (*std_options).parallel_workers,
                        pgrx::pg_sys::max_parallel_maintenance_workers,
                    )
                } else {
                    pgrx::pg_sys::max_parallel_maintenance_workers
                }
            }
        }

        let request = unsafe { compute_parallel_workers(heap_relation, index_relation) };
        if request <= 0 {
            return None;
        }

        unsafe {
            pgrx::pg_sys::EnterParallelMode();
        }
        let pcxt = unsafe {
            pgrx::pg_sys::CreateParallelContext(
                c"vchord".as_ptr(),
                c"vchordrq_parallel_build_main".as_ptr(),
                request,
            )
        };

        let snapshot = if isconcurrent {
            unsafe { pgrx::pg_sys::RegisterSnapshot(pgrx::pg_sys::GetTransactionSnapshot()) }
        } else {
            &raw mut pgrx::pg_sys::SnapshotAnyData
        };

        fn estimate_chunk(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.space_for_chunks += x.next_multiple_of(pgrx::pg_sys::ALIGNOF_BUFFER as _);
        }
        fn estimate_keys(e: &mut pgrx::pg_sys::shm_toc_estimator, x: usize) {
            e.number_of_keys += x;
        }
        let est_tablescandesc =
            unsafe { pgrx::pg_sys::table_parallelscan_estimate(heap_relation, snapshot) };
        unsafe {
            estimate_chunk(&mut (*pcxt).estimator, size_of::<VchordrqShared>());
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, est_tablescandesc);
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, 8 + cache.len());
            estimate_keys(&mut (*pcxt).estimator, 1);
        }

        unsafe {
            pgrx::pg_sys::InitializeParallelDSM(pcxt);
            if (*pcxt).seg.is_null() {
                if is_mvcc_snapshot(snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        let vchordrqshared = unsafe {
            let vchordrqshared =
                pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, size_of::<VchordrqShared>())
                    .cast::<VchordrqShared>();
            vchordrqshared.write(VchordrqShared {
                heaprelid: (*heap_relation).rd_id,
                indexrelid: (*index_relation).rd_id,
                isconcurrent,
                workersdonecv: std::mem::zeroed(),
                mutex: std::mem::zeroed(),
                nparticipantsdone: 0,
                indtuples: 0,
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).workersdonecv);
            pgrx::pg_sys::SpinLockInit(&raw mut (*vchordrqshared).mutex);
            vchordrqshared
        };

        let tablescandesc = unsafe {
            let tablescandesc = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_tablescandesc)
                .cast::<pgrx::pg_sys::ParallelTableScanDescData>();
            pgrx::pg_sys::table_parallelscan_initialize(heap_relation, tablescandesc, snapshot);
            tablescandesc
        };

        let vchordrqcached = unsafe {
            let x = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, 8 + cache.len()).cast::<u8>();
            (x as *mut u64).write_unaligned(cache.len() as _);
            std::ptr::copy(cache.as_ptr(), x.add(8), cache.len());
            x
        };

        unsafe {
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000001, vchordrqshared.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000002, tablescandesc.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000003, vchordrqcached.cast());
        }

        unsafe {
            pgrx::pg_sys::LaunchParallelWorkers(pcxt);
        }

        let nworkers_launched = unsafe { (*pcxt).nworkers_launched };

        unsafe {
            if nworkers_launched == 0 {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(pcxt);
                if is_mvcc_snapshot(snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(pcxt);
                pgrx::pg_sys::ExitParallelMode();
                return None;
            }
        }

        Some(Self {
            pcxt,
            nparticipants: nworkers_launched + 1,
            snapshot,
            vchordrqshared,
            tablescandesc,
            vchordrqcached,
        })
    }

    pub fn wait(&self) {
        unsafe {
            pgrx::pg_sys::WaitForParallelWorkersToAttach(self.pcxt);
        }
    }
}

impl Drop for VchordrqLeader {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            unsafe {
                pgrx::pg_sys::WaitForParallelWorkersToFinish(self.pcxt);
                if is_mvcc_snapshot(self.snapshot) {
                    pgrx::pg_sys::UnregisterSnapshot(self.snapshot);
                }
                pgrx::pg_sys::DestroyParallelContext(self.pcxt);
                pgrx::pg_sys::ExitParallelMode();
            }
        }
    }
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vchordrq_parallel_build_main(
    _seg: *mut pgrx::pg_sys::dsm_segment,
    toc: *mut pgrx::pg_sys::shm_toc,
) {
    let vchordrqshared = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000001, false).cast::<VchordrqShared>()
    };
    let tablescandesc = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000002, false)
            .cast::<pgrx::pg_sys::ParallelTableScanDescData>()
    };
    let vchordrqcached = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000003, false)
            .cast::<u8>()
            .cast_const()
    };
    let heap_lockmode;
    let index_lockmode;
    if unsafe { !(*vchordrqshared).isconcurrent } {
        heap_lockmode = pgrx::pg_sys::ShareLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE;
    } else {
        heap_lockmode = pgrx::pg_sys::ShareUpdateExclusiveLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::RowExclusiveLock as pgrx::pg_sys::LOCKMODE;
    }
    let heap = unsafe { pgrx::pg_sys::table_open((*vchordrqshared).heaprelid, heap_lockmode) };
    let index = unsafe { pgrx::pg_sys::index_open((*vchordrqshared).indexrelid, index_lockmode) };
    let index_info = unsafe { pgrx::pg_sys::BuildIndexInfo(index) };
    unsafe {
        (*index_info).ii_Concurrent = (*vchordrqshared).isconcurrent;
    }

    unsafe {
        parallel_build(
            index,
            heap,
            index_info,
            tablescandesc,
            vchordrqshared,
            vchordrqcached,
            |_| (),
        );
    }

    unsafe {
        pgrx::pg_sys::index_close(index, index_lockmode);
        pgrx::pg_sys::table_close(heap, heap_lockmode);
    }
}

unsafe fn parallel_build(
    index_relation: pgrx::pg_sys::Relation,
    heap_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    vchordrqshared: *mut VchordrqShared,
    vchordrqcached: *const u8,
    mut callback: impl FnMut(u64),
) {
    use vchordrq_cached::VchordrqCachedReader;
    let cached = VchordrqCachedReader::deserialize_ref(unsafe {
        let bytes = (vchordrqcached as *const u64).read_unaligned();
        std::slice::from_raw_parts(vchordrqcached.add(8), bytes as _)
    });

    let index = unsafe { PostgresRelation::new(index_relation) };

    let scan = unsafe { pgrx::pg_sys::table_beginscan_parallel(heap_relation, tablescandesc) };
    let opfamily = unsafe { opfamily(index_relation) };
    let heap = Heap {
        heap_relation,
        index_relation,
        index_info,
        opfamily,
        scan,
    };
    match cached {
        VchordrqCachedReader::_0(_) => {
            heap.traverse(true, |(ctid, store)| {
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    crate::index::vchordrq::algo::insert(opfamily, &index, payload, vector, true);
                }
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    callback(indtuples);
                }
            });
        }
        VchordrqCachedReader::_1(cached) => {
            let index = CachingRelation {
                cache: cached,
                relation: index,
            };
            heap.traverse(true, |(ctid, store)| {
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    crate::index::vchordrq::algo::insert(opfamily, &index, payload, vector, true);
                }
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    callback(indtuples);
                }
            });
        }
    }
    unsafe {
        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
        (*vchordrqshared).nparticipantsdone += 1;
        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
        pgrx::pg_sys::ConditionVariableSignal(&raw mut (*vchordrqshared).workersdonecv);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambuildempty(_index_relation: pgrx::pg_sys::Relation) {
    pgrx::error!("Unlogged indexes are not supported.");
}

unsafe fn options(
    index_relation: pgrx::pg_sys::Relation,
) -> (VectorOptions, VchordrqIndexingOptions) {
    let att = unsafe { &mut *(*index_relation).rd_att };
    #[cfg(any(
        feature = "pg13",
        feature = "pg14",
        feature = "pg15",
        feature = "pg16",
        feature = "pg17"
    ))]
    let atts = unsafe { att.attrs.as_slice(att.natts as _) };
    #[cfg(feature = "pg18")]
    let atts = unsafe {
        let ptr = att
            .compact_attrs
            .as_ptr()
            .add(att.natts as _)
            .cast::<pgrx::pg_sys::FormData_pg_attribute>();
        std::slice::from_raw_parts(ptr, att.natts as _)
    };
    if atts.is_empty() {
        pgrx::error!("indexing on no columns is not supported");
    }
    if atts.len() != 1 {
        pgrx::error!("multicolumn index is not supported");
    }
    // get dims
    let typmod = Typmod::parse_from_i32(atts[0].atttypmod).unwrap();
    let dims = if let Some(dims) = typmod.dims() {
        dims.get()
    } else {
        pgrx::error!(
            "Dimensions type modifier of a vector column is needed for building the index."
        );
    };
    // get v, d
    let opfamily = unsafe { opfamily(index_relation) };
    let vector = VectorOptions {
        dims,
        v: opfamily.vector_kind(),
        d: opfamily.distance_kind(),
    };
    // get indexing, segment, optimizing
    let rabitq = 'rabitq: {
        let reloption = unsafe { (*index_relation).rd_options as *const Reloption };
        if reloption.is_null() || unsafe { (*reloption).options == 0 } {
            break 'rabitq Default::default();
        }
        let s = unsafe { Reloption::options(reloption) }.to_string_lossy();
        match toml::from_str::<VchordrqIndexingOptions>(&s) {
            Ok(p) => p,
            Err(e) => pgrx::error!("failed to parse options: {}", e),
        }
    };
    (vector, rabitq)
}

fn make_internal_build(
    vector_options: VectorOptions,
    internal_build: VchordrqInternalBuildOptions,
    mut samples: Vec<Vec<f32>>,
    reporter: &mut PostgresReporter,
) -> Vec<Structure<Vec<f32>>> {
    use std::iter::once;
    k_means::preprocess(internal_build.build_threads as _, &mut samples, |sample| {
        *sample = rabitq::rotate::rotate(sample)
    });
    let mut result = Vec::<Structure<Vec<f32>>>::new();
    for w in internal_build.lists.iter().rev().copied().chain(once(1)) {
        let input = if let Some(structure) = result.last() {
            &structure.centroids
        } else {
            &samples
        };
        let num_threads = internal_build.build_threads as _;
        let num_points = input.len();
        let num_dims = vector_options.dims as usize;
        let num_lists = w as usize;
        let num_iterations = internal_build.kmeans_iterations as _;
        if result.is_empty() {
            let percentage = 0;
            let default = BuildPhase::from_code(BuildPhaseCode::InternalBuild);
            let phase =
                BuildPhase::new(BuildPhaseCode::InternalBuild, 1 + percentage).unwrap_or(default);
            reporter.phase(phase);
        }
        if num_lists > 1 {
            pgrx::info!(
                "clustering: starting, using {num_threads} threads, clustering {num_points} vectors of {num_dims} dimension into {num_lists} clusters, in {num_iterations} iterations"
            );
        }
        let centroids = k_means::k_means(
            num_threads,
            |i| {
                pgrx::check_for_interrupts!();
                if result.is_empty() {
                    let percentage =
                        ((i as f64 / num_iterations as f64) * 100.0).clamp(0.0, 100.0) as u16;
                    let default = BuildPhase::from_code(BuildPhaseCode::InternalBuild);
                    let phase = BuildPhase::new(BuildPhaseCode::InternalBuild, 1 + percentage)
                        .unwrap_or(default);
                    reporter.phase(phase);
                }
                if num_lists > 1 {
                    pgrx::info!("clustering: iteration {}", i + 1);
                }
            },
            num_lists,
            num_dims,
            input,
            internal_build.spherical_centroids,
            num_iterations,
        );
        if result.is_empty() {
            let percentage = 100;
            let default = BuildPhase::from_code(BuildPhaseCode::InternalBuild);
            let phase =
                BuildPhase::new(BuildPhaseCode::InternalBuild, 1 + percentage).unwrap_or(default);
            reporter.phase(phase);
        }
        if num_lists > 1 {
            pgrx::info!("clustering: finished");
        }
        if let Some(structure) = result.last() {
            let mut children = vec![Vec::new(); centroids.len()];
            for i in 0..structure.len() as u32 {
                let target = k_means::k_means_lookup(&structure.centroids[i as usize], &centroids);
                children[target].push(i);
            }
            let (centroids, children) = std::iter::zip(centroids, children)
                .filter(|(_, x)| !x.is_empty())
                .unzip::<_, _, Vec<_>, Vec<_>>();
            result.push(Structure {
                centroids,
                children,
            });
        } else {
            let children = vec![Vec::new(); centroids.len()];
            result.push(Structure {
                centroids,
                children,
            });
        }
    }
    result
}

#[allow(clippy::collapsible_else_if)]
fn make_external_build(
    vector_options: VectorOptions,
    _opfamily: Opfamily,
    external_build: VchordrqExternalBuildOptions,
) -> Vec<Structure<Vec<f32>>> {
    use std::collections::BTreeMap;
    let VchordrqExternalBuildOptions { table } = external_build;
    let mut parents = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    pgrx::spi::Spi::connect(|client| {
        use crate::datatype::memory_vector::VectorOutput;
        use pgrx::pg_sys::panic::ErrorReportable;
        use vector::VectorBorrowed;
        let schema_query = "SELECT n.nspname::TEXT 
            FROM pg_catalog.pg_extension e
            LEFT JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname = 'vector';";
        let pgvector_schema: String = client
            .select(schema_query, None, &[])
            .unwrap_or_report()
            .first()
            .get_by_name("nspname")
            .expect("external build: cannot get schema of pgvector")
            .expect("external build: cannot get schema of pgvector");
        let dump_query =
            format!("SELECT id, parent, vector::{pgvector_schema}.vector FROM {table};");
        let centroids = client.select(&dump_query, None, &[]).unwrap_or_report();
        for row in centroids {
            let id: Option<i32> = row.get_by_name("id").unwrap();
            let parent: Option<i32> = row.get_by_name("parent").unwrap();
            let vector: Option<VectorOutput> = row.get_by_name("vector").unwrap();
            let id = id.expect("external build: id could not be NULL");
            let vector = vector.expect("external build: vector could not be NULL");
            let pop = parents.insert(id, parent);
            if pop.is_some() {
                pgrx::error!(
                    "external build: there are at least two lines have same id, id = {id}"
                );
            }
            if vector_options.dims != vector.as_borrowed().dims() {
                pgrx::error!("external build: incorrect dimension, id = {id}");
            }
            vectors.insert(id, rabitq::rotate::rotate(vector.as_borrowed().slice()));
        }
    });
    if parents.len() >= 2 && parents.values().all(|x| x.is_none()) {
        // if there are more than one vertex and no edges,
        // assume there is an implicit root
        let n = parents.len();
        let mut result = Vec::new();
        result.push(Structure {
            centroids: vectors.values().cloned().collect::<Vec<_>>(),
            children: vec![Vec::new(); n],
        });
        result.push(Structure {
            centroids: vec![{
                // compute the vector on root, without normalizing it
                let mut sum = vec![0.0f32; vector_options.dims as _];
                for vector in vectors.values() {
                    f32::vector_add_inplace(&mut sum, vector);
                }
                f32::vector_mul_scalar_inplace(&mut sum, 1.0 / n as f32);
                sum
            }],
            children: vec![(0..n as u32).collect()],
        });
        return result;
    }
    let mut children = parents
        .keys()
        .map(|x| (*x, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    let mut root = None;
    for (&id, &parent) in parents.iter() {
        if let Some(parent) = parent {
            if let Some(parent) = children.get_mut(&parent) {
                parent.push(id);
            } else {
                pgrx::error!("external build: parent does not exist, id = {id}, parent = {parent}");
            }
        } else {
            if let Some(root) = root {
                pgrx::error!("external build: two root, id = {root}, id = {id}");
            } else {
                root = Some(id);
            }
        }
    }
    let Some(root) = root else {
        pgrx::error!("external build: there are no root");
    };
    let mut heights = BTreeMap::<_, _>::new();
    fn dfs_for_heights(
        heights: &mut BTreeMap<i32, Option<u32>>,
        children: &BTreeMap<i32, Vec<i32>>,
        u: i32,
    ) {
        if heights.contains_key(&u) {
            pgrx::error!("external build: detect a cycle, id = {u}");
        }
        heights.insert(u, None);
        let mut height = None;
        for &v in children[&u].iter() {
            dfs_for_heights(heights, children, v);
            let new = heights[&v].unwrap() + 1;
            if let Some(height) = height {
                if height != new {
                    pgrx::error!("external build: two heights, id = {u}");
                }
            } else {
                height = Some(new);
            }
        }
        if height.is_none() {
            height = Some(1);
        }
        heights.insert(u, height);
    }
    dfs_for_heights(&mut heights, &children, root);
    let heights = heights
        .into_iter()
        .map(|(k, v)| (k, v.expect("not a connected graph")))
        .collect::<BTreeMap<_, _>>();
    if !(1..=8).contains(&(heights[&root] - 1)) {
        pgrx::error!(
            "external build: unexpected tree height, height = {}",
            heights[&root]
        );
    }
    let mut cursors = vec![0_u32; 1 + heights[&root] as usize];
    let mut labels = BTreeMap::new();
    for id in parents.keys().copied() {
        let height = heights[&id];
        let cursor = cursors[height as usize];
        labels.insert(id, (height, cursor));
        cursors[height as usize] += 1;
    }
    fn extract(
        height: u32,
        labels: &BTreeMap<i32, (u32, u32)>,
        vectors: &BTreeMap<i32, Vec<f32>>,
        children: &BTreeMap<i32, Vec<i32>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
        labels
            .iter()
            .filter(|(_, (h, _))| *h == height)
            .map(|(id, _)| {
                (
                    vectors[id].clone(),
                    children[id].iter().map(|id| labels[id].1).collect(),
                )
            })
            .unzip()
    }
    let mut result = Vec::new();
    for height in 1..=heights[&root] {
        let (centroids, children) = extract(height, &labels, &vectors, &children);
        result.push(Structure {
            centroids,
            children,
        });
    }
    result
}

pub trait InternalBuild: VectorOwned {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32>;

    fn build_from_vecf32(x: &[f32]) -> Self;
}

impl InternalBuild for VectOwned<f32> {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        vector.slice().to_vec()
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(x.to_vec())
    }
}

impl InternalBuild for VectOwned<f16> {
    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        f16::vector_to_f32(vector.slice())
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(f16::vector_from_f32(x))
    }
}

struct CachingRelation<'a, R> {
    cache: vchordrq_cached::VchordrqCachedReader1<'a>,
    relation: R,
}

impl<R: Clone> Clone for CachingRelation<'_, R> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache,
            relation: self.relation.clone(),
        }
    }
}

enum CachingRelationReadGuard<'a, G: Deref> {
    Wrapping(G),
    Cached(u32, &'a G::Target),
}

impl<G: PageGuard + Deref> PageGuard for CachingRelationReadGuard<'_, G> {
    fn id(&self) -> u32 {
        match self {
            CachingRelationReadGuard::Wrapping(x) => x.id(),
            CachingRelationReadGuard::Cached(id, _) => *id,
        }
    }
}

impl<G: Deref> Deref for CachingRelationReadGuard<'_, G> {
    type Target = G::Target;

    fn deref(&self) -> &Self::Target {
        match self {
            CachingRelationReadGuard::Wrapping(x) => x,
            CachingRelationReadGuard::Cached(_, page) => page,
        }
    }
}

impl<R: Relation> Relation for CachingRelation<'_, R> {
    type Page = R::Page;
}

impl<R: RelationRead<Page = PostgresPage<vchordrq::Opaque>>> RelationReadTypes
    for CachingRelation<'_, R>
{
    type ReadGuard<'a> = CachingRelationReadGuard<'a, R::ReadGuard<'a>>;
}

impl<R: RelationRead<Page = PostgresPage<vchordrq::Opaque>>> RelationRead
    for CachingRelation<'_, R>
{
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        if let Some(x) = self.cache.get(id) {
            CachingRelationReadGuard::Cached(id, x)
        } else {
            CachingRelationReadGuard::Wrapping(self.relation.read(id))
        }
    }
}

impl<R: RelationWrite<Page = PostgresPage<vchordrq::Opaque>>> RelationWriteTypes
    for CachingRelation<'_, R>
{
    type WriteGuard<'a> = R::WriteGuard<'a>;
}

impl<R: RelationWrite<Page = PostgresPage<vchordrq::Opaque>>> RelationWrite
    for CachingRelation<'_, R>
{
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.relation.write(id, tracking_freespace)
    }

    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> Self::WriteGuard<'_> {
        self.relation.extend(opaque, tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.relation.search(freespace)
    }
}
