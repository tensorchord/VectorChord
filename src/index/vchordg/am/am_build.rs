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
use crate::index::storage::PostgresRelation;
use crate::index::vchordg::am::{Reloption, ctid_to_key, kv_to_pointer};
use crate::index::vchordg::opclass::{Opfamily, opfamily};
use crate::index::vchordg::types::VchordgIndexingOptions;
use pgrx::pg_sys::{Datum, ItemPointerData};
use std::ffi::CStr;
use vchordg::types::*;

#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum BuildPhaseCode {
    Initializing = 0,
    Inserting = 1,
}

pub struct BuildPhase(BuildPhaseCode, u16);

impl BuildPhase {
    pub const fn new(code: BuildPhaseCode, k: u16) -> Option<Self> {
        match (code, k) {
            (BuildPhaseCode::Initializing, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::Inserting, 0) => Some(BuildPhase(code, k)),
            _ => None,
        }
    }
    pub const fn name(self) -> &'static CStr {
        match self {
            BuildPhase(BuildPhaseCode::Initializing, k) => {
                static RAW: [&CStr; 1] = [c"initializing"];
                RAW[k as usize]
            }
            BuildPhase(BuildPhaseCode::Inserting, k) => {
                static RAW: [&CStr; 1] = [c"inserting tuples from table to index"];
                RAW[k as usize]
            }
        }
    }
    pub const fn from_code(code: BuildPhaseCode) -> Self {
        Self(code, 0)
    }
    pub const fn from_value(value: u32) -> Option<Self> {
        const INITIALIZING: u16 = BuildPhaseCode::Initializing as _;
        const INSERTING: u16 = BuildPhaseCode::Inserting as _;
        let k = value as u16;
        match (value >> 16) as u16 {
            INITIALIZING => Self::new(BuildPhaseCode::Initializing, k),
            INSERTING => Self::new(BuildPhaseCode::Inserting, k),
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
    let (vector_options, vchordg_options) = unsafe { options(index_relation) };
    if let Err(errors) = Validate::validate(&vector_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    if let Err(errors) = Validate::validate(&vchordg_options) {
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
    crate::index::vchordg::algo::build(vector_options, vchordg_options.index, &index);
    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Inserting));
    let cache = vchordg_cached::VchordgCached::_0 {};
    if let Some(leader) = unsafe {
        VchordgLeader::enter(
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
                leader.vchordgshared,
                leader.vchordgcached,
                |indtuples| {
                    reporter.tuples_done(indtuples);
                },
            );
            leader.wait();
            let nparticipants = leader.nparticipants;
            loop {
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*leader.vchordgshared).mutex);
                if (*leader.vchordgshared).nparticipantsdone == nparticipants {
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordgshared).mutex);
                    break;
                }
                pgrx::pg_sys::SpinLockRelease(&raw mut (*leader.vchordgshared).mutex);
                pgrx::pg_sys::ConditionVariableSleep(
                    &raw mut (*leader.vchordgshared).workersdonecv,
                    pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                );
            }
            reporter.tuples_done((*leader.vchordgshared).indtuples);
            reporter.tuples_total((*leader.vchordgshared).indtuples);
            pgrx::pg_sys::ConditionVariableCancelSleep();
        }
    } else {
        let mut indtuples = 0;
        reporter.tuples_done(indtuples);
        heap.traverse(true, |(ctid, store)| {
            for (vector, extra) in store {
                let key = ctid_to_key(ctid);
                let payload = kv_to_pointer((key, extra));
                crate::index::vchordg::algo::insert(opfamily, &index, payload, vector);
            }
            indtuples += 1;
            reporter.tuples_done(indtuples);
        });
        reporter.tuples_total(indtuples);
    }
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

struct VchordgShared {
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

fn is_mvcc_snapshot(snapshot: *mut pgrx::pg_sys::SnapshotData) -> bool {
    matches!(
        unsafe { (*snapshot).snapshot_type },
        pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
            | pgrx::pg_sys::SnapshotType::SNAPSHOT_HISTORIC_MVCC
    )
}

struct VchordgLeader {
    pcxt: *mut pgrx::pg_sys::ParallelContext,
    nparticipants: i32,
    snapshot: pgrx::pg_sys::Snapshot,
    vchordgshared: *mut VchordgShared,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    vchordgcached: *const u8,
}

impl VchordgLeader {
    pub unsafe fn enter(
        heap_relation: pgrx::pg_sys::Relation,
        index_relation: pgrx::pg_sys::Relation,
        isconcurrent: bool,
        cache: vchordg_cached::VchordgCached,
    ) -> Option<Self> {
        let _cache = cache.serialize();
        #[expect(clippy::drop_non_drop)]
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
                c"vchordg_parallel_build_main".as_ptr(),
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
            estimate_chunk(&mut (*pcxt).estimator, size_of::<VchordgShared>());
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

        let vchordgshared = unsafe {
            let vchordgshared =
                pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, size_of::<VchordgShared>())
                    .cast::<VchordgShared>();
            vchordgshared.write(VchordgShared {
                heaprelid: (*heap_relation).rd_id,
                indexrelid: (*index_relation).rd_id,
                isconcurrent,
                workersdonecv: std::mem::zeroed(),
                mutex: std::mem::zeroed(),
                nparticipantsdone: 0,
                indtuples: 0,
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordgshared).workersdonecv);
            pgrx::pg_sys::SpinLockInit(&raw mut (*vchordgshared).mutex);
            vchordgshared
        };

        let tablescandesc = unsafe {
            let tablescandesc = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_tablescandesc)
                .cast::<pgrx::pg_sys::ParallelTableScanDescData>();
            pgrx::pg_sys::table_parallelscan_initialize(heap_relation, tablescandesc, snapshot);
            tablescandesc
        };

        let vchordgcached = unsafe {
            let x = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, 8 + cache.len()).cast::<u8>();
            (x as *mut u64).write_unaligned(cache.len() as _);
            std::ptr::copy(cache.as_ptr(), x.add(8), cache.len());
            x
        };

        unsafe {
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000001, vchordgshared.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000002, tablescandesc.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000003, vchordgcached.cast());
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
            vchordgshared,
            tablescandesc,
            vchordgcached,
        })
    }

    pub fn wait(&self) {
        unsafe {
            pgrx::pg_sys::WaitForParallelWorkersToAttach(self.pcxt);
        }
    }
}

impl Drop for VchordgLeader {
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
pub unsafe extern "C-unwind" fn vchordg_parallel_build_main(
    _seg: *mut pgrx::pg_sys::dsm_segment,
    toc: *mut pgrx::pg_sys::shm_toc,
) {
    let vchordgshared = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000001, false).cast::<VchordgShared>()
    };
    let tablescandesc = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000002, false)
            .cast::<pgrx::pg_sys::ParallelTableScanDescData>()
    };
    let vchordgcached = unsafe {
        pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000003, false)
            .cast::<u8>()
            .cast_const()
    };
    let heap_lockmode;
    let index_lockmode;
    if unsafe { !(*vchordgshared).isconcurrent } {
        heap_lockmode = pgrx::pg_sys::ShareLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE;
    } else {
        heap_lockmode = pgrx::pg_sys::ShareUpdateExclusiveLock as pgrx::pg_sys::LOCKMODE;
        index_lockmode = pgrx::pg_sys::RowExclusiveLock as pgrx::pg_sys::LOCKMODE;
    }
    let heap = unsafe { pgrx::pg_sys::table_open((*vchordgshared).heaprelid, heap_lockmode) };
    let index = unsafe { pgrx::pg_sys::index_open((*vchordgshared).indexrelid, index_lockmode) };
    let index_info = unsafe { pgrx::pg_sys::BuildIndexInfo(index) };
    unsafe {
        (*index_info).ii_Concurrent = (*vchordgshared).isconcurrent;
    }

    unsafe {
        parallel_build(
            index,
            heap,
            index_info,
            tablescandesc,
            vchordgshared,
            vchordgcached,
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
    vchordgshared: *mut VchordgShared,
    vchordgcached: *const u8,
    mut callback: impl FnMut(u64),
) {
    use vchordg_cached::VchordgCachedReader;
    let cached = VchordgCachedReader::deserialize_ref(unsafe {
        let bytes = (vchordgcached as *const u64).read_unaligned();
        std::slice::from_raw_parts(vchordgcached.add(8), bytes as _)
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
        VchordgCachedReader::_0(_) => {
            heap.traverse(true, move |(ctid, store)| {
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    crate::index::vchordg::algo::insert(opfamily, &index, payload, vector);
                }
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordgshared).mutex);
                        (*vchordgshared).indtuples += 1;
                        indtuples = (*vchordgshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordgshared).mutex);
                    }
                    callback(indtuples);
                }
            });
        }
    }
    unsafe {
        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordgshared).mutex);
        (*vchordgshared).nparticipantsdone += 1;
        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordgshared).mutex);
        pgrx::pg_sys::ConditionVariableSignal(&raw mut (*vchordgshared).workersdonecv);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C-unwind" fn ambuildempty(_index_relation: pgrx::pg_sys::Relation) {
    pgrx::error!("Unlogged indexes are not supported.");
}

unsafe fn options(
    index_relation: pgrx::pg_sys::Relation,
) -> (VectorOptions, VchordgIndexingOptions) {
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
        match toml::from_str::<VchordgIndexingOptions>(&s) {
            Ok(p) => p,
            Err(e) => pgrx::error!("failed to parse options: {}", e),
        }
    };
    (vector, rabitq)
}

mod vchordg_cached {
    pub type Tag = u64;

    use algo::tuples::RefChecker;
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct VchordgCachedHeader0 {}

    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct VchordgCachedHeader1 {
        mapping_s: usize,
        mapping_e: usize,
        pages_s: usize,
        pages_e: usize,
    }

    pub enum VchordgCached {
        _0 {},
    }

    impl VchordgCached {
        pub fn serialize(&self) -> Vec<u8> {
            let mut buffer = Vec::new();
            match self {
                VchordgCached::_0 {} => {
                    buffer.extend((0 as Tag).to_ne_bytes());
                    buffer.extend(std::iter::repeat_n(0, size_of::<VchordgCachedHeader0>()));
                    buffer[size_of::<Tag>()..][..size_of::<VchordgCachedHeader0>()]
                        .copy_from_slice(VchordgCachedHeader0 {}.as_bytes());
                }
            }
            buffer
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum VchordgCachedReader<'a> {
        #[allow(dead_code)]
        _0(VchordgCachedReader0<'a>),
    }

    #[derive(Debug, Clone, Copy)]
    pub struct VchordgCachedReader0<'a> {
        #[allow(dead_code)]
        header: &'a VchordgCachedHeader0,
    }

    impl<'a> VchordgCachedReader<'a> {
        pub fn deserialize_ref(source: &'a [u8]) -> Self {
            let tag = u64::from_ne_bytes(std::array::from_fn(|i| source[i]));
            match tag {
                0 => {
                    let checker = RefChecker::new(source);
                    let header: &VchordgCachedHeader0 = checker.prefix(size_of::<Tag>());
                    Self::_0(VchordgCachedReader0 { header })
                }
                _ => panic!("bad bytes"),
            }
        }
    }
}
