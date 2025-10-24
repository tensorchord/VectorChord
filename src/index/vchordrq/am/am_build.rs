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
use crate::index::sample::{HeapSampler, Sample, Sampler, Tuple};
use crate::index::storage::{PostgresPage, PostgresRelation};
use crate::index::traverse::{HeapTraverser, Traverser};
use crate::index::vchordrq::am::Reloption;
use crate::index::vchordrq::build::{Normalize, Normalized};
use crate::index::vchordrq::opclass::{Opfamily, opfamily};
use crate::index::vchordrq::types::*;
use index::relation::{
    Page, PageGuard, Relation, RelationRead, RelationReadTypes, RelationWrite, RelationWriteTypes,
};
use k_means::reduction::*;
use k_means::square::Square;
use pgrx::pg_sys::ItemPointerData;
use simd::Floating;
use std::collections::HashMap;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::Deref;
use vchordrq::types::*;
use vchordrq::{InsertChooser, MaintainChooser};
use vector::vect::VectOwned;

#[derive(Debug, Clone, Copy)]
#[repr(u16)]
pub enum BuildPhaseCode {
    Initializing = 0,
    DefaultBuild = 1,
    InternalBuild = 2,
    ExternalBuild = 3,
    Build = 4,
    Inserting = 5,
    Compacting = 6,
}

pub struct BuildPhase(BuildPhaseCode, u16);

impl BuildPhase {
    pub const fn new(code: BuildPhaseCode, k: u16) -> Option<Self> {
        match (code, k) {
            (BuildPhaseCode::Initializing, 0) => Some(BuildPhase(code, k)),
            (BuildPhaseCode::DefaultBuild, 0) => Some(BuildPhase(code, k)),
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
            BuildPhase(BuildPhaseCode::DefaultBuild, k) => {
                static RAW: [&CStr; 1] = [c"initializing index, by default build"];
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
        const DEFAULT_BUILD: u16 = BuildPhaseCode::DefaultBuild as _;
        const INTERNAL_BUILD: u16 = BuildPhaseCode::InternalBuild as _;
        const EXTERNAL_BUILD: u16 = BuildPhaseCode::ExternalBuild as _;
        const BUILD: u16 = BuildPhaseCode::Build as _;
        const INSERTING: u16 = BuildPhaseCode::Inserting as _;
        const COMPACTING: u16 = BuildPhaseCode::Compacting as _;
        let k = value as u16;
        match (value >> 16) as u16 {
            INITIALIZING => Self::new(BuildPhaseCode::Initializing, k),
            DEFAULT_BUILD => Self::new(BuildPhaseCode::DefaultBuild, k),
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
struct PostgresReporter {
    _phantom: PhantomData<*mut ()>,
}

impl PostgresReporter {
    fn phase(&self, phase: BuildPhase) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_SUBPHASE as _,
                (phase.into_value() as i64) + 1,
            );
        }
    }
    fn tuples_total(&self, tuples_total: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_TOTAL as _,
                tuples_total as _,
            );
        }
    }
    fn tuples_done(&self, tuples_done: u64) {
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
    let (vector_options, vchordrq_options) = unsafe { options(index_relation) };
    if let Err(errors) = Validate::validate(&vector_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    if let Err(errors) = Validate::validate(&vchordrq_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    let opfamily = unsafe { opfamily(index_relation) };
    let reporter = PostgresReporter {
        _phantom: PhantomData,
    };
    reporter.tuples_total(unsafe { (*(*index_relation).rd_rel).reltuples as u64 });
    let mut structures = match vchordrq_options.build.source.clone() {
        VchordrqBuildSourceOptions::Default(default_build) => {
            reporter.phase(BuildPhase::from_code(BuildPhaseCode::DefaultBuild));
            make_default_build(vector_options, default_build)
        }
        VchordrqBuildSourceOptions::Internal(internal_build) => {
            reporter.phase(BuildPhase::from_code(BuildPhaseCode::InternalBuild));
            let snapshot = if unsafe { (*index_info).ii_Concurrent } {
                unsafe { pgrx::pg_sys::RegisterSnapshot(pgrx::pg_sys::GetTransactionSnapshot()) }
            } else {
                &raw mut pgrx::pg_sys::SnapshotAnyData
            };
            let sampler = unsafe { HeapSampler::new(index_relation, heap_relation, snapshot) };
            let mut sample = sampler.sample();
            let (approximation, reduction, k_means_dim) = match internal_build.kmeans_algorithm {
                KMeansAlgorithm::Lloyd {} => (false, false, vector_options.dims),
                KMeansAlgorithm::Reduction { dim: None } => (true, false, vector_options.dims),
                KMeansAlgorithm::Reduction { dim: Some(d) } => (true, true, d),
            };
            assert!(
                k_means_dim <= vector_options.dims,
                "invalid K-means reduction dimension {} > {}",
                k_means_dim,
                vector_options.dims
            );
            let mut samples_idx = Vec::new();
            pgrx::info!("start collecting samples from the table");
            let samples = 'a: {
                let mut samples = Square::new(k_means_dim as _);
                let Some(max_number_of_samples) = internal_build
                    .lists
                    .last()
                    .map(|x| x.saturating_mul(internal_build.sampling_factor))
                else {
                    break 'a samples;
                };
                while samples.len() < max_number_of_samples as usize {
                    if let Some(mut tuple) = sample.next() {
                        let (values, is_nulls) = tuple.build();
                        let datum = (!is_nulls[0]).then_some(values[0]);
                        if let Some(datum) = datum {
                            let vectors = unsafe { opfamily.store(datum) };
                            if let Some(vectors) = vectors {
                                for (vector, _) in vectors {
                                    let mut x = match vector {
                                        OwnedVector::Vecf32(x) => VectOwned::normalize(x),
                                        OwnedVector::Vecf16(x) => VectOwned::normalize(x),
                                    };
                                    assert_eq!(
                                        vector_options.dims,
                                        x.len() as u32,
                                        "invalid vector dimensions"
                                    );
                                    if reduction {
                                        x = rabitq::rotate::rotate(&x)[..k_means_dim as usize]
                                            .to_vec();
                                    }
                                    samples.push_slice(x.as_slice());
                                    samples_idx.push(tuple.id());
                                }
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else {
                        break;
                    }
                }
                samples.truncate(max_number_of_samples as usize);
                samples_idx.truncate(max_number_of_samples as usize);
                samples
            };
            let tree = match approximation {
                true => {
                    let (approximate_tree, allocation) = make_internal_approximate_build(
                        VectorOptions {
                            dims: k_means_dim,
                            ..vector_options
                        },
                        internal_build.clone(),
                        samples,
                        &reporter,
                    );
                    let mut fetcher = HeapFetcher::new_with_tmp_heapfetch(
                        index_relation,
                        heap_relation,
                        snapshot,
                    );
                    restore_from_approximate_tree(
                        vector_options,
                        internal_build,
                        reduction,
                        opfamily,
                        &mut fetcher,
                        samples_idx,
                        approximate_tree,
                        allocation,
                    )
                }
                false => make_internal_build(vector_options, internal_build, samples, &reporter),
            };
            if is_mvcc_snapshot(snapshot) {
                unsafe {
                    pgrx::pg_sys::UnregisterSnapshot(snapshot);
                }
            }
            tree
        }
        VchordrqBuildSourceOptions::External(external_build) => {
            reporter.phase(BuildPhase::from_code(BuildPhaseCode::ExternalBuild));
            make_external_build(vector_options, opfamily, external_build)
        }
    };
    pgrx::info!("the K-means structure tree is constructed");
    for structure in structures.iter_mut() {
        for centroid in structure.centroids.iter_mut() {
            *centroid = rabitq::rotate::rotate(centroid);
        }
    }
    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Build));
    let index = unsafe { PostgresRelation::new(index_relation) };
    crate::index::vchordrq::dispatch::build(
        vector_options,
        vchordrq_options.index,
        &index,
        structures,
    );
    let cached = if vchordrq_options.build.pin >= 0 {
        let mut trace = vchordrq::cache(&index, vchordrq_options.build.pin);
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
    }
    .serialize();
    if let Some(leader) = unsafe {
        VchordrqLeader::enter(
            c"vchordrq_parallel_build_main",
            heap_relation,
            index_relation,
            (*index_info).ii_Concurrent,
            &cached,
        )
    } {
        drop(cached);
        unsafe {
            leader.wait();
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
                || {
                    #[allow(clippy::needless_late_init)]
                    let order;
                    // enter the barrier
                    let shared = leader.vchordrqshared;
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).nparticipants = leader.nparticipants as u32;
                    order = (*shared).barrier_enter_0 as u32;
                    (*shared).barrier_enter_0 += 1;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_enter_0,
                    );
                    // leave the barrier
                    let total = leader.nparticipants;
                    loop {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                        if (*shared).barrier_enter_0 == total {
                            pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                            break;
                        }
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        pgrx::pg_sys::ConditionVariableSleep(
                            &raw mut (*shared).condvar_barrier_enter_0,
                            pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                        );
                    }
                    pgrx::pg_sys::ConditionVariableCancelSleep();
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_leave_0 = true;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_leave_0,
                    );
                    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Inserting));
                    order
                },
                |indtuples| {
                    reporter.tuples_done(indtuples);
                    reporter.tuples_total(indtuples);
                    // enter the barrier
                    let shared = leader.vchordrqshared;
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_enter_1 += 1;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_enter_1,
                    );
                    // leave the barrier
                    let total = leader.nparticipants;
                    loop {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                        if (*shared).barrier_enter_1 == total {
                            pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                            break;
                        }
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        pgrx::pg_sys::ConditionVariableSleep(
                            &raw mut (*shared).condvar_barrier_enter_1,
                            pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                        );
                    }
                    pgrx::pg_sys::ConditionVariableCancelSleep();
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_leave_1 = true;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_leave_1,
                    );
                    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Compacting));
                },
                || {
                    // enter the barrier
                    let shared = leader.vchordrqshared;
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_enter_2 += 1;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_enter_2,
                    );
                    // leave the barrier
                    let total = leader.nparticipants;
                    loop {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                        if (*shared).barrier_enter_2 == total {
                            pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                            break;
                        }
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        pgrx::pg_sys::ConditionVariableSleep(
                            &raw mut (*shared).condvar_barrier_enter_2,
                            pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                        );
                    }
                    pgrx::pg_sys::ConditionVariableCancelSleep();
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    (*shared).barrier_leave_2 = true;
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableBroadcast(
                        &raw mut (*shared).condvar_barrier_leave_2,
                    );
                },
            );
        }
    } else {
        unsafe {
            sequential_build(
                index_relation,
                heap_relation,
                index_info,
                &cached,
                |indtuples| {
                    reporter.tuples_done(indtuples);
                },
                || {
                    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Inserting));
                },
                |indtuples| {
                    reporter.tuples_done(indtuples);
                    reporter.tuples_total(indtuples);
                    reporter.phase(BuildPhase::from_code(BuildPhaseCode::Compacting));
                },
                || {},
            );
        }
    }
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

struct VchordrqShared {
    /* immutable state */
    heaprelid: pgrx::pg_sys::Oid,
    indexrelid: pgrx::pg_sys::Oid,
    isconcurrent: bool,

    /* locking */
    mutex: pgrx::pg_sys::slock_t,
    condvar_barrier_enter_0: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_leave_0: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_enter_1: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_leave_1: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_enter_2: pgrx::pg_sys::ConditionVariable,
    condvar_barrier_leave_2: pgrx::pg_sys::ConditionVariable,

    /* mutable state */
    barrier_enter_0: i32,
    nparticipants: u32,
    indtuples: u64,
    barrier_leave_0: bool,
    barrier_enter_1: i32,
    barrier_leave_1: bool,
    barrier_enter_2: i32,
    barrier_leave_2: bool,
}

mod vchordrq_cached {
    pub const ALIGN: usize = 8;
    pub type Tag = u64;

    use crate::index::storage::PostgresPage;
    use index::tuples::RefChecker;
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
        main: &'static CStr,
        heap_relation: pgrx::pg_sys::Relation,
        index_relation: pgrx::pg_sys::Relation,
        isconcurrent: bool,
        vchordrq_cached: &[u8],
    ) -> Option<Self> {
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
            pgrx::pg_sys::CreateParallelContext(c"vchord".as_ptr(), main.as_ptr(), request)
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
            estimate_chunk(&mut (*pcxt).estimator, 8 + vchordrq_cached.len());
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
                nparticipants: 0,
                condvar_barrier_enter_0: std::mem::zeroed(),
                condvar_barrier_leave_0: std::mem::zeroed(),
                condvar_barrier_enter_1: std::mem::zeroed(),
                condvar_barrier_leave_1: std::mem::zeroed(),
                condvar_barrier_enter_2: std::mem::zeroed(),
                condvar_barrier_leave_2: std::mem::zeroed(),
                barrier_enter_0: 0,
                barrier_leave_0: false,
                barrier_enter_1: 0,
                barrier_leave_1: false,
                barrier_enter_2: 0,
                barrier_leave_2: false,
                mutex: std::mem::zeroed(),
                indtuples: 0,
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_enter_0);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_leave_0);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_enter_1);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_leave_1);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_enter_2);
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).condvar_barrier_leave_2);
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
            let x =
                pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, 8 + vchordrq_cached.len()).cast::<u8>();
            (x as *mut u64).write_unaligned(vchordrq_cached.len() as _);
            std::ptr::copy(vchordrq_cached.as_ptr(), x.add(8), vchordrq_cached.len());
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
    let _ = rand::rng().reseed();
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
            || {
                #[allow(clippy::needless_late_init)]
                let order;
                // enter the barrier
                let shared = vchordrqshared;
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                order = (*shared).barrier_enter_0 as u32;
                (*shared).barrier_enter_0 += 1;
                pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                pgrx::pg_sys::ConditionVariableBroadcast(
                    &raw mut (*shared).condvar_barrier_enter_0,
                );
                // leave the barrier
                loop {
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    if (*shared).barrier_leave_0 {
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        break;
                    }
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableSleep(
                        &raw mut (*shared).condvar_barrier_leave_0,
                        pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                    );
                }
                pgrx::pg_sys::ConditionVariableCancelSleep();
                order
            },
            |_| {
                // enter the barrier
                let shared = vchordrqshared;
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                (*shared).barrier_enter_1 += 1;
                pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                pgrx::pg_sys::ConditionVariableBroadcast(
                    &raw mut (*shared).condvar_barrier_enter_1,
                );
                // leave the barrier
                loop {
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    if (*shared).barrier_leave_1 {
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        break;
                    }
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableSleep(
                        &raw mut (*shared).condvar_barrier_leave_1,
                        pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                    );
                }
                pgrx::pg_sys::ConditionVariableCancelSleep();
            },
            || {
                // enter the barrier
                let shared = vchordrqshared;
                pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                (*shared).barrier_enter_2 += 1;
                pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                pgrx::pg_sys::ConditionVariableBroadcast(
                    &raw mut (*shared).condvar_barrier_enter_2,
                );
                // leave the barrier
                loop {
                    pgrx::pg_sys::SpinLockAcquire(&raw mut (*shared).mutex);
                    if (*shared).barrier_leave_2 {
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                        break;
                    }
                    pgrx::pg_sys::SpinLockRelease(&raw mut (*shared).mutex);
                    pgrx::pg_sys::ConditionVariableSleep(
                        &raw mut (*shared).condvar_barrier_leave_2,
                        pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN as _,
                    );
                }
                pgrx::pg_sys::ConditionVariableCancelSleep();
            },
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
    sync_0: impl FnOnce() -> u32,
    sync_1: impl FnOnce(u64),
    sync_2: impl FnOnce(),
) {
    use vchordrq_cached::VchordrqCachedReader;

    let cached = VchordrqCachedReader::deserialize_ref(unsafe {
        let bytes = (vchordrqcached as *const u64).read_unaligned();
        std::slice::from_raw_parts(vchordrqcached.add(8), bytes as _)
    });

    let index = unsafe { PostgresRelation::new(index_relation) };

    let scan = unsafe { pgrx::pg_sys::table_beginscan_parallel(heap_relation, tablescandesc) };
    let opfamily = unsafe { opfamily(index_relation) };
    let traverser = unsafe { HeapTraverser::new(heap_relation, index_relation, index_info, scan) };

    struct IdChooser(u32);
    impl InsertChooser for IdChooser {
        fn choose(&mut self, n: NonZero<usize>) -> usize {
            self.0 as usize % n.get()
        }
    }

    struct ChooseSome {
        n: usize,
        k: usize,
    }
    impl MaintainChooser for ChooseSome {
        fn choose(&mut self, i: usize) -> bool {
            i % self.n == self.k
        }
    }

    let check = || {
        pgrx::check_for_interrupts!();
    };

    let order = sync_0();

    match cached {
        VchordrqCachedReader::_0(_) => {
            traverser.traverse(true, |tuple: &mut dyn crate::index::traverse::Tuple| {
                let ctid = tuple.id();
                let (values, is_nulls) = tuple.build();
                let value = unsafe { (!is_nulls.add(0).read()).then_some(values.add(0).read()) };
                let store = value
                    .and_then(|x| unsafe { opfamily.store(x) })
                    .unwrap_or_default();
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    let mut chooser = IdChooser(order);
                    let bump = bumpalo::Bump::new();
                    crate::index::vchordrq::dispatch::insert(
                        opfamily,
                        &index,
                        payload,
                        vector,
                        true,
                        true,
                        &mut chooser,
                        &bump,
                    );
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
                relation: index.clone(),
            };
            traverser.traverse(true, |tuple: &mut dyn crate::index::traverse::Tuple| {
                let ctid = tuple.id();
                let (values, is_nulls) = tuple.build();
                let value = unsafe { (!is_nulls.add(0).read()).then_some(values.add(0).read()) };
                let store = value
                    .and_then(|x| unsafe { opfamily.store(x) })
                    .unwrap_or_default();
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    let mut chooser = IdChooser(order);
                    let bump = bumpalo::Bump::new();
                    crate::index::vchordrq::dispatch::insert(
                        opfamily,
                        &index,
                        payload,
                        vector,
                        true,
                        true,
                        &mut chooser,
                        &bump,
                    );
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

    sync_1(unsafe { (*vchordrqshared).indtuples });

    let mut chooser = ChooseSome {
        n: unsafe { (*vchordrqshared).nparticipants as usize },
        k: order as usize,
    };
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_2();
}

unsafe fn sequential_build(
    index_relation: pgrx::pg_sys::Relation,
    heap_relation: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    vchordrqcached: &[u8],
    mut callback: impl FnMut(u64),
    sync_0: impl FnOnce(),
    sync_1: impl FnOnce(u64),
    sync_2: impl FnOnce(),
) {
    use vchordrq_cached::VchordrqCachedReader;

    let cached = VchordrqCachedReader::deserialize_ref(vchordrqcached);

    let index = unsafe { PostgresRelation::new(index_relation) };

    let opfamily = unsafe { opfamily(index_relation) };
    let traverser = unsafe {
        HeapTraverser::new(
            heap_relation,
            index_relation,
            index_info,
            std::ptr::null_mut(),
        )
    };

    struct ChooseZero;
    impl InsertChooser for ChooseZero {
        fn choose(&mut self, _: NonZero<usize>) -> usize {
            0
        }
    }

    struct ChooseAll;
    impl MaintainChooser for ChooseAll {
        fn choose(&mut self, _: usize) -> bool {
            true
        }
    }

    let check = || {
        pgrx::check_for_interrupts!();
    };

    sync_0();

    let mut indtuples = 0;
    match cached {
        VchordrqCachedReader::_0(_) => {
            traverser.traverse(true, |tuple: &mut dyn crate::index::traverse::Tuple| {
                let ctid = tuple.id();
                let (values, is_nulls) = tuple.build();
                let value = unsafe { (!is_nulls.add(0).read()).then_some(values.add(0).read()) };
                let store = value
                    .and_then(|x| unsafe { opfamily.store(x) })
                    .unwrap_or_default();
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    let mut chooser = ChooseZero;
                    let bump = bumpalo::Bump::new();
                    crate::index::vchordrq::dispatch::insert(
                        opfamily,
                        &index,
                        payload,
                        vector,
                        true,
                        true,
                        &mut chooser,
                        &bump,
                    );
                }
                indtuples += 1;
                callback(indtuples);
            });
        }
        VchordrqCachedReader::_1(cached) => {
            let index = CachingRelation {
                cache: cached,
                relation: index.clone(),
            };
            traverser.traverse(true, |tuple: &mut dyn crate::index::traverse::Tuple| {
                let ctid = tuple.id();
                let (values, is_nulls) = tuple.build();
                let value = unsafe { (!is_nulls.add(0).read()).then_some(values.add(0).read()) };
                let store = value
                    .and_then(|x| unsafe { opfamily.store(x) })
                    .unwrap_or_default();
                for (vector, extra) in store {
                    let key = ctid_to_key(ctid);
                    let payload = kv_to_pointer((key, extra));
                    let mut chooser = ChooseZero;
                    let bump = bumpalo::Bump::new();
                    crate::index::vchordrq::dispatch::insert(
                        opfamily,
                        &index,
                        payload,
                        vector,
                        true,
                        true,
                        &mut chooser,
                        &bump,
                    );
                }
                indtuples += 1;
                callback(indtuples);
            });
        }
    }

    sync_1(indtuples);

    let mut chooser = ChooseAll;
    crate::index::vchordrq::dispatch::maintain(opfamily, &index, &mut chooser, check);

    sync_2();
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

fn make_default_build(
    vector_options: VectorOptions,
    _default_build: VchordrqDefaultBuildOptions,
) -> Vec<Structure<Normalized>> {
    vec![Structure::<Normalized> {
        centroids: vec![vec![0.0f32; vector_options.dims as usize]],
        children: vec![vec![]],
    }]
}

fn make_internal_approximate_build(
    vector_options: VectorOptions,
    internal_build: VchordrqInternalBuildOptions,
    samples: Square,
    reporter: &PostgresReporter,
) -> (Vec<Structure<Normalized>>, HashMap<usize, usize>) {
    let mut result = vec![];
    let (top_list, bottom_list) = if internal_build.lists.len() == 1 {
        let top_k = (internal_build.lists[0] as f64).sqrt().floor() as u32;
        let top_k = top_k.clamp(1, (samples.len() as f64).sqrt().floor() as u32);
        (top_k, internal_build.lists[0])
    } else {
        (internal_build.lists[0], internal_build.lists[1])
    };
    let num_iterations = internal_build.kmeans_iterations as _;
    pgrx::info!(
        "approximation: building a two-level index with {} top-level clusters and {} bottom-level clusters.",
        top_list,
        bottom_list
    );
    let k_means_report = |i: u32| {
        let percentage = ((i + 1) as f64 / (top_list + 1) as f64 * 100.0).clamp(0.0, 100.0) as u16;
        let default = BuildPhase::from_code(BuildPhaseCode::InternalBuild);
        let phase =
            BuildPhase::new(BuildPhaseCode::InternalBuild, 1 + percentage).unwrap_or(default);
        reporter.phase(phase);
    };
    pgrx::info!("approximation: top clustering started");
    let mut f = k_means::k_means(
        vector_options.dims as _,
        samples,
        top_list as usize,
        internal_build.build_threads as _,
        [7; 32],
    );
    if internal_build.spherical_centroids {
        f.sphericalize();
    }
    for i in 0..num_iterations {
        pgrx::check_for_interrupts!();
        if result.is_empty() {
            let percentage = ((i as f64 / num_iterations as f64) * 100.0).clamp(0.0, 100.0) as u16;
            let default = BuildPhase::from_code(BuildPhaseCode::InternalBuild);
            let phase =
                BuildPhase::new(BuildPhaseCode::InternalBuild, 1 + percentage).unwrap_or(default);
            reporter.phase(phase);
        }
        pgrx::info!("clustering: iteration {}", i + 1);
        f.assign();
        f.update();
        if internal_build.spherical_centroids {
            f.sphericalize();
        }
    }
    let final_assign = f.assign();
    let (mid_centroids, samples) = f.finish();
    pgrx::info!("approximation: top clustering finished");
    k_means_report(0);
    let alloc = final_assign.into_iter().enumerate().fold(
        vec![vec![]; mid_centroids.len()],
        |mut acc, (i, target)| {
            acc[target].push(i);
            acc
        },
    );
    let alloc_size = alloc.iter().map(|x| x.len() as u32).collect::<Vec<_>>();
    let keep_indices: Vec<usize> = alloc_size
        .iter()
        .enumerate()
        .filter_map(|(i, size)| if *size > 0 { Some(i) } else { None })
        .collect();
    let mid_centroids: Vec<_> = keep_indices
        .iter()
        .map(|&i| mid_centroids[i].to_vec())
        .collect();
    let alloc: Vec<_> = keep_indices.iter().map(|&i| alloc[i].clone()).collect();
    let alloc_size: Vec<_> = keep_indices.iter().map(|&i| alloc_size[i]).collect();
    let alloc_lists = successive_quotients_allocate(bottom_list, alloc_size);
    pgrx::info!("approximation: bottom clustering started");
    let mut mid_children = vec![Vec::new(); mid_centroids.len()];
    let mut bottom_centroids = vec![];
    let mut bottom_children = vec![];
    let mut offset = 0;
    let mut global_alloc: HashMap<usize, usize> = HashMap::new();
    for (i, nlist) in alloc_lists.into_iter().enumerate() {
        pgrx::check_for_interrupts!();
        let alloc_i = if let Some(a) = alloc.get(i) {
            a.clone()
        } else {
            unreachable!()
        };
        let sub_samples = {
            let mut s = Square::new(vector_options.dims as _);
            for j in alloc_i {
                s.push_slice(&samples[j]);
            }
            s
        };
        let mut f = k_means::k_means(
            vector_options.dims as _,
            sub_samples,
            nlist as usize,
            internal_build.build_threads as _,
            [7; 32],
        );
        if internal_build.spherical_centroids {
            f.sphericalize();
        }
        for _ in 0..num_iterations {
            f.assign();
            f.update();
            if internal_build.spherical_centroids {
                f.sphericalize();
            }
        }
        let final_assign = f.assign();
        let (sub_centroids, sub_samples) = f.finish();
        let sub_alloc =
            final_assign
                .into_iter()
                .fold(vec![0_usize; sub_centroids.len()], |mut acc, target| {
                    acc[target] += 1;
                    acc
                });
        let sub_centroids = {
            let it = sub_centroids
                .into_iter()
                .enumerate()
                .filter(|(j, _)| sub_alloc[*j] > 0)
                .map(|(_, c)| c);
            let mut s = Square::new(vector_options.dims as _);
            for c in it {
                s.push_slice(c);
            }
            s
        };
        for j in 0..sub_samples.len() {
            let si = alloc[i][j];
            let target = k_means_lookup(&sub_samples[j], &sub_centroids);
            global_alloc.insert(si, offset + target);
        }
        k_means_report(i as u32 + 1);
        mid_children[i] = (offset as u32..offset as u32 + sub_centroids.len() as u32).collect();
        offset += sub_centroids.len();
        bottom_children.extend(vec![Vec::new(); sub_centroids.len()]);
        bottom_centroids.extend(sub_centroids.into_iter().map(|x| x.to_vec()));
    }
    pgrx::info!("approximation: bottom clustering finished");
    result.push(Structure {
        centroids: bottom_centroids.clone(),
        children: bottom_children,
    });
    result.push(Structure {
        centroids: mid_centroids.clone(),
        children: mid_children,
    });
    (result, global_alloc)
}

fn restore_from_approximate_tree(
    vector_options: VectorOptions,
    internal_build: VchordrqInternalBuildOptions,
    reduction: bool,
    opfamily: Opfamily,
    fetcher: &mut HeapFetcher,
    samples_idx: Vec<ItemPointerData>,
    tree: Vec<Structure<Normalized>>,
    allocation: HashMap<usize, usize>,
) -> Vec<Structure<Normalized>> {
    let drop_top = internal_build.lists.len() == 1;
    let num_iterations = internal_build.kmeans_iterations as _;
    let mut result = Vec::<Structure<Normalized>>::new();
    let centroids = if reduction {
        pgrx::info!(
            "approximation: restoring the precise index structure and dimension from {} samples.",
            samples_idx.len()
        );
        let dim = vector_options.dims as usize;
        let bottom = tree.first().unwrap();
        let top = tree.last().unwrap();
        pgrx::info!("approximation: start recovering samples from the table");
        let mut centroids = traverse_heap_again(
            opfamily,
            fetcher,
            dim,
            bottom.len(),
            internal_build.spherical_centroids,
            samples_idx,
            allocation,
        );
        pgrx::info!("approximation: samples recovered from the table");
        result.push(Structure {
            centroids: centroids.clone(),
            children: bottom.children.clone(),
        });

        if !drop_top {
            let mut mid_centroids = vec![vec![0.0_f32; dim]; top.len()];
            for (i, children) in top.children.iter().enumerate() {
                for c in children.iter().copied() {
                    mid_centroids[i] = f32::vector_add(&mid_centroids[i], &centroids[c as usize]);
                }
                k_means_centroids_inplace(
                    &mut mid_centroids[i],
                    children.len() as u32,
                    internal_build.spherical_centroids,
                );
            }
            result.push(Structure {
                centroids: mid_centroids.clone(),
                children: top.children.clone(),
            });
            centroids = mid_centroids;
        }
        centroids
    } else {
        pgrx::info!(
            "approximation: restoring the precise index structure from {} samples.",
            samples_idx.len()
        );
        if !drop_top {
            result = tree;
        } else {
            result = tree.into_iter().rev().skip(1).collect();
        }
        result.last().unwrap().centroids.clone()
    };
    let samples = {
        let mut s = Square::new(vector_options.dims as _);
        for c in centroids.iter() {
            s.push_slice(c.as_slice());
        }
        s
    };
    let mut f = k_means::k_means(
        vector_options.dims as _,
        samples,
        1,
        internal_build.build_threads as _,
        [7; 32],
    );
    if internal_build.spherical_centroids {
        f.sphericalize();
    }
    for _ in 0..num_iterations {
        f.assign();
        f.update();
        if internal_build.spherical_centroids {
            f.sphericalize();
        }
    }
    let top_centroids = f.finish().0;
    result.push(Structure {
        centroids: top_centroids.into_iter().map(|x| x.to_vec()).collect(),
        children: vec![(0..centroids.len() as u32).collect(); 1],
    });
    result
}

fn traverse_heap_again(
    opfamily: Opfamily,
    fetcher: &mut HeapFetcher,
    dim: usize,
    centroids_len: usize,
    is_spherical: bool,
    samples_idx: Vec<ItemPointerData>,
    allocation: HashMap<usize, usize>,
) -> Vec<Vec<f32>> {
    use crate::index::fetcher::Tuple;
    let mut centroids = vec![vec![0.0_f32; dim]; centroids_len];
    let mut idx_alloc = 0_usize;
    let mut count = vec![0_u32; centroids_len];
    for ctid in samples_idx.into_iter() {
        let key = ctid_to_key(ctid);
        let Some(mut tuple) = fetcher.fetch(key) else {
            unreachable!()
        };
        let (values, is_nulls) = tuple.build();
        let datum = (!is_nulls[0]).then_some(values[0]);
        if let Some(datum) = datum {
            let vectors = unsafe { opfamily.store(datum) };
            if let Some(vectors) = vectors {
                for (vector, _) in vectors {
                    let x = match vector {
                        OwnedVector::Vecf32(x) => VectOwned::normalize(x),
                        OwnedVector::Vecf16(x) => VectOwned::normalize(x),
                    };
                    let centroid_id = allocation[&idx_alloc];
                    centroids[centroid_id] = f32::vector_add(&centroids[centroid_id], &x);
                    count[centroid_id] += 1;
                    idx_alloc += 1;
                }
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }
    for i in 0..centroids_len {
        k_means_centroids_inplace(&mut centroids[i], count[i], is_spherical);
    }
    centroids
}

fn make_internal_build(
    vector_options: VectorOptions,
    internal_build: VchordrqInternalBuildOptions,
    samples: Square,
    reporter: &PostgresReporter,
) -> Vec<Structure<Normalized>> {
    use std::iter::once;
    let mut result = Vec::<Structure<Normalized>>::new();
    let mut samples = Some(samples);
    for w in internal_build.lists.iter().rev().copied().chain(once(1)) {
        let input = if let Some(structure) = result.last() {
            let mut input = Square::new(vector_options.dims as _);
            for slice in structure.centroids.iter() {
                input.push_slice(slice);
            }
            input
        } else if let Some(samples) = samples.take() {
            samples
        } else {
            unreachable!()
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
        let mut f = k_means::k_means(num_dims, input, num_lists, num_threads, [7; 32]);
        if internal_build.spherical_centroids {
            f.sphericalize();
        }
        for i in 0..num_iterations {
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
            f.assign();
            f.update();
            if internal_build.spherical_centroids {
                f.sphericalize();
            }
        }
        let centroids = f.finish().0;
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
                let target = k_means_lookup(&structure.centroids[i as usize], &centroids);
                children[target].push(i);
            }
            let (centroids, children) = std::iter::zip(&centroids, children)
                .filter(|(_, children)| !children.is_empty())
                .map(|(centroids, children)| (centroids.to_vec(), children))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            result.push(Structure {
                centroids,
                children,
            });
        } else {
            let children = vec![Vec::new(); centroids.len()];
            result.push(Structure {
                centroids: centroids.into_iter().map(|x| x.to_vec()).collect(),
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
) -> Vec<Structure<Normalized>> {
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
            vectors.insert(id, vector.as_borrowed().slice().to_vec());
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
        vectors: &BTreeMap<i32, Normalized>,
        children: &BTreeMap<i32, Vec<i32>>,
    ) -> (Vec<Normalized>, Vec<Vec<u32>>) {
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
