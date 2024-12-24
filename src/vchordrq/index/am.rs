use crate::postgres::{Page, Relation};
use crate::vchordrq::algorithm::build::{HeapRelation, Reporter};
use crate::vchordrq::algorithm::tuples::Vector;
use crate::vchordrq::algorithm::{self, PageGuard};
use crate::vchordrq::algorithm::{RelationRead, RelationWrite};
use crate::vchordrq::index::am_options::{Opfamily, Reloption};
use crate::vchordrq::index::am_scan::Scanner;
use crate::vchordrq::index::utils::{ctid_to_pointer, pointer_to_ctid};
use crate::vchordrq::index::{am_options, am_scan};
use crate::vchordrq::types::{OwnedVector, VectorKind};
use base::search::Pointer;
use base::vector::VectOwned;
use half::f16;
use pgrx::datum::Internal;
use pgrx::pg_sys::Datum;
use std::collections::HashMap;
use std::ops::Deref;

static mut RELOPT_KIND_VCHORDRQ: pgrx::pg_sys::relopt_kind::Type = 0;

pub unsafe fn init() {
    unsafe {
        (&raw mut RELOPT_KIND_VCHORDRQ).write(pgrx::pg_sys::add_reloption_kind());
        pgrx::pg_sys::add_string_reloption(
            (&raw const RELOPT_KIND_VCHORDRQ).read(),
            c"options".as_ptr(),
            c"Vector index options, represented as a TOML string.".as_ptr(),
            c"".as_ptr(),
            None,
            pgrx::pg_sys::AccessExclusiveLock as pgrx::pg_sys::LOCKMODE,
        );
    }
}

#[pgrx::pg_extern(sql = "")]
fn _vchordrq_amhandler(_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> Internal {
    type T = pgrx::pg_sys::IndexAmRoutine;
    unsafe {
        let index_am_routine = pgrx::pg_sys::palloc0(size_of::<T>()) as *mut T;
        index_am_routine.write(AM_HANDLER);
        Internal::from(Some(Datum::from(index_am_routine)))
    }
}

const AM_HANDLER: pgrx::pg_sys::IndexAmRoutine = {
    let mut am_routine =
        unsafe { std::mem::MaybeUninit::<pgrx::pg_sys::IndexAmRoutine>::zeroed().assume_init() };

    am_routine.type_ = pgrx::pg_sys::NodeTag::T_IndexAmRoutine;

    am_routine.amsupport = 1;
    am_routine.amcanorderbyop = true;

    #[cfg(feature = "pg17")]
    {
        am_routine.amcanbuildparallel = true;
    }

    // Index access methods that set `amoptionalkey` to `false`
    // must index all tuples, even if the first column is `NULL`.
    // However, PostgreSQL does not generate a path if there is no
    // index clauses, even if there is a `ORDER BY` clause.
    // So we have to set it to `true` and set costs of every path
    // for vector index scans without `ORDER BY` clauses a large number
    // and throw errors if someone really wants such a path.
    am_routine.amoptionalkey = true;

    am_routine.amvalidate = Some(amvalidate);
    am_routine.amoptions = Some(amoptions);
    am_routine.amcostestimate = Some(amcostestimate);

    am_routine.ambuild = Some(ambuild);
    am_routine.ambuildempty = Some(ambuildempty);
    am_routine.aminsert = Some(aminsert);
    am_routine.ambulkdelete = Some(ambulkdelete);
    am_routine.amvacuumcleanup = Some(amvacuumcleanup);

    am_routine.ambeginscan = Some(ambeginscan);
    am_routine.amrescan = Some(amrescan);
    am_routine.amgettuple = Some(amgettuple);
    am_routine.amendscan = Some(amendscan);

    am_routine
};

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvalidate(_opclass_oid: pgrx::pg_sys::Oid) -> bool {
    true
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amoptions(reloptions: Datum, validate: bool) -> *mut pgrx::pg_sys::bytea {
    let rdopts = unsafe {
        pgrx::pg_sys::build_reloptions(
            reloptions,
            validate,
            (&raw const RELOPT_KIND_VCHORDRQ).read(),
            size_of::<Reloption>(),
            Reloption::TAB.as_ptr(),
            Reloption::TAB.len() as _,
        )
    };
    rdopts as *mut pgrx::pg_sys::bytea
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amcostestimate(
    _root: *mut pgrx::pg_sys::PlannerInfo,
    path: *mut pgrx::pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pgrx::pg_sys::Cost,
    index_total_cost: *mut pgrx::pg_sys::Cost,
    index_selectivity: *mut pgrx::pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    unsafe {
        if (*path).indexorderbys.is_null() && (*path).indexclauses.is_null() {
            *index_startup_cost = f64::MAX;
            *index_total_cost = f64::MAX;
            *index_selectivity = 0.0;
            *index_correlation = 0.0;
            *index_pages = 0.0;
            return;
        }
        *index_startup_cost = 0.0;
        *index_total_cost = 0.0;
        *index_selectivity = 1.0;
        *index_correlation = 1.0;
        *index_pages = 0.0;
    }
}

#[derive(Debug, Clone)]
struct PgReporter {}

impl Reporter for PgReporter {
    fn tuples_total(&mut self, tuples_total: u64) {
        unsafe {
            pgrx::pg_sys::pgstat_progress_update_param(
                pgrx::pg_sys::PROGRESS_CREATEIDX_TUPLES_TOTAL as _,
                tuples_total as _,
            );
        }
    }
}

impl PgReporter {
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
pub unsafe extern "C" fn ambuild(
    heap: pgrx::pg_sys::Relation,
    index: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
) -> *mut pgrx::pg_sys::IndexBuildResult {
    use validator::Validate;
    #[derive(Debug, Clone)]
    pub struct Heap {
        heap: pgrx::pg_sys::Relation,
        index: pgrx::pg_sys::Relation,
        index_info: *mut pgrx::pg_sys::IndexInfo,
        opfamily: Opfamily,
    }
    impl<V: Vector> HeapRelation<V> for Heap {
        fn traverse<F>(&self, progress: bool, callback: F)
        where
            F: FnMut((Pointer, V)),
        {
            pub struct State<'a, F> {
                pub this: &'a Heap,
                pub callback: F,
            }
            #[pgrx::pg_guard]
            unsafe extern "C" fn call<F, V: Vector>(
                _index: pgrx::pg_sys::Relation,
                ctid: pgrx::pg_sys::ItemPointer,
                values: *mut Datum,
                is_null: *mut bool,
                _tuple_is_alive: bool,
                state: *mut core::ffi::c_void,
            ) where
                F: FnMut((Pointer, V)),
            {
                let state = unsafe { &mut *state.cast::<State<F>>() };
                let opfamily = state.this.opfamily;
                let vector = unsafe { opfamily.datum_to_vector(*values.add(0), *is_null.add(0)) };
                let pointer = unsafe { ctid_to_pointer(ctid.read()) };
                if let Some(vector) = vector {
                    (state.callback)((pointer, V::from_owned(vector)));
                }
            }
            let table_am = unsafe { &*(*self.heap).rd_tableam };
            let mut state = State {
                this: self,
                callback,
            };
            unsafe {
                table_am.index_build_range_scan.unwrap()(
                    self.heap,
                    self.index,
                    self.index_info,
                    true,
                    false,
                    progress,
                    0,
                    pgrx::pg_sys::InvalidBlockNumber,
                    Some(call::<F, V>),
                    (&mut state) as *mut State<F> as *mut _,
                    std::ptr::null_mut(),
                );
            }
        }

        fn opfamily(&self) -> Opfamily {
            self.opfamily
        }
    }
    let (vector_options, vchordrq_options) = unsafe { am_options::options(index) };
    if let Err(errors) = Validate::validate(&vector_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    if vector_options.dims == 0 {
        pgrx::error!("error while validating options: dimension cannot be 0");
    }
    if vector_options.dims > 60000 {
        pgrx::error!("error while validating options: dimension is too large");
    }
    if let Err(errors) = Validate::validate(&vchordrq_options) {
        pgrx::error!("error while validating options: {}", errors);
    }
    let opfamily = unsafe { am_options::opfamily(index) };
    let heap_relation = Heap {
        heap,
        index,
        index_info,
        opfamily,
    };
    let mut reporter = PgReporter {};
    let index_relation = unsafe { Relation::new(index) };
    match opfamily.vector_kind() {
        VectorKind::Vecf32 => algorithm::build::build::<VectOwned<f32>, Heap, _>(
            vector_options,
            vchordrq_options,
            heap_relation.clone(),
            index_relation.clone(),
            reporter.clone(),
        ),
        VectorKind::Vecf16 => algorithm::build::build::<VectOwned<f16>, Heap, _>(
            vector_options,
            vchordrq_options,
            heap_relation.clone(),
            index_relation.clone(),
            reporter.clone(),
        ),
    }
    let cache = {
        let n = index_relation.len();
        let mut dir = HashMap::<u32, usize>::with_capacity(n as _);
        let mut pages = Vec::<Box<Page>>::new();
        {
            use crate::vchordrq::algorithm::tuples::{Height1Tuple, MetaTuple};
            let mut read = |id| {
                let result = index_relation.read(id);
                dir.insert(id, pages.len());
                pages.push(result.clone_into_boxed());
                result
            };
            let meta_guard = read(0);
            let meta_tuple = meta_guard
                .get(1)
                .map(rkyv::check_archived_root::<MetaTuple>)
                .expect("data corruption")
                .expect("data corruption");
            let mut firsts = vec![meta_tuple.first];
            let mut make_firsts = |firsts| {
                let mut results = Vec::new();
                for first in firsts {
                    let mut current = first;
                    while current != u32::MAX {
                        let h1_guard = read(current);
                        for i in 1..=h1_guard.len() {
                            let h1_tuple = h1_guard
                                .get(i)
                                .map(rkyv::check_archived_root::<Height1Tuple>)
                                .expect("data corruption")
                                .expect("data corruption");
                            results.push(h1_tuple.first);
                        }
                        current = h1_guard.get_opaque().next;
                    }
                }
                results
            };
            for _ in (1..meta_tuple.height_of_root).rev() {
                firsts = make_firsts(firsts);
            }
        }
        (dir, pages)
    };
    if let Some(leader) =
        unsafe { VchordrqLeader::enter(heap, index, (*index_info).ii_Concurrent, &cache) }
    {
        unsafe {
            parallel_build(
                index,
                heap,
                index_info,
                leader.tablescandesc,
                leader.vchordrqshared,
                Some(reporter),
                &*leader.cache_0,
                &*leader.cache_1,
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
                    pgrx::pg_sys::WaitEventIPC::WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN,
                );
            }
            pgrx::pg_sys::ConditionVariableCancelSleep();
        }
    } else {
        let mut indtuples = 0;
        reporter.tuples_done(indtuples);
        let relation = unsafe { Relation::new(index) };
        let relation = CachingRelation {
            cache: &(cache.0, cache.1.iter().map(|x| x.as_ref()).collect()),
            relation,
        };
        match opfamily.vector_kind() {
            VectorKind::Vecf32 => {
                HeapRelation::<VectOwned<f32>>::traverse(
                    &heap_relation,
                    true,
                    |(pointer, vector)| {
                        algorithm::insert::insert::<VectOwned<f32>>(
                            relation.clone(),
                            pointer,
                            vector,
                            opfamily.distance_kind(),
                            true,
                        );
                        indtuples += 1;
                        reporter.tuples_done(indtuples);
                    },
                );
            }
            VectorKind::Vecf16 => {
                HeapRelation::<VectOwned<f16>>::traverse(
                    &heap_relation,
                    true,
                    |(pointer, vector)| {
                        algorithm::insert::insert::<VectOwned<f16>>(
                            relation.clone(),
                            pointer,
                            vector,
                            opfamily.distance_kind(),
                            true,
                        );
                        indtuples += 1;
                        reporter.tuples_done(indtuples);
                    },
                );
            }
        }
    }
    unsafe { pgrx::pgbox::PgBox::<pgrx::pg_sys::IndexBuildResult>::alloc0().into_pg() }
}

#[derive(Clone)]
struct CachingRelation<'a, R> {
    cache: &'a (HashMap<u32, usize>, Vec<&'a Page>),
    relation: R,
}

enum CachingRelationReadGuard<'a, G> {
    Wrapping(G),
    Cached(u32, &'a Page),
}

impl<G: PageGuard> PageGuard for CachingRelationReadGuard<'_, G> {
    fn id(&self) -> u32 {
        match self {
            CachingRelationReadGuard::Wrapping(x) => x.id(),
            CachingRelationReadGuard::Cached(id, _) => *id,
        }
    }
}

impl<G: Deref<Target = Page>> Deref for CachingRelationReadGuard<'_, G> {
    type Target = Page;

    fn deref(&self) -> &Self::Target {
        match self {
            CachingRelationReadGuard::Wrapping(x) => x,
            CachingRelationReadGuard::Cached(_, page) => page,
        }
    }
}

impl<R: RelationRead> RelationRead for CachingRelation<'_, R> {
    type ReadGuard<'a>
        = CachingRelationReadGuard<'a, R::ReadGuard<'a>>
    where
        Self: 'a;

    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        if let Some(&x) = self.cache.0.get(&id) {
            CachingRelationReadGuard::Cached(id, self.cache.1[x])
        } else {
            CachingRelationReadGuard::Wrapping(self.relation.read(id))
        }
    }
}

impl<R: RelationWrite> RelationWrite for CachingRelation<'_, R> {
    type WriteGuard<'a>
        = R::WriteGuard<'a>
    where
        Self: 'a;

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.relation.write(id, tracking_freespace)
    }

    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.relation.extend(tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.relation.search(freespace)
    }
}

struct VchordrqShared {
    /* Immutable state */
    heaprelid: pgrx::pg_sys::Oid,
    indexrelid: pgrx::pg_sys::Oid,
    isconcurrent: bool,
    est_cache_0: usize,
    est_cache_1: usize,

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

struct VchordrqLeader {
    pcxt: *mut pgrx::pg_sys::ParallelContext,
    nparticipants: i32,
    vchordrqshared: *mut VchordrqShared,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    cache_0: *const [u8],
    cache_1: *const [u8],
    snapshot: pgrx::pg_sys::Snapshot,
}

impl VchordrqLeader {
    pub unsafe fn enter(
        heap: pgrx::pg_sys::Relation,
        index: pgrx::pg_sys::Relation,
        isconcurrent: bool,
        cache: &(HashMap<u32, usize>, Vec<Box<Page>>),
    ) -> Option<Self> {
        let cache_mapping: Vec<u8> = bincode::serialize(&cache.0).unwrap();

        unsafe fn compute_parallel_workers(
            heap: pgrx::pg_sys::Relation,
            index: pgrx::pg_sys::Relation,
        ) -> i32 {
            unsafe {
                if pgrx::pg_sys::plan_create_index_workers((*heap).rd_id, (*index).rd_id) == 0 {
                    return 0;
                }
                if !(*heap).rd_options.is_null() {
                    let std_options = (*heap).rd_options.cast::<pgrx::pg_sys::StdRdOptions>();
                    std::cmp::min(
                        (*std_options).parallel_workers,
                        pgrx::pg_sys::max_parallel_maintenance_workers,
                    )
                } else {
                    pgrx::pg_sys::max_parallel_maintenance_workers
                }
            }
        }

        let request = unsafe { compute_parallel_workers(heap, index) };
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
            unsafe { pgrx::pg_sys::table_parallelscan_estimate(heap, snapshot) };
        let est_cache_0 = cache_mapping.len();
        let est_cache_1 = cache.1.len() * size_of::<Page>();
        unsafe {
            estimate_chunk(&mut (*pcxt).estimator, size_of::<VchordrqShared>());
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, est_tablescandesc);
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, est_cache_0);
            estimate_keys(&mut (*pcxt).estimator, 1);
            estimate_chunk(&mut (*pcxt).estimator, est_cache_1);
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
                heaprelid: (*heap).rd_id,
                indexrelid: (*index).rd_id,
                isconcurrent,
                workersdonecv: std::mem::zeroed(),
                mutex: std::mem::zeroed(),
                nparticipantsdone: 0,
                indtuples: 0,
                est_cache_0,
                est_cache_1,
            });
            pgrx::pg_sys::ConditionVariableInit(&raw mut (*vchordrqshared).workersdonecv);
            pgrx::pg_sys::SpinLockInit(&raw mut (*vchordrqshared).mutex);
            vchordrqshared
        };

        let tablescandesc = unsafe {
            let tablescandesc = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_tablescandesc)
                .cast::<pgrx::pg_sys::ParallelTableScanDescData>();
            pgrx::pg_sys::table_parallelscan_initialize(heap, tablescandesc, snapshot);
            tablescandesc
        };

        let cache_0 = unsafe {
            let cache_0 = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_cache_0).cast::<u8>();
            std::ptr::copy(cache_mapping.as_ptr(), cache_0, est_cache_0);
            core::ptr::slice_from_raw_parts(cache_0, est_cache_0)
        };

        let cache_1 = unsafe {
            let cache_1 = pgrx::pg_sys::shm_toc_allocate((*pcxt).toc, est_cache_1).cast::<u8>();
            for i in 0..cache.1.len() {
                std::ptr::copy(
                    (cache.1[i].deref() as *const Page).cast::<u8>(),
                    cache_1.cast::<Page>().add(i).cast(),
                    size_of::<Page>(),
                );
            }
            core::ptr::slice_from_raw_parts(cache_1, est_cache_1)
        };

        unsafe {
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000001, vchordrqshared.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000002, tablescandesc.cast());
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000003, cache_0 as _);
            pgrx::pg_sys::shm_toc_insert((*pcxt).toc, 0xA000000000000004, cache_1 as _);
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
            vchordrqshared,
            tablescandesc,
            cache_0,
            cache_1,
            snapshot,
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
pub unsafe extern "C" fn vchordrq_parallel_build_main(
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
    let cache_0 = unsafe {
        std::slice::from_raw_parts(
            pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000003, false).cast::<u8>(),
            (*vchordrqshared).est_cache_0,
        )
    };
    let cache_1 = unsafe {
        std::slice::from_raw_parts(
            pgrx::pg_sys::shm_toc_lookup(toc, 0xA000000000000004, false).cast::<u8>(),
            (*vchordrqshared).est_cache_1,
        )
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
            None,
            cache_0,
            cache_1,
        );
    }

    unsafe {
        pgrx::pg_sys::index_close(index, index_lockmode);
        pgrx::pg_sys::table_close(heap, heap_lockmode);
    }
}

unsafe fn parallel_build(
    index: *mut pgrx::pg_sys::RelationData,
    heap: pgrx::pg_sys::Relation,
    index_info: *mut pgrx::pg_sys::IndexInfo,
    tablescandesc: *mut pgrx::pg_sys::ParallelTableScanDescData,
    vchordrqshared: *mut VchordrqShared,
    mut reporter: Option<PgReporter>,
    cache_0: &[u8],
    cache_1: &[u8],
) {
    #[derive(Debug, Clone)]
    pub struct Heap {
        heap: pgrx::pg_sys::Relation,
        index: pgrx::pg_sys::Relation,
        index_info: *mut pgrx::pg_sys::IndexInfo,
        opfamily: Opfamily,
        scan: *mut pgrx::pg_sys::TableScanDescData,
    }
    impl<V: Vector> HeapRelation<V> for Heap {
        fn traverse<F>(&self, progress: bool, callback: F)
        where
            F: FnMut((Pointer, V)),
        {
            pub struct State<'a, F> {
                pub this: &'a Heap,
                pub callback: F,
            }
            #[pgrx::pg_guard]
            unsafe extern "C" fn call<F, V: Vector>(
                _index: pgrx::pg_sys::Relation,
                ctid: pgrx::pg_sys::ItemPointer,
                values: *mut Datum,
                is_null: *mut bool,
                _tuple_is_alive: bool,
                state: *mut core::ffi::c_void,
            ) where
                F: FnMut((Pointer, V)),
            {
                let state = unsafe { &mut *state.cast::<State<F>>() };
                let opfamily = state.this.opfamily;
                let vector = unsafe { opfamily.datum_to_vector(*values.add(0), *is_null.add(0)) };
                let pointer = unsafe { ctid_to_pointer(ctid.read()) };
                if let Some(vector) = vector {
                    (state.callback)((pointer, V::from_owned(vector)));
                }
            }
            let table_am = unsafe { &*(*self.heap).rd_tableam };
            let mut state = State {
                this: self,
                callback,
            };
            unsafe {
                table_am.index_build_range_scan.unwrap()(
                    self.heap,
                    self.index,
                    self.index_info,
                    true,
                    false,
                    progress,
                    0,
                    pgrx::pg_sys::InvalidBlockNumber,
                    Some(call::<F, V>),
                    (&mut state) as *mut State<F> as *mut _,
                    self.scan,
                );
            }
        }

        fn opfamily(&self) -> Opfamily {
            self.opfamily
        }
    }

    let index_relation = unsafe { Relation::new(index) };
    let index_relation = CachingRelation {
        cache: {
            let cache_0: HashMap<u32, usize> = bincode::deserialize(cache_0).unwrap();
            assert!(cache_1.len() % size_of::<Page>() == 0);
            let n = cache_1.len() / size_of::<Page>();
            let cache_1 = unsafe {
                (0..n)
                    .map(|i| &*cache_1.as_ptr().cast::<Page>().add(i))
                    .collect::<Vec<&Page>>()
            };
            &(cache_0, cache_1)
        },
        relation: index_relation,
    };

    let scan = unsafe { pgrx::pg_sys::table_beginscan_parallel(heap, tablescandesc) };
    let opfamily = unsafe { am_options::opfamily(index) };
    let heap_relation = Heap {
        heap,
        index,
        index_info,
        opfamily,
        scan,
    };
    match opfamily.vector_kind() {
        VectorKind::Vecf32 => {
            HeapRelation::<VectOwned<f32>>::traverse(&heap_relation, true, |(pointer, vector)| {
                algorithm::insert::insert::<VectOwned<f32>>(
                    index_relation.clone(),
                    pointer,
                    vector,
                    opfamily.distance_kind(),
                    true,
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
                }
            });
        }
        VectorKind::Vecf16 => {
            HeapRelation::<VectOwned<f16>>::traverse(&heap_relation, true, |(pointer, vector)| {
                algorithm::insert::insert::<VectOwned<f16>>(
                    index_relation.clone(),
                    pointer,
                    vector,
                    opfamily.distance_kind(),
                    true,
                );
                unsafe {
                    let indtuples;
                    {
                        pgrx::pg_sys::SpinLockAcquire(&raw mut (*vchordrqshared).mutex);
                        (*vchordrqshared).indtuples += 1;
                        indtuples = (*vchordrqshared).indtuples;
                        pgrx::pg_sys::SpinLockRelease(&raw mut (*vchordrqshared).mutex);
                    }
                    if let Some(reporter) = reporter.as_mut() {
                        reporter.tuples_done(indtuples);
                    }
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
pub unsafe extern "C" fn ambuildempty(_index: pgrx::pg_sys::Relation) {
    pgrx::error!("Unlogged indexes are not supported.");
}

#[cfg(feature = "pg13")]
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
    index: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    let opfamily = unsafe { am_options::opfamily(index) };
    let vector = unsafe { opfamily.datum_to_vector(*values.add(0), *is_null.add(0)) };
    if let Some(vector) = vector {
        let pointer = ctid_to_pointer(unsafe { heap_tid.read() });
        match opfamily.vector_kind() {
            VectorKind::Vecf32 => algorithm::insert::insert::<VectOwned<f32>>(
                unsafe { Relation::new(index) },
                pointer,
                VectOwned::<f32>::from_owned(vector),
                opfamily.distance_kind(),
                false,
            ),
            VectorKind::Vecf16 => algorithm::insert::insert::<VectOwned<f16>>(
                unsafe { Relation::new(index) },
                pointer,
                VectOwned::<f16>::from_owned(vector),
                opfamily.distance_kind(),
                false,
            ),
        }
    }
    false
}

#[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
#[pgrx::pg_guard]
pub unsafe extern "C" fn aminsert(
    index: pgrx::pg_sys::Relation,
    values: *mut Datum,
    is_null: *mut bool,
    heap_tid: pgrx::pg_sys::ItemPointer,
    _heap: pgrx::pg_sys::Relation,
    _check_unique: pgrx::pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pgrx::pg_sys::IndexInfo,
) -> bool {
    let opfamily = unsafe { am_options::opfamily(index) };
    let vector = unsafe { opfamily.datum_to_vector(*values.add(0), *is_null.add(0)) };
    if let Some(vector) = vector {
        let pointer = ctid_to_pointer(unsafe { heap_tid.read() });
        match opfamily.vector_kind() {
            VectorKind::Vecf32 => algorithm::insert::insert::<VectOwned<f32>>(
                unsafe { Relation::new(index) },
                pointer,
                VectOwned::<f32>::from_owned(vector),
                opfamily.distance_kind(),
                false,
            ),
            VectorKind::Vecf16 => algorithm::insert::insert::<VectOwned<f16>>(
                unsafe { Relation::new(index) },
                pointer,
                VectOwned::<f16>::from_owned(vector),
                opfamily.distance_kind(),
                false,
            ),
        }
    }
    false
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambeginscan(
    index: pgrx::pg_sys::Relation,
    n_keys: std::os::raw::c_int,
    n_orderbys: std::os::raw::c_int,
) -> pgrx::pg_sys::IndexScanDesc {
    use pgrx::memcxt::PgMemoryContexts::CurrentMemoryContext;

    let scan = unsafe { pgrx::pg_sys::RelationGetIndexScan(index, n_keys, n_orderbys) };
    unsafe {
        let scanner = am_scan::scan_make(None, None, false);
        (*scan).opaque = CurrentMemoryContext.leak_and_drop_on_delete(scanner).cast();
    }
    scan
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amrescan(
    scan: pgrx::pg_sys::IndexScanDesc,
    keys: pgrx::pg_sys::ScanKey,
    _n_keys: std::os::raw::c_int,
    orderbys: pgrx::pg_sys::ScanKey,
    _n_orderbys: std::os::raw::c_int,
) {
    unsafe {
        if !keys.is_null() && (*scan).numberOfKeys > 0 {
            std::ptr::copy(keys, (*scan).keyData, (*scan).numberOfKeys as _);
        }
        if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
            std::ptr::copy(orderbys, (*scan).orderByData, (*scan).numberOfOrderBys as _);
        }
        let opfamily = am_options::opfamily((*scan).indexRelation);
        let (orderbys, spheres) = {
            let mut orderbys = Vec::new();
            let mut spheres = Vec::new();
            if (*scan).numberOfOrderBys == 0 && (*scan).numberOfKeys == 0 {
                pgrx::error!(
                    "vector search with no WHERE clause and no ORDER BY clause is not supported"
                );
            }
            for i in 0..(*scan).numberOfOrderBys {
                let data = (*scan).orderByData.add(i as usize);
                let value = (*data).sk_argument;
                let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                match (*data).sk_strategy {
                    1 => orderbys.push(opfamily.datum_to_vector(value, is_null)),
                    _ => unreachable!(),
                }
            }
            for i in 0..(*scan).numberOfKeys {
                let data = (*scan).keyData.add(i as usize);
                let value = (*data).sk_argument;
                let is_null = ((*data).sk_flags & pgrx::pg_sys::SK_ISNULL as i32) != 0;
                match (*data).sk_strategy {
                    2 => spheres.push(opfamily.datum_to_sphere(value, is_null)),
                    _ => unreachable!(),
                }
            }
            (orderbys, spheres)
        };
        let (vector, threshold, recheck) = am_scan::scan_build(orderbys, spheres, opfamily);
        let scanner = (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked();
        let scanner = std::mem::replace(scanner, am_scan::scan_make(vector, threshold, recheck));
        am_scan::scan_release(scanner);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amgettuple(
    scan: pgrx::pg_sys::IndexScanDesc,
    direction: pgrx::pg_sys::ScanDirection::Type,
) -> bool {
    if direction != pgrx::pg_sys::ScanDirection::ForwardScanDirection {
        pgrx::error!("vector search without a forward scan direction is not supported");
    }
    // https://www.postgresql.org/docs/current/index-locking.html
    // If heap entries referenced physical pointers are deleted before
    // they are consumed by PostgreSQL, PostgreSQL will received wrong
    // physical pointers: no rows or irreverent rows are referenced.
    if unsafe { (*(*scan).xs_snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC
    {
        pgrx::error!("scanning with a non-MVCC-compliant snapshot is not supported");
    }
    let scanner = unsafe { (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked() };
    let relation = unsafe { Relation::new((*scan).indexRelation) };
    if let Some((pointer, recheck)) =
        am_scan::scan_next(scanner, relation, unsafe { (*scan).heapRelation })
    {
        let ctid = pointer_to_ctid(pointer);
        unsafe {
            (*scan).xs_heaptid = ctid;
            (*scan).xs_recheckorderby = false;
            (*scan).xs_recheck = recheck;
        }
        true
    } else {
        false
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amendscan(scan: pgrx::pg_sys::IndexScanDesc) {
    unsafe {
        let scanner = (*scan).opaque.cast::<Scanner>().as_mut().unwrap_unchecked();
        let scanner = std::mem::replace(scanner, am_scan::scan_make(None, None, false));
        am_scan::scan_release(scanner);
    }
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn ambulkdelete(
    info: *mut pgrx::pg_sys::IndexVacuumInfo,
    stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
    callback: pgrx::pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::os::raw::c_void,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    let mut stats = stats;
    if stats.is_null() {
        stats = unsafe {
            pgrx::pg_sys::palloc0(size_of::<pgrx::pg_sys::IndexBulkDeleteResult>()).cast()
        };
    }
    let opfamily = unsafe { am_options::opfamily((*info).index) };
    let callback = callback.unwrap();
    let callback = |p: Pointer| unsafe { callback(&mut pointer_to_ctid(p), callback_state) };
    match opfamily.vector_kind() {
        VectorKind::Vecf32 => algorithm::vacuum::vacuum::<VectOwned<f32>>(
            unsafe { Relation::new((*info).index) },
            || unsafe {
                pgrx::pg_sys::vacuum_delay_point();
            },
            callback,
        ),
        VectorKind::Vecf16 => algorithm::vacuum::vacuum::<VectOwned<f16>>(
            unsafe { Relation::new((*info).index) },
            || unsafe {
                pgrx::pg_sys::vacuum_delay_point();
            },
            callback,
        ),
    }
    stats
}

#[pgrx::pg_guard]
pub unsafe extern "C" fn amvacuumcleanup(
    _info: *mut pgrx::pg_sys::IndexVacuumInfo,
    _stats: *mut pgrx::pg_sys::IndexBulkDeleteResult,
) -> *mut pgrx::pg_sys::IndexBulkDeleteResult {
    std::ptr::null_mut()
}

pub(super) unsafe fn fetch_vector(
    opfamily: Opfamily,
    heap_relation: pgrx::pg_sys::Relation,
    attnum: i16,
    payload: u64,
) -> Option<OwnedVector> {
    unsafe {
        let slot = pgrx::pg_sys::table_slot_create(heap_relation, std::ptr::null_mut());
        let table_am = (*heap_relation).rd_tableam;
        let fetch_row_version = (*table_am).tuple_fetch_row_version.unwrap();
        let mut ctid = pointer_to_ctid(Pointer::new(payload));
        fetch_row_version(
            heap_relation,
            &mut ctid,
            &raw mut pgrx::pg_sys::SnapshotAnyData,
            slot,
        );
        assert!(attnum > 0);
        if attnum > (*slot).tts_nvalid {
            pgrx::pg_sys::slot_getsomeattrs(slot, attnum as i32);
        }
        let result = opfamily.datum_to_vector(
            (*slot).tts_values.add(attnum as usize - 1).read(),
            (*slot).tts_isnull.add(attnum as usize - 1).read(),
        );
        if !slot.is_null() {
            pgrx::pg_sys::ExecDropSingleTupleTableSlot(slot);
        }
        result
    }
}

pub(super) unsafe fn get_attribute_number_from_index(
    index: pgrx::pg_sys::Relation,
) -> pgrx::pg_sys::AttrNumber {
    unsafe {
        let a = (*index).rd_index;
        let natts = (*a).indnatts;
        assert!(natts == 1);
        (*a).indkey.values.as_slice(natts as _)[0]
    }
}
