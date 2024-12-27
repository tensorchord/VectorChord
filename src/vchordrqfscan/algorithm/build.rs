use crate::postgres::BufferWriteGuard;
use crate::postgres::Relation;
use crate::vchordrqfscan::algorithm::rabitq;
use crate::vchordrqfscan::algorithm::tuples::*;
use crate::vchordrqfscan::index::am_options::Opfamily;
use crate::vchordrqfscan::types::VchordrqfscanBuildOptions;
use crate::vchordrqfscan::types::VchordrqfscanExternalBuildOptions;
use crate::vchordrqfscan::types::VchordrqfscanIndexingOptions;
use crate::vchordrqfscan::types::VchordrqfscanInternalBuildOptions;
use crate::vchordrqfscan::types::VectorOptions;
use base::distance::DistanceKind;
use base::search::Pointer;
use base::simd::ScalarLike;
use rand::Rng;
use rkyv::ser::serializers::AllocSerializer;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait HeapRelation {
    fn traverse<F>(&self, progress: bool, callback: F)
    where
        F: FnMut((Pointer, Vec<f32>));
    fn opfamily(&self) -> Opfamily;
}

pub trait Reporter {
    fn tuples_total(&mut self, tuples_total: u64);
}

pub fn build<T: HeapRelation, R: Reporter>(
    vector_options: VectorOptions,
    vchordrqfscan_options: VchordrqfscanIndexingOptions,
    heap_relation: T,
    relation: Relation,
    mut reporter: R,
) {
    let dims = vector_options.dims;
    let is_residual =
        vchordrqfscan_options.residual_quantization && vector_options.d == DistanceKind::L2;
    let structures = match vchordrqfscan_options.build {
        VchordrqfscanBuildOptions::External(external_build) => Structure::extern_build(
            vector_options.clone(),
            heap_relation.opfamily(),
            external_build.clone(),
        ),
        VchordrqfscanBuildOptions::Internal(internal_build) => {
            let mut tuples_total = 0_u64;
            let samples = {
                let mut rand = rand::thread_rng();
                let max_number_of_samples = internal_build
                    .lists
                    .last()
                    .unwrap()
                    .saturating_mul(internal_build.sampling_factor);
                let mut samples = Vec::new();
                let mut number_of_samples = 0_u32;
                heap_relation.traverse(false, |(_, vector)| {
                    assert_eq!(dims as usize, vector.len(), "invalid vector dimensions");
                    if number_of_samples < max_number_of_samples {
                        samples.push(vector);
                        number_of_samples += 1;
                    } else {
                        let index = rand.gen_range(0..max_number_of_samples) as usize;
                        samples[index] = vector;
                    }
                    tuples_total += 1;
                });
                samples
            };
            reporter.tuples_total(tuples_total);
            Structure::internal_build(vector_options.clone(), internal_build.clone(), samples)
        }
    };
    let mut meta = Tape::create(&relation, false);
    assert_eq!(meta.first(), 0);
    let mut forwards = Tape::<std::convert::Infallible>::create(&relation, false);
    assert_eq!(forwards.first(), 1);
    let mut vectors = Tape::create(&relation, true);
    assert_eq!(vectors.first(), 2);
    let mut pointer_of_means = Vec::<Vec<(u32, u16)>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let pointer = vectors.push(&VectorTuple {
                payload: None,
                vector: structures[i].means[j].clone(),
            });
            level.push(pointer);
        }
        pointer_of_means.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let tape = Tape::<Height0Tuple>::create(&relation, false);
                level.push(tape.first());
            } else {
                let mut tape = Tape::<Height1Tuple>::create(&relation, false);
                let mut cache = Vec::new();
                let h2_mean = &structures[i].means[j];
                let h2_children = &structures[i].children[j];
                for child in h2_children.iter().copied() {
                    let h1_mean = &structures[i - 1].means[child as usize];
                    let code = if is_residual {
                        rabitq::code(dims, &f32::vector_sub(h1_mean, h2_mean))
                    } else {
                        rabitq::code(dims, h1_mean)
                    };
                    cache.push((child, code));
                    if cache.len() == 32 {
                        let group = std::mem::take(&mut cache);
                        let codes = std::array::from_fn(|k| group[k].1.clone());
                        let packed = rabitq::pack_codes(dims, codes);
                        tape.push(&Height1Tuple {
                            mask: [true; 32],
                            mean: std::array::from_fn(|k| {
                                pointer_of_means[i - 1][group[k].0 as usize]
                            }),
                            first: std::array::from_fn(|k| {
                                pointer_of_firsts[i - 1][group[k].0 as usize]
                            }),
                            dis_u_2: packed.dis_u_2,
                            factor_ppc: packed.factor_ppc,
                            factor_ip: packed.factor_ip,
                            factor_err: packed.factor_err,
                            t: packed.t,
                        });
                    }
                }
                if !cache.is_empty() {
                    let group = std::mem::take(&mut cache);
                    let codes = std::array::from_fn(|k| {
                        if k < group.len() {
                            group[k].1.clone()
                        } else {
                            rabitq::dummy_code(dims)
                        }
                    });
                    let packed = rabitq::pack_codes(dims, codes);
                    tape.push(&Height1Tuple {
                        mask: std::array::from_fn(|k| k < group.len()),
                        mean: std::array::from_fn(|k| {
                            if k < group.len() {
                                pointer_of_means[i - 1][group[k].0 as usize]
                            } else {
                                Default::default()
                            }
                        }),
                        first: std::array::from_fn(|k| {
                            if k < group.len() {
                                pointer_of_firsts[i - 1][group[k].0 as usize]
                            } else {
                                Default::default()
                            }
                        }),
                        dis_u_2: packed.dis_u_2,
                        factor_ppc: packed.factor_ppc,
                        factor_ip: packed.factor_ip,
                        factor_err: packed.factor_err,
                        t: packed.t,
                    });
                }
                level.push(tape.first());
            }
        }
        pointer_of_firsts.push(level);
    }
    forwards.head.get_opaque_mut().skip = vectors.first();
    meta.push(&MetaTuple {
        dims,
        height_of_root: structures.len() as u32,
        is_residual,
        vectors_first: vectors.first(),
        forwards_first: forwards.first(),
        mean: pointer_of_means.last().unwrap()[0],
        first: pointer_of_firsts.last().unwrap()[0],
    });
}

struct Structure {
    means: Vec<Vec<f32>>,
    children: Vec<Vec<u32>>,
}

impl Structure {
    fn len(&self) -> usize {
        self.children.len()
    }
    fn internal_build(
        vector_options: VectorOptions,
        internal_build: VchordrqfscanInternalBuildOptions,
        mut samples: Vec<Vec<f32>>,
    ) -> Vec<Self> {
        use std::iter::once;
        for sample in samples.iter_mut() {
            *sample = crate::projection::project(sample);
        }
        let mut result = Vec::<Self>::new();
        for w in internal_build.lists.iter().rev().copied().chain(once(1)) {
            let means = crate::utils::parallelism::RayonParallelism::scoped(
                internal_build.build_threads as _,
                Arc::new(|| {
                    pgrx::check_for_interrupts!();
                }),
                |parallelism| {
                    crate::utils::k_means::k_means(
                        parallelism,
                        w as usize,
                        vector_options.dims as usize,
                        if let Some(structure) = result.last() {
                            &structure.means
                        } else {
                            &samples
                        },
                        internal_build.spherical_centroids,
                        10,
                    )
                },
            )
            .expect("failed to create thread pool");
            if let Some(structure) = result.last() {
                let mut children = vec![Vec::new(); means.len()];
                for i in 0..structure.len() as u32 {
                    let target =
                        crate::utils::k_means::k_means_lookup(&structure.means[i as usize], &means);
                    children[target].push(i);
                }
                let (means, children) = std::iter::zip(means, children)
                    .filter(|(_, x)| !x.is_empty())
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                result.push(Structure { means, children });
            } else {
                let children = vec![Vec::new(); means.len()];
                result.push(Structure { means, children });
            }
        }
        result
    }
    fn extern_build(
        vector_options: VectorOptions,
        _opfamily: Opfamily,
        external_build: VchordrqfscanExternalBuildOptions,
    ) -> Vec<Self> {
        use std::collections::BTreeMap;
        let VchordrqfscanExternalBuildOptions { table } = external_build;
        let query = format!("SELECT id, parent, vector FROM {table};");
        let mut parents = BTreeMap::new();
        let mut vectors = BTreeMap::new();
        pgrx::spi::Spi::connect(|client| {
            use crate::datatype::memory_pgvector_vector::PgvectorVectorOutput;
            use base::vector::VectorBorrowed;
            use pgrx::pg_sys::panic::ErrorReportable;
            let table = client.select(&query, None, None).unwrap_or_report();
            for row in table {
                let id: Option<i32> = row.get_by_name("id").unwrap();
                let parent: Option<i32> = row.get_by_name("parent").unwrap();
                let vector: Option<PgvectorVectorOutput> = row.get_by_name("vector").unwrap();
                let id = id.expect("extern build: id could not be NULL");
                let vector = vector.expect("extern build: vector could not be NULL");
                let pop = parents.insert(id, parent);
                if pop.is_some() {
                    pgrx::error!(
                        "external build: there are at least two lines have same id, id = {id}"
                    );
                }
                if vector_options.dims != vector.as_borrowed().dims() {
                    pgrx::error!("extern build: incorrect dimension, id = {id}");
                }
                vectors.insert(id, crate::projection::project(vector.as_borrowed().slice()));
            }
        });
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
                    pgrx::error!(
                        "external build: parent does not exist, id = {id}, parent = {parent}"
                    );
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
            pgrx::error!("extern build: there are no root");
        };
        let mut heights = BTreeMap::<_, _>::new();
        fn dfs_for_heights(
            heights: &mut BTreeMap<i32, Option<u32>>,
            children: &BTreeMap<i32, Vec<i32>>,
            u: i32,
        ) {
            if heights.contains_key(&u) {
                pgrx::error!("extern build: detect a cycle, id = {u}");
            }
            heights.insert(u, None);
            let mut height = None;
            for &v in children[&u].iter() {
                dfs_for_heights(heights, children, v);
                let new = heights[&v].unwrap() + 1;
                if let Some(height) = height {
                    if height != new {
                        pgrx::error!("extern build: two heights, id = {u}");
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
                "extern build: unexpected tree height, height = {}",
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
                .filter(|(_, &(h, _))| h == height)
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
            let (means, children) = extract(height, &labels, &vectors, &children);
            result.push(Structure { means, children });
        }
        result
    }
}

struct Tape<'a, T> {
    relation: &'a Relation,
    head: BufferWriteGuard,
    first: u32,
    tracking_freespace: bool,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, T> Tape<'a, T> {
    fn create(relation: &'a Relation, tracking_freespace: bool) -> Self {
        let head = relation.extend(tracking_freespace);
        let first = head.id();
        Self {
            relation,
            head,
            first,
            tracking_freespace,
            _phantom: PhantomData,
        }
    }
    fn first(&self) -> u32 {
        self.first
    }
}

impl<T> Tape<'_, T>
where
    T: rkyv::Serialize<AllocSerializer<8192>>,
{
    fn push(&mut self, x: &T) -> (u32, u16) {
        let bytes = rkyv::to_bytes(x).expect("failed to serialize");
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            let next = self.relation.extend(self.tracking_freespace);
            self.head.get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.alloc(&bytes) {
                (self.head.id(), i)
            } else {
                panic!("tuple is too large to fit in a fresh page")
            }
        }
    }
}
