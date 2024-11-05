use crate::algorithm::k_means;
use crate::algorithm::rabitq;
use crate::algorithm::tuples::*;
use crate::index::am_options::Opfamily;
use crate::postgres::BufferWriteGuard;
use crate::postgres::Relation;
use crate::types::RabbitholeBuildOptions;
use crate::types::RabbitholeExternalBuildOptions;
use crate::types::RabbitholeIndexingOptions;
use crate::types::RabbitholeInternalBuildOptions;
use base::distance::DistanceKind;
use base::index::VectorOptions;
use base::scalar::ScalarLike;
use base::search::Pointer;
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
    fn tuples_total(&mut self, tuples_total: usize);
    fn tuples_done(&mut self, tuples_done: usize);
}

pub fn build<T: HeapRelation, R: Reporter>(
    vector_options: VectorOptions,
    rabbithole_options: RabbitholeIndexingOptions,
    heap_relation: T,
    relation: Relation,
    mut reporter: R,
) {
    let dims = vector_options.dims;
    let is_residual =
        rabbithole_options.residual_quantization && vector_options.d == DistanceKind::L2;
    let structure = match rabbithole_options.build {
        RabbitholeBuildOptions::External(external_build) => Structure::extern_build(
            vector_options.clone(),
            heap_relation.opfamily(),
            external_build.clone(),
        ),
        RabbitholeBuildOptions::Internal(internal_build) => {
            let mut tuples_total = 0_usize;
            let samples = {
                let mut rand = rand::thread_rng();
                let max_number_of_samples = internal_build.nlist_1.saturating_mul(256);
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
    let h3_len = structure.h3_len();
    let h2_len = structure.h2_len();
    let h1_len = structure.h1_len();
    let mut meta = Tape::create(&relation, false);
    assert_eq!(meta.first(), 0);
    let mut forwards = Tape::<std::convert::Infallible>::create(&relation, false);
    assert_eq!(forwards.first(), 1);
    let mut vectors = Tape::create(&relation, true);
    assert_eq!(vectors.first(), 2);
    let h3_means = (0..h3_len)
        .map(|i| {
            vectors.push(&VectorTuple {
                payload: None,
                vector: structure.h3_means(i).clone(),
            })
        })
        .collect::<Vec<_>>();
    let h2_means = (0..h2_len)
        .map(|i| {
            vectors.push(&VectorTuple {
                payload: None,
                vector: structure.h2_means(i).clone(),
            })
        })
        .collect::<Vec<_>>();
    let h1_means = (0..h1_len)
        .map(|i| {
            vectors.push(&VectorTuple {
                payload: None,
                vector: structure.h1_means(i).clone(),
            })
        })
        .collect::<Vec<_>>();
    let h1_firsts = (0..h1_len)
        .map(|_| {
            let tape = Tape::<Height0Tuple>::create(&relation, false);
            tape.first()
        })
        .collect::<Vec<_>>();
    let h2_firsts = (0..h2_len)
        .map(|i| {
            let mut tape = Tape::<Height1Tuple>::create(&relation, false);
            let mut cache = Vec::new();
            let h2_mean = structure.h2_means(i);
            let children = structure.h2_children(i);
            for child in children.iter().copied() {
                let h1_mean = structure.h1_means(child);
                let code = if is_residual {
                    rabitq::code(dims, &f32::vector_sub(h1_mean, h2_mean))
                } else {
                    rabitq::code(dims, h1_mean)
                };
                cache.push((child, code));
                if cache.len() == 32 {
                    let group = std::mem::take(&mut cache);
                    let code = std::array::from_fn(|i| group[i].1.clone());
                    let packed = rabitq::pack_codes(dims, code);
                    tape.push(&Height1Tuple {
                        mask: [true; 32],
                        mean: std::array::from_fn(|i| h1_means[group[i].0 as usize]),
                        first: std::array::from_fn(|i| h1_firsts[group[i].0 as usize]),
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
                let codes = std::array::from_fn(|i| {
                    if i < group.len() {
                        group[i].1.clone()
                    } else {
                        rabitq::dummy_code(dims)
                    }
                });
                let packed = rabitq::pack_codes(dims, codes);
                tape.push(&Height1Tuple {
                    mask: std::array::from_fn(|i| i < group.len()),
                    mean: std::array::from_fn(|i| {
                        if i < group.len() {
                            h1_means[group[i].0 as usize]
                        } else {
                            Default::default()
                        }
                    }),
                    first: std::array::from_fn(|i| {
                        if i < group.len() {
                            h1_firsts[group[i].0 as usize]
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
            tape.first()
        })
        .collect::<Vec<_>>();
    let h3_firsts = (0..h3_len)
        .map(|i| {
            let mut tape = Tape::<Height2Tuple>::create(&relation, false);
            let mut cache = Vec::new();
            let h3_mean = structure.h3_means(i);
            let children = structure.h3_children(i);
            for child in children.iter().copied() {
                let h2_mean = structure.h2_means(child);
                let code = if is_residual {
                    rabitq::code(dims, &f32::vector_sub(h2_mean, h3_mean))
                } else {
                    rabitq::code(dims, h2_mean)
                };
                cache.push((child, code));
                if cache.len() == 32 {
                    let group = std::mem::take(&mut cache);
                    let code = std::array::from_fn(|i| group[i].1.clone());
                    let packed = rabitq::pack_codes(dims, code);
                    tape.push(&Height2Tuple {
                        mask: [true; 32],
                        mean: std::array::from_fn(|i| h2_means[group[i].0 as usize]),
                        first: std::array::from_fn(|i| h2_firsts[group[i].0 as usize]),
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
                let codes = std::array::from_fn(|i| {
                    if i < group.len() {
                        group[i].1.clone()
                    } else {
                        rabitq::dummy_code(dims)
                    }
                });
                let packed = rabitq::pack_codes(dims, codes);
                tape.push(&Height2Tuple {
                    mask: std::array::from_fn(|i| i < group.len()),
                    mean: std::array::from_fn(|i| {
                        if i < group.len() {
                            h2_means[group[i].0 as usize]
                        } else {
                            Default::default()
                        }
                    }),
                    first: std::array::from_fn(|i| {
                        if i < group.len() {
                            h2_firsts[group[i].0 as usize]
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
            tape.first()
        })
        .collect::<Vec<_>>();
    forwards.head.get_mut().get_opaque_mut().fast_forward = vectors.first();
    meta.push(&MetaTuple {
        dims,
        is_residual,
        vectors_first: vectors.first(),
        forwards_first: forwards.first(),
        mean: h3_means[0],
        first: h3_firsts[0],
    });
}

struct Structure {
    h3_means: Vec<Vec<f32>>,
    h3_children: Vec<Vec<u32>>,
    h2_means: Vec<Vec<f32>>,
    h2_children: Vec<Vec<u32>>,
    h1_means: Vec<Vec<f32>>,
    h1_children: Vec<Vec<u32>>,
}

impl Structure {
    fn internal_build(
        vector_options: VectorOptions,
        internal_build: RabbitholeInternalBuildOptions,
        samples: Vec<Vec<f32>>,
    ) -> Self {
        let h1_means = crate::algorithm::parallelism::RayonParallelism::scoped(
            internal_build.build_threads as _,
            Arc::new(|| {
                pgrx::check_for_interrupts!();
            }),
            |parallelism| {
                k_means::k_means(
                    parallelism,
                    internal_build.nlist_1 as usize,
                    vector_options.dims as usize,
                    samples,
                    internal_build.spherical_centroids,
                    10,
                )
            },
        )
        .expect("failed to create thread pool");
        let h1_children = vec![Vec::new(); h1_means.len()];
        let h2_means = crate::algorithm::parallelism::RayonParallelism::scoped(
            internal_build.build_threads as _,
            Arc::new(|| {
                pgrx::check_for_interrupts!();
            }),
            |parallelism| {
                k_means::k_means(
                    parallelism,
                    internal_build.nlist_2 as usize,
                    vector_options.dims as usize,
                    h1_means.clone(),
                    internal_build.spherical_centroids,
                    10,
                )
            },
        )
        .expect("failed to create thread pool");
        let mut h2_children = vec![Vec::new(); h2_means.len()];
        for i in 0..h1_means.len() as u32 {
            let target: usize = k_means::k_means_lookup(&h1_means[i as usize], &h2_means);
            h2_children[target].push(i);
        }
        let (h2_means, h2_children) = std::iter::zip(h2_means, h2_children)
            .filter(|(_, x)| !x.is_empty())
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let h3_means = crate::algorithm::parallelism::RayonParallelism::scoped(
            internal_build.build_threads as _,
            Arc::new(|| {
                pgrx::check_for_interrupts!();
            }),
            |parallelism| {
                k_means::k_means(
                    parallelism,
                    1,
                    vector_options.dims as usize,
                    h2_means.clone(),
                    internal_build.spherical_centroids,
                    10,
                )
            },
        )
        .expect("failed to create thread pool");
        let mut h3_children = vec![Vec::new(); h3_means.len()];
        for i in 0..h2_means.len() as u32 {
            let target = k_means::k_means_lookup(&h2_means[i as usize], &h3_means);
            h3_children[target].push(i);
        }
        let (h3_means, h3_children) = std::iter::zip(h3_means, h3_children)
            .filter(|(_, x)| !x.is_empty())
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Structure {
            h3_means: h3_means.into_iter().map(|x| rabitq::project(&x)).collect(),
            h3_children,
            h2_means: h2_means.into_iter().map(|x| rabitq::project(&x)).collect(),
            h2_children,
            h1_means: h1_means.into_iter().map(|x| rabitq::project(&x)).collect(),
            h1_children,
        }
    }
    fn extern_build(
        vector_options: VectorOptions,
        _opfamily: Opfamily,
        external_build: RabbitholeExternalBuildOptions,
    ) -> Self {
        use std::collections::BTreeMap;
        let RabbitholeExternalBuildOptions { table } = external_build;
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
                vectors.insert(id, rabitq::project(vector.slice()));
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
        if heights[&root] != 3 {
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
        let (h3_means, h3_children) = extract(3, &labels, &vectors, &children);
        let (h2_means, h2_children) = extract(2, &labels, &vectors, &children);
        let (h1_means, h1_children) = extract(1, &labels, &vectors, &children);
        Self {
            h3_means,
            h3_children,
            h2_means,
            h2_children,
            h1_means,
            h1_children,
        }
    }
    fn h3_len(&self) -> u32 {
        self.h3_means.len() as _
    }
    fn h3_means(&self, i: u32) -> &Vec<f32> {
        &self.h3_means[i as usize]
    }
    fn h3_children(&self, i: u32) -> &Vec<u32> {
        &self.h3_children[i as usize]
    }
    fn h2_len(&self) -> u32 {
        self.h2_means.len() as _
    }
    fn h2_means(&self, i: u32) -> &Vec<f32> {
        &self.h2_means[i as usize]
    }
    fn h2_children(&self, i: u32) -> &Vec<u32> {
        &self.h2_children[i as usize]
    }
    fn h1_len(&self) -> u32 {
        self.h1_means.len() as _
    }
    fn h1_means(&self, i: u32) -> &Vec<f32> {
        &self.h1_means[i as usize]
    }
    #[allow(dead_code)]
    fn h1_children(&self, i: u32) -> &Vec<u32> {
        &self.h1_children[i as usize]
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

impl<'a, T> Tape<'a, T>
where
    T: rkyv::Serialize<AllocSerializer<8192>>,
{
    fn push(&mut self, x: &T) -> (u32, u16) {
        let bytes = rkyv::to_bytes(x).expect("failed to serialize");
        if let Some(i) = self.head.get_mut().alloc(&bytes) {
            (self.head.id(), i)
        } else {
            let next = self.relation.extend(self.tracking_freespace);
            self.head.get_mut().get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.get_mut().alloc(&bytes) {
                (self.head.id(), i)
            } else {
                panic!("tuple is too large to fit in a fresh page")
            }
        }
    }
}
