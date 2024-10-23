use crate::algorithm::rabitq;
use crate::algorithm::tuples::*;
use crate::index::utils::load_proj_vectors;
use crate::postgres::BufferWriteGuard;
use crate::postgres::Relation;
use crate::types::ExternalCentroids;
use crate::types::RabbitholeIndexingOptions;
use base::distance::DistanceKind;
use base::index::VectorOptions;
use base::scalar::ScalarLike;
use base::search::Pointer;
use common::vec2::Vec2;
use rand::Rng;
use rkyv::ser::serializers::AllocSerializer;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub trait HeapRelation {
    fn traverse<F>(&self, callback: F)
    where
        F: FnMut((Pointer, Option<u32>, Vec<f32>));
}

pub trait Reporter {
    fn tuples_total(&mut self, tuples_total: usize);
    fn tuples_done(&mut self, tuples_done: usize);
}

pub fn build<T: HeapRelation, R: Reporter>(
    vector_options: VectorOptions,
    rabbithole_options: RabbitholeIndexingOptions,
    heap_relation: T,
    index_relation: Relation,
    mut reporter: R,
) {
    let dims = vector_options.dims;
    let is_residual =
        rabbithole_options.residual_quantization && vector_options.d == DistanceKind::L2;
    let structure = match &rabbithole_options.external_centroids {
        Some(_) => Structure::load(vector_options.clone(), rabbithole_options.clone()),
        None => {
            let mut tuples_total = 0_usize;
            let samples = {
                let mut rand = rand::thread_rng();
                let max_number_of_samples = rabbithole_options.nlist.saturating_mul(256);
                let mut samples = Vec::new();
                let mut number_of_samples = 0_u32;
                heap_relation.traverse(|(_, _, vector)| {
                    pgrx::check_for_interrupts!();
                    assert_eq!(dims as usize, vector.len(), "invalid vector dimensions",);
                    let vector = rabitq::project(&vector);
                    if number_of_samples < max_number_of_samples {
                        samples.extend(vector);
                        number_of_samples += 1;
                    } else {
                        let index = rand.gen_range(0..max_number_of_samples) as usize;
                        let start = index * dims as usize;
                        let end = start + dims as usize;
                        samples[start..end].copy_from_slice(&vector);
                    }
                    tuples_total += 1;
                });
                Vec2::from_vec((number_of_samples as _, dims as _), samples)
            };
            reporter.tuples_total(tuples_total);
            Structure::compute(vector_options.clone(), rabbithole_options.clone(), samples)
        }
    };
    let h2_len = structure.h2_len();
    let h1_len = structure.h1_len();
    let mut meta = Tape::create(&index_relation);
    assert_eq!(meta.first(), 0);
    let mut vectors = Tape::create(&index_relation);
    assert_eq!(vectors.first(), 1);
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
            let tape = Tape::<Height0Tuple>::create(&index_relation);
            tape.first()
        })
        .collect::<Vec<_>>();
    let h2_firsts = (0..h2_len)
        .map(|i| {
            let mut tape = Tape::<Height1Tuple>::create(&index_relation);
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
    meta.push(&MetaTuple {
        dims,
        is_residual,
        vectors_first: vectors.first(),
        mean: h2_means[0],
        first: h2_firsts[0],
    });
    drop(meta);
    let mut heads = Vec::new();
    for i in 0..structure.h1_len() {
        heads.push(h1_firsts[i as usize]);
    }
    let mut tuples_done = 0;
    heap_relation.traverse(|(payload, extra, vector)| {
        pgrx::check_for_interrupts!();
        tuples_done += 1;
        reporter.tuples_done(tuples_done);
        assert_eq!(dims as usize, vector.len(), "invalid vector dimensions");
        let vector = rabitq::project(&vector);
        let h0_vector = vectors.push(&VectorTuple {
            vector: vector.clone(),
            payload: Some(payload.as_u64()),
        });
        let h0_payload = payload.as_u64();
        let h2_id = 0_u32;
        let h1_id = {
            let mut target = (0_u32, f32::INFINITY);
            for &i in structure.h2_children(h2_id) {
                let dis = f32::reduce_sum_of_d2(&vector, structure.h1_means(i));
                if dis < target.1 {
                    target = (i, dis);
                }
            }
            target.0
        };
        let code = if is_residual {
            rabitq::code(dims, &f32::vector_sub(&vector, structure.h1_means(h1_id)))
        } else {
            rabitq::code(dims, &vector)
        };
        let mut write = index_relation.write(heads[h1_id as usize]);
        let page = write.get_mut();
        if page.len() != 0 {
            let flag = put(
                page.get_mut(page.len()).expect("data corruption"),
                dims,
                &code,
                h0_vector,
                h0_payload,
                extra,
            );
            if flag {
                return;
            }
        }
        let tuple = rkyv::to_bytes::<_, 8192>(&Height0Tuple {
            mask: [false; 32],
            mean: [(0, 0); 32],
            payload: [0; 32],
            dis_u_2: [0.0; 32],
            factor_ppc: [0.0; 32],
            factor_ip: [0.0; 32],
            factor_err: [0.0; 32],
            t: vec![0; (dims.div_ceil(4) * 16) as usize],
            extra: [None; 32],
        })
        .unwrap();
        if let Some(i) = page.alloc(&tuple) {
            let flag = put(
                page.get_mut(i).expect("data corruption"),
                dims,
                &code,
                h0_vector,
                h0_payload,
                extra,
            );
            assert!(flag, "a put fails even on a fresh tuple");
            return;
        }
        let mut extend = index_relation.extend();
        heads[h1_id as usize] = extend.id();
        page.get_opaque_mut().next = extend.id();
        let page = extend.get_mut();
        if let Some(i) = page.alloc(&tuple) {
            let flag = put(
                page.get_mut(i).expect("data corruption"),
                dims,
                &code,
                h0_vector,
                h0_payload,
                extra,
            );
            assert!(flag, "a put fails even on a fresh tuple");
            return;
        }
        panic!("a tuple cannot even be fit in a fresh page")
    });
}

struct Structure {
    h2_mean: Vec<f32>,
    h2_children: Vec<u32>,
    h1_means: Vec<Vec<f32>>,
    h1_children: Vec<Vec<u32>>,
}

impl Structure {
    fn compute(
        vector_options: VectorOptions,
        rabbithole_options: RabbitholeIndexingOptions,
        samples: Vec2<f32>,
    ) -> Self {
        let dims = vector_options.dims;
        let h1_means = base::parallelism::RayonParallelism::scoped(
            rabbithole_options.build_threads as _,
            Arc::new(AtomicBool::new(false)),
            |parallelism| {
                let raw = k_means::k_means(
                    parallelism,
                    rabbithole_options.nlist as usize,
                    samples,
                    rabbithole_options.spherical_centroids,
                    10,
                    false,
                );
                let mut centroids = Vec::new();
                for i in 0..rabbithole_options.nlist {
                    centroids.push(raw[(i as usize,)].to_vec());
                }
                centroids
            },
        )
        .expect("k_means panics")
        .expect("k_means interrupted");
        let h2_mean = {
            let mut centroid = vec![0.0; dims as _];
            for i in 0..rabbithole_options.nlist {
                for j in 0..dims {
                    centroid[j as usize] += h1_means[i as usize][j as usize];
                }
            }
            for j in 0..dims {
                centroid[j as usize] /= rabbithole_options.nlist as f32;
            }
            centroid
        };
        Structure {
            h2_mean,
            h2_children: (0..rabbithole_options.nlist).collect(),
            h1_means,
            h1_children: (0..rabbithole_options.nlist).map(|_| Vec::new()).collect(),
        }
    }
    fn load(vector_options: VectorOptions, rabbithole_options: RabbitholeIndexingOptions) -> Self {
        let dims = vector_options.dims;
        let h1_means = match &rabbithole_options.external_centroids {
            Some(ExternalCentroids {
                table,
                h1_means_column: h1,
                ..
            }) => load_proj_vectors(table, h1, rabbithole_options.nlist, vector_options.dims),
            _ => unreachable!(),
        };
        let h1_children = match &rabbithole_options.external_centroids {
            Some(ExternalCentroids {
                table,
                h1_children_column: Some(h1),
                ..
            }) => load_proj_vectors(table, h1, 1, vector_options.dims)
                .into_iter()
                .map(|v| v.into_iter().map(|f| f as u32).collect())
                .collect(),
            _ => (0..rabbithole_options.nlist).map(|_| Vec::new()).collect(),
        };
        let h2_mean = match &rabbithole_options.external_centroids {
            Some(ExternalCentroids {
                table,
                h2_mean_column: Some(h2),
                ..
            }) => load_proj_vectors(table, h2, 1, vector_options.dims)
                .pop()
                .expect("load h2_mean panic"),
            _ => {
                let mut centroid = vec![0.0; dims as _];
                for i in 0..rabbithole_options.nlist {
                    for j in 0..dims {
                        centroid[j as usize] += h1_means[i as usize][j as usize];
                    }
                }
                for j in 0..dims {
                    centroid[j as usize] /= rabbithole_options.nlist as f32;
                }
                centroid
            }
        };
        let h2_children = match &rabbithole_options.external_centroids {
            Some(ExternalCentroids {
                table,
                h2_children_column: Some(h2),
                ..
            }) => load_proj_vectors(table, h2, 1, vector_options.dims)
                .pop()
                .expect("load h2_children panic")
                .into_iter()
                .map(|f| f as u32)
                .collect(),
            _ => (0..rabbithole_options.nlist).collect(),
        };
        Structure {
            h2_mean,
            h2_children,
            h1_means,
            h1_children,
        }
    }
    fn h2_len(&self) -> u32 {
        1
    }
    fn h2_means(&self, i: u32) -> &Vec<f32> {
        assert!(i == 0);
        &self.h2_mean
    }
    fn h2_children(&self, i: u32) -> &Vec<u32> {
        assert!(i == 0);
        &self.h2_children
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
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, T> Tape<'a, T> {
    fn create(relation: &'a Relation) -> Self {
        let head = relation.extend();
        let first = head.id();
        Self {
            relation,
            head,
            first,
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
            let next = self.relation.extend();
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
