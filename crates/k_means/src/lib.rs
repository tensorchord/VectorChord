use rabitq::block::BlockCode;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use simd::Floating;
use simd::fast_scan::{any_pack, padding_pack};

pub fn preprocess<T: Send>(num_threads: usize, x: &mut [T], f: impl Fn(&mut T) + Sync) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_scoped(
            |thread| thread.run(),
            move |pool| {
                pool.install(|| {
                    x.par_iter_mut().for_each(&f);
                });
            },
        )
        .expect("failed to build thread pool")
}

pub fn k_means(
    num_threads: usize,
    mut check: impl FnMut(usize),
    c: usize,
    dims: usize,
    samples: &[Vec<f32>],
    is_spherical: bool,
    iterations: usize,
) -> Vec<Vec<f32>> {
    assert!(c > 0);
    assert!(dims > 0);
    let n = samples.len();
    if n <= c {
        quick_centers(c, dims, samples.to_vec(), is_spherical)
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_scoped(
                |thread| thread.run(),
                move |pool| {
                    let compute = |centroids: &[Vec<f32>]| {
                        if n >= 1024 && c >= 1024 {
                            rabitq_index(dims, n, c, samples, centroids)
                        } else {
                            flat_index(dims, n, c, samples, centroids)
                        }
                    };
                    let mut lloyd_k_means =
                        pool.install(|| LloydKMeans::new(c, dims, samples, is_spherical, compute));
                    for i in 0..iterations {
                        check(i);
                        if pool.install(|| lloyd_k_means.iterate()) {
                            break;
                        }
                    }
                    pool.install(|| lloyd_k_means.finish())
                },
            )
            .expect("failed to build thread pool")
    }
}

pub fn k_means_lookup(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    assert_ne!(centroids.len(), 0);
    let mut result = (f32::INFINITY, 0);
    for i in 0..centroids.len() {
        let dis = f32::reduce_sum_of_d2(vector, &centroids[i]);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

fn quick_centers(
    c: usize,
    dims: usize,
    samples: Vec<Vec<f32>>,
    is_spherical: bool,
) -> Vec<Vec<f32>> {
    let n = samples.len();
    assert!(c >= n);
    if c == 1 && n == 0 {
        return vec![vec![0.0; dims]];
    }
    let mut rng = rand::rng();
    let mut centroids = samples;
    for _ in n..c {
        let r = (0..dims)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        centroids.push(r);
    }
    if is_spherical {
        for i in 0..c {
            let centroid = &mut centroids[i];
            let l = f32::reduce_sum_of_x2(centroid).sqrt();
            f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
        }
    }
    centroids
}

fn rabitq_index(
    dims: usize,
    n: usize,
    c: usize,
    samples: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Vec<usize> {
    struct Branch {
        dis_u_2: f32,
        factor_ppc: f32,
        factor_ip: f32,
        factor_err: f32,
        signs: Vec<bool>,
    }
    let branches = {
        let mut branches = Vec::new();
        for centroid in centroids {
            let code = rabitq::code(dims as _, centroid);
            branches.push(Branch {
                dis_u_2: code.dis_u_2,
                factor_ppc: code.factor_ppc,
                factor_ip: code.factor_ip,
                factor_err: code.factor_err,
                signs: code.signs,
            });
        }
        branches
    };
    struct Block {
        dis_u_2: [f32; 32],
        factor_ppc: [f32; 32],
        factor_ip: [f32; 32],
        factor_err: [f32; 32],
        elements: Vec<[u8; 16]>,
    }
    impl Block {
        fn code(&self) -> BlockCode<'_> {
            (
                &self.dis_u_2,
                &self.factor_ppc,
                &self.factor_ip,
                &self.factor_err,
                &self.elements,
            )
        }
    }
    let mut blocks = Vec::new();
    for chunk in branches.chunks(32) {
        blocks.push(Block {
            dis_u_2: any_pack(chunk.iter().map(|x| x.dis_u_2)),
            factor_ppc: any_pack(chunk.iter().map(|x| x.factor_ppc)),
            factor_ip: any_pack(chunk.iter().map(|x| x.factor_ip)),
            factor_err: any_pack(chunk.iter().map(|x| x.factor_err)),
            elements: padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs))),
        });
    }
    (0..n)
        .into_par_iter()
        .map(|i| {
            let lut = rabitq::block::preprocess(&samples[i]);
            let mut result = (f32::INFINITY, 0);
            for block in 0..c.div_ceil(32) {
                let returns = rabitq::block::process_l2(&lut, blocks[block].code());
                let lowerbound = returns.map(|(rough, err)| rough - err * 1.9);
                for j in block * 32..std::cmp::min(block * 32 + 32, c) {
                    if lowerbound[j - block * 32] < result.0 {
                        let dis = f32::reduce_sum_of_d2(&samples[i], &centroids[j]);
                        if dis <= result.0 {
                            result = (dis, j);
                        }
                    }
                }
            }
            result.1
        })
        .collect::<Vec<_>>()
}

fn flat_index(
    _dims: usize,
    n: usize,
    c: usize,
    samples: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Vec<usize> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut result = (f32::INFINITY, 0);
            for j in 0..c {
                let dis_2 = f32::reduce_sum_of_d2(&samples[i], &centroids[j]);
                if dis_2 <= result.0 {
                    result = (dis_2, j);
                }
            }
            result.1
        })
        .collect::<Vec<_>>()
}

struct LloydKMeans<'a, F> {
    dims: usize,
    c: usize,
    is_spherical: bool,
    centroids: Vec<Vec<f32>>,
    assign: Vec<usize>,
    rng: StdRng,
    samples: &'a [Vec<f32>],
    compute: F,
}

const DELTA: f32 = 9.7656e-4_f32;

impl<'a, F: Fn(&[Vec<f32>]) -> Vec<usize>> LloydKMeans<'a, F> {
    fn new(c: usize, dims: usize, samples: &'a [Vec<f32>], is_spherical: bool, compute: F) -> Self {
        let n = samples.len();

        let mut rng = StdRng::from_seed([7; 32]);
        let mut centroids = Vec::with_capacity(c);

        for index in rand::seq::index::sample(&mut rng, n, c).into_iter() {
            centroids.push(samples[index].clone());
        }

        let assign = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut result = (f32::INFINITY, 0);
                for j in 0..c {
                    let dis_2 = f32::reduce_sum_of_d2(&samples[i], &centroids[j]);
                    if dis_2 <= result.0 {
                        result = (dis_2, j);
                    }
                }
                result.1
            })
            .collect::<Vec<_>>();

        Self {
            dims,
            c,
            is_spherical,
            centroids,
            assign,
            rng,
            samples,
            compute,
        }
    }

    fn iterate(&mut self) -> bool {
        let dims = self.dims;
        let c = self.c;
        let rand = &mut self.rng;
        let samples = self.samples;
        let n = samples.len();

        let (sum, mut count) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![vec![f32::zero(); dims]; c], vec![0.0f32; c]),
                |(mut sum, mut count), i| {
                    f32::vector_add_inplace(&mut sum[self.assign[i]], &samples[i]);
                    count[self.assign[i]] += 1.0;
                    (sum, count)
                },
            )
            .reduce(
                || (vec![vec![f32::zero(); dims]; c], vec![0.0f32; c]),
                |(mut sum, mut count), (sum_1, count_1)| {
                    for i in 0..c {
                        f32::vector_add_inplace(&mut sum[i], &sum_1[i]);
                        count[i] += count_1[i];
                    }
                    (sum, count)
                },
            );

        let mut centroids = (0..c)
            .into_par_iter()
            .map(|i| f32::vector_mul_scalar(&sum[i], 1.0 / count[i]))
            .collect::<Vec<_>>();

        for i in 0..c {
            if count[i] != 0.0f32 {
                continue;
            }
            let mut o = 0;
            loop {
                let alpha = rand.random_range(0.0..1.0f32);
                let beta = (count[o] - 1.0) / (n - c) as f32;
                if alpha < beta {
                    break;
                }
                o = (o + 1) % c;
            }
            centroids[i] = centroids[o].clone();
            vector_mul_scalars_inplace(&mut centroids[i], [1.0 + DELTA, 1.0 - DELTA]);
            vector_mul_scalars_inplace(&mut centroids[o], [1.0 - DELTA, 1.0 + DELTA]);
            count[i] = count[o] / 2.0;
            count[o] -= count[i];
        }

        if self.is_spherical {
            (&mut centroids).into_par_iter().for_each(|centroid| {
                let l = f32::reduce_sum_of_x2(centroid).sqrt();
                f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
            });
        }

        let assign = (self.compute)(&centroids);

        let result = (0..n).all(|i| assign[i] == self.assign[i]);

        self.centroids = centroids;
        self.assign = assign;

        result
    }

    fn finish(self) -> Vec<Vec<f32>> {
        self.centroids
    }
}

fn vector_mul_scalars_inplace(this: &mut [f32], scalars: [f32; 2]) {
    let n: usize = this.len();
    for i in 0..n {
        if i % 2 == 0 {
            this[i] *= scalars[0];
        } else {
            this[i] *= scalars[1];
        }
    }
}
