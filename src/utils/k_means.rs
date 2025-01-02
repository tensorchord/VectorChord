use super::parallelism::{ParallelIterator, Parallelism};
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simd::Floating;

pub fn k_means<P: Parallelism>(
    parallelism: &P,
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
        let compute = |parallelism: &P, centroids: &[Vec<f32>]| {
            if n >= 1000 && c >= 1000 {
                rabitq_index(parallelism, dims, n, c, samples, centroids)
            } else {
                flat_index(parallelism, dims, n, c, samples, centroids)
            }
        };
        let mut lloyd_k_means =
            LloydKMeans::new(parallelism, c, dims, samples, is_spherical, compute);
        for _ in 0..iterations {
            parallelism.check();
            if lloyd_k_means.iterate() {
                break;
            }
        }
        lloyd_k_means.finish()
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
    let mut rng = rand::thread_rng();
    let mut centroids = samples;
    for _ in n..c {
        let r = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
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

fn rabitq_index<P: Parallelism>(
    parallelism: &P,
    dims: usize,
    n: usize,
    c: usize,
    samples: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Vec<usize> {
    let mut a0 = Vec::new();
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();
    let mut a3 = Vec::new();
    let mut a4 = Vec::new();
    for vectors in centroids.chunks(32) {
        use simd::fast_scan::pack;
        let x = std::array::from_fn::<_, 32, _>(|i| {
            if let Some(vector) = vectors.get(i) {
                rabitq::block::code(dims as _, vector)
            } else {
                rabitq::block::dummy_code(dims as _)
            }
        });
        a0.push(x.each_ref().map(|x| x.dis_u_2));
        a1.push(x.each_ref().map(|x| x.factor_ppc));
        a2.push(x.each_ref().map(|x| x.factor_ip));
        a3.push(x.each_ref().map(|x| x.factor_err));
        a4.push(pack(dims.div_ceil(4) as _, x.map(|x| x.signs)).collect::<Vec<_>>());
    }
    parallelism
        .rayon_into_par_iter(0..n)
        .map(|i| {
            use distance::Distance;
            let lut = rabitq::block::fscan_preprocess(&samples[i]);
            let mut result = (Distance::INFINITY, 0);
            for block in 0..c.div_ceil(32) {
                let lowerbound = rabitq::block::fscan_process_lowerbound_l2(
                    dims as _,
                    &lut,
                    (&a0[block], &a1[block], &a2[block], &a3[block], &a4[block]),
                    1.9,
                );
                for j in block * 32..std::cmp::min(block * 32 + 32, c) {
                    if lowerbound[j - block * 32] < result.0 {
                        let dis =
                            Distance::from_f32(f32::reduce_sum_of_d2(&samples[i], &centroids[j]));
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

fn flat_index<P: Parallelism>(
    parallelism: &P,
    _dims: usize,
    n: usize,
    c: usize,
    samples: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Vec<usize> {
    parallelism
        .rayon_into_par_iter(0..n)
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

struct LloydKMeans<'a, P, F> {
    parallelism: &'a P,
    dims: usize,
    c: usize,
    is_spherical: bool,
    centroids: Vec<Vec<f32>>,
    assign: Vec<usize>,
    rng: StdRng,
    samples: &'a [Vec<f32>],
    compute: F,
}

const DELTA: f32 = f16::EPSILON.to_f32_const();

impl<'a, P: Parallelism, F: Fn(&P, &[Vec<f32>]) -> Vec<usize>> LloydKMeans<'a, P, F> {
    fn new(
        parallelism: &'a P,
        c: usize,
        dims: usize,
        samples: &'a [Vec<f32>],
        is_spherical: bool,
        compute: F,
    ) -> Self {
        let n = samples.len();

        let mut rng = StdRng::from_entropy();
        let mut centroids = Vec::with_capacity(c);

        for index in rand::seq::index::sample(&mut rng, n, c).into_iter() {
            centroids.push(samples[index].clone());
        }

        let assign = parallelism
            .rayon_into_par_iter(0..n)
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
            parallelism,
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

        let (sum, mut count) = self
            .parallelism
            .rayon_into_par_iter(0..n)
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

        let mut centroids = self
            .parallelism
            .rayon_into_par_iter(0..c)
            .map(|i| f32::vector_mul_scalar(&sum[i], 1.0 / count[i]))
            .collect::<Vec<_>>();

        for i in 0..c {
            if count[i] != 0.0f32 {
                continue;
            }
            let mut o = 0;
            loop {
                let alpha = rand.gen_range(0.0..1.0f32);
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
            self.parallelism
                .rayon_into_par_iter(&mut centroids)
                .for_each(|centroid| {
                    let l = f32::reduce_sum_of_x2(centroid).sqrt();
                    f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
                });
        }

        let assign = (self.compute)(self.parallelism, &centroids);

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
