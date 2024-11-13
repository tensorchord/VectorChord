use crate::algorithm::parallelism::{ParallelIterator, Parallelism};
use base::scalar::*;
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn k_means<S: ScalarLike>(
    parallelism: &impl Parallelism,
    c: usize,
    dims: usize,
    samples: Vec<Vec<S>>,
    is_spherical: bool,
    iterations: usize,
) -> Vec<Vec<S>> {
    assert!(c > 0);
    assert!(dims > 0);
    let n = samples.len();
    if n <= c {
        quick_centers(c, dims, samples, is_spherical)
    } else {
        let mut lloyd_k_means = LloydKMeans::new(parallelism, c, dims, samples, is_spherical);
        for _ in 0..iterations {
            parallelism.check();
            if lloyd_k_means.iterate() {
                break;
            }
        }
        lloyd_k_means.finish()
    }
}

pub fn k_means_lookup<S: ScalarLike>(vector: &[S], centroids: &[Vec<S>]) -> usize {
    assert_ne!(centroids.len(), 0);
    let mut result = (f32::INFINITY, 0);
    for i in 0..centroids.len() {
        let dis = S::reduce_sum_of_d2(vector, &centroids[i]);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

fn quick_centers<S: ScalarLike>(
    c: usize,
    dims: usize,
    samples: Vec<Vec<S>>,
    is_spherical: bool,
) -> Vec<Vec<S>> {
    let n = samples.len();
    assert!(c >= n);
    let mut rng = rand::thread_rng();
    let mut centroids = samples;
    for _ in n..c {
        let r = (0..dims)
            .map(|_| S::from_f32(rng.gen_range(-1.0f32..1.0f32)))
            .collect();
        centroids.push(r);
    }
    if is_spherical {
        for i in 0..c {
            let centroid = &mut centroids[i];
            let l = S::reduce_sum_of_x2(centroid).sqrt();
            S::vector_mul_scalar_inplace(centroid, 1.0 / l);
        }
    }
    centroids
}

struct LloydKMeans<'a, P, S> {
    parallelism: &'a P,
    dims: usize,
    c: usize,
    is_spherical: bool,
    centroids: Vec<Vec<S>>,
    assign: Vec<usize>,
    rng: StdRng,
    samples: Vec<Vec<S>>,
}

const DELTA: f32 = f16::EPSILON.to_f32_const();

impl<'a, P: Parallelism, S: ScalarLike> LloydKMeans<'a, P, S> {
    fn new(
        parallelism: &'a P,
        c: usize,
        dims: usize,
        samples: Vec<Vec<S>>,
        is_spherical: bool,
    ) -> Self {
        let n = samples.len();

        let mut rng = StdRng::from_entropy();
        let mut centroids = Vec::with_capacity(c);

        for index in rand::seq::index::sample(&mut rng, n, c).into_iter() {
            centroids.push(samples[index].clone());
        }

        let assign = parallelism
            .into_par_iter(0..n)
            .map(|i| {
                let mut result = (f32::INFINITY, 0);
                for j in 0..c {
                    let dis_2 = S::reduce_sum_of_d2(&samples[i], &centroids[j]);
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
        }
    }

    fn iterate(&mut self) -> bool {
        let dims = self.dims;
        let c = self.c;
        let rand = &mut self.rng;
        let samples = &self.samples;
        let n = samples.len();

        let (sum, mut count) = self
            .parallelism
            .into_par_iter(0..n)
            .fold(
                || (vec![vec![S::zero(); dims]; c], vec![0.0f32; c]),
                |(mut sum, mut count), i| {
                    S::vector_add_inplace(&mut sum[self.assign[i]], &samples[i]);
                    count[self.assign[i]] += 1.0;
                    (sum, count)
                },
            )
            .reduce(
                || (vec![vec![S::zero(); dims]; c], vec![0.0f32; c]),
                |(mut sum, mut count), (sum_1, count_1)| {
                    for i in 0..c {
                        S::vector_add_inplace(&mut sum[i], &sum_1[i]);
                        count[i] += count_1[i];
                    }
                    (sum, count)
                },
            );

        let mut centroids = self
            .parallelism
            .into_par_iter(0..c)
            .map(|i| S::vector_mul_scalar(&sum[i], 1.0 / count[i]))
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
            S::kmeans_helper(&mut centroids[i], 1.0 + DELTA, 1.0 - DELTA);
            S::kmeans_helper(&mut centroids[o], 1.0 - DELTA, 1.0 + DELTA);
            count[i] = count[o] / 2.0;
            count[o] -= count[i];
        }

        if self.is_spherical {
            self.parallelism
                .into_par_iter(&mut centroids)
                .for_each(|centroid| {
                    let l = S::reduce_sum_of_x2(centroid).sqrt();
                    S::vector_mul_scalar_inplace(centroid, 1.0 / l);
                });
        }

        let assign = self
            .parallelism
            .into_par_iter(0..n)
            .map(|i| {
                let mut result = (f32::INFINITY, 0);
                for j in 0..c {
                    let dis_2 = S::reduce_sum_of_d2(&samples[i], &centroids[j]);
                    if dis_2 <= result.0 {
                        result = (dis_2, j);
                    }
                }
                result.1
            })
            .collect::<Vec<_>>();

        let result = (0..n).all(|i| assign[i] == self.assign[i]);

        self.centroids = centroids;
        self.assign = assign;

        result
    }

    fn finish(self) -> Vec<Vec<S>> {
        self.centroids
    }
}
