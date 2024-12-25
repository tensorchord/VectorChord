#![allow(clippy::ptr_arg)]

use super::parallelism::{ParallelIterator, Parallelism};
use base::simd::*;
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn k_means<P: Parallelism>(
    parallelism: &P,
    c: usize,
    dims: usize,
    samples: &Vec<Vec<f32>>,
    is_spherical: bool,
    iterations: usize,
) -> Vec<Vec<f32>> {
    assert!(c > 0);
    assert!(dims > 0);
    let n = samples.len();
    if n <= c {
        quick_centers(c, dims, samples.clone(), is_spherical)
    } else {
        let compute = |parallelism: &P, centroids: &Vec<Vec<f32>>| {
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
        let r = (0..dims)
            .map(|_| f32::from_f32(rng.gen_range(-1.0f32..1.0f32)))
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

fn rabitq_index<P: Parallelism>(
    parallelism: &P,
    dims: usize,
    n: usize,
    c: usize,
    samples: &Vec<Vec<f32>>,
    centroids: &Vec<Vec<f32>>,
) -> Vec<usize> {
    fn code_alpha(vector: &[f32]) -> (f32, f32, f32, f32) {
        let dims = vector.len();
        let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
        let sum_of_x2 = f32::reduce_sum_of_x2(vector);
        let dis_u = sum_of_x2.sqrt();
        let x0 = sum_of_abs_x / (sum_of_x2 * (dims as f32)).sqrt();
        let x_x0 = dis_u / x0;
        let fac_norm = (dims as f32).sqrt();
        let max_x1 = 1.0f32 / (dims as f32 - 1.0).sqrt();
        let factor_err = 2.0f32 * max_x1 * (x_x0 * x_x0 - dis_u * dis_u).sqrt();
        let factor_ip = -2.0f32 / fac_norm * x_x0;
        let cnt_pos = vector
            .iter()
            .map(|x| x.scalar_is_sign_positive() as i32)
            .sum::<i32>();
        let cnt_neg = vector
            .iter()
            .map(|x| x.scalar_is_sign_negative() as i32)
            .sum::<i32>();
        let factor_ppc = factor_ip * (cnt_pos - cnt_neg) as f32;
        (sum_of_x2, factor_ppc, factor_ip, factor_err)
    }
    fn code_beta(vector: &[f32]) -> Vec<u8> {
        let dims = vector.len();
        let mut code = Vec::new();
        for i in 0..dims {
            code.push(vector[i].scalar_is_sign_positive() as u8);
        }
        code
    }
    let mut a0 = Vec::new();
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();
    let mut a3 = Vec::new();
    let mut a4 = Vec::new();
    for vectors in centroids.chunks(32) {
        use base::simd::fast_scan::b4::pack;
        let code_alphas = std::array::from_fn::<_, 32, _>(|i| {
            if let Some(vector) = vectors.get(i) {
                code_alpha(vector)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            }
        });
        let code_betas = std::array::from_fn::<_, 32, _>(|i| {
            let mut result = vec![0_u8; dims.div_ceil(4)];
            if let Some(vector) = vectors.get(i) {
                let mut c = code_beta(vector);
                c.resize(dims.next_multiple_of(4), 0);
                for i in 0..dims.div_ceil(4) {
                    for j in 0..4 {
                        result[i] |= c[i * 4 + j] << j;
                    }
                }
            }
            result
        });
        a0.push(code_alphas.map(|x| x.0));
        a1.push(code_alphas.map(|x| x.1));
        a2.push(code_alphas.map(|x| x.2));
        a3.push(code_alphas.map(|x| x.3));
        a4.push(pack(dims.div_ceil(4) as _, code_betas).collect::<Vec<_>>());
    }
    parallelism
        .into_par_iter(0..n)
        .map(|i| {
            fn generate(mut qvector: Vec<u8>) -> Vec<u8> {
                let dims = qvector.len() as u32;
                let t = dims.div_ceil(4);
                qvector.resize(qvector.len().next_multiple_of(4), 0);
                let mut lut = vec![0u8; t as usize * 16];
                for i in 0..t as usize {
                    unsafe {
                        // this hint is used to skip bound checks
                        std::hint::assert_unchecked(4 * i + 3 < qvector.len());
                        std::hint::assert_unchecked(16 * i + 15 < lut.len());
                    }
                    let t0 = qvector[4 * i + 0];
                    let t1 = qvector[4 * i + 1];
                    let t2 = qvector[4 * i + 2];
                    let t3 = qvector[4 * i + 3];
                    lut[16 * i + 0b0000] = 0;
                    lut[16 * i + 0b0001] = t0;
                    lut[16 * i + 0b0010] = t1;
                    lut[16 * i + 0b0011] = t1 + t0;
                    lut[16 * i + 0b0100] = t2;
                    lut[16 * i + 0b0101] = t2 + t0;
                    lut[16 * i + 0b0110] = t2 + t1;
                    lut[16 * i + 0b0111] = t2 + t1 + t0;
                    lut[16 * i + 0b1000] = t3;
                    lut[16 * i + 0b1001] = t3 + t0;
                    lut[16 * i + 0b1010] = t3 + t1;
                    lut[16 * i + 0b1011] = t3 + t1 + t0;
                    lut[16 * i + 0b1100] = t3 + t2;
                    lut[16 * i + 0b1101] = t3 + t2 + t0;
                    lut[16 * i + 0b1110] = t3 + t2 + t1;
                    lut[16 * i + 0b1111] = t3 + t2 + t1 + t0;
                }
                lut
            }
            fn fscan_process_lowerbound(
                dims: u32,
                lut: &(f32, f32, f32, f32, Vec<u8>),
                (dis_u_2, factor_ppc, factor_ip, factor_err, t): (
                    &[f32; 32],
                    &[f32; 32],
                    &[f32; 32],
                    &[f32; 32],
                    &[u8],
                ),
                epsilon: f32,
            ) -> [Distance; 32] {
                use base::simd::fast_scan::b4::fast_scan_b4;
                let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
                let r = fast_scan_b4(dims.div_ceil(4), t, s);
                std::array::from_fn(|i| {
                    let rough = dis_u_2[i]
                        + dis_v_2
                        + b * factor_ppc[i]
                        + ((2.0 * r[i] as f32) - qvector_sum) * factor_ip[i] * k;
                    let err = factor_err[i] * dis_v_2.sqrt();
                    Distance::from_f32(rough - epsilon * err)
                })
            }
            use base::distance::Distance;
            use base::simd::quantize;

            let lut = {
                let vector = &samples[i];
                let dis_v_2 = f32::reduce_sum_of_x2(vector);
                let (k, b, qvector) =
                    quantize::quantize(f32::vector_to_f32_borrowed(vector).as_ref(), 15.0);
                let qvector_sum = if vector.len() <= 4369 {
                    u8::reduce_sum_of_x_as_u16(&qvector) as f32
                } else {
                    u8::reduce_sum_of_x_as_u32(&qvector) as f32
                };
                (dis_v_2, b, k, qvector_sum, generate(qvector))
            };

            let mut result = (Distance::INFINITY, 0);
            for block in 0..c.div_ceil(32) {
                let lowerbound = fscan_process_lowerbound(
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
    samples: &Vec<Vec<f32>>,
    centroids: &Vec<Vec<f32>>,
) -> Vec<usize> {
    parallelism
        .into_par_iter(0..n)
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
    samples: &'a Vec<Vec<f32>>,
    compute: F,
}

const DELTA: f32 = f16::EPSILON.to_f32_const();

impl<'a, P: Parallelism, F: Fn(&P, &Vec<Vec<f32>>) -> Vec<usize>> LloydKMeans<'a, P, F> {
    fn new(
        parallelism: &'a P,
        c: usize,
        dims: usize,
        samples: &'a Vec<Vec<f32>>,
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
            .into_par_iter(0..n)
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
            .into_par_iter(0..n)
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
            .into_par_iter(0..c)
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
            f32::kmeans_helper(&mut centroids[i], 1.0 + DELTA, 1.0 - DELTA);
            f32::kmeans_helper(&mut centroids[o], 1.0 - DELTA, 1.0 + DELTA);
            count[i] = count[o] / 2.0;
            count[o] -= count[i];
        }

        if self.is_spherical {
            self.parallelism
                .into_par_iter(&mut centroids)
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
