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

use crate::square::Square;
use crate::{KMeans, This};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use simd::Floating;

struct RaBitQ {
    this: This,
}

impl KMeans for RaBitQ {
    fn this(&mut self) -> &mut This {
        &mut self.this
    }

    fn assign(&mut self) {
        let this = &mut self.this;
        this.pool.install(|| {
            use rabitq::packing::{pack_to_u4, padding_pack};

            let metadata = this
                .centroids
                .par_iter()
                .map(rabitq::bit::code_metadata)
                .collect::<Vec<_>>();

            let blocks = this
                .centroids
                .par_iter()
                .chunks(32)
                .map(|chunk| {
                    let f = |x: &&_| pack_to_u4(&rabitq::bit::code_elements(x));
                    padding_pack(chunk.iter().map(f))
                })
                .collect::<Vec<_>>();

            this.targets
                .par_iter_mut()
                .zip(this.samples.into_par_iter())
                .for_each(|(target, sample)| {
                    let lut = rabitq::bit::block::preprocess(sample);
                    let mut result = (f32::INFINITY, 0);
                    let mut sum = [0u32; 32];
                    for (j, centroid) in this.centroids.into_iter().enumerate() {
                        if j % 32 == 0 {
                            sum = rabitq::bit::block::accumulate(&blocks[j / 32], &lut.1);
                        }
                        let (rough, err) =
                            rabitq::bit::block::half_process_l2(sum[j % 32], metadata[j], lut.0);
                        let lowerbound = rough - err * 1.9;
                        if lowerbound < result.0 {
                            let dis_2 = f32::reduce_sum_of_d2(sample, centroid);
                            if dis_2 <= result.0 {
                                result = (dis_2, j);
                            }
                        }
                    }
                    *target = result.1;
                });
        });
    }

    fn update(&mut self) {
        let this = &mut self.this;
        this.pool.install(|| {
            const DELTA: f32 = 9.7656e-4_f32;

            let d = this.d;
            let n = this.samples.len();
            let c = this.c;

            let list = rayon::broadcast({
                |ctx| {
                    let mut sum = Square::from_zeros(d, c);
                    let mut count = vec![0.0f32; c];
                    for i in (ctx.index()..this.samples.len()).step_by(ctx.num_threads()) {
                        let target = this.targets[i];
                        let sample = &this.samples[i];
                        f32::vector_add_inplace(&mut sum[target], sample);
                        count[target] += 1.0;
                    }
                    (sum, count)
                }
            });
            let mut sum = Square::from_zeros(d, c);
            let mut count = vec![0.0f32; c];
            for (sum_1, count_1) in list {
                for i in 0..c {
                    f32::vector_add_inplace(&mut sum[i], &sum_1[i]);
                    count[i] += count_1[i];
                }
            }

            sum.par_iter_mut()
                .enumerate()
                .for_each(|(i, sum)| f32::vector_mul_scalar_inplace(sum, 1.0 / count[i]));

            this.centroids = sum;

            for i in 0..c {
                if count[i] != 0.0f32 {
                    continue;
                }
                let mut o = 0;
                loop {
                    let alpha = this.rng.random_range(0.0..1.0f32);
                    let beta = (count[o] - 1.0) / (n - c) as f32;
                    if alpha < beta {
                        break;
                    }
                    o = (o + 1) % c;
                }
                this.centroids.copy_within(o..o + 1, i);
                vector_mul_scalars_inplace(&mut this.centroids[i], [1.0 + DELTA, 1.0 - DELTA]);
                vector_mul_scalars_inplace(&mut this.centroids[o], [1.0 - DELTA, 1.0 + DELTA]);
                count[i] = count[o] / 2.0;
                count[o] -= count[i];
            }
        });
    }

    fn finish(self: Box<Self>) -> Square {
        let mut this = self.this;
        this.pool.install(|| {
            this.centroids.par_iter_mut().for_each(|centroid| {
                rabitq::rotate::rotate_reversed_inplace(centroid);
            });
        });
        this.centroids
    }
}

pub fn new(
    d: usize,
    mut samples: Square,
    c: usize,
    num_threads: usize,
    seed: [u8; 32],
) -> Box<dyn KMeans> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build thread pool");
    let mut rng = StdRng::from_seed(seed);

    pool.install(|| {
        samples.par_iter_mut().for_each(|sample| {
            rabitq::rotate::rotate_inplace(sample);
        });
    });

    let mut centroids = Square::new(d);

    for index in rand::seq::index::sample(&mut rng, samples.len(), c.min(samples.len())) {
        centroids.push_slice(&samples[index]);
    }

    if centroids.is_empty() && c == 1 {
        centroids.push_iter(std::iter::repeat_n(0.0, d as _));
    }

    while centroids.len() < c {
        centroids.push_iter((0..d).map(|_| rng.random_range(-1.0f32..1.0f32)));
    }

    let targets = vec![0; samples.len()];

    Box::new(RaBitQ {
        this: This {
            pool,
            d,
            c,
            centroids,
            targets,
            rng,
            samples,
        },
    })
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
