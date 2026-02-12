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

use crate::index::{flat_index as prefect_index, rabitq_index as index};
use crate::square::{Square, SquareMut};
use crate::{KMeans, This};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

struct RaBitQ<'a> {
    this: This<'a>,
    samples: SquareMut<'a>,
    centroids: Square,
    targets: Vec<usize>,
}

impl<'a> KMeans for RaBitQ<'a> {
    fn prefect_index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        let index = prefect_index(&self.centroids);
        Box::new(move |sample| {
            let rotated = rabitq::rotate::rotate(sample);
            let sample = rotated.as_slice();
            index(sample)
        })
    }

    fn index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        let index = index(self.this.pool, &self.centroids);
        Box::new(move |sample| {
            let rotated = rabitq::rotate::rotate(sample);
            let sample = rotated.as_slice();
            index(sample)
        })
    }

    fn assign(&mut self) {
        let this = &mut self.this;
        let samples = &mut self.samples;
        let centroids = &self.centroids;
        let targets = &mut self.targets;
        let index = index(this.pool, centroids);
        this.pool.install(|| {
            targets
                .par_iter_mut()
                .zip(samples.par_iter_mut())
                .for_each(|(target, sample)| {
                    *target = index(sample).1;
                });
        });
    }

    fn update(&mut self) {
        crate::index::update(
            &mut self.this,
            &self.samples,
            &self.targets,
            &mut self.centroids,
        );
    }

    fn finish(mut self: Box<Self>) -> Square {
        self.this.pool.install(|| {
            self.centroids.par_iter_mut().for_each(|centroid| {
                rabitq::rotate::rotate_reversed_inplace(centroid);
            });
        });
        self.centroids
    }
}

pub fn new<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    mut samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    let mut rng = StdRng::from_seed(seed);

    pool.install(|| {
        samples.par_iter_mut().for_each(|sample| {
            rabitq::rotate::rotate_inplace(sample);
        });
    });

    let mut centroids = Square::with_capacity(d, c);

    for index in rand::seq::index::sample(&mut rng, samples.len(), c.min(samples.len())) {
        centroids.push_slice(&samples[index]);
    }

    if centroids.is_empty() && c == 1 {
        centroids.push_iter(std::iter::repeat_n(0.0, d as _));
    }

    while centroids.len() < c {
        centroids.push_iter((0..d).map(|_| rng.random_range(-1.0f32..1.0f32)));
    }

    pool.install(|| {
        if is_spherical {
            use simd::Floating;
            (&mut centroids).into_par_iter().for_each(|centroid| {
                let l = f32::reduce_sum_of_x2(centroid).sqrt();
                f32::vector_mul_scalar_inplace(centroid, 1.0 / l);
            });
        }
    });

    let targets = vec![0; samples.len()];

    Box::new(RaBitQ {
        this: This {
            pool,
            d,
            c,
            rng,
            is_spherical,
        },
        samples,
        centroids,
        targets,
    })
}
