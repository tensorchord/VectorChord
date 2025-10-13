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

use crate::square::{Square, SquareMut};
use crate::{KMeans, This};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

struct Quick {
    this: This,
}

impl KMeans for Quick {
    fn this(&mut self) -> &mut This {
        &mut self.this
    }

    fn assign(&mut self) {}

    fn update(&mut self) {}

    fn index(&mut self) -> Box<dyn Fn(&[f32]) -> u32> {
        let this = self.this();
        let centroids = this.centroids.clone();
        Box::new(move |sample| crate::flat::k_means_lookup(sample, &centroids) as u32)
    }

    fn finish(self: Box<Self>) -> Square {
        let this = self.this;
        this.centroids
    }
}

pub fn new<'a>(
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    num_threads: usize,
    seed: [u8; 32],
) -> Box<dyn KMeans + 'a> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build thread pool");
    let mut rng = StdRng::from_seed(seed);

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

    let targets = vec![0; samples.len()];

    Box::new(Quick {
        this: This {
            pool,
            rng,
            d,
            c,
            centroids,
            targets,
        },
    })
}
