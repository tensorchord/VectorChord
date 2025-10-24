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

struct Quick {
    this: This,
}

impl KMeans for Quick {
    fn this(&mut self) -> &mut This {
        &mut self.this
    }

    fn assign(&mut self) -> Vec<usize> {
        self.this.targets.clone()
    }

    fn update(&mut self) {}

    fn finish(self: Box<Self>) -> (Square, Square) {
        let this = self.this;
        (this.centroids, this.samples)
    }
}

pub fn new(
    d: usize,
    samples: Square,
    c: usize,
    num_threads: usize,
    seed: [u8; 32],
) -> Box<dyn KMeans> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build thread pool");
    let mut rng = StdRng::from_seed(seed);

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

    Box::new(Quick {
        this: This {
            pool,
            rng,
            d,
            samples,
            c,
            centroids,
            targets,
        },
    })
}
