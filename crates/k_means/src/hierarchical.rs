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
use always_equal::AlwaysEqual;
use distance::Distance;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::collections::BinaryHeap;

struct Hierarchical<'a> {
    this: This<'a>,
    partitions: Vec<Box<dyn KMeans + 'a>>,
    coarse_centroids: Square,
    offsets: Vec<usize>,
}

impl<'a> KMeans for Hierarchical<'a> {
    fn prefect_index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        Box::new(prefect_index(&self.partitions, &self.offsets))
    }

    fn index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        Box::new(index(
            &self.partitions,
            &self.coarse_centroids,
            &self.offsets,
        ))
    }

    fn assign(&mut self) {
        for partial in self.partitions.iter_mut() {
            partial.assign();
        }
    }

    fn update(&mut self) {
        for partial in self.partitions.iter_mut() {
            partial.update();
        }
    }

    fn finish(self: Box<Self>) -> Square {
        let mut centroids = Square::new(self.this.d);
        for k_means in self.partitions {
            let partial_centroids = k_means.finish();
            for centroid in partial_centroids.into_iter() {
                centroids.push_slice(centroid);
            }
        }
        centroids
    }
}

fn prefect_index(
    partitions: &[Box<dyn KMeans + '_>],
    offsets: &[usize],
) -> impl Fn(&[f32]) -> (f32, usize) + Sync {
    let indexes = partitions
        .iter()
        .map(|p| p.prefect_index())
        .collect::<Vec<_>>();
    move |sample| {
        let mut result = (f32::INFINITY, 0);
        for (id, index) in indexes.iter().enumerate() {
            let partial_result = index(sample);
            if partial_result.0 <= result.0 {
                result = (partial_result.0, offsets[id] + partial_result.1);
            }
        }
        result
    }
}

fn index(
    partitions: &[Box<dyn KMeans + '_>],
    coarse_centroids: &Square,
    offsets: &[usize],
) -> impl Fn(&[f32]) -> (f32, usize) + Sync {
    let indexes = partitions.iter().map(|p| p.index()).collect::<Vec<_>>();
    move |sample| {
        use simd::Floating;
        let mut result = (f32::INFINITY, 0);
        for i in 0..coarse_centroids.len() {
            let dis = f32::reduce_sum_of_d2(sample, &coarse_centroids[i]);
            if dis <= result.0 {
                result = (dis, i);
            }
        }
        let id = result.1;
        let result = indexes[id](sample);
        (result.0, offsets[id] + result.1)
    }
}

const COARSE_SAMPLING_FACTOR: usize = 256;
const COARSE_ITERATIONS: usize = 10;

pub fn new<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    mut samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    let mut rng = StdRng::from_seed(seed);

    let mut coarse_samples = {
        let mut coarse_samples = Square::new(d);
        let s = c.isqrt().saturating_mul(COARSE_SAMPLING_FACTOR);
        for index in rand::seq::index::sample(&mut rng, samples.len(), s.min(samples.len())) {
            coarse_samples.push_slice(&samples[index]);
        }
        coarse_samples
    };
    let coarse_k_means = {
        let mut coarse_k_means = crate::lloyd_k_means(
            pool,
            d,
            coarse_samples.as_mut_view(),
            c.isqrt(),
            seed,
            is_spherical,
        );
        for _ in 0..COARSE_ITERATIONS {
            coarse_k_means.assign();
            coarse_k_means.update();
        }
        coarse_k_means
    };
    let coarse_assign = {
        let coarse_index = coarse_k_means.prefect_index();
        let mut coarse_assign = vec![0; samples.len()];
        pool.install(|| {
            coarse_assign
                .par_iter_mut()
                .zip(samples.par_iter_mut())
                .for_each(|(target, sample)| {
                    *target = coarse_index(sample).1;
                });
        });
        coarse_assign
    };
    let coarse_centroids = coarse_k_means.finish();

    let (weight, groups) = {
        let mut weight = vec![0_usize; coarse_centroids.len()];
        let mut groups = vec![vec![]; coarse_centroids.len()];
        for (i, &target) in coarse_assign.iter().enumerate() {
            weight[target] += 1;
            groups[target].push(i);
        }
        (weight, groups)
    };
    let seats = modified_webster_method(c, &weight);

    let mut partitions = vec![];
    let mut offsets = vec![];
    let mut offset = 0;
    for (samples, c) in std::iter::zip(partition(samples, &groups), seats) {
        partitions.push(crate::lloyd_k_means(
            pool,
            d,
            samples,
            c,
            seed,
            is_spherical,
        ));
        offsets.push(offset);
        offset += c;
    }

    Box::new(Hierarchical {
        this: This {
            pool,
            d,
            c,
            rng,
            is_spherical,
        },
        coarse_centroids,
        partitions,
        offsets,
    })
}

// https://en.wikipedia.org/wiki/Sainte-Lagu%C3%AB_method
fn modified_webster_method(n: usize, weight: &[usize]) -> Vec<usize> {
    assert!(n >= weight.len());
    let mut seats = vec![0_usize; weight.len()];
    let mut quotients = Vec::new();
    for index in 0..weight.len() {
        seats[index] += 1;
        let quotient = weight[index] as f64 / (seats[index] as f64 + 0.5);
        quotients.push((Distance::from_f32(quotient as _), AlwaysEqual(index)));
    }
    let mut quotients = BinaryHeap::<_>::from(quotients);
    for _ in weight.len()..n {
        let Some((_, AlwaysEqual(index))) = quotients.pop() else {
            break;
        };
        seats[index] += 1;
        let quotient = weight[index] as f64 / (seats[index] as f64 + 0.5);
        quotients.push((Distance::from_f32(quotient as _), AlwaysEqual(index)));
    }
    seats
}

fn partition<'a>(mut a: SquareMut<'a>, groups: &[impl AsRef<[usize]>]) -> Vec<SquareMut<'a>> {
    let n = a.len();
    let permutation = groups
        .iter()
        .flat_map(AsRef::as_ref)
        .copied()
        .collect::<Vec<_>>();
    let mut marked = vec![false; a.len()];
    let mut buffer = vec![0.0; a.d()];
    for i in 0..n {
        if marked[i] {
            continue;
        }
        buffer.copy_from_slice(&a[i]);
        let (mut src, mut dst) = (permutation[i], i);
        while src != i {
            a.copy_within(src..src + 1, dst);
            marked[dst] = true;
            (src, dst) = (permutation[src], src);
        }
        a[dst].copy_from_slice(&buffer);
        marked[dst] = true;
    }
    let mut result = Vec::with_capacity(groups.len());
    let (d, mut p) = a.into_inner();
    for group in groups {
        let group = group.as_ref();
        let head;
        (head, p) = std::mem::take(&mut p).split_at_mut(group.len() * d);
        result.push(SquareMut::new(d, head));
    }
    result
}

#[test]
fn test_modified_webster_method() {
    let seats = modified_webster_method(51, &[10, 10, 10, 11, 9]);
    assert_eq!(seats[0], 10);
    assert_eq!(seats[1], 10);
    assert_eq!(seats[2], 10);
    assert_eq!(seats[3], 12);
    assert_eq!(seats[4], 9);
}

#[test]
fn test_partition() {
    fn gen_random_alloc(rows: usize, groups: usize, rng: &mut impl RngExt) -> Vec<Vec<usize>> {
        let mut idx: Vec<usize> = (0..rows).collect();
        idx.shuffle(rng);
        let mut alloc = Vec::with_capacity(groups);
        let mut start = 0;
        for g in 0..groups {
            let rem = groups - g;
            let take = if rem == 1 {
                rows - start
            } else {
                rng.random_range(1..=(rows - start - (rem - 1)))
            };
            alloc.push(idx[start..start + take].to_vec());
            start += take;
        }
        alloc
    }

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(7);
    for trial in 0..if cfg!(not(miri)) { 1000 } else { 1 } {
        let d = rng.random_range(1..10);
        let rows = rng.random_range(1000..2000);
        let groups = rng.random_range(10..=rows.min(20));
        let mut s = {
            let mut result = Square::with_capacity(d, rows);
            for _ in 0..rows {
                result.push_iter((0..d).map(|_| rng.random_range(-1000.0..1000.0)));
            }
            result
        };
        let golden: Vec<Vec<f32>> = (0..s.len()).map(|i| s[i].to_vec()).collect();
        let alloc = gen_random_alloc(rows, groups, &mut rng);
        let views = partition(s.as_mut_view(), &alloc);
        assert_eq!(views.len(), alloc.len(), "trial {}", trial);

        for (g, group) in alloc.iter().enumerate() {
            let v = &views[g];
            assert_eq!(v.len(), group.len(), "trial {}, group {}", trial, g);
            assert_eq!(v.d(), d);
            for (r, &row_idx) in group.iter().enumerate() {
                assert_eq!(
                    v.row(r),
                    &golden[row_idx][..],
                    "trial {}, group {}, row {}",
                    trial,
                    g,
                    r
                );
            }
        }
    }
}
