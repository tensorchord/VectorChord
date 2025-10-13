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
    this: This,
    top_centroids: Square,
    partition_start: Vec<usize>,
    bottom_k_means: Vec<Box<dyn KMeans + 'a>>,
}

impl<'a> KMeans for Hierarchical<'a> {
    fn this(&mut self) -> &mut This {
        &mut self.this
    }

    fn assign(&mut self) {
        for k_means in &mut self.bottom_k_means {
            k_means.assign();
        }
    }

    fn update(&mut self) {
        for k_means in &mut self.bottom_k_means {
            k_means.update();
        }
    }

    fn sphericalize(&mut self) {
        for k_means in &mut self.bottom_k_means {
            k_means.sphericalize();
        }
    }

    fn index(&mut self) -> Box<dyn Fn(&[f32]) -> u32 + '_> {
        let top_centroids = self.top_centroids.clone();
        let top = move |sample: &[f32]| crate::flat::k_means_lookup(sample, &top_centroids) as u32;
        let bottom_centroids: Vec<_> = self
            .bottom_k_means
            .iter_mut()
            .map(|k_means| k_means.index())
            .collect();
        let partition_start = self.partition_start.clone();
        let index = move |sample: &[f32]| {
            let top_id = top(sample) as usize;
            let bottom_id = bottom_centroids[top_id](sample) as usize;
            (partition_start[top_id] + bottom_id) as u32
        };
        Box::new(index)
    }

    fn finish(self: Box<Self>) -> Square {
        let mut ret = Square::new(self.this.d);
        for k_means in self.bottom_k_means {
            let centroids = k_means.finish();
            for centroid in centroids.into_iter() {
                ret.push_slice(centroid);
            }
        }
        ret
    }
}

const LOCAL_SAMPLE_FACTOR: usize = 256;
const LOCAL_NUM_ITERATIONS: usize = 10;

pub fn new<'a>(
    d: usize,
    mut samples: SquareMut<'a>,
    c: usize,
    num_threads: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build thread pool");
    let centroids = Square::new(d);
    let targets = vec![0; samples.len()];
    let samples_len = samples.len();
    let top_list =
        ((c as f64).sqrt().floor() as u32).clamp(1, (samples_len as f64).sqrt().floor() as u32);
    let top_samples_len = LOCAL_SAMPLE_FACTOR * (top_list as usize);
    let mut top_samples = Square::new(d);
    let mut rng = StdRng::from_seed(seed);
    for index in
        rand::seq::index::sample(&mut rng, samples.len(), top_samples_len.min(samples.len()))
    {
        top_samples.push_slice(&samples[index]);
    }
    let mut f = crate::k_means(
        d,
        top_samples.as_mut_view(),
        top_list as usize,
        num_threads,
        seed,
    );
    if is_spherical {
        f.sphericalize();
    }
    for _ in 0..LOCAL_NUM_ITERATIONS {
        f.assign();
        f.update();
        if is_spherical {
            f.sphericalize();
        }
    }
    let top_centroids = f.finish();
    let mut final_assign = vec![0; samples.len()];
    pool.install(|| {
        final_assign
            .par_iter_mut()
            .zip(samples.par_iter_mut())
            .for_each(|(target, sample)| {
                *target = crate::flat::k_means_lookup(sample, &top_centroids);
            });
    });
    let alloc = final_assign.into_iter().enumerate().fold(
        vec![vec![]; top_centroids.len()],
        |mut acc, (i, target)| {
            acc[target].push(i);
            acc
        },
    );
    let alloc_size = alloc.iter().map(|x| x.len() as u32).collect::<Vec<_>>();
    let keep_indices: Vec<usize> = alloc_size
        .iter()
        .enumerate()
        .filter_map(|(i, size)| if *size > 0 { Some(i) } else { None })
        .collect();
    let alloc: Vec<_> = keep_indices.iter().map(|&i| alloc[i].clone()).collect();
    let alloc_size: Vec<_> = keep_indices.iter().map(|&i| alloc_size[i]).collect();
    let alloc_lists = successive_quotients_allocate(c, alloc_size);

    let mut bottom_k_means = vec![];
    let mut partition_start = vec![];
    let mut offset = 0;
    let all_sub_samples = partition_mut(samples, &alloc);
    for (sub_samples, nlist) in all_sub_samples.into_iter().zip(alloc_lists) {
        partition_start.push(offset);
        offset += nlist as usize;
        let f = crate::k_means(d, sub_samples, nlist as usize, num_threads, seed);
        bottom_k_means.push(f);
    }
    Box::new(Hierarchical {
        this: This {
            pool,
            d,
            c,
            centroids,
            targets,
            rng,
        },
        top_centroids,
        partition_start,
        bottom_k_means,
    })
}

/// Allocate clusters to different parts according to the given proportions
///
/// See: https://en.wikipedia.org/wiki/Sainte-Lagu%C3%AB_method
fn successive_quotients_allocate(all_clusters: usize, proportion: Vec<u32>) -> Vec<u32> {
    let mut alloc_lists = vec![1u32; proportion.len()];
    let mut diff = all_clusters as i32 - proportion.len() as i32;
    assert!(diff >= 0);
    let mut priorities: BinaryHeap<(AlwaysEqual<usize>, Distance)> = proportion
        .iter()
        .enumerate()
        .map(|(i, x)| {
            (
                AlwaysEqual(i),
                Distance::from_f32(*x as f32 / (alloc_lists[i] as f32 + 0.5)),
            )
        })
        .collect();
    while diff > 0 {
        let index = priorities.pop().unwrap().0.0;
        alloc_lists[index] += 1;
        priorities.push((
            AlwaysEqual(index),
            Distance::from_f32(proportion[index] as f32 / (alloc_lists[index] as f32 + 0.5)),
        ));
        diff -= 1;
    }
    alloc_lists
}

pub fn partition_mut<'a>(
    mut a: SquareMut<'a>,
    groups: &[impl AsRef<[usize]>],
) -> Vec<SquareMut<'a>> {
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

#[cfg(test)]
mod partition_tests {
    use super::*;
    use rand::prelude::*;

    fn mk_square_random(d: usize, rows: usize, rng: &mut impl Rng) -> Square {
        let mut s = Square::with_capacity(d, rows);
        for _ in 0..rows {
            let row: Vec<f32> = (0..d).map(|_| rng.random_range(-1000.0..1000.0)).collect();
            s.push_slice(&row);
        }
        s
    }

    fn gen_random_alloc(rows: usize, groups: usize, rng: &mut impl Rng) -> Vec<Vec<usize>> {
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
    #[test]
    fn random_partition() {
        let mut rng = StdRng::seed_from_u64(7);
        for trial in 0..1000 {
            let d = rng.random_range(1..10);
            let rows = rng.random_range(1000..2000);
            let groups = rng.random_range(10..=rows.min(20));
            let mut s = mk_square_random(d, rows, &mut rng);
            let golden: Vec<Vec<f32>> = (0..s.len()).map(|i| s[i].to_vec()).collect();
            let alloc = gen_random_alloc(rows, groups, &mut rng);
            let views = partition_mut(s.as_mut_view(), &alloc);
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
}
