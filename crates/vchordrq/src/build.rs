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

use crate::operator::{Operator, Vector};
use crate::tape::TapeWriter;
use crate::tape_writer::{DirectoryTapeWriter, H1TapeWriter};
use crate::tuples::*;
use crate::types::*;
use crate::{Branch, Opaque};
use index::relation::{Page, RelationWrite};
use vector::{VectorBorrowed, VectorOwned};

pub fn build<R: RelationWrite, O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: &R,
    structures: Vec<Structure<O::Vector>>,
) where
    R::Page: Page<Opaque = Opaque>,
{
    let dims = vector_options.dims;
    let is_residual = vchordrq_options.residual_quantization;
    let mut meta = TapeWriter::<_, MetaTuple>::create(index, false);
    assert_eq!(meta.first(), 0);
    let mut freepages = TapeWriter::<_, FreepagesTuple>::create(index, false);
    freepages.push(FreepagesTuple {});
    let mut centroids = TapeWriter::<_, CentroidTuple<O::Vector>>::create(index, false);
    let vectors = (0..vchordrq_options.degree_of_parallelism)
        .map(|_| TapeWriter::<_, VectorTuple<O::Vector>>::create(index, true).first())
        .collect::<Vec<_>>();
    let mut pointer_of_centroids = Vec::<Vec<(Vec<u32>, u16)>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].centroids[j].as_borrowed();
            let (slices, metadata) = O::Vector::split(vector);
            let mut chain = Ok(metadata);
            let mut prefetch = Vec::new();
            for i in (0..slices.len()).rev() {
                let (id, head) = centroids.push(match chain {
                    Ok(metadata) => CentroidTuple::_0 {
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(head) => CentroidTuple::_1 {
                        elements: slices[i].to_vec(),
                        head,
                    },
                });
                chain = Err(head);
                prefetch.push(id);
            }
            prefetch.reverse();
            level.push((
                prefetch,
                chain.expect_err("internal error: 0-dimensional vector"),
            ));
        }
        pointer_of_centroids.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let frozen_tape = TapeWriter::<_, FrozenTuple>::create(index, false);
                let appendable_tape = TapeWriter::<_, AppendableTuple>::create(index, false);
                let frozen_first = { frozen_tape }.first();

                let mut directory_tape = DirectoryTapeWriter::create(index, false);
                directory_tape.push(&[frozen_first]);
                let directory_tape = directory_tape.into_inner();

                let mut jump = TapeWriter::<_, JumpTuple>::create(index, false);
                jump.push(JumpTuple {
                    directory_first: { directory_tape }.first(),
                    frozen_first,
                    appendable_first: { appendable_tape }.first(),
                    centroid_prefetch: pointer_of_centroids[i][j].0.clone(),
                    centroid_head: pointer_of_centroids[i][j].1,
                    tuples: 0,
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::create(index, O::Vector::count(dims as _), false);
                let centroid = structures[i].centroids[j].as_borrowed();
                for child in structures[i].children[j].iter().copied() {
                    let vector = structures[i - 1].centroids[child as usize].as_borrowed();
                    let (code, delta) = O::build(vector, is_residual.then_some(centroid.own()));
                    tape.push(Branch {
                        code,
                        delta,
                        prefetch: pointer_of_centroids[i - 1][child as usize].0.clone(),
                        head: pointer_of_centroids[i - 1][child as usize].1,
                        extra: pointer_of_firsts[i - 1][child as usize],
                        norm: norm::<O::Vector>(vector),
                    });
                }
                let (mut tape, chunk) = tape.into_inner();
                H1TapeWriter::flush(&mut tape, O::Vector::count(dims as _), chunk);
                level.push(tape.first());
            }
        }
        pointer_of_firsts.push(level);
    }
    meta.push(MetaTuple {
        dims,
        epsilon: vchordrq_options.epsilon,
        height_of_root: structures.len() as u32,
        is_residual,
        rerank_in_heap: vchordrq_options.rerank_in_table,
        centroids_first: centroids.first(),
        vectors_first: vectors,
        centroid_prefetch: pointer_of_centroids
            .last()
            .expect("internal error: empty structure")[0]
            .0
            .clone(),
        centroid_head: pointer_of_centroids
            .last()
            .expect("internal error: empty structure")[0]
            .1,
        centroid_norm: norm::<O::Vector>(
            structures
                .last()
                .expect("internal error: empty structure")
                .centroids[0]
                .as_borrowed(),
        ),
        first: pointer_of_firsts
            .last()
            .expect("internal error: empty structure")[0],
        freepages_first: freepages.first(),
        cells: structures.iter().map(|s| s.len() as _).collect(),
    });
}

fn norm<V: Vector>(vector: V::Borrowed<'_>) -> f32 {
    V::squared_norm(vector).sqrt()
}
