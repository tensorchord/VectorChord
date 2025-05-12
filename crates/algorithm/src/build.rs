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

use crate::operator::{Accessor2, Operator, Vector};
use crate::tape::TapeWriter;
use crate::tape_writer::H1TapeWriter;
use crate::tuples::*;
use crate::types::*;
use crate::{Branch, RelationWrite};
use vector::VectorOwned;

pub fn build<R: RelationWrite, O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: &R,
    structures: Vec<Structure<O::Vector>>,
) {
    if vchordrq_options.residual_quantization && !O::SUPPORTS_RESIDUAL {
        panic!("residual_quantization can be enabled only if distance type is L2");
    }
    let dims = vector_options.dims;
    let is_residual = vchordrq_options.residual_quantization;
    let mut meta = TapeWriter::<_, MetaTuple>::create(index, false);
    assert_eq!(meta.first(), 0);
    let freepage = TapeWriter::<_, FreepageTuple>::create(index, false);
    let mut vectors = TapeWriter::<_, VectorTuple<O::Vector>>::create(index, true);
    let mut pointer_of_means = Vec::<Vec<(Vec<u32>, u16)>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].means[j].as_borrowed();
            let (slices, metadata) = O::Vector::split(vector);
            let mut chain = Ok(metadata);
            let mut prefetch = Vec::new();
            for i in (0..slices.len()).rev() {
                let (id, head) = vectors.push(match chain {
                    Ok(metadata) => VectorTuple::_0 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(head) => VectorTuple::_1 {
                        payload: None,
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
        pointer_of_means.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let directory_tape = TapeWriter::<_, DirectoryTuple>::create(index, false);
                let appendable_tape = TapeWriter::<_, AppendableTuple>::create(index, false);
                let mut jump = TapeWriter::<_, JumpTuple>::create(index, false);
                jump.push(JumpTuple {
                    directory_first: directory_tape.first(),
                    appendable_first: appendable_tape.first(),
                    tuples: 0,
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::create(index, O::Vector::count(dims as _), false);
                let h2_mean = structures[i].means[j].as_borrowed();
                let h2_children = structures[i].children[j].as_slice();
                for child in h2_children.iter().copied() {
                    let h1_mean = structures[i - 1].means[child as usize].as_borrowed();
                    let code = if is_residual {
                        let mut residual_accessor = O::ResidualAccessor::default();
                        residual_accessor
                            .push(O::Vector::unpack(h1_mean).0, O::Vector::unpack(h2_mean).0);
                        let residual = residual_accessor
                            .finish(O::Vector::unpack(h1_mean).1, O::Vector::unpack(h2_mean).1);
                        O::Vector::code(residual.as_borrowed())
                    } else {
                        O::Vector::code(h1_mean)
                    };
                    tape.push(Branch {
                        head: pointer_of_means[i - 1][child as usize].1,
                        dis_u_2: code.dis_u_2,
                        factor_ppc: code.factor_ppc,
                        factor_ip: code.factor_ip,
                        factor_err: code.factor_err,
                        signs: code.signs,
                        prefetch: pointer_of_means[i - 1][child as usize].0.clone(),
                        extra: pointer_of_firsts[i - 1][child as usize],
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
        height_of_root: structures.len() as u32,
        is_residual,
        rerank_in_heap: vchordrq_options.rerank_in_table,
        vectors_first: vectors.first(),
        root_prefetch: pointer_of_means
            .last()
            .expect("internal error: empty structure")[0]
            .0
            .clone(),
        root_head: pointer_of_means
            .last()
            .expect("internal error: empty structure")[0]
            .1,
        root_first: pointer_of_firsts
            .last()
            .expect("internal error: empty structure")[0],
        freepage_first: freepage.first(),
        cells: structures.iter().map(|s| s.len() as _).collect(),
    });
}
