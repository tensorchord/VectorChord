use crate::RelationWrite;
use crate::operator::{Accessor2, Operator, Vector};
use crate::tape::*;
use crate::tuples::*;
use crate::types::*;
use vector::VectorOwned;

pub fn build<O: Operator>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: impl RelationWrite,
    structures: Vec<Structure<O::Vector>>,
) {
    let dims = vector_options.dims;
    let is_residual = vchordrq_options.residual_quantization && O::SUPPORTS_RESIDUAL;
    let mut meta = TapeWriter::<_, _, MetaTuple>::create(|| index.extend(false));
    assert_eq!(meta.first(), 0);
    let freepage = TapeWriter::<_, _, FreepageTuple>::create(|| index.extend(false));
    let mut vectors = TapeWriter::<_, _, VectorTuple<O::Vector>>::create(|| index.extend(true));
    let mut pointer_of_means = Vec::<Vec<IndexPointer>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            let vector = structures[i].means[j].as_borrowed();
            let (metadata, slices) = O::Vector::vector_split(vector);
            let mut chain = Ok(metadata);
            for i in (0..slices.len()).rev() {
                chain = Err(vectors.push(match chain {
                    Ok(metadata) => VectorTuple::_0 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        metadata,
                    },
                    Err(pointer) => VectorTuple::_1 {
                        payload: None,
                        elements: slices[i].to_vec(),
                        pointer,
                    },
                }));
            }
            level.push(chain.err().unwrap());
        }
        pointer_of_means.push(level);
    }
    let mut pointer_of_firsts = Vec::<Vec<u32>>::new();
    for i in 0..structures.len() {
        let mut level = Vec::new();
        for j in 0..structures[i].len() {
            if i == 0 {
                let tape = TapeWriter::<_, _, H0Tuple>::create(|| index.extend(false));
                let mut jump = TapeWriter::<_, _, JumpTuple>::create(|| index.extend(false));
                jump.push(JumpTuple {
                    first: tape.first(),
                });
                level.push(jump.first());
            } else {
                let mut tape = H1TapeWriter::<_, _>::create(|| index.extend(false));
                let h2_mean = structures[i].means[j].as_borrowed();
                let h2_children = structures[i].children[j].as_slice();
                for child in h2_children.iter().copied() {
                    let h1_mean = structures[i - 1].means[child as usize].as_borrowed();
                    let code = if is_residual {
                        let mut residual_accessor = O::ResidualAccessor::default();
                        residual_accessor.push(
                            O::Vector::elements_and_metadata(h1_mean).0,
                            O::Vector::elements_and_metadata(h2_mean).0,
                        );
                        let residual = residual_accessor.finish(
                            O::Vector::elements_and_metadata(h1_mean).1,
                            O::Vector::elements_and_metadata(h2_mean).1,
                        );
                        O::Vector::code(residual.as_borrowed())
                    } else {
                        O::Vector::code(h1_mean)
                    };
                    tape.push(H1Branch {
                        mean: pointer_of_means[i - 1][child as usize],
                        dis_u_2: code.dis_u_2,
                        factor_ppc: code.factor_ppc,
                        factor_ip: code.factor_ip,
                        factor_err: code.factor_err,
                        signs: code.signs,
                        first: pointer_of_firsts[i - 1][child as usize],
                    });
                }
                let tape = tape.into_inner();
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
        root_mean: pointer_of_means.last().unwrap()[0],
        root_first: pointer_of_firsts.last().unwrap()[0],
        freepage_first: freepage.first(),
    });
}
