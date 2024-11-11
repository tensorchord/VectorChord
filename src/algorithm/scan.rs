use crate::algorithm::rabitq;
use crate::algorithm::rabitq::fscan_process_lowerbound;
use crate::algorithm::tuples::*;
use crate::index::utils::distance;
use crate::postgres::Relation;
use base::always_equal::AlwaysEqual;
use base::distance::Distance;
use base::distance::DistanceKind;
use base::scalar::ScalarLike;
use base::search::Pointer;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn scan(
    relation: Relation,
    vector: Vec<f32>,
    distance_kind: DistanceKind,
    nprobe_1: u32,
    epsilon: f32,
) -> impl Iterator<Item = (Distance, Pointer)> {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get()
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    let dims = meta_tuple.dims;
    assert_eq!(dims as usize, vector.len(), "invalid vector dimensions");
    let vector = rabitq::project(&vector);
    let is_residual = meta_tuple.is_residual;
    let default_lut = if !is_residual {
        Some(rabitq::fscan_preprocess(&vector))
    } else {
        None
    };
    let lists: Vec<_> = vec![(
        meta_tuple.first,
        if is_residual {
            let vector_guard = relation.read(meta_tuple.mean.0);
            let vector_tuple = vector_guard
                .get()
                .get(meta_tuple.mean.1)
                .map(rkyv::check_archived_root::<VectorTuple>)
                .expect("data corruption")
                .expect("data corruption");
            Some(vector_tuple.vector.to_vec())
        } else {
            None
        },
    )];
    let lists: Vec<_> = {
        let mut results = Vec::new();
        for list in lists {
            let lut = if is_residual {
                &rabitq::fscan_preprocess(&f32::vector_sub(&vector, list.1.as_ref().unwrap()))
            } else {
                default_lut.as_ref().unwrap()
            };
            let mut current = list.0;
            while current != u32::MAX {
                let h1_guard = relation.read(current);
                for i in 1..=h1_guard.get().len() {
                    let h1_tuple = h1_guard
                        .get()
                        .get(i)
                        .map(rkyv::check_archived_root::<Height1Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    let lowerbounds = fscan_process_lowerbound(
                        distance_kind,
                        dims,
                        lut,
                        (
                            &h1_tuple.dis_u_2,
                            &h1_tuple.factor_ppc,
                            &h1_tuple.factor_ip,
                            &h1_tuple.factor_err,
                            &h1_tuple.t,
                        ),
                        epsilon,
                    );
                    for j in 0..32 {
                        if h1_tuple.mask[j] {
                            results.push((
                                Reverse(lowerbounds[j]),
                                AlwaysEqual(h1_tuple.mean[j]),
                                AlwaysEqual(h1_tuple.first[j]),
                            ));
                        }
                    }
                }
                current = h1_guard.get().get_opaque().next;
            }
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        std::iter::from_fn(|| {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                let vector_guard = relation.read(mean.0);
                let vector_tuple = vector_guard
                    .get()
                    .get(mean.1)
                    .map(rkyv::check_archived_root::<VectorTuple>)
                    .expect("data corruption")
                    .expect("data corruption");
                let dis_u = distance(distance_kind, &vector, &vector_tuple.vector);
                cache.push((
                    Reverse(dis_u),
                    AlwaysEqual(first),
                    AlwaysEqual(if is_residual {
                        Some(vector_tuple.vector.to_vec())
                    } else {
                        None
                    }),
                ));
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean))
        })
        .take(nprobe_1 as usize)
        .collect()
    };
    {
        let mut results = Vec::new();
        for list in lists {
            let lut = if is_residual {
                &rabitq::fscan_preprocess(&f32::vector_sub(&vector, list.1.as_ref().unwrap()))
            } else {
                default_lut.as_ref().unwrap()
            };
            let mut current = list.0;
            while current != u32::MAX {
                let h0_guard = relation.read(current);
                for i in 1..=h0_guard.get().len() {
                    let h0_tuple = h0_guard
                        .get()
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    let lowerbounds = fscan_process_lowerbound(
                        distance_kind,
                        dims,
                        lut,
                        (
                            &h0_tuple.dis_u_2,
                            &h0_tuple.factor_ppc,
                            &h0_tuple.factor_ip,
                            &h0_tuple.factor_err,
                            &h0_tuple.t,
                        ),
                        epsilon,
                    );
                    for j in 0..32 {
                        if h0_tuple.mask[j] {
                            results.push((
                                Reverse(lowerbounds[j]),
                                AlwaysEqual(h0_tuple.mean[j]),
                                AlwaysEqual(h0_tuple.payload[j]),
                            ));
                        }
                    }
                }
                current = h0_guard.get().get_opaque().next;
            }
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
        std::iter::from_fn(move || {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(pay_u)) = heap.pop().unwrap();
                let vector_guard = relation.read(mean.0);
                let Some(vector_tuple) = vector_guard.get().get(mean.1) else {
                    // fails consistency check
                    continue;
                };
                let vector_tuple = rkyv::check_archived_root::<VectorTuple>(vector_tuple)
                    .expect("data corruption");
                if vector_tuple.payload != Some(pay_u) {
                    // fails consistency check
                    continue;
                }
                let dis_u = distance(distance_kind, &vector, &vector_tuple.vector);
                cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            }
            let (Reverse(dis_u), AlwaysEqual(pay_u)) = cache.pop()?;
            Some((dis_u, Pointer::new(pay_u)))
        })
    }
}
