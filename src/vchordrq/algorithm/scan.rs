use super::RelationRead;
use crate::vchordrq::algorithm::rabitq::fscan_process_lowerbound;
use crate::vchordrq::algorithm::tuples::*;
use crate::vchordrq::algorithm::vectors;
use base::always_equal::AlwaysEqual;
use base::distance::Distance;
use base::distance::DistanceKind;
use base::search::Pointer;
use base::vector::VectorBorrowed;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn scan<V: Vector>(
    relation: impl RelationRead + Clone,
    vector: V,
    distance_kind: DistanceKind,
    probes: Vec<u32>,
    epsilon: f32,
) -> impl Iterator<Item = (Distance, Pointer)> {
    let vector = vector.as_borrowed();
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    let dims = meta_tuple.dims;
    let height_of_root = meta_tuple.height_of_root;
    assert_eq!(dims, vector.dims(), "invalid vector dimensions");
    assert_eq!(height_of_root as usize, 1 + probes.len(), "invalid probes");
    let vector = V::random_projection(vector);
    let is_residual = meta_tuple.is_residual;
    let default_lut = if !is_residual {
        Some(V::rabitq_fscan_preprocess(vector.as_borrowed()))
    } else {
        None
    };
    let mut lists: Vec<_> = vec![{
        let Some((_, original)) = vectors::vector_dist::<V>(
            relation.clone(),
            vector.as_borrowed(),
            meta_tuple.mean,
            None,
            None,
            is_residual,
        ) else {
            panic!("data corruption")
        };
        (meta_tuple.first, original)
    }];
    let make_lists = |lists: Vec<(u32, Option<V>)>, probes| {
        let mut results = Vec::new();
        for list in lists {
            let lut = if is_residual {
                &V::rabitq_fscan_preprocess(
                    V::residual(
                        vector.as_borrowed(),
                        list.1.as_ref().map(|x| x.as_borrowed()).unwrap(),
                    )
                    .as_borrowed(),
                )
            } else {
                default_lut.as_ref().unwrap()
            };
            let mut current = list.0;
            while current != u32::MAX {
                let h1_guard = relation.read(current);
                for i in 1..=h1_guard.len() {
                    let h1_tuple = h1_guard
                        .get(i)
                        .map(rkyv::check_archived_root::<Height1Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    let lowerbounds = fscan_process_lowerbound(
                        distance_kind,
                        dims,
                        lut,
                        (
                            h1_tuple.dis_u_2,
                            h1_tuple.factor_ppc,
                            h1_tuple.factor_ip,
                            h1_tuple.factor_err,
                            &h1_tuple.t,
                        ),
                        epsilon,
                    );
                    results.push((
                        Reverse(lowerbounds),
                        AlwaysEqual(h1_tuple.mean),
                        AlwaysEqual(h1_tuple.first),
                    ));
                }
                current = h1_guard.get_opaque().next;
            }
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        std::iter::from_fn(|| {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                let Some((Some(dis_u), original)) = vectors::vector_dist::<V>(
                    relation.clone(),
                    vector.as_borrowed(),
                    mean,
                    None,
                    Some(distance_kind),
                    is_residual,
                ) else {
                    panic!("data corruption")
                };
                cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(original)));
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop()?;
            Some((first, mean))
        })
        .take(probes as usize)
        .collect()
    };
    for i in (1..meta_tuple.height_of_root).rev() {
        lists = make_lists(lists, probes[i as usize - 1]);
    }
    drop(meta_guard);
    {
        let mut results = Vec::new();
        for list in lists {
            let lut = if is_residual {
                &V::rabitq_fscan_preprocess(
                    V::residual(
                        vector.as_borrowed(),
                        list.1.as_ref().map(|x| x.as_borrowed()).unwrap(),
                    )
                    .as_borrowed(),
                )
            } else {
                default_lut.as_ref().unwrap()
            };
            let mut current = list.0;
            while current != u32::MAX {
                let h0_guard = relation.read(current);
                for i in 1..=h0_guard.len() {
                    let h0_tuple = h0_guard
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    let lowerbounds = fscan_process_lowerbound(
                        distance_kind,
                        dims,
                        lut,
                        (
                            h0_tuple.dis_u_2,
                            h0_tuple.factor_ppc,
                            h0_tuple.factor_ip,
                            h0_tuple.factor_err,
                            &h0_tuple.t,
                        ),
                        epsilon,
                    );
                    results.push((
                        Reverse(lowerbounds),
                        AlwaysEqual(h0_tuple.mean),
                        AlwaysEqual(h0_tuple.payload),
                    ));
                }
                current = h0_guard.get_opaque().next;
            }
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
        std::iter::from_fn(move || {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(pay_u)) = heap.pop().unwrap();
                let Some((Some(dis_u), _)) = vectors::vector_dist::<V>(
                    relation.clone(),
                    vector.as_borrowed(),
                    mean,
                    Some(pay_u),
                    Some(distance_kind),
                    false,
                ) else {
                    continue;
                };
                cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            }
            let (Reverse(dis_u), AlwaysEqual(pay_u)) = cache.pop()?;
            Some((dis_u, Pointer::new(pay_u)))
        })
    }
}
