use super::RelationWrite;
use crate::vchordrq::algorithm::rabitq::fscan_process_lowerbound;
use crate::vchordrq::algorithm::tuples::*;
use crate::vchordrq::algorithm::vectors;
use crate::vchordrq::algorithm::PageGuard;
use base::always_equal::AlwaysEqual;
use base::distance::Distance;
use base::distance::DistanceKind;
use base::search::Pointer;
use base::vector::VectorBorrowed;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn insert<V: Vector>(
    relation: impl RelationWrite + Clone,
    payload: Pointer,
    vector: V,
    distance_kind: DistanceKind,
    in_building: bool,
) {
    let vector = vector.as_borrowed();
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    let dims = meta_tuple.dims;
    assert_eq!(dims, vector.dims(), "invalid vector dimensions");
    let vector = V::random_projection(vector);
    let vector = vector.as_borrowed();
    let is_residual = meta_tuple.is_residual;
    let default_lut = if !is_residual {
        Some(V::rabitq_fscan_preprocess(vector))
    } else {
        None
    };
    let h0_vector = {
        let (metadata, slices) = V::vector_split(vector);
        let mut chain = Err(metadata);
        for i in (0..slices.len()).rev() {
            let tuple = rkyv::to_bytes::<_, 8192>(&VectorTuple::<V> {
                slice: slices[i].to_vec(),
                payload: Some(payload.as_u64()),
                chain,
            })
            .unwrap();
            chain = Ok(append(
                relation.clone(),
                meta_tuple.vectors_first,
                &tuple,
                true,
                true,
                true,
            ));
        }
        chain.ok().unwrap()
    };
    let h0_payload = payload.as_u64();
    let mut list = {
        let Some((_, original)) = vectors::vector_dist::<V>(
            relation.clone(),
            vector,
            meta_tuple.mean,
            None,
            None,
            is_residual,
        ) else {
            panic!("data corruption")
        };
        (meta_tuple.first, original)
    };
    let make_list = |list: (u32, Option<V>)| {
        let mut results = Vec::new();
        {
            let lut = if is_residual {
                &V::rabitq_fscan_preprocess(
                    V::residual(vector, list.1.as_ref().map(|x| x.as_borrowed()).unwrap())
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
                        1.9,
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
        {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                let Some((Some(dis_u), original)) = vectors::vector_dist::<V>(
                    relation.clone(),
                    vector,
                    mean,
                    None,
                    Some(distance_kind),
                    is_residual,
                ) else {
                    panic!("data corruption")
                };
                cache.push((Reverse(dis_u), AlwaysEqual(first), AlwaysEqual(original)));
            }
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop().unwrap();
            (first, mean)
        }
    };
    for _ in (1..meta_tuple.height_of_root).rev() {
        list = make_list(list);
    }
    let code = if is_residual {
        V::rabitq_code(
            dims,
            V::residual(vector, list.1.as_ref().map(|x| x.as_borrowed()).unwrap()).as_borrowed(),
        )
    } else {
        V::rabitq_code(dims, vector)
    };
    let tuple = rkyv::to_bytes::<_, 8192>(&Height0Tuple {
        mean: h0_vector,
        payload: h0_payload,
        dis_u_2: code.dis_u_2,
        factor_ppc: code.factor_ppc,
        factor_ip: code.factor_ip,
        factor_err: code.factor_err,
        t: code.t(),
    })
    .unwrap();
    append(
        relation.clone(),
        list.0,
        &tuple,
        false,
        in_building,
        in_building,
    );
}

fn append(
    relation: impl RelationWrite,
    first: u32,
    tuple: &[u8],
    tracking_freespace: bool,
    skipping_traversal: bool,
    updating_skip: bool,
) -> (u32, u16) {
    if tracking_freespace {
        if let Some(mut write) = relation.search(tuple.len()) {
            let i = write.alloc(tuple).unwrap();
            return (write.id(), i);
        }
    }
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = relation.read(current);
        if read.freespace() as usize >= tuple.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = relation.write(current, tracking_freespace);
            if let Some(i) = write.alloc(tuple) {
                return (current, i);
            }
            if write.get_opaque().next == u32::MAX {
                let mut extend = relation.extend(tracking_freespace);
                write.get_opaque_mut().next = extend.id();
                drop(write);
                if let Some(i) = extend.alloc(tuple) {
                    let result = (extend.id(), i);
                    drop(extend);
                    if updating_skip {
                        let mut past = relation.write(first, tracking_freespace);
                        let skip = &mut past.get_opaque_mut().skip;
                        assert!(*skip != u32::MAX);
                        *skip = std::cmp::max(*skip, result.0);
                    }
                    return result;
                } else {
                    panic!("a tuple cannot even be fit in a fresh page");
                }
            }
            if skipping_traversal && current == first && write.get_opaque().skip != first {
                current = write.get_opaque().skip;
            } else {
                current = write.get_opaque().next;
            }
        } else {
            if skipping_traversal && current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}
