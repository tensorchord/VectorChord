use crate::postgres::Relation;
use crate::vchordrqfscan::algorithm::rabitq;
use crate::vchordrqfscan::algorithm::rabitq::distance;
use crate::vchordrqfscan::algorithm::rabitq::fscan_process_lowerbound;
use crate::vchordrqfscan::algorithm::tuples::*;
use base::always_equal::AlwaysEqual;
use base::distance::Distance;
use base::distance::DistanceKind;
use base::search::Pointer;
use base::simd::ScalarLike;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub fn insert(relation: Relation, payload: Pointer, vector: Vec<f32>, distance_kind: DistanceKind) {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    let dims = meta_tuple.dims;
    assert_eq!(dims as usize, vector.len(), "invalid vector dimensions");
    let vector = crate::projection::project(&vector);
    let is_residual = meta_tuple.is_residual;
    let default_lut = if !is_residual {
        Some(rabitq::fscan_preprocess(&vector))
    } else {
        None
    };
    let h0_vector = 'h0_vector: {
        let tuple = rkyv::to_bytes::<_, 8192>(&VectorTuple {
            vector: vector.clone(),
            payload: Some(payload.as_u64()),
        })
        .unwrap();
        if let Some(mut write) = relation.search(tuple.len()) {
            let i = write.alloc(&tuple).unwrap();
            break 'h0_vector (write.id(), i);
        }
        let mut current = relation.read(meta_tuple.forwards_first).get_opaque().skip;
        let mut changed = false;
        loop {
            let read = relation.read(current);
            let flag = 'flag: {
                if read.freespace() as usize >= tuple.len() {
                    break 'flag true;
                }
                if read.get_opaque().next == u32::MAX {
                    break 'flag true;
                }
                false
            };
            if flag {
                drop(read);
                let mut write = relation.write(current, true);
                if let Some(i) = write.alloc(&tuple) {
                    break (current, i);
                }
                if write.get_opaque().next == u32::MAX {
                    if changed {
                        relation
                            .write(meta_tuple.forwards_first, false)
                            .get_opaque_mut()
                            .skip = write.id();
                    }
                    let mut extend = relation.extend(true);
                    write.get_opaque_mut().next = extend.id();
                    if let Some(i) = extend.alloc(&tuple) {
                        break (extend.id(), i);
                    } else {
                        panic!("a tuple cannot even be fit in a fresh page");
                    }
                }
                current = write.get_opaque().next;
            } else {
                current = read.get_opaque().next;
            }
            changed = true;
        }
    };
    let h0_payload = payload.as_u64();
    let mut list = (
        meta_tuple.first,
        if is_residual {
            let vector_guard = relation.read(meta_tuple.mean.0);
            let vector_tuple = vector_guard
                .get(meta_tuple.mean.1)
                .map(rkyv::check_archived_root::<VectorTuple>)
                .expect("data corruption")
                .expect("data corruption");
            Some(vector_tuple.vector.to_vec())
        } else {
            None
        },
    );
    let make_list = |list: (u32, Option<Vec<f32>>)| {
        let mut results = Vec::new();
        {
            let lut = if is_residual {
                &rabitq::fscan_preprocess(&f32::vector_sub(&vector, list.1.as_ref().unwrap()))
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
                            &h1_tuple.dis_u_2,
                            &h1_tuple.factor_ppc,
                            &h1_tuple.factor_ip,
                            &h1_tuple.factor_err,
                            &h1_tuple.t,
                        ),
                        1.9,
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
                current = h1_guard.get_opaque().next;
            }
        }
        let mut heap = BinaryHeap::from(results);
        let mut cache = BinaryHeap::<(Reverse<Distance>, _, _)>::new();
        {
            while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
                let (_, AlwaysEqual(mean), AlwaysEqual(first)) = heap.pop().unwrap();
                let vector_guard = relation.read(mean.0);
                let vector_tuple = vector_guard
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
            let (_, AlwaysEqual(first), AlwaysEqual(mean)) = cache.pop().unwrap();
            (first, mean)
        }
    };
    for _ in (1..meta_tuple.height_of_root).rev() {
        list = make_list(list);
    }
    let code = if is_residual {
        rabitq::code(dims, &f32::vector_sub(&vector, list.1.as_ref().unwrap()))
    } else {
        rabitq::code(dims, &vector)
    };
    let dummy = rkyv::to_bytes::<_, 8192>(&Height0Tuple {
        mask: [false; 32],
        mean: [(0, 0); 32],
        payload: [0; 32],
        dis_u_2: [0.0f32; 32],
        factor_ppc: [0.0f32; 32],
        factor_ip: [0.0f32; 32],
        factor_err: [0.0f32; 32],
        t: vec![0; (dims.div_ceil(4) * 16) as usize],
    })
    .unwrap();
    let first = list.0;
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = relation.read(current);
        let flag = 'flag: {
            for i in 1..=read.len() {
                let h0_tuple = read
                    .get(i)
                    .map(rkyv::check_archived_root::<Height0Tuple>)
                    .expect("data corruption")
                    .expect("data corruption");
                if h0_tuple.mask.iter().any(|x| *x) {
                    break 'flag true;
                }
            }
            if read.freespace() as usize >= dummy.len() {
                break 'flag true;
            }
            if read.get_opaque().next == u32::MAX {
                break 'flag true;
            }
            false
        };
        if flag {
            drop(read);
            let mut write = relation.write(current, false);
            for i in 1..=write.len() {
                let flag = put(
                    write.get_mut(i).expect("data corruption"),
                    dims,
                    &code,
                    h0_vector,
                    h0_payload,
                );
                if flag {
                    return;
                }
            }
            if let Some(i) = write.alloc(&dummy) {
                let flag = put(
                    write.get_mut(i).expect("data corruption"),
                    dims,
                    &code,
                    h0_vector,
                    h0_payload,
                );
                assert!(flag, "a put fails even on a fresh tuple");
                return;
            }
            if write.get_opaque().next == u32::MAX {
                let mut extend = relation.extend(false);
                write.get_opaque_mut().next = extend.id();
                if let Some(i) = extend.alloc(&dummy) {
                    let flag = put(
                        extend.get_mut(i).expect("data corruption"),
                        dims,
                        &code,
                        h0_vector,
                        h0_payload,
                    );
                    assert!(flag, "a put fails even on a fresh tuple");
                    return;
                } else {
                    panic!("a tuple cannot even be fit in a fresh page");
                }
            }
            current = write.get_opaque().next;
        } else {
            current = read.get_opaque().next;
        }
    }
}
