use crate::algorithm::rabitq;
use crate::algorithm::tuples::*;
use crate::postgres::Relation;
use base::distance::Distance;
use base::scalar::ScalarLike;
use base::search::Pointer;

pub fn insert(relation: Relation, payload: Pointer, extra: Option<u32>, vector: Vec<f32>) {
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
    let h0_vector = {
        let tuple = rkyv::to_bytes::<_, 8192>(&VectorTuple {
            vector: vector.clone(),
            payload: Some(payload.as_u64()),
        })
        .unwrap();
        let mut current = meta_tuple.vectors_first;
        loop {
            let read = relation.read(current);
            let flag = 'flag: {
                if read.get().freespace() as usize >= tuple.len() {
                    break 'flag true;
                }
                if read.get().get_opaque().next == u32::MAX {
                    break 'flag true;
                }
                false
            };
            if flag {
                drop(read);
                let mut write = relation.write(current);
                if let Some(i) = write.get_mut().alloc(&tuple) {
                    break (current, i);
                }
                if write.get().get_opaque().next == u32::MAX {
                    let mut extend = relation.extend();
                    write.get_mut().get_opaque_mut().next = extend.id();
                    if let Some(i) = extend.get_mut().alloc(&tuple) {
                        break (extend.id(), i);
                    } else {
                        panic!("a tuple cannot even be fit in a fresh page");
                    }
                }
                current = write.get().get_opaque().next;
            } else {
                current = read.get().get_opaque().next;
            }
        }
    };
    let h0_payload = payload.as_u64();
    let list = (meta_tuple.first,);
    let list = {
        let mut result = None::<(Distance, u32, Option<Vec<_>>)>;
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
                for j in 0..32 {
                    if h1_tuple.mask[j] {
                        let vector_guard = relation.read(h1_tuple.mean[j].0);
                        let vector_tuple = vector_guard
                            .get()
                            .get(h1_tuple.mean[j].1)
                            .map(rkyv::check_archived_root::<VectorTuple>)
                            .expect("data corruption")
                            .expect("data corruption");
                        let dis = Distance::from_f32(f32::reduce_sum_of_d2(
                            &vector,
                            &vector_tuple.vector,
                        ));
                        if result.is_none() || dis < result.as_ref().unwrap().0 {
                            result = Some((
                                dis,
                                h1_tuple.first[j],
                                if is_residual {
                                    Some(vector_tuple.vector.to_vec())
                                } else {
                                    None
                                },
                            ));
                        }
                    }
                }
            }
            current = h1_guard.get().get_opaque().next;
        }
        let result = result.unwrap();
        (result.1, result.2)
    };
    let code = if is_residual {
        rabitq::code(dims, &f32::vector_sub(&vector, list.1.as_ref().unwrap()))
    } else {
        rabitq::code(dims, &vector)
    };
    let dummy = rkyv::to_bytes::<_, 8192>(&Height0Tuple {
        mask: [false; 32],
        mean: [(0, 0); 32],
        payload: [0; 32],
        extra: [None; 32],
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
            for i in 1..=read.get().len() {
                let h0_tuple = read
                    .get()
                    .get(i)
                    .map(rkyv::check_archived_root::<Height0Tuple>)
                    .expect("data corruption")
                    .expect("data corruption");
                if h0_tuple.mask.iter().any(|x| *x) {
                    break 'flag true;
                }
            }
            if read.get().freespace() as usize >= dummy.len() {
                break 'flag true;
            }
            if read.get().get_opaque().next == u32::MAX {
                break 'flag true;
            }
            false
        };
        if flag {
            drop(read);
            let mut write = relation.write(current);
            for i in 1..=write.get().len() {
                let flag = put(
                    write.get_mut().get_mut(i).expect("data corruption"),
                    dims,
                    &code,
                    h0_vector,
                    h0_payload,
                    extra,
                );
                if flag {
                    return;
                }
            }
            if let Some(i) = write.get_mut().alloc(&dummy) {
                let flag = put(
                    write.get_mut().get_mut(i).expect("data corruption"),
                    dims,
                    &code,
                    h0_vector,
                    h0_payload,
                    extra,
                );
                assert!(flag, "a put fails even on a fresh tuple");
                return;
            }
            if write.get().get_opaque().next == u32::MAX {
                let mut extend = relation.extend();
                write.get_mut().get_opaque_mut().next = extend.id();
                if let Some(i) = extend.get_mut().alloc(&dummy) {
                    let flag = put(
                        extend.get_mut().get_mut(i).expect("data corruption"),
                        dims,
                        &code,
                        h0_vector,
                        h0_payload,
                        extra,
                    );
                    assert!(flag, "a put fails even on a fresh tuple");
                    return;
                } else {
                    panic!("a tuple cannot even be fit in a fresh page");
                }
            }
            current = write.get().get_opaque().next;
        } else {
            current = read.get().get_opaque().next;
        }
    }
}
