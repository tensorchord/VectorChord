use crate::algorithm::tuples::*;
use crate::postgres::Relation;
use base::search::Pointer;

pub fn vacuum(relation: Relation, delay: impl Fn(), callback: impl Fn(Pointer) -> bool) {
    // step 1: vacuum height_0_tuple
    {
        let h2_firsts = {
            let meta_guard = relation.read(0);
            let meta_tuple = meta_guard
                .get()
                .get(1)
                .map(rkyv::check_archived_root::<MetaTuple>)
                .expect("data corruption")
                .expect("data corruption");
            vec![meta_tuple.first]
        };
        let h1_firsts = {
            let mut results = Vec::new();
            for first in h2_firsts {
                let mut current = first;
                while current != u32::MAX {
                    let h2_guard = relation.read(current);
                    for i in 1..=h2_guard.get().len() {
                        let h2_tuple = h2_guard
                            .get()
                            .get(i)
                            .map(rkyv::check_archived_root::<Height2Tuple>)
                            .expect("data corruption")
                            .expect("data corruption");
                        for j in 0..32 {
                            if h2_tuple.mask[j] {
                                results.push(h2_tuple.first[j]);
                            }
                        }
                    }
                    current = h2_guard.get().get_opaque().next;
                }
            }
            results
        };
        let h0_firsts = {
            let mut results = Vec::new();
            for first in h1_firsts {
                let mut current = first;
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
                                results.push(h1_tuple.first[j]);
                            }
                        }
                    }
                    current = h1_guard.get().get_opaque().next;
                }
            }
            results
        };
        for first in h0_firsts {
            let mut current = first;
            while current != u32::MAX {
                delay();
                let mut h0_guard = relation.write(current);
                for i in 1..=h0_guard.get().len() {
                    let h0_tuple = h0_guard
                        .get()
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    let flag = 'flag: {
                        for j in 0..32 {
                            if h0_tuple.mask[j] && callback(Pointer::new(h0_tuple.payload[j])) {
                                break 'flag true;
                            }
                        }
                        false
                    };
                    if flag {
                        // todo: use mutable API
                        let mut temp = h0_guard
                            .get()
                            .get(i)
                            .map(rkyv::from_bytes::<Height0Tuple>)
                            .expect("data corruption")
                            .expect("data corruption");
                        for j in 0..32 {
                            if temp.mask[j] && callback(Pointer::new(temp.payload[j])) {
                                temp.mask[j] = false;
                            }
                        }
                        let temp = rkyv::to_bytes::<_, 8192>(&temp).expect("failed to serialize");
                        h0_guard
                            .get_mut()
                            .get_mut(i)
                            .expect("data corruption")
                            .copy_from_slice(&temp);
                    }
                }
                // todo: cross-tuple vacuum so that we can skip a tuple
                current = h0_guard.get().get_opaque().next;
            }
        }
    }
    // step 2: vacuum vector_tuple
    {
        let mut current = {
            let meta_guard = relation.read(0);
            let meta_tuple = meta_guard
                .get()
                .get(1)
                .map(rkyv::check_archived_root::<MetaTuple>)
                .expect("data corruption")
                .expect("data corruption");
            meta_tuple.vectors_first
        };
        while current != u32::MAX {
            delay();
            let read = relation.read(current);
            let flag = 'flag: {
                for i in 1..=read.get().len() {
                    let Some(vector_tuple) = read.get().get(i) else {
                        continue;
                    };
                    let vector_tuple = rkyv::check_archived_root::<VectorTuple>(vector_tuple)
                        .expect("data corruption");
                    if let Some(payload) = vector_tuple.payload.as_ref().copied() {
                        if callback(Pointer::new(payload)) {
                            break 'flag true;
                        }
                    }
                }
                false
            };
            if flag {
                drop(read);
                let mut write = relation.write(current);
                for i in 1..=write.get().len() {
                    let Some(vector_tuple) = write.get().get(i) else {
                        continue;
                    };
                    let vector_tuple = rkyv::check_archived_root::<VectorTuple>(vector_tuple)
                        .expect("data corruption");
                    if let Some(payload) = vector_tuple.payload.as_ref().copied() {
                        if callback(Pointer::new(payload)) {
                            write.get_mut().free(i);
                        }
                    }
                }
                current = write.get().get_opaque().next;
            } else {
                current = read.get().get_opaque().next;
            }
        }
    }
}
