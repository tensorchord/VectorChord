use crate::postgres::Relation;
use crate::vchordrq::algorithm::tuples::*;
use base::search::Pointer;

pub fn vacuum<V: Vector>(relation: Relation, delay: impl Fn(), callback: impl Fn(Pointer) -> bool) {
    // step 1: vacuum height_0_tuple
    {
        let meta_guard = relation.read(0);
        let meta_tuple = meta_guard
            .get()
            .get(1)
            .map(rkyv::check_archived_root::<MetaTuple>)
            .expect("data corruption")
            .expect("data corruption");
        let mut firsts = vec![meta_tuple.first];
        let make_firsts = |firsts| {
            let mut results = Vec::new();
            for first in firsts {
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
                        results.push(h1_tuple.first);
                    }
                    current = h1_guard.get().get_opaque().next;
                }
            }
            results
        };
        for _ in (1..meta_tuple.height_of_root).rev() {
            firsts = make_firsts(firsts);
        }
        for first in firsts {
            let mut current = first;
            while current != u32::MAX {
                delay();
                let mut h0_guard = relation.write(current, false);
                let mut reconstruct_removes = Vec::new();
                for i in 1..=h0_guard.get().len() {
                    let h0_tuple = h0_guard
                        .get()
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    if callback(Pointer::new(h0_tuple.payload)) {
                        reconstruct_removes.push(i);
                    }
                }
                h0_guard.get_mut().reconstruct(&reconstruct_removes);
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
                    let vector_tuple =
                        unsafe { rkyv::archived_root::<VectorTuple<V>>(vector_tuple) };
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
                let mut write = relation.write(current, true);
                for i in 1..=write.get().len() {
                    let Some(vector_tuple) = write.get().get(i) else {
                        continue;
                    };
                    let vector_tuple =
                        unsafe { rkyv::archived_root::<VectorTuple<V>>(vector_tuple) };
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
