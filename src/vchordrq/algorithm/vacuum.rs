use super::RelationWrite;
use crate::vchordrq::algorithm::tuples::*;
use base::search::Pointer;

pub fn vacuum<V: Vector>(
    relation: impl RelationWrite,
    delay: impl Fn(),
    callback: impl Fn(Pointer) -> bool,
) {
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
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
                for i in 1..=h1_guard.len() {
                    let h1_tuple = h1_guard
                        .get(i)
                        .map(rkyv::check_archived_root::<Height1Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    results.push(h1_tuple.first);
                }
                current = h1_guard.get_opaque().next;
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
            for i in 1..=h0_guard.len() {
                let h0_tuple = h0_guard
                    .get(i)
                    .map(rkyv::check_archived_root::<Height0Tuple>)
                    .expect("data corruption")
                    .expect("data corruption");
                if callback(Pointer::new(h0_tuple.payload)) {
                    reconstruct_removes.push(i);
                }
            }
            h0_guard.reconstruct(&reconstruct_removes);
            current = h0_guard.get_opaque().next;
        }
    }
}
