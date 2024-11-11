use crate::algorithm::tuples::*;
use crate::postgres::Relation;
use std::fmt::Write;

pub fn prewarm(relation: Relation) -> String {
    let mut message = String::new();
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get()
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    let lists = vec![meta_tuple.first];
    writeln!(message, "number of h3 tuples: {}", lists.len()).unwrap();
    writeln!(message, "number of h3 pages: {}", 1).unwrap();
    let mut counter = 0_usize;
    let lists: Vec<_> = {
        let mut results = Vec::new();
        for list in lists {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
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
                            let mean = h2_tuple.mean[j];
                            let vector_guard = relation.read(mean.0);
                            let vector_tuple = vector_guard
                                .get()
                                .get(mean.1)
                                .map(rkyv::check_archived_root::<VectorTuple>)
                                .expect("data corruption")
                                .expect("data corruption");
                            let _ = vector_tuple;
                        }
                    }
                }
                current = h2_guard.get().get_opaque().next;
            }
        }
        results
    };
    writeln!(message, "number of h2 tuples: {}", lists.len()).unwrap();
    writeln!(message, "number of h2 pages: {}", counter).unwrap();
    let mut counter = 0_usize;
    let lists: Vec<_> = {
        let mut results = Vec::new();
        for list in lists {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
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
                            let mean = h1_tuple.mean[j];
                            let vector_guard = relation.read(mean.0);
                            let vector_tuple = vector_guard
                                .get()
                                .get(mean.1)
                                .map(rkyv::check_archived_root::<VectorTuple>)
                                .expect("data corruption")
                                .expect("data corruption");
                            let _ = vector_tuple;
                        }
                    }
                }
                current = h1_guard.get().get_opaque().next;
            }
        }
        results
    };
    writeln!(message, "number of h1 tuples: {}", lists.len()).unwrap();
    writeln!(message, "number of h1 pages: {}", counter).unwrap();
    let mut counter = 0_usize;
    let lists: Vec<_> = {
        let mut results = Vec::new();
        for list in lists {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
                let h0_guard = relation.read(current);
                for i in 1..=h0_guard.get().len() {
                    let h0_tuple = h0_guard
                        .get()
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    for j in 0..32 {
                        if h0_tuple.mask[j] {
                            results.push(());
                        }
                    }
                }
                current = h0_guard.get().get_opaque().next;
            }
        }
        results
    };
    writeln!(message, "number of h0 tuples: {}", lists.len()).unwrap();
    writeln!(message, "number of h0 pages: {}", counter).unwrap();
    message
}
