use super::RelationRead;
use crate::vchordrq::algorithm::tuples::*;
use crate::vchordrq::algorithm::vectors;
use std::fmt::Write;

pub fn prewarm<V: Vector>(relation: impl RelationRead + Clone, height: i32) -> String {
    let mut message = String::new();
    let meta_guard = relation.read(0);
    let meta_tuple = meta_guard
        .get(1)
        .map(rkyv::check_archived_root::<MetaTuple>)
        .expect("data corruption")
        .expect("data corruption");
    writeln!(message, "height of root: {}", meta_tuple.height_of_root).unwrap();
    let prewarm_max_height = if height < 0 { 0 } else { height as u32 };
    if prewarm_max_height > meta_tuple.height_of_root {
        return message;
    }
    let mut lists = {
        let mut results = Vec::new();
        let counter = 1_usize;
        {
            vectors::vector_warm::<V>(relation.clone(), meta_tuple.mean);
            results.push(meta_tuple.first);
        }
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
        results
    };
    let mut make_lists = |lists| {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for list in lists {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
                pgrx::check_for_interrupts!();
                let h1_guard = relation.read(current);
                for i in 1..=h1_guard.len() {
                    let h1_tuple = h1_guard
                        .get(i)
                        .map(rkyv::check_archived_root::<Height1Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    vectors::vector_warm::<V>(relation.clone(), h1_tuple.mean);
                    results.push(h1_tuple.first);
                }
                current = h1_guard.get_opaque().next;
            }
        }
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
        results
    };
    for _ in (std::cmp::max(1, prewarm_max_height)..meta_tuple.height_of_root).rev() {
        lists = make_lists(lists);
    }
    if prewarm_max_height == 0 {
        let mut counter = 0_usize;
        let mut results = Vec::new();
        for list in lists {
            let mut current = list;
            while current != u32::MAX {
                counter += 1;
                pgrx::check_for_interrupts!();
                let h0_guard = relation.read(current);
                for i in 1..=h0_guard.len() {
                    let _h0_tuple = h0_guard
                        .get(i)
                        .map(rkyv::check_archived_root::<Height0Tuple>)
                        .expect("data corruption")
                        .expect("data corruption");
                    results.push(());
                }
                current = h0_guard.get_opaque().next;
            }
        }
        writeln!(message, "number of tuples: {}", results.len()).unwrap();
        writeln!(message, "number of pages: {}", counter).unwrap();
    }
    message
}
