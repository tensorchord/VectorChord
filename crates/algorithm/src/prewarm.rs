use crate::operator::Operator;
use crate::pipe::Pipe;
use crate::tuples::*;
use crate::{Page, RelationRead, vectors};
use std::fmt::Write;

pub fn prewarm<O: Operator>(index: impl RelationRead, height: i32, check: impl Fn()) -> String {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let height_of_root = meta_tuple.height_of_root();
    let root_mean = meta_tuple.root_mean();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);

    let mut message = String::new();
    writeln!(message, "height of root: {}", height_of_root).unwrap();
    let prewarm_max_height = if height < 0 { 0 } else { height as u32 };
    if prewarm_max_height > height_of_root {
        return message;
    }
    type State = Vec<u32>;
    let mut state: State = {
        let mut nodes = Vec::new();
        {
            vectors::access_1::<O, _>(index.clone(), root_mean, ());
            nodes.push(root_first);
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", nodes.len()).unwrap();
        writeln!(message, "number of tuples: {}", 1).unwrap();
        writeln!(message, "number of pages: {}", 1).unwrap();
        nodes
    };
    let mut step = |state: State| {
        let mut counter_pages = 0_usize;
        let mut counter_tuples = 0_usize;
        let mut nodes = Vec::new();
        for list in state {
            let mut current = list;
            while current != u32::MAX {
                counter_pages += 1;
                check();
                let h1_guard = index.read(current);
                for i in 1..=h1_guard.len() {
                    counter_tuples += 1;
                    let h1_tuple = h1_guard
                        .get(i)
                        .expect("data corruption")
                        .pipe(read_tuple::<H1Tuple>);
                    match h1_tuple {
                        H1TupleReader::_0(h1_tuple) => {
                            for mean in h1_tuple.mean().iter().copied() {
                                vectors::access_1::<O, _>(index.clone(), mean, ());
                            }
                            for first in h1_tuple.first().iter().copied() {
                                nodes.push(first);
                            }
                        }
                        H1TupleReader::_1(_) => (),
                    }
                }
                current = h1_guard.get_opaque().next;
            }
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", nodes.len()).unwrap();
        writeln!(message, "number of tuples: {}", counter_tuples).unwrap();
        writeln!(message, "number of pages: {}", counter_pages).unwrap();
        nodes
    };
    for _ in (std::cmp::max(1, prewarm_max_height)..height_of_root).rev() {
        state = step(state);
    }
    if prewarm_max_height == 0 {
        let mut counter_pages = 0_usize;
        let mut counter_tuples = 0_usize;
        let mut counter_nodes = 0_usize;
        for list in state {
            let jump_guard = index.read(list);
            let jump_tuple = jump_guard
                .get(1)
                .expect("data corruption")
                .pipe(read_tuple::<JumpTuple>);
            let first = jump_tuple.first();
            let mut current = first;
            while current != u32::MAX {
                counter_pages += 1;
                check();
                let h0_guard = index.read(current);
                for i in 1..=h0_guard.len() {
                    counter_tuples += 1;
                    let h0_tuple = h0_guard
                        .get(i)
                        .expect("data corruption")
                        .pipe(read_tuple::<H0Tuple>);
                    match h0_tuple {
                        H0TupleReader::_0(_h0_tuple) => {
                            counter_nodes += 1;
                        }
                        H0TupleReader::_1(_h0_tuple) => {
                            counter_nodes += 32;
                        }
                        H0TupleReader::_2(_) => (),
                    }
                }
                current = h0_guard.get_opaque().next;
            }
        }
        writeln!(message, "------------------------").unwrap();
        writeln!(message, "number of nodes: {}", counter_nodes).unwrap();
        writeln!(message, "number of tuples: {}", counter_tuples).unwrap();
        writeln!(message, "number of pages: {}", counter_pages).unwrap();
    }
    message
}
