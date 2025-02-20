use crate::operator::Operator;
use crate::pipe::Pipe;
use crate::tuples::*;
use crate::{Page, RelationWrite};
use std::num::NonZeroU64;

pub fn bulkdelete<O: Operator>(
    index: impl RelationWrite,
    check: impl Fn(),
    callback: impl Fn(NonZeroU64) -> bool,
) {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    let vectors_first = meta_tuple.vectors_first();
    drop(meta_guard);
    {
        type State = Vec<u32>;
        let mut state: State = vec![root_first];
        let step = |state: State| {
            let mut results = Vec::new();
            for first in state {
                let mut current = first;
                while current != u32::MAX {
                    let h1_guard = index.read(current);
                    for i in 1..=h1_guard.len() {
                        let h1_tuple = h1_guard
                            .get(i)
                            .expect("data corruption")
                            .pipe(read_tuple::<H1Tuple>);
                        match h1_tuple {
                            H1TupleReader::_0(h1_tuple) => {
                                for first in h1_tuple.first().iter().copied() {
                                    results.push(first);
                                }
                            }
                            H1TupleReader::_1(_) => (),
                        }
                    }
                    current = h1_guard.get_opaque().next;
                }
            }
            results
        };
        for _ in (1..height_of_root).rev() {
            state = step(state);
        }
        for first in state {
            let jump_guard = index.read(first);
            let jump_tuple = jump_guard
                .get(1)
                .expect("data corruption")
                .pipe(read_tuple::<JumpTuple>);
            let first = jump_tuple.first();
            let mut current = first;
            while current != u32::MAX {
                check();
                let read = index.read(current);
                let flag = 'flag: {
                    for i in 1..=read.len() {
                        let h0_tuple = read
                            .get(i)
                            .expect("data corruption")
                            .pipe(read_tuple::<H0Tuple>);
                        match h0_tuple {
                            H0TupleReader::_0(h0_tuple) => {
                                let p = h0_tuple.payload();
                                if let Some(payload) = p {
                                    if callback(payload) {
                                        break 'flag true;
                                    }
                                }
                            }
                            H0TupleReader::_1(h0_tuple) => {
                                let p = h0_tuple.payload();
                                for j in 0..32 {
                                    if let Some(payload) = p[j] {
                                        if callback(payload) {
                                            break 'flag true;
                                        }
                                    }
                                }
                            }
                            H0TupleReader::_2(_) => (),
                        }
                    }
                    false
                };
                if flag {
                    drop(read);
                    let mut write = index.write(current, false);
                    for i in 1..=write.len() {
                        let h0_tuple = write
                            .get_mut(i)
                            .expect("data corruption")
                            .pipe(write_tuple::<H0Tuple>);
                        match h0_tuple {
                            H0TupleWriter::_0(mut h0_tuple) => {
                                let p = h0_tuple.payload();
                                if let Some(payload) = *p {
                                    if callback(payload) {
                                        *p = None;
                                    }
                                }
                            }
                            H0TupleWriter::_1(mut h0_tuple) => {
                                let p = h0_tuple.payload();
                                for j in 0..32 {
                                    if let Some(payload) = p[j] {
                                        if callback(payload) {
                                            p[j] = None;
                                        }
                                    }
                                }
                            }
                            H0TupleWriter::_2(_) => (),
                        }
                    }
                    current = write.get_opaque().next;
                } else {
                    current = read.get_opaque().next;
                }
            }
        }
    }
    {
        let first = vectors_first;
        let mut current = first;
        while current != u32::MAX {
            check();
            let read = index.read(current);
            let flag = 'flag: {
                for i in 1..=read.len() {
                    if let Some(vector_bytes) = read.get(i) {
                        let vector_tuple = vector_bytes.pipe(read_tuple::<VectorTuple<O::Vector>>);
                        let p = vector_tuple.payload();
                        if let Some(payload) = p {
                            if callback(payload) {
                                break 'flag true;
                            }
                        }
                    }
                }
                false
            };
            if flag {
                drop(read);
                let mut write = index.write(current, true);
                for i in 1..=write.len() {
                    if let Some(vector_bytes) = write.get(i) {
                        let vector_tuple = vector_bytes.pipe(read_tuple::<VectorTuple<O::Vector>>);
                        let p = vector_tuple.payload();
                        if let Some(payload) = p {
                            if callback(payload) {
                                write.free(i);
                            }
                        }
                    };
                }
                current = write.get_opaque().next;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}
