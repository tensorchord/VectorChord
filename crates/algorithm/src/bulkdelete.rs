use crate::operator::{FunctionalAccessor, Operator};
use crate::tuples::*;
use crate::{Page, RelationWrite, tape};
use std::num::NonZero;

pub fn bulkdelete<O: Operator>(
    index: impl RelationWrite,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
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
                tape::read_h1_tape(
                    index.clone(),
                    first,
                    || {
                        fn push<T>(_: &mut (), _: &[T]) {}
                        fn finish<T>(_: (), _: (&T, &T, &T, &T)) -> [(); 32] {
                            [(); 32]
                        }
                        FunctionalAccessor::new((), push, finish)
                    },
                    |(), _, first| results.push(first),
                    |_| check(),
                );
            }
            results
        };
        for _ in (1..height_of_root).rev() {
            state = step(state);
        }
        for first in state {
            let jump_guard = index.read(first);
            let jump_bytes = jump_guard.get(1).expect("data corruption");
            let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
            {
                let mut current = jump_tuple.frozen_first();
                while current != u32::MAX {
                    check();
                    let read = index.read(current);
                    let flag = 'flag: {
                        for i in 1..=read.len() {
                            let bytes = read.get(i).expect("data corruption");
                            let tuple = FrozenTuple::deserialize_ref(bytes);
                            if let FrozenTupleReader::_0(tuple) = tuple {
                                for p in tuple.payload().iter() {
                                    if Some(true) == p.map(&callback) {
                                        break 'flag true;
                                    }
                                }
                            }
                        }
                        false
                    };
                    if flag {
                        drop(read);
                        let mut write = index.write(current, false);
                        for i in 1..=write.len() {
                            let bytes = write.get_mut(i).expect("data corruption");
                            let tuple = FrozenTuple::deserialize_mut(bytes);
                            if let FrozenTupleWriter::_0(mut tuple) = tuple {
                                for p in tuple.payload().iter_mut() {
                                    if Some(true) == p.map(&callback) {
                                        *p = None;
                                    }
                                }
                            }
                        }
                        current = write.get_opaque().next;
                    } else {
                        current = read.get_opaque().next;
                    }
                }
            }
            {
                let mut current = jump_tuple.appendable_first();
                while current != u32::MAX {
                    check();
                    let read = index.read(current);
                    let flag = 'flag: {
                        for i in 1..=read.len() {
                            let bytes = read.get(i).expect("data corruption");
                            let tuple = AppendableTuple::deserialize_ref(bytes);
                            let p = tuple.payload();
                            if Some(true) == p.map(&callback) {
                                break 'flag true;
                            }
                        }
                        false
                    };
                    if flag {
                        drop(read);
                        let mut write = index.write(current, false);
                        for i in 1..=write.len() {
                            let bytes = write.get_mut(i).expect("data corruption");
                            let mut tuple = AppendableTuple::deserialize_mut(bytes);
                            let p = tuple.payload();
                            if Some(true) == p.map(&callback) {
                                *p = None;
                            }
                        }
                        current = write.get_opaque().next;
                    } else {
                        current = read.get_opaque().next;
                    }
                }
            }
        }
    }
    {
        let mut current = vectors_first;
        while current != u32::MAX {
            check();
            let read = index.read(current);
            let flag = 'flag: {
                for i in 1..=read.len() {
                    if let Some(bytes) = read.get(i) {
                        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                        let p = tuple.payload();
                        if Some(true) == p.map(&callback) {
                            break 'flag true;
                        }
                    }
                }
                false
            };
            if flag {
                drop(read);
                let mut write = index.write(current, true);
                for i in 1..=write.len() {
                    if let Some(bytes) = write.get(i) {
                        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
                        let p = tuple.payload();
                        if Some(true) == p.map(&callback) {
                            write.free(i);
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
