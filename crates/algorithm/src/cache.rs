use crate::pipe::Pipe;
use crate::tuples::{H1Tuple, H1TupleReader, MetaTuple, read_tuple};
use crate::{Page, RelationRead};

pub fn cache(index: impl RelationRead) -> Vec<u32> {
    let mut trace = Vec::<u32>::new();
    let mut read = |id| {
        let result = index.read(id);
        trace.push(id);
        result
    };
    let meta_guard = read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    drop(meta_guard);
    type State = Vec<u32>;
    let mut state: State = vec![root_first];
    let mut step = |state: State| {
        let mut results = Vec::new();
        for first in state {
            let mut current = first;
            while current != u32::MAX {
                let h1_guard = read(current);
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
        let _ = read(first);
    }
    trace
}
