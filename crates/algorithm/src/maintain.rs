use crate::operator::Operator;
use crate::pipe::Pipe;
use crate::tape::*;
use crate::tuples::*;
use crate::{Page, RelationWrite, freepages};
use simd::fast_scan::unpack;

pub fn maintain<O: Operator>(index: impl RelationWrite, check: impl Fn()) {
    let meta_guard = index.read(0);
    let meta_tuple = meta_guard.get(1).unwrap().pipe(read_tuple::<MetaTuple>);
    let dims = meta_tuple.dims();
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    let freepage_first = meta_tuple.freepage_first();
    drop(meta_guard);

    let firsts = {
        type State = Vec<u32>;
        let mut state: State = vec![root_first];
        let step = |state: State| {
            let mut results = Vec::new();
            for first in state {
                let mut current = first;
                while current != u32::MAX {
                    check();
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
        state
    };

    for first in firsts {
        let mut jump_guard = index.write(first, false);
        let mut jump_tuple = jump_guard
            .get_mut(1)
            .expect("data corruption")
            .pipe(write_tuple::<JumpTuple>);

        let mut tape = H0TapeWriter::<_, _>::create(|| {
            if let Some(id) = freepages::fetch(index.clone(), freepage_first) {
                let mut write = index.write(id, false);
                write.clear();
                write
            } else {
                index.extend(false)
            }
        });

        let mut trace = Vec::new();

        let first = *jump_tuple.first();
        let mut current = first;
        let mut computing = None;
        while current != u32::MAX {
            check();
            trace.push(current);
            let h0_guard = index.read(current);
            for i in 1..=h0_guard.len() {
                let h0_tuple = h0_guard
                    .get(i)
                    .expect("data corruption")
                    .pipe(read_tuple::<H0Tuple>);
                match h0_tuple {
                    H0TupleReader::_0(h0_tuple) => {
                        if let Some(payload) = h0_tuple.payload() {
                            tape.push(H0Branch {
                                mean: h0_tuple.mean(),
                                dis_u_2: h0_tuple.code().0,
                                factor_ppc: h0_tuple.code().1,
                                factor_ip: h0_tuple.code().2,
                                factor_err: h0_tuple.code().3,
                                signs: h0_tuple
                                    .code()
                                    .4
                                    .iter()
                                    .flat_map(|x| {
                                        std::array::from_fn::<_, 64, _>(|i| *x & (1 << i) != 0)
                                    })
                                    .take(dims as _)
                                    .collect::<Vec<_>>(),
                                payload,
                            });
                        }
                    }
                    H0TupleReader::_1(h0_tuple) => {
                        let computing = &mut computing.take().unwrap_or_else(Vec::new);
                        computing.extend_from_slice(h0_tuple.elements());
                        let unpacked = unpack(computing);
                        for j in 0..32 {
                            if let Some(payload) = h0_tuple.payload()[j] {
                                tape.push(H0Branch {
                                    mean: h0_tuple.mean()[j],
                                    dis_u_2: h0_tuple.metadata().0[j],
                                    factor_ppc: h0_tuple.metadata().1[j],
                                    factor_ip: h0_tuple.metadata().2[j],
                                    factor_err: h0_tuple.metadata().3[j],
                                    signs: unpacked[j]
                                        .iter()
                                        .flat_map(|&x| {
                                            [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0]
                                        })
                                        .collect(),
                                    payload,
                                });
                            }
                        }
                    }
                    H0TupleReader::_2(h0_tuple) => {
                        let computing = computing.get_or_insert_with(Vec::new);
                        computing.extend_from_slice(h0_tuple.elements());
                    }
                }
            }
            current = h0_guard.get_opaque().next;
            drop(h0_guard);
        }

        let tape = tape.into_inner();
        let new = tape.first();
        drop(tape);

        *jump_tuple.first() = new;
        drop(jump_guard);

        freepages::mark(index.clone(), freepage_first, &trace);
    }
}
