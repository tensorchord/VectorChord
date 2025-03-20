use crate::operator::{FunctionalAccessor, Operator};
use crate::tape::{self, TapeWriter};
use crate::tuples::*;
use crate::{Branch, DerefMut, Page, PageGuard, RelationWrite, freepages};
use simd::fast_scan::{padding_pack, unpack};
use std::num::NonZeroU64;

pub fn maintain<O: Operator>(index: impl RelationWrite, check: impl Fn()) {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let height_of_root = meta_tuple.height_of_root();
    let root_first = meta_tuple.root_first();
    let freepage_first = meta_tuple.freepage_first();
    drop(meta_guard);

    let state = {
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
        state
    };

    for first in state {
        let mut jump_guard = index.write(first, false);
        let jump_bytes = jump_guard.get_mut(1).expect("data corruption");
        let mut jump_tuple = JumpTuple::deserialize_mut(jump_bytes);

        let expand = || {
            if let Some(id) = freepages::fetch(index.clone(), freepage_first) {
                let mut write = index.write(id, false);
                write.clear();
                write
            } else {
                index.extend(false)
            }
        };

        let mut tape = FrozenTapeWriter::<_, _>::create(expand);

        let mut trace = Vec::new();

        let mut tuples = 0_u64;
        let mut callback = |code: (_, _, _, _, _), mean, payload| {
            tape.push(Branch {
                mean,
                dis_u_2: code.0,
                factor_ppc: code.1,
                factor_ip: code.2,
                factor_err: code.3,
                signs: code.4,
                extra: payload,
            });
            tuples += 1;
        };
        let mut step = |id| {
            check();
            trace.push(id);
        };
        tape::read_frozen_tape(
            index.clone(),
            *jump_tuple.frozen_first(),
            || {
                FunctionalAccessor::new(
                    Vec::<[u8; 16]>::new(),
                    Vec::<[u8; 16]>::extend_from_slice,
                    |elements: Vec<_>, input: (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32])| {
                        let unpacked = unpack(&elements);
                        std::array::from_fn(|i| {
                            let f = |&x| [x & 1 != 0, x & 2 != 0, x & 4 != 0, x & 8 != 0];
                            let signs = unpacked[i].iter().flat_map(f).collect::<Vec<_>>();
                            (input.0[i], input.1[i], input.2[i], input.3[i], signs)
                        })
                    },
                )
            },
            &mut callback,
            &mut step,
        );
        tape::read_appendable_tape(
            index.clone(),
            *jump_tuple.appendable_first(),
            |code| {
                let signs = code
                    .4
                    .iter()
                    .flat_map(|x| std::array::from_fn::<_, 64, _>(|i| *x & (1 << i) != 0))
                    .take(dims as _)
                    .collect::<Vec<_>>();
                (code.0, code.1, code.2, code.3, signs)
            },
            &mut callback,
            &mut step,
        );

        let (frozen_tape, branches) = tape.into_inner();

        let mut appendable_tape = TapeWriter::create(expand);

        for branch in branches {
            appendable_tape.push(AppendableTuple {
                mean: branch.mean,
                dis_u_2: branch.dis_u_2,
                factor_ppc: branch.factor_ppc,
                factor_ip: branch.factor_ip,
                factor_err: branch.factor_err,
                payload: Some(branch.extra),
                elements: rabitq::pack_to_u64(&branch.signs),
            });
        }

        *jump_tuple.frozen_first() = { frozen_tape }.first();
        *jump_tuple.appendable_first() = { appendable_tape }.first();
        *jump_tuple.tuples() = tuples;

        drop(jump_guard);

        freepages::mark(index.clone(), freepage_first, &trace);
    }
}

struct FrozenTapeWriter<G, E> {
    tape: TapeWriter<G, E, FrozenTuple>,
    branches: Vec<Branch<NonZeroU64>>,
}

impl<G, E> FrozenTapeWriter<G, E>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
{
    fn create(extend: E) -> Self {
        Self {
            tape: TapeWriter::create(extend),
            branches: Vec::new(),
        }
    }
    fn push(&mut self, branch: Branch<NonZeroU64>) {
        self.branches.push(branch);
        if self.branches.len() == 32 {
            let chunk = std::array::from_fn::<_, 32, _>(|_| self.branches.pop().unwrap());
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                if FrozenTuple::estimate_size_0(remain.len()) <= freespace as usize {
                    self.tape.tape_put(FrozenTuple::_0 {
                        mean: chunk.each_ref().map(|x| x.mean),
                        dis_u_2: chunk.each_ref().map(|x| x.dis_u_2),
                        factor_ppc: chunk.each_ref().map(|x| x.factor_ppc),
                        factor_ip: chunk.each_ref().map(|x| x.factor_ip),
                        factor_err: chunk.each_ref().map(|x| x.factor_err),
                        payload: chunk.each_ref().map(|x| Some(x.extra)),
                        elements: remain,
                    });
                    break;
                }
                if let Some(w) = FrozenTuple::fit_1(freespace) {
                    let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                    self.tape.tape_put(FrozenTuple::_1 {
                        elements: left.to_vec(),
                    });
                    remain = right.to_vec();
                } else {
                    self.tape.tape_move();
                }
            }
        }
    }
    fn into_inner(self) -> (TapeWriter<G, E, FrozenTuple>, Vec<Branch<NonZeroU64>>) {
        (self.tape, self.branches)
    }
}
