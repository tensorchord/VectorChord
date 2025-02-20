use crate::operator::Accessor1;
use crate::pipe::Pipe;
use crate::tuples::*;
use crate::{Page, PageGuard, RelationRead, RelationWrite};
use distance::Distance;
use simd::fast_scan::{any_pack, padding_pack};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::ops::DerefMut;

pub struct TapeWriter<G, E, T> {
    head: G,
    first: u32,
    extend: E,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<G, E, T> TapeWriter<G, E, T>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
{
    pub fn create(extend: E) -> Self {
        let mut head = extend();
        head.get_opaque_mut().skip = head.id();
        let first = head.id();
        Self {
            head,
            first,
            extend,
            _phantom: PhantomData,
        }
    }
    pub fn first(&self) -> u32 {
        self.first
    }
    fn freespace(&self) -> u16 {
        self.head.freespace()
    }
    fn tape_move(&mut self) {
        if self.head.len() == 0 {
            panic!("tuple is too large to fit in a fresh page");
        }
        let next = (self.extend)();
        self.head.get_opaque_mut().next = next.id();
        self.head = next;
    }
}

impl<G, E, T> TapeWriter<G, E, T>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
    T: Tuple,
{
    pub fn push(&mut self, x: T) -> IndexPointer {
        let bytes = serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            pair_to_pointer((self.head.id(), i))
        } else {
            let next = (self.extend)();
            self.head.get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.alloc(&bytes) {
                pair_to_pointer((self.head.id(), i))
            } else {
                panic!("tuple is too large to fit in a fresh page")
            }
        }
    }
    fn tape_put(&mut self, x: T) -> IndexPointer {
        let bytes = serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            pair_to_pointer((self.head.id(), i))
        } else {
            panic!("tuple is too large to fit in this page")
        }
    }
}

pub struct H1Branch {
    pub mean: IndexPointer,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
    pub first: u32,
}

pub struct H1TapeWriter<G, E> {
    tape: TapeWriter<G, E, H1Tuple>,
    branches: Vec<H1Branch>,
}

impl<G, E> H1TapeWriter<G, E>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
{
    pub fn create(extend: E) -> Self {
        Self {
            tape: TapeWriter::create(extend),
            branches: Vec::new(),
        }
    }
    pub fn push(&mut self, branch: H1Branch) {
        self.branches.push(branch);
        if self.branches.len() == 32 {
            let chunk = std::array::from_fn::<_, 32, _>(|_| self.branches.pop().unwrap());
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                match (H1Tuple::fit_0(freespace), H1Tuple::fit_1(freespace)) {
                    (Some(w), _) if w >= remain.len() => {
                        self.tape.tape_put(H1Tuple::_0 {
                            mean: chunk.each_ref().map(|x| x.mean),
                            dis_u_2: chunk.each_ref().map(|x| x.dis_u_2),
                            factor_ppc: chunk.each_ref().map(|x| x.factor_ppc),
                            factor_ip: chunk.each_ref().map(|x| x.factor_ip),
                            factor_err: chunk.each_ref().map(|x| x.factor_err),
                            first: chunk.each_ref().map(|x| x.first),
                            len: chunk.len() as _,
                            elements: remain,
                        });
                        break;
                    }
                    (_, Some(w)) => {
                        let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                        self.tape.tape_put(H1Tuple::_1 {
                            elements: left.to_vec(),
                        });
                        remain = right.to_vec();
                    }
                    (_, None) => self.tape.tape_move(),
                }
            }
        }
    }
    pub fn into_inner(mut self) -> TapeWriter<G, E, H1Tuple> {
        let chunk = self.branches;
        if !chunk.is_empty() {
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                match (H1Tuple::fit_0(freespace), H1Tuple::fit_1(freespace)) {
                    (Some(w), _) if w >= remain.len() => {
                        self.tape.push(H1Tuple::_0 {
                            mean: any_pack(chunk.iter().map(|x| x.mean)),
                            dis_u_2: any_pack(chunk.iter().map(|x| x.dis_u_2)),
                            factor_ppc: any_pack(chunk.iter().map(|x| x.factor_ppc)),
                            factor_ip: any_pack(chunk.iter().map(|x| x.factor_ip)),
                            factor_err: any_pack(chunk.iter().map(|x| x.factor_err)),
                            first: any_pack(chunk.iter().map(|x| x.first)),
                            len: chunk.len() as _,
                            elements: remain,
                        });
                        break;
                    }
                    (_, Some(w)) => {
                        let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                        self.tape.tape_put(H1Tuple::_1 {
                            elements: left.to_vec(),
                        });
                        remain = right.to_vec();
                    }
                    (_, None) => self.tape.tape_move(),
                }
            }
        }
        self.tape
    }
}

pub struct H0Branch {
    pub mean: IndexPointer,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<bool>,
    pub payload: NonZeroU64,
}

pub struct H0TapeWriter<G, E> {
    tape: TapeWriter<G, E, H0Tuple>,
    branches: Vec<H0Branch>,
}

impl<G, E> H0TapeWriter<G, E>
where
    G: PageGuard + DerefMut,
    G::Target: Page,
    E: Fn() -> G,
{
    pub fn create(extend: E) -> Self {
        Self {
            tape: TapeWriter::create(extend),
            branches: Vec::new(),
        }
    }
    pub fn push(&mut self, branch: H0Branch) {
        self.branches.push(branch);
        if self.branches.len() == 32 {
            let chunk = std::array::from_fn::<_, 32, _>(|_| self.branches.pop().unwrap());
            let mut remain = padding_pack(chunk.iter().map(|x| rabitq::pack_to_u4(&x.signs)));
            loop {
                let freespace = self.tape.freespace();
                match (H0Tuple::fit_1(freespace), H0Tuple::fit_2(freespace)) {
                    (Some(w), _) if w >= remain.len() => {
                        self.tape.push(H0Tuple::_1 {
                            mean: chunk.each_ref().map(|x| x.mean),
                            dis_u_2: chunk.each_ref().map(|x| x.dis_u_2),
                            factor_ppc: chunk.each_ref().map(|x| x.factor_ppc),
                            factor_ip: chunk.each_ref().map(|x| x.factor_ip),
                            factor_err: chunk.each_ref().map(|x| x.factor_err),
                            payload: chunk.each_ref().map(|x| Some(x.payload)),
                            elements: remain,
                        });
                        break;
                    }
                    (_, Some(w)) => {
                        let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                        self.tape.tape_put(H0Tuple::_2 {
                            elements: left.to_vec(),
                        });
                        remain = right.to_vec();
                    }
                    (_, None) => self.tape.tape_move(),
                }
            }
        }
    }
    pub fn into_inner(mut self) -> TapeWriter<G, E, H0Tuple> {
        for x in self.branches {
            self.tape.push(H0Tuple::_0 {
                mean: x.mean,
                dis_u_2: x.dis_u_2,
                factor_ppc: x.factor_ppc,
                factor_ip: x.factor_ip,
                factor_err: x.factor_err,
                payload: Some(x.payload),
                elements: rabitq::pack_to_u64(&x.signs),
            });
        }
        self.tape
    }
}

pub fn access_1<A>(
    index: impl RelationRead,
    first: u32,
    make_block_accessor: impl Fn() -> A + Copy,
    mut callback: impl FnMut(Distance, IndexPointer, u32),
) where
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [Distance; 32],
        >,
{
    assert!(first != u32::MAX);
    let mut current = first;
    let mut computing = None;
    while current != u32::MAX {
        let h1_guard = index.read(current);
        for i in 1..=h1_guard.len() {
            let h1_tuple = h1_guard
                .get(i)
                .expect("data corruption")
                .pipe(read_tuple::<H1Tuple>);
            match h1_tuple {
                H1TupleReader::_0(h1_tuple) => {
                    let mut compute = computing.take().unwrap_or_else(make_block_accessor);
                    compute.push(h1_tuple.elements());
                    let lowerbounds = compute.finish(h1_tuple.metadata());
                    for i in 0..h1_tuple.len() {
                        callback(
                            lowerbounds[i as usize],
                            h1_tuple.mean()[i as usize],
                            h1_tuple.first()[i as usize],
                        );
                    }
                }
                H1TupleReader::_1(h1_tuple) => {
                    let computing = computing.get_or_insert_with(make_block_accessor);
                    computing.push(h1_tuple.elements());
                }
            }
        }
        current = h1_guard.get_opaque().next;
    }
}

pub fn access_0<A>(
    index: impl RelationRead,
    first: u32,
    make_block_accessor: impl Fn() -> A + Copy,
    compute_binary: impl Fn((f32, f32, f32, f32, &[u64])) -> Distance,
    mut callback: impl FnMut(Distance, IndexPointer, NonZeroU64),
) where
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [Distance; 32],
        >,
{
    assert!(first != u32::MAX);
    let mut current = first;
    let mut computing = None;
    while current != u32::MAX {
        let h0_guard = index.read(current);
        for i in 1..=h0_guard.len() {
            let h0_tuple = h0_guard
                .get(i)
                .expect("data corruption")
                .pipe(read_tuple::<H0Tuple>);
            match h0_tuple {
                H0TupleReader::_0(h0_tuple) => {
                    let lowerbound = compute_binary(h0_tuple.code());
                    if let Some(payload) = h0_tuple.payload() {
                        callback(lowerbound, h0_tuple.mean(), payload);
                    }
                }
                H0TupleReader::_1(h0_tuple) => {
                    let mut compute = computing.take().unwrap_or_else(make_block_accessor);
                    compute.push(h0_tuple.elements());
                    let lowerbounds = compute.finish(h0_tuple.metadata());
                    for j in 0..32 {
                        if let Some(payload) = h0_tuple.payload()[j] {
                            callback(lowerbounds[j], h0_tuple.mean()[j], payload);
                        }
                    }
                }
                H0TupleReader::_2(h0_tuple) => {
                    let computing = computing.get_or_insert_with(make_block_accessor);
                    computing.push(h0_tuple.elements());
                }
            }
        }
        current = h0_guard.get_opaque().next;
    }
}

pub fn append(
    index: impl RelationWrite,
    first: u32,
    bytes: &[u8],
    tracking_freespace: bool,
) -> IndexPointer {
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, tracking_freespace);
            if write.get_opaque().next == u32::MAX {
                if let Some(i) = write.alloc(bytes) {
                    return pair_to_pointer((current, i));
                }
                let mut extend = index.extend(tracking_freespace);
                write.get_opaque_mut().next = extend.id();
                drop(write);
                let fresh = extend.id();
                if let Some(i) = extend.alloc(bytes) {
                    drop(extend);
                    let mut past = index.write(first, tracking_freespace);
                    past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                    return pair_to_pointer((fresh, i));
                } else {
                    panic!("a tuple cannot even be fit in a fresh page");
                }
            }
            if current == first && write.get_opaque().skip != first {
                current = write.get_opaque().skip;
            } else {
                current = write.get_opaque().next;
            }
        } else {
            if current == first && read.get_opaque().skip != first {
                current = read.get_opaque().skip;
            } else {
                current = read.get_opaque().next;
            }
        }
    }
}
