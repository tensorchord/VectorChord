// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use crate::operator::Accessor1;
use crate::tuples::*;
use crate::{Page, PageGuard, PrefetcherSequenceFamily, RelationRead, RelationWrite};
use rabitq::binary::BinaryCode;
use std::marker::PhantomData;
use std::num::NonZero;

pub struct TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    head: R::WriteGuard<'a>,
    first: u32,
    index: &'a R,
    tracking_freespace: bool,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        let mut head = index.extend(tracking_freespace);
        head.get_opaque_mut().skip = head.id();
        let first = head.id();
        Self {
            head,
            first,
            index,
            tracking_freespace,
            _phantom: PhantomData,
        }
    }
    pub fn first(&self) -> u32 {
        self.first
    }
    pub fn freespace(&self) -> u16 {
        self.head.freespace()
    }
    pub fn tape_move(&mut self) {
        if self.head.len() == 0 {
            panic!("implementation: a clear page cannot accommodate a single tuple");
        }
        let next = self.index.extend(self.tracking_freespace);
        self.head.get_opaque_mut().next = next.id();
        self.head = next;
    }
}

impl<'a, R, T> TapeWriter<'a, R, T>
where
    R: RelationWrite + 'a,
    T: Tuple,
{
    pub fn push(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            let next = self.index.extend(self.tracking_freespace);
            self.head.get_opaque_mut().next = next.id();
            self.head = next;
            if let Some(i) = self.head.alloc(&bytes) {
                (self.head.id(), i)
            } else {
                panic!("implementation: a free page cannot accommodate a single tuple")
            }
        }
    }
    pub fn tape_put(&mut self, x: T) -> (u32, u16) {
        let bytes = T::serialize(&x);
        if let Some(i) = self.head.alloc(&bytes) {
            (self.head.id(), i)
        } else {
            panic!("implementation: a free page cannot accommodate a single tuple")
        }
    }
}

pub fn read_directory_tape<'r, R>(
    iter: impl Iterator<Item = R::ReadGuard<'r>>,
) -> impl Iterator<Item = u32>
where
    R: RelationRead + 'r,
{
    use std::pin::Pin;
    use std::ptr::NonNull;

    #[pin_project::pin_project]
    struct State<'r, R: RelationRead + 'r, I> {
        slice: NonNull<[u32]>,
        #[pin]
        now: Option<(R::ReadGuard<'r>, u16)>,
        iter: I,
    }

    impl<'r, R: RelationRead + 'r, I: Iterator<Item = R::ReadGuard<'r>>> State<'r, R, I> {
        fn init(self: Pin<&mut Self>) {
            let mut this = self.project();
            let now = this.iter.next().map(|guard| (guard, 0));
            this.now.set(now);
        }

        fn next(mut self: Pin<&mut Self>) -> Option<u32> {
            loop {
                let mut this = self.as_mut().project();
                // Safety: If the slice is not empty, the function will return immediately,
                // so the guard will not be moved or dropped and the slice remains valid. If
                // the slice is empty, a pointer is trivially never dangling, so it's safe
                // to use.
                #[allow(unsafe_code)]
                if let Some((first, more)) = unsafe { this.slice.as_ref() }.split_first() {
                    *this.slice = more.into();
                    return Some(*first);
                }
                // Safety: `guard` is never moved in this block
                #[allow(unsafe_code)]
                if let Some((guard, i)) = unsafe { this.now.as_mut().get_unchecked_mut() } {
                    if *i < guard.len() {
                        *i += 1;
                        let bytes = guard.get(*i).expect("data corruption");
                        let tuple = DirectoryTuple::deserialize_ref(bytes);
                        *this.slice = match tuple {
                            DirectoryTupleReader::_0(tuple) => tuple.elements(),
                            DirectoryTupleReader::_1(tuple) => tuple.elements(),
                        }
                        .into();
                        continue;
                    }
                } else {
                    return None;
                }
                let now = this.iter.next().map(|guard| (guard, 0));
                this.now.set(now);
            }
        }
    }

    let mut state = Box::pin(State::<'r, R, _> {
        slice: NonNull::from(&mut []),
        now: None,
        iter,
    });

    impl<'r, R: RelationRead + 'r, I: Iterator<Item = R::ReadGuard<'r>>> Iterator
        for Pin<Box<State<'r, R, I>>>
    {
        type Item = u32;

        fn next(&mut self) -> Option<u32> {
            self.as_mut().next()
        }
    }

    state.as_mut().init();

    state
}

pub fn by_directory<'r, R>(
    p: &mut impl PrefetcherSequenceFamily<'r, R>,
    iter: impl Iterator<Item = u32>,
) -> impl Iterator<Item = R::ReadGuard<'r>>
where
    R: RelationRead + 'r,
{
    let mut t = p.prefetch(iter.peekable());
    std::iter::from_fn(move || {
        use crate::prefetcher::Prefetcher;
        let (_, mut x) = t.next()?;
        let ret = x.pop().expect("should be at least one element");
        assert!(x.pop().is_none(), "should be at most one element");
        Some(ret)
    })
}

pub fn by_next<'r, R>(index: &'r R, first: u32) -> impl Iterator<Item = R::ReadGuard<'r>>
where
    R: RelationRead + 'r,
{
    let mut current = first;
    std::iter::from_fn(move || {
        if current != u32::MAX {
            let guard = index.read(current);
            current = guard.get_opaque().next;
            Some(guard)
        } else {
            None
        }
    })
}

pub fn read_h1_tape<'r, R, A, T>(
    iter: impl Iterator<Item = R::ReadGuard<'r>>,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, u32, &'a [u32]),
) where
    R: RelationRead + 'r,
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [T; 32],
        >,
{
    let mut x = None;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = H1Tuple::deserialize_ref(bytes);
            match tuple {
                H1TupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish(tuple.metadata());
                    let prefetch: [_; 32] = fix_0(tuple.prefetch());
                    for (j, value) in values.into_iter().enumerate() {
                        if j < tuple.len() as usize {
                            callback(value, tuple.head()[j], tuple.first()[j], fix_1(prefetch[j]));
                        }
                    }
                }
                H1TupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
    }
}

pub fn read_frozen_tape<'r, R, A, T>(
    iter: impl Iterator<Item = R::ReadGuard<'r>>,
    accessor: impl Fn() -> A,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32]),
) where
    R: RelationRead + 'r,
    A: for<'a> Accessor1<
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            Output = [T; 32],
        >,
{
    let mut x = None;
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = FrozenTuple::deserialize_ref(bytes);
            match tuple {
                FrozenTupleReader::_0(tuple) => {
                    let mut x = x.take().unwrap_or_else(&accessor);
                    x.push(tuple.elements());
                    let values = x.finish(tuple.metadata());
                    let prefetch: [_; 32] = fix_0(tuple.prefetch());
                    for (j, value) in values.into_iter().enumerate() {
                        if let Some(payload) = tuple.payload()[j] {
                            callback(value, tuple.mean()[j], payload, fix_1(prefetch[j]));
                        }
                    }
                }
                FrozenTupleReader::_1(tuple) => {
                    x.get_or_insert_with(&accessor).push(tuple.elements());
                }
            }
        }
    }
}

pub fn read_appendable_tape<'r, R, T>(
    iter: impl Iterator<Item = R::ReadGuard<'r>>,
    mut access: impl for<'a> FnMut(BinaryCode<'a>) -> T,
    mut callback: impl for<'a> FnMut(T, u16, NonZero<u64>, &'a [u32]),
) where
    R: RelationRead + 'r,
{
    for guard in iter {
        for i in 1..=guard.len() {
            let bytes = guard.get(i).expect("data corruption");
            let tuple = AppendableTuple::deserialize_ref(bytes);
            if let Some(payload) = tuple.payload() {
                let value = access(tuple.code());
                callback(value, tuple.head(), payload, tuple.prefetch());
            }
        }
    }
}

#[allow(clippy::collapsible_else_if)]
pub fn append(
    index: &(impl RelationRead + RelationWrite),
    first: u32,
    bytes: &[u8],
    tracking_freespace: bool,
) -> (u32, u16) {
    assert!(first != u32::MAX);
    let mut current = first;
    loop {
        let read = index.read(current);
        if read.freespace() as usize >= bytes.len() || read.get_opaque().next == u32::MAX {
            drop(read);
            let mut write = index.write(current, tracking_freespace);
            if write.get_opaque().next == u32::MAX {
                if let Some(i) = write.alloc(bytes) {
                    return (current, i);
                }
                let mut extend = index.extend(tracking_freespace);
                write.get_opaque_mut().next = extend.id();
                drop(write);
                let fresh = extend.id();
                if let Some(i) = extend.alloc(bytes) {
                    drop(extend);
                    let mut past = index.write(first, tracking_freespace);
                    past.get_opaque_mut().skip = fresh.max(past.get_opaque().skip);
                    return (fresh, i);
                } else {
                    panic!("implementation: a clear page cannot accommodate a single tuple");
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

fn fix_0<T>(x: &[[T; 32]]) -> [&[T]; 32] {
    let step = x.len();
    let flat = x.as_flattened();
    std::array::from_fn(|i| &flat[i * step..][..step])
}

fn fix_1(x: &[u32]) -> &[u32] {
    if let Some(i) = x.iter().position(|&x| x == u32::MAX) {
        &x[..i]
    } else {
        x
    }
}
