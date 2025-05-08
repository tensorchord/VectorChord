use crate::operator::Operator;
use crate::{Fetch, Heap, ReadStream, RelationPrefetch, RelationRead, RelationReadStream};
use std::collections::{BinaryHeap, VecDeque, binary_heap, vec_deque};
use std::iter::Chain;
use std::marker::PhantomData;

pub const WINDOW_SIZE: usize = 32;
const _: () = assert!(WINDOW_SIZE > 0);

pub trait Prefetcher: IntoIterator
where
    Self::Item: Fetch,
{
    type R: RelationRead;
    type O: Operator;
    fn pop_if(
        &mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(
        Self::Item,
        Vec<<Self::R as RelationRead>::ReadGuard<'_>>,
        Option<<Self::O as Operator>::Vector>,
    )>;
}

pub struct PlainPrefetcher<R, H, F, O> {
    relation: R,
    heap: H,
    filter: F,
    with_fetch: bool,
    _phantom: PhantomData<O>,
}

impl<R, H: Heap, F: FnMut(&H::Item, bool) -> (bool, Option<O::Vector>), O: Operator>
    PlainPrefetcher<R, H, F, O>
{
    pub fn new(relation: R, vec: Vec<H::Item>, filter: F, with_fetch: bool) -> Self {
        Self {
            relation,
            heap: Heap::make(vec),
            filter,
            with_fetch,
            _phantom: PhantomData,
        }
    }
}

impl<R, H: Heap, F: FnMut(&H::Item, bool) -> (bool, Option<O::Vector>), O: Operator> IntoIterator
    for PlainPrefetcher<R, H, F, O>
{
    type Item = H::Item;

    type IntoIter = H::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.heap.into_iter()
    }
}

impl<R: RelationRead, H: Heap, F: FnMut(&H::Item, bool) -> (bool, Option<O::Vector>), O: Operator>
    Prefetcher for PlainPrefetcher<R, H, F, O>
where
    H::Item: Fetch + Ord,
{
    type R = R;
    type O = O;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(H::Item, Vec<R::ReadGuard<'s>>, Option<O::Vector>)> {
        let (e, raw) = self.heap.pop_if_with(|m| {
            let (filtered, raw) = (self.filter)(m, self.with_fetch);
            (predicate(m) && filtered, raw)
        })?;
        let list = e.fetch().iter().map(|&id| self.relation.read(id)).collect();
        Some((e, list, raw))
    }
}

pub struct SimplePrefetcher<R, T, F, O> {
    relation: R,
    window: VecDeque<T>,
    heap: BinaryHeap<T>,
    filter: F,
    with_fetch: bool,
    _phantom: PhantomData<O>,
}

impl<R, T: Ord, F: FnMut(&T, bool) -> (bool, Option<O::Vector>), O: Operator>
    SimplePrefetcher<R, T, F, O>
{
    pub fn new(relation: R, vec: Vec<T>, filter: F, with_fetch: bool) -> Self {
        Self {
            relation,
            window: VecDeque::new(),
            heap: BinaryHeap::from(vec),
            filter,
            with_fetch,
            _phantom: PhantomData,
        }
    }
}

impl<R, T, F: FnMut(&T, bool) -> (bool, Option<O::Vector>), O: Operator> IntoIterator
    for SimplePrefetcher<R, T, F, O>
{
    type Item = T;

    type IntoIter = Chain<vec_deque::IntoIter<T>, binary_heap::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.window.into_iter().chain(self.heap)
    }
}

impl<
    R: RelationRead + RelationPrefetch,
    T: Fetch + Ord,
    F: FnMut(&T, bool) -> (bool, Option<O::Vector>),
    O: Operator,
> Prefetcher for SimplePrefetcher<R, T, F, O>
{
    type R = R;
    type O = O;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>, Option<O::Vector>)> {
        while self.window.len() < WINDOW_SIZE
            && let Some(e) = self.heap.pop()
        {
            for id in e.fetch().iter().copied() {
                self.relation.prefetch(id);
            }
            self.window.push_back(e);
        }
        let (e, raw) = vec_deque_pop_front_if_with(&mut self.window, |m| {
            let (filtered, raw) = (self.filter)(m, self.with_fetch);
            (predicate(m) && filtered, raw)
        })?;
        let list = e.fetch().iter().map(|&id| self.relation.read(id)).collect();
        Some((e, list, raw))
    }
}

pub struct StreamPrefetcherHeap<T>(BinaryHeap<T>);

impl<T: Ord> Iterator for StreamPrefetcherHeap<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

pub struct StreamPrefetcher<'r, R, T, F, O>
where
    R: RelationReadStream + 'r,
    T: Fetch + Ord,
{
    stream: R::ReadStream<'r, StreamPrefetcherHeap<T>>,
    filter: F,
    with_fetch: bool,
    _phantom: PhantomData<O>,
}

impl<
    'r,
    R: RelationReadStream,
    T: Fetch + Ord,
    F: FnMut(&T, bool) -> (bool, Option<O::Vector>),
    O: Operator,
> StreamPrefetcher<'r, R, T, F, O>
{
    pub fn new(relation: &'r R, vec: Vec<T>, filter: F, with_fetch: bool) -> Self {
        let stream = relation.read_stream(StreamPrefetcherHeap(BinaryHeap::from(vec)));
        Self {
            stream,
            filter,
            with_fetch,
            _phantom: PhantomData,
        }
    }
}

impl<
    'r,
    R: RelationReadStream,
    T: Fetch + Ord,
    F: FnMut(&T, bool) -> (bool, Option<O::Vector>),
    O: Operator,
> IntoIterator for StreamPrefetcher<'r, R, T, F, O>
{
    type Item = T;

    type IntoIter = <<R as RelationReadStream>::ReadStream<
        'r,
        StreamPrefetcherHeap<T>,
    > as ReadStream<T>>::Inner;

    fn into_iter(self) -> Self::IntoIter {
        self.stream.into_inner()
    }
}

impl<
    'r,
    R: RelationReadStream,
    T: Fetch + Ord,
    F: FnMut(&T, bool) -> (bool, Option<O::Vector>),
    O: Operator,
> Prefetcher for StreamPrefetcher<'r, R, T, F, O>
{
    type R = R;
    type O = O;
    fn pop_if<'s>(
        &'s mut self,
        predicate: impl FnOnce(&Self::Item) -> bool,
    ) -> Option<(T, Vec<R::ReadGuard<'s>>, Option<O::Vector>)> {
        self.stream.next_if_with(|m| {
            let (filtered, raw) = (self.filter)(m, self.with_fetch);
            (predicate(m) && filtered, raw)
        })
    }
}

// Emulate unstable library feature `vec_deque_pop_if`.
// See https://github.com/rust-lang/rust/issues/135889.
#[allow(dead_code)]
fn vec_deque_pop_front_if<T>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&T) -> bool,
) -> Option<T> {
    let first = this.front()?;
    if predicate(first) {
        this.pop_front()
    } else {
        None
    }
}

fn vec_deque_pop_front_if_with<T, A>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&T) -> (bool, A),
) -> Option<(T, A)> {
    let first = this.front()?;
    let (result, another) = predicate(first);
    if result {
        this.pop_front().map(|peek| (peek, another))
    } else {
        None
    }
}
