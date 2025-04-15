use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{IndexPointer, Page, RelationRead, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::VectorOwned;

type Result<T> = (
    Reverse<Distance>,
    AlwaysEqual<T>,
    AlwaysEqual<NonZero<u64>>,
    AlwaysEqual<IndexPointer>,
);

type Rerank = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

pub struct Reranker<T, F> {
    heap: BinaryHeap<Result<T>>,
    cache: BinaryHeap<(Reverse<Distance>, AlwaysEqual<NonZero<u64>>)>,
    f: F,
}

impl<T, F: FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>> Iterator for Reranker<T, F> {
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((Reverse(_), AlwaysEqual(_), AlwaysEqual(pay_u), AlwaysEqual(mean))) =
            pop_if(&mut self.heap, |(d, ..)| {
                Some(*d) > self.cache.peek().map(|(d, ..)| *d)
            })
        {
            if let Some(dis_u) = (self.f)(mean, pay_u) {
                self.cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            };
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = self.cache.pop()?;
        Some((dis_u, pay_u))
    }
}

impl<T, F> Reranker<T, F> {
    pub fn finish(
        self,
    ) -> (
        impl Iterator<Item = Result<T>>,
        impl Iterator<Item = Rerank>,
    ) {
        (self.heap.into_iter(), self.cache.into_iter())
    }
}

pub fn rerank_index<O: Operator, T>(
    index: impl RelationRead,
    vector: O::Vector,
    results: Vec<Result<T>>,
) -> Reranker<T, impl FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>> {
    Reranker {
        heap: BinaryHeap::from(results),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        f: move |mean, pay_u| {
            vectors::read_for_h0_tuple::<O, _>(
                index.clone(),
                mean,
                pay_u,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            )
        },
    }
}

pub fn rerank_heap<O: Operator, T>(
    vector: O::Vector,
    results: Vec<Result<T>>,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector>,
) -> Reranker<T, impl FnMut(IndexPointer, NonZero<u64>) -> Option<Distance>> {
    Reranker {
        heap: BinaryHeap::from(results),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        f: move |_: IndexPointer, pay_u| {
            let vector = O::Vector::unpack(vector.as_borrowed());
            let vec_u = fetch(pay_u)?;
            let vec_u = O::Vector::unpack(vec_u.as_borrowed());
            let mut accessor = O::DistanceAccessor::default();
            accessor.push(vector.0, vec_u.0);
            let dis_u = accessor.finish(vector.1, vec_u.1);
            Some(dis_u)
        },
    }
}

fn pop_if<T: Ord>(
    heap: &mut BinaryHeap<T>,
    mut predicate: impl FnMut(&mut T) -> bool,
) -> Option<T> {
    use std::collections::binary_heap::PeekMut;
    let mut peek = heap.peek_mut()?;
    if predicate(&mut peek) {
        Some(PeekMut::pop(peek))
    } else {
        None
    }
}
