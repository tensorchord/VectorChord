use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{IndexPointer, Page, RelationRead, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::VectorOwned;

type Results = Vec<(
    Reverse<Distance>,
    AlwaysEqual<NonZeroU64>,
    AlwaysEqual<IndexPointer>,
)>;

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

pub fn rerank_index<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    results: Results,
) -> impl Iterator<Item = (Distance, NonZeroU64)> {
    let mut heap = BinaryHeap::from(results);
    let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
    std::iter::from_fn(move || {
        while let Some((_, AlwaysEqual(pay_u), AlwaysEqual(mean))) =
            pop_if(&mut heap, |x| Some(x.0) > cache.peek().map(|x| x.0))
        {
            if let Some(dis_u) = vectors::read_for_h0_tuple::<O, _>(
                index.clone(),
                mean,
                pay_u,
                LAccess::new(
                    O::Vector::elements_and_metadata(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            ) {
                cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            };
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = cache.pop()?;
        Some((dis_u, pay_u))
    })
}

pub fn rerank_heap<O: Operator, F>(
    vector: O::Vector,
    results: Results,
    mut fetch: F,
) -> impl Iterator<Item = (Distance, NonZeroU64)>
where
    F: FnMut(NonZeroU64) -> Option<O::Vector>,
{
    let mut heap = BinaryHeap::from(results);
    let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
    std::iter::from_fn(move || {
        while let Some((_, AlwaysEqual(pay_u), AlwaysEqual(_))) =
            pop_if(&mut heap, |x| Some(x.0) > cache.peek().map(|x| x.0))
        {
            let vector = O::Vector::elements_and_metadata(vector.as_borrowed());
            if let Some(vec_u) = fetch(pay_u) {
                let vec_u = O::Vector::elements_and_metadata(vec_u.as_borrowed());
                let mut accessor = O::DistanceAccessor::default();
                accessor.push(vector.0, vec_u.0);
                let dis_u = accessor.finish(vector.1, vec_u.1);
                cache.push((Reverse(dis_u), AlwaysEqual(pay_u)));
            }
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = cache.pop()?;
        Some((dis_u, pay_u))
    })
}

pub fn skip(results: Results) -> impl Iterator<Item = (Distance, NonZeroU64)> {
    let results = BinaryHeap::from(results);
    results
        .into_iter()
        .map(|(Reverse(x), AlwaysEqual(y), _)| (x, y))
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
