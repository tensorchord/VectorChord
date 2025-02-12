use crate::operator::*;
use crate::tuples::*;
use crate::{RelationRead, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;
use vector::VectorOwned;

pub fn rerank_index<O: Operator>(
    index: impl RelationRead,
    vector: O::Vector,
    results: Vec<(
        Reverse<Distance>,
        AlwaysEqual<IndexPointer>,
        AlwaysEqual<NonZeroU64>,
    )>,
) -> impl Iterator<Item = (Distance, NonZeroU64)> {
    let mut heap = BinaryHeap::from(results);
    let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
    std::iter::from_fn(move || {
        while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
            let (_, AlwaysEqual(mean), AlwaysEqual(pay_u)) = heap.pop().unwrap();
            if let Some(dis_u) = vectors::access_0::<O, _>(
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
    results: Vec<(
        Reverse<Distance>,
        AlwaysEqual<IndexPointer>,
        AlwaysEqual<NonZeroU64>,
    )>,
    fetch: F,
) -> impl Iterator<Item = (Distance, NonZeroU64)>
where
    F: Fn(NonZeroU64) -> Option<O::Vector>,
{
    let mut heap = BinaryHeap::from(results);
    let mut cache = BinaryHeap::<(Reverse<Distance>, _)>::new();
    std::iter::from_fn(move || {
        let vector = O::Vector::elements_and_metadata(vector.as_borrowed());
        while !heap.is_empty() && heap.peek().map(|x| x.0) > cache.peek().map(|x| x.0) {
            let (_, AlwaysEqual(_), AlwaysEqual(pay_u)) = heap.pop().unwrap();
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
