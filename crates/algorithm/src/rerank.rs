use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::prefetcher::Prefetcher;
use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, RelationRead, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use vector::VectorOwned;

type Extra<'b> = &'b mut (NonZero<u64>, u16, &'b mut [u32]);

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

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

pub struct Reranker<T, F, P> {
    prefetcher: P,
    cache: BinaryHeap<Result>,
    f: F,
    _phantom: PhantomData<fn(T) -> T>,
}

impl<'b, T, F, P> Iterator for Reranker<T, F, P>
where
    F: FnMut(NonZero<u64>, Vec<<P::R as RelationRead>::ReadGuard<'_>>, u16) -> Option<Distance>,
    P: Prefetcher<Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(((_, AlwaysEqual(&mut (payload, head, ..))), list)) = self
            .prefetcher
            .pop_if(|((d, _), ..)| Some(*d) > self.cache.peek().map(|(d, ..)| *d))
        {
            if let Some(distance) = (self.f)(payload, list, head) {
                self.cache.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let (Reverse(distance), AlwaysEqual(payload)) = self.cache.pop()?;
        Some((distance, payload))
    }
}

impl<T, F, P> Reranker<T, F, P> {
    pub fn finish(self) -> (P, impl Iterator<Item = Result>) {
        (self.prefetcher, self.cache.into_iter())
    }
}

pub fn rerank_index<
    'b,
    O: Operator,
    T,
    P: Prefetcher<Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
>(
    vector: O::Vector,
    prefetcher: P,
    mut prefilter: impl FnMut(NonZero<u64>) -> bool,
) -> Reranker<
    T,
    impl FnMut(NonZero<u64>, Vec<<P::R as RelationRead>::ReadGuard<'_>>, u16) -> Option<Distance>,
    P,
> {
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, P::R, _, _, _>(move |payload, list, head| {
            if !prefilter(payload) {
                return None;
            }
            vectors::read_for_h0_tuple::<P::R, O, _>(
                head,
                list.into_iter(),
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    O: Operator,
    T,
    P: Prefetcher<Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<Extra<'b>>)>,
>(
    vector: O::Vector,
    prefetcher: P,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<
    T,
    impl FnMut(NonZero<u64>, Vec<<P::R as RelationRead>::ReadGuard<'_>>, u16) -> Option<Distance>,
    P,
> {
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        f: id_4::<_, P::R, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.as_borrowed());
            let vector = fetch(payload)?;
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default();
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            Some(distance)
        }),
        _phantom: PhantomData,
    }
}
