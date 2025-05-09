use crate::Sequence;
use std::collections::BinaryHeap;
use std::num::NonZero;

pub struct SortHeap<T> {
    inner: Vec<T>,
    t: NonZero<usize>,
}

impl<T> SortHeap<T> {
    pub fn peek(&self) -> Option<&T> {
        self.inner.last()
    }
}

pub enum FastHeap<T> {
    Sorted(SortHeap<T>),
    Binary(BinaryHeap<T>),
}

impl<T: Ord> FastHeap<T> {
    pub fn from_vec(vec: Vec<T>) -> Self {
        let n = vec.len();
        if let Some(t) = NonZero::new(n / 384) {
            let mut inner = vec;
            let index = n - t.get();
            turboselect::select_nth_unstable(&mut inner, index);
            inner[index..].sort_unstable();
            Self::Sorted(SortHeap { inner, t })
        } else {
            Self::Binary(BinaryHeap::from(vec))
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        match self {
            FastHeap::Sorted(SortHeap { inner, t }) => {
                let Some(k) = inner.pop() else { unreachable!() };
                if let Some(value) = NonZero::new(t.get() - 1) {
                    *t = value;
                } else {
                    *self = FastHeap::Binary(std::mem::take(inner).into());
                }
                Some(k)
            }
            FastHeap::Binary(x) => x.pop(),
        }
    }
    pub fn peek(&self) -> Option<&T> {
        match self {
            FastHeap::Sorted(x) => x.peek(),
            FastHeap::Binary(x) => x.peek(),
        }
    }
}

impl<T: Ord> From<Vec<T>> for FastHeap<T> {
    fn from(value: Vec<T>) -> Self {
        Self::from_vec(value)
    }
}

impl<T: Ord> Sequence for FastHeap<T> {
    type Item = T;
    type Inner = std::vec::IntoIter<T>;
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
    fn next_if(&mut self, predicate: impl FnOnce(&T) -> bool) -> Option<T> {
        let first = self.peek()?;
        if predicate(first) { self.pop() } else { None }
    }
    fn into_inner(self) -> Self::Inner {
        match self {
            FastHeap::Sorted(sort_heap) => sort_heap.inner.into_iter(),
            FastHeap::Binary(binary_heap) => binary_heap.into_vec().into_iter(),
        }
    }
}

#[test]
fn test_fast_heap() {
    for _ in 0..1000 {
        let sequence = (0..10000)
            .map(|_| rand::random::<i32>())
            .collect::<Vec<_>>();
        let answer = {
            let mut x = sequence.clone();
            x.sort_by_key(|x| std::cmp::Reverse(*x));
            x
        };
        let result = {
            let mut x = FastHeap::from_vec(sequence.clone());
            std::iter::from_fn(|| x.pop()).collect::<Vec<_>>()
        };
        assert_eq!(answer, result);
    }
}

#[test]
fn test_issue_209() {
    let mut heap = FastHeap::from_vec(vec![0]);
    assert_eq!(heap.pop(), Some(0));
    assert_eq!(heap.pop(), None);
}
