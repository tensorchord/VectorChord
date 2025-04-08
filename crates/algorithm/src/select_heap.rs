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

pub enum SelectHeap<T> {
    Sorted(SortHeap<T>),
    Binary(BinaryHeap<T>),
}

impl<T: Ord> SelectHeap<T> {
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
            SelectHeap::Sorted(SortHeap { inner, t }) => {
                let Some(k) = inner.pop() else { unreachable!() };
                if let Some(value) = NonZero::new(t.get() - 1) {
                    *t = value;
                } else {
                    *self = SelectHeap::Binary(std::mem::take(inner).into());
                }
                Some(k)
            }
            SelectHeap::Binary(x) => x.pop(),
        }
    }
    pub fn peek(&self) -> Option<&T> {
        match self {
            SelectHeap::Sorted(x) => x.peek(),
            SelectHeap::Binary(x) => x.peek(),
        }
    }
}

#[test]
fn test_select_heap() {
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
            let mut x = SelectHeap::from_vec(sequence.clone());
            std::iter::from_fn(|| x.pop()).collect::<Vec<_>>()
        };
        assert_eq!(answer, result);
    }
}

#[test]
fn test_issue_209() {
    let mut heap = SelectHeap::from_vec(vec![0]);
    assert_eq!(heap.pop(), Some(0));
    assert_eq!(heap.pop(), None);
}
