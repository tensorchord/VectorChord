pub struct SelectHeap<T> {
    threshold: usize,
    inner: Vec<T>,
}

impl<T: Ord> SelectHeap<T> {
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let n = vec.len();
        if n != 0 {
            let threshold = n.saturating_sub(n.div_ceil(384));
            turboselect::select_nth_unstable(&mut vec, threshold);
            vec[threshold..].sort_unstable();
            Self {
                threshold,
                inner: vec,
            }
        } else {
            Self {
                threshold: 0,
                inner: Vec::new(),
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.inner.len() <= self.threshold {
            heapify::pop_heap(&mut self.inner);
        }
        let t = self.inner.pop();
        if self.inner.len() == self.threshold {
            heapify::make_heap(&mut self.inner);
        }
        t
    }
    pub fn peek(&self) -> Option<&T> {
        if self.inner.len() <= self.threshold {
            self.inner.first()
        } else {
            self.inner.last()
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
