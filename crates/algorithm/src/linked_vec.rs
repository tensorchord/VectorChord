pub struct LinkedVec<T> {
    inner: Vec<Vec<T>>,
    last: Vec<T>,
}

impl<T> LinkedVec<T> {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            last: Vec::with_capacity(4096),
        }
    }
    pub fn push(&mut self, value: T) {
        if self.last.len() == self.last.capacity() {
            self.reserve();
        }
        #[allow(unsafe_code)]
        unsafe {
            std::hint::assert_unchecked(self.last.len() != self.last.capacity());
        }
        self.last.push(value);
    }
    #[cold]
    fn reserve(&mut self) {
        let fresh = Vec::with_capacity(self.last.capacity() * 4);
        self.inner.push(core::mem::replace(&mut self.last, fresh));
    }
    pub fn into_vec(self) -> Vec<T> {
        let mut last = self.last;
        last.reserve(self.inner.iter().map(Vec::len).sum::<usize>());
        for x in self.inner {
            last.extend(x);
        }
        last
    }
}
