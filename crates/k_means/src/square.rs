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

#[derive(Debug, Clone)]
pub struct Square {
    d: usize,
    p: Vec<f32>,
}

impl Square {
    pub fn d(&self) -> usize {
        self.d
    }
    pub fn len(&self) -> usize {
        self.p.len() / self.d
    }
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }
    pub fn with_capacity(d: usize, p: usize) -> Self {
        Self {
            d,
            p: Vec::with_capacity(usize::saturating_mul(d, p)),
        }
    }
    pub fn new(d: usize) -> Self {
        Self { d, p: Vec::new() }
    }
    pub fn push_slice(&mut self, slice: &[f32]) {
        assert_eq!(slice.len(), self.d);
        self.p.extend_from_slice(slice);
    }
    pub fn push_iter(&mut self, iter: impl ExactSizeIterator<Item = f32>) {
        assert_eq!(iter.len(), self.d);
        self.p.extend(iter);
    }
    pub fn copy_within<R: std::ops::RangeBounds<usize>>(&mut self, src: R, dest: usize) {
        let src_start = src.start_bound().map(|x| self.d * x);
        let src_end = src.end_bound().map(|x| self.d * x);
        self.p.copy_within((src_start, src_end), self.d * dest);
    }
    pub fn from_zeros(d: usize, p: usize) -> Self {
        Self {
            d,
            p: vec![0.0; d * p],
        }
    }
    pub fn truncate(&mut self, len: usize) {
        self.p.truncate(self.d * len);
    }
    #[inline]
    pub fn as_mut_view(&mut self) -> SquareMut<'_> {
        SquareMut {
            d: self.d,
            p: self.p.as_mut_slice(),
        }
    }
}

impl std::ops::Index<usize> for Square {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        &self.p[self.d * index..][..self.d]
    }
}

impl std::ops::IndexMut<usize> for Square {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.p[self.d * index..][..self.d]
    }
}

impl<'a> IntoIterator for &'a Square {
    type Item = &'a [f32];

    type IntoIter = std::slice::ChunksExact<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.p.chunks_exact(self.d)
    }
}

impl<'a> IntoIterator for &'a mut Square {
    type Item = &'a mut [f32];

    type IntoIter = std::slice::ChunksExactMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.p.chunks_exact_mut(self.d)
    }
}

impl<'a> rayon::prelude::IntoParallelIterator for &'a Square {
    type Item = &'a [f32];

    type Iter = rayon::slice::ChunksExact<'a, f32>;

    fn into_par_iter(self) -> Self::Iter {
        rayon::slice::ParallelSlice::par_chunks_exact(self.p.as_slice(), self.d)
    }
}

impl<'a> rayon::prelude::IntoParallelIterator for &'a mut Square {
    type Item = &'a mut [f32];

    type Iter = rayon::slice::ChunksExactMut<'a, f32>;

    fn into_par_iter(self) -> Self::Iter {
        rayon::slice::ParallelSliceMut::par_chunks_exact_mut(self.p.as_mut_slice(), self.d)
    }
}

#[derive(Debug)]
pub struct SquareMut<'a> {
    d: usize,
    p: &'a mut [f32],
}
impl<'a> SquareMut<'a> {
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.p.len() / self.d
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.p.is_empty()
    }
    pub fn new(d: usize, p: &'a mut [f32]) -> Self {
        Self { d, p }
    }
    #[inline]
    pub fn iter_mut<'b>(&'b mut self) -> std::slice::ChunksExactMut<'b, f32>
    where
        'a: 'b,
    {
        self.p.chunks_exact_mut(self.d)
    }
    #[inline]
    pub fn par_iter_mut<'b>(&'b mut self) -> rayon::slice::ChunksExactMut<'b, f32>
    where
        'a: 'b,
    {
        rayon::slice::ParallelSliceMut::par_chunks_exact_mut(self.p, self.d)
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let d = self.d;
        &self.p[i * d..(i + 1) * d]
    }
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let d = self.d;
        &mut self.p[i * d..(i + 1) * d]
    }
    pub fn copy_within<R: std::ops::RangeBounds<usize>>(&mut self, src: R, dest: usize) {
        let src_start = src.start_bound().map(|x| self.d * x);
        let src_end = src.end_bound().map(|x| self.d * x);
        self.p.copy_within((src_start, src_end), self.d * dest);
    }
    #[inline]
    pub fn into_inner(self) -> (usize, &'a mut [f32]) {
        (self.d, self.p)
    }
}

impl std::ops::Index<usize> for SquareMut<'_> {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        &self.p[self.d * index..][..self.d]
    }
}

impl std::ops::IndexMut<usize> for SquareMut<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.p[self.d * index..][..self.d]
    }
}

impl<'a> rayon::prelude::IntoParallelIterator for &'a mut SquareMut<'a> {
    type Item = &'a mut [f32];

    type Iter = rayon::slice::ChunksExactMut<'a, f32>;

    fn into_par_iter(self) -> Self::Iter {
        rayon::slice::ParallelSliceMut::par_chunks_exact_mut(self.p, self.d)
    }
}
