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

use crate::{VectorBorrowed, VectorOwned};
use distance::Distance;
use simd::Floating;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct VectOwned<S>(Vec<S>);

impl<S: Floating> VectOwned<S> {
    #[inline(always)]
    pub fn new(slice: Vec<S>) -> Self {
        Self::new_checked(slice).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(slice: Vec<S>) -> Option<Self> {
        if !(1..=65535).contains(&slice.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(slice) })
    }

    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    /// * `slice.len()` must be less than 65536.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: Vec<S>) -> Self {
        Self(slice)
    }

    #[inline(always)]
    pub fn slice(&self) -> &[S] {
        self.0.as_slice()
    }

    #[inline(always)]
    pub fn slice_mut(&mut self) -> &mut [S] {
        self.0.as_mut_slice()
    }

    #[inline(always)]
    pub fn into_vec(self) -> Vec<S> {
        self.0
    }
}

impl<S: Floating> VectorOwned for VectOwned<S> {
    type Borrowed<'a> = VectBorrowed<'a, S>;

    #[inline(always)]
    fn as_borrowed(&self) -> VectBorrowed<'_, S> {
        VectBorrowed(self.0.as_slice())
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct VectBorrowed<'a, S>(&'a [S]);

impl<'a, S: Floating> VectBorrowed<'a, S> {
    #[inline(always)]
    pub fn new(slice: &'a [S]) -> Self {
        Self::new_checked(slice).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(slice: &'a [S]) -> Option<Self> {
        if !(1..=65535).contains(&slice.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(slice) })
    }

    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    /// * `slice.len()` must be less than 65536.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: &'a [S]) -> Self {
        Self(slice)
    }

    #[inline(always)]
    pub fn slice(&self) -> &'a [S] {
        self.0
    }
}

impl<S: Floating> VectorBorrowed for VectBorrowed<'_, S> {
    type Owned = VectOwned<S>;

    #[inline(always)]
    fn dim(&self) -> u32 {
        self.0.len() as u32
    }

    #[inline(always)]
    fn own(&self) -> VectOwned<S> {
        VectOwned(self.0.to_vec())
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        S::reduce_sum_of_x2(self.0).sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        Distance::from(-S::reduce_sum_of_xy(self.slice(), rhs.slice()))
    }

    #[inline(always)]
    fn operator_l2s(self, rhs: Self) -> Distance {
        Distance::from(S::reduce_sum_of_d2(self.slice(), rhs.slice()))
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        let xy = S::reduce_sum_of_xy(self.slice(), rhs.slice());
        let x2 = S::reduce_sum_of_x2(self.0);
        let y2 = S::reduce_sum_of_x2(rhs.0);
        Distance::from(1.0 - xy / (x2 * y2).sqrt())
    }

    #[inline(always)]
    fn operator_hamming(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_jaccard(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn function_normalize(&self) -> VectOwned<S> {
        let mut data = self.0.to_vec();
        let l = S::reduce_sum_of_x2(&data).sqrt();
        S::vector_mul_scalar_inplace(&mut data, 1.0 / l);
        VectOwned(data)
    }

    fn operator_add(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_add(self.slice(), rhs.slice()))
    }

    fn operator_sub(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_sub(self.slice(), rhs.slice()))
    }

    fn operator_mul(&self, rhs: Self) -> Self::Owned {
        VectOwned::new(S::vector_mul(self.slice(), rhs.slice()))
    }

    fn operator_and(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_or(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_xor(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }
}

impl<S: Floating> PartialEq for VectBorrowed<'_, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        let n = self.0.len();
        for i in 0..n {
            if self.0[i] != other.0[i] {
                return false;
            }
        }
        true
    }
}

impl<S: Floating> PartialOrd for VectBorrowed<'_, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.len() != other.0.len() {
            return None;
        }
        let n = self.0.len();
        for i in 0..n {
            match PartialOrd::partial_cmp(&self.0[i], &other.0[i])? {
                Ordering::Less => return Some(Ordering::Less),
                Ordering::Equal => continue,
                Ordering::Greater => return Some(Ordering::Greater),
            }
        }
        Some(Ordering::Equal)
    }
}
