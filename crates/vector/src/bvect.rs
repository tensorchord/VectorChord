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

pub const BVECTOR_WIDTH: u32 = u64::BITS;

// When using binary vector, please ensure that the padding bits are always zero.
#[derive(Debug, Clone)]
pub struct BVectOwned {
    dim: u32,
    data: Vec<u64>,
}

impl BVectOwned {
    #[inline(always)]
    pub fn new(dim: u32, data: Vec<u64>) -> Self {
        Self::new_checked(dim, data).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(dim: u32, data: Vec<u64>) -> Option<Self> {
        if !(1..=65535).contains(&dim) {
            return None;
        }
        if data.len() != dim.div_ceil(BVECTOR_WIDTH) as usize {
            return None;
        }
        if dim % BVECTOR_WIDTH != 0 && data[data.len() - 1] >> (dim % BVECTOR_WIDTH) != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        unsafe {
            Some(Self::new_unchecked(dim, data))
        }
    }

    /// # Safety
    ///
    /// * `dim` must be in `1..=65535`.
    /// * `data` must be of the correct length.
    /// * The padding bits must be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(dim: u32, data: Vec<u64>) -> Self {
        Self { dim, data }
    }
}

impl VectorOwned for BVectOwned {
    type Borrowed<'a> = BVectBorrowed<'a>;

    #[inline(always)]
    fn as_borrowed(&self) -> BVectBorrowed<'_> {
        BVectBorrowed {
            dim: self.dim,
            data: &self.data,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BVectBorrowed<'a> {
    dim: u32,
    data: &'a [u64],
}

impl<'a> BVectBorrowed<'a> {
    #[inline(always)]
    pub fn new(dim: u32, data: &'a [u64]) -> Self {
        Self::new_checked(dim, data).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(dim: u32, data: &'a [u64]) -> Option<Self> {
        if !(1..=65535).contains(&dim) {
            return None;
        }
        if data.len() != dim.div_ceil(BVECTOR_WIDTH) as usize {
            return None;
        }
        if dim % BVECTOR_WIDTH != 0 && data[data.len() - 1] >> (dim % BVECTOR_WIDTH) != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        unsafe {
            Some(Self::new_unchecked(dim, data))
        }
    }

    /// # Safety
    ///
    /// * `dim` must be in `1..=65535`.
    /// * `data` must be of the correct length.
    /// * The padding bits must be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(dim: u32, data: &'a [u64]) -> Self {
        Self { dim, data }
    }

    #[inline(always)]
    pub fn data(&self) -> &'a [u64] {
        self.data
    }

    #[inline(always)]
    pub fn get(&self, index: u32) -> bool {
        assert!(index < self.dim);
        self.data[(index / BVECTOR_WIDTH) as usize] & (1 << (index % BVECTOR_WIDTH)) != 0
    }

    #[inline(always)]
    pub fn iter(self) -> impl Iterator<Item = bool> + 'a {
        let mut index = 0_u32;
        std::iter::from_fn(move || {
            if index < self.dim {
                let result = self.data[(index / BVECTOR_WIDTH) as usize]
                    & (1 << (index % BVECTOR_WIDTH))
                    != 0;
                index += 1;
                Some(result)
            } else {
                None
            }
        })
    }
}

impl VectorBorrowed for BVectBorrowed<'_> {
    type Owned = BVectOwned;

    #[inline(always)]
    fn dim(&self) -> u32 {
        self.dim
    }

    fn own(&self) -> BVectOwned {
        BVectOwned {
            dim: self.dim,
            data: self.data.to_vec(),
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        (simd::bit::reduce_sum_of_x(self.data) as f32).sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        Distance::from(-(simd::bit::reduce_sum_of_and(self.data, rhs.data) as f32))
    }

    #[inline(always)]
    fn operator_l2s(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_cos(self, _: Self) -> Distance {
        unimplemented!()
    }

    #[inline(always)]
    fn operator_hamming(self, rhs: Self) -> Distance {
        Distance::from(simd::bit::reduce_sum_of_xor(self.data, rhs.data) as f32)
    }

    #[inline(always)]
    fn operator_jaccard(self, rhs: Self) -> Distance {
        let (and, or) = simd::bit::reduce_sum_of_and_or(self.data, rhs.data);
        Distance::from(1.0 - (and as f32 / or as f32))
    }

    #[inline(always)]
    fn function_normalize(&self) -> BVectOwned {
        unimplemented!()
    }

    fn operator_add(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_sub(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_mul(&self, _: Self) -> Self::Owned {
        unimplemented!()
    }

    fn operator_and(&self, rhs: Self) -> Self::Owned {
        assert_eq!(self.dim, rhs.dim);
        let data = simd::bit::vector_and(self.data, rhs.data);
        BVectOwned::new(self.dim, data)
    }

    fn operator_or(&self, rhs: Self) -> Self::Owned {
        assert_eq!(self.dim, rhs.dim);
        let data = simd::bit::vector_or(self.data, rhs.data);
        BVectOwned::new(self.dim, data)
    }

    fn operator_xor(&self, rhs: Self) -> Self::Owned {
        assert_eq!(self.dim, rhs.dim);
        let data = simd::bit::vector_xor(self.data, rhs.data);
        BVectOwned::new(self.dim, data)
    }
}

impl PartialEq for BVectBorrowed<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.dim != other.dim {
            return false;
        }
        for (&l, &r) in self.data.iter().zip(other.data.iter()) {
            let l = l.reverse_bits();
            let r = r.reverse_bits();
            if l != r {
                return false;
            }
        }
        true
    }
}

impl PartialOrd for BVectBorrowed<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        if self.dim != other.dim {
            return None;
        }
        for (&l, &r) in self.data.iter().zip(other.data.iter()) {
            let l = l.reverse_bits();
            let r = r.reverse_bits();
            match l.cmp(&r) {
                Ordering::Equal => (),
                x => return Some(x),
            }
        }
        Some(Ordering::Equal)
    }
}
