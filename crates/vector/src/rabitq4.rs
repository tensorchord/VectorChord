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

#[derive(Debug, Clone)]
pub struct Rabitq4Owned {
    dim: u32,
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    packed_code: Vec<u8>,
}

impl Rabitq4Owned {
    #[inline(always)]
    pub fn new(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Self {
        Self::new_checked(
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        )
        .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Option<Self> {
        if !(1..=65535).contains(&dim) {
            return None;
        }
        if dim.div_ceil(2) as usize != packed_code.len() {
            return None;
        }
        if dim % 2 == 1 && packed_code.last().copied().unwrap_or_default() >> 4 != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(
                dim,
                sum_of_x2,
                norm_of_lattice,
                sum_of_code,
                sum_of_abs_x,
                packed_code,
            )
        })
    }

    /// # Safety
    ///
    /// * `dim` must not be zero.
    /// * `dim` must be less than 65536.
    /// * `dim` must be equal to 1/2 of `code.len()`, rounding to infinity.
    /// * `packed_code` is filled with zero bits correctly.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: Vec<u8>,
    ) -> Self {
        Self {
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        }
    }
}

impl VectorOwned for Rabitq4Owned {
    type Borrowed<'a> = Rabitq4Borrowed<'a>;

    #[inline(always)]
    fn as_borrowed(&self) -> Rabitq4Borrowed<'_> {
        Rabitq4Borrowed {
            dim: self.dim,
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            packed_code: self.packed_code.as_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rabitq4Borrowed<'a> {
    dim: u32,
    sum_of_x2: f32,
    norm_of_lattice: f32,
    sum_of_code: f32,
    sum_of_abs_x: f32,
    packed_code: &'a [u8],
}

impl<'a> Rabitq4Borrowed<'a> {
    #[inline(always)]
    pub fn new(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Self {
        Self::new_checked(
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        )
        .expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Option<Self> {
        if !(1..=65535).contains(&packed_code.len()) {
            return None;
        }
        if dim.div_ceil(2) as usize != packed_code.len() {
            return None;
        }
        if dim % 2 == 1 && packed_code.last().copied().unwrap_or_default() >> 4 != 0 {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe {
            Self::new_unchecked(
                dim,
                sum_of_x2,
                norm_of_lattice,
                sum_of_code,
                sum_of_abs_x,
                packed_code,
            )
        })
    }

    /// # Safety
    ///
    /// * `dim` must not be zero.
    /// * `dim` must be less than 65536.
    /// * `dim` must be equal to 1/2 of `code.len()`, rounding to infinity.
    /// * `packed_code` is filled with zero bits correctly.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        dim: u32,
        sum_of_x2: f32,
        norm_of_lattice: f32,
        sum_of_code: f32,
        sum_of_abs_x: f32,
        packed_code: &'a [u8],
    ) -> Self {
        Self {
            dim,
            sum_of_x2,
            norm_of_lattice,
            sum_of_code,
            sum_of_abs_x,
            packed_code,
        }
    }

    #[inline(always)]
    pub fn sum_of_x2(&self) -> f32 {
        self.sum_of_x2
    }

    #[inline(always)]
    pub fn norm_of_lattice(&self) -> f32 {
        self.norm_of_lattice
    }

    #[inline(always)]
    pub fn sum_of_code(&self) -> f32 {
        self.sum_of_code
    }

    #[inline(always)]
    pub fn sum_of_abs_x(&self) -> f32 {
        self.sum_of_abs_x
    }

    #[inline(always)]
    pub fn packed_code(&self) -> &'a [u8] {
        self.packed_code
    }

    #[inline(always)]
    pub fn unpacked_code(&self) -> impl Iterator<Item = u8> {
        self.packed_code
            .iter()
            .flat_map(|x| [x & 0xf, x >> 4])
            .take(self.dim as _)
    }
}

impl VectorBorrowed for Rabitq4Borrowed<'_> {
    type Owned = Rabitq4Owned;

    #[inline(always)]
    fn dim(&self) -> u32 {
        self.dim
    }

    #[inline(always)]
    fn own(&self) -> Rabitq4Owned {
        Rabitq4Owned {
            dim: self.dim,
            sum_of_x2: self.sum_of_x2,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x,
            packed_code: self.packed_code.to_owned(),
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        self.sum_of_x2.sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        assert_eq!(self.packed_code.len(), rhs.packed_code.len());
        let n = self.packed_code.len() as u32;
        let value = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_dot(
                n,
                value,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_l2s(self, rhs: Self) -> Distance {
        assert_eq!(self.packed_code.len(), rhs.packed_code.len());
        let n = self.packed_code.len() as u32;
        let value = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_l2(
                n,
                value,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        assert_eq!(self.packed_code.len(), rhs.packed_code.len());
        let n = self.packed_code.len() as u32;
        let value = rabitq::halfbyte::binary::accumulate(self.packed_code, rhs.packed_code);
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_cos(
                n,
                value,
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: self.sum_of_x2,
                    norm_of_lattice: self.norm_of_lattice,
                    sum_of_code: self.sum_of_code,
                },
                rabitq::halfbyte::CodeMetadata {
                    dis_u_2: rhs.sum_of_x2,
                    norm_of_lattice: rhs.norm_of_lattice,
                    sum_of_code: rhs.sum_of_code,
                },
            )
            .0,
        )
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
    fn function_normalize(&self) -> Rabitq4Owned {
        Rabitq4Owned {
            dim: self.dim,
            sum_of_x2: 1.0,
            norm_of_lattice: self.norm_of_lattice,
            sum_of_code: self.sum_of_code,
            sum_of_abs_x: self.sum_of_abs_x / self.sum_of_x2.sqrt(),
            packed_code: self.packed_code.to_owned(),
        }
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
