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

pub mod bvect;
pub mod scalar8;
pub mod svect;
pub mod vect;

pub trait VectorOwned: Clone + 'static {
    type Borrowed<'a>: VectorBorrowed<Owned = Self>;

    fn as_borrowed(&self) -> Self::Borrowed<'_>;

    fn zero(dims: u32) -> Self;
}

pub trait VectorBorrowed: Copy {
    type Owned: VectorOwned;

    fn own(&self) -> Self::Owned;

    fn dims(&self) -> u32;

    fn norm(&self) -> f32;

    fn operator_dot(self, rhs: Self) -> distance::Distance;

    fn operator_l2(self, rhs: Self) -> distance::Distance;

    fn operator_cos(self, rhs: Self) -> distance::Distance;

    fn operator_hamming(self, rhs: Self) -> distance::Distance;

    fn operator_jaccard(self, rhs: Self) -> distance::Distance;

    fn function_normalize(&self) -> Self::Owned;

    fn operator_add(&self, rhs: Self) -> Self::Owned;

    fn operator_sub(&self, rhs: Self) -> Self::Owned;

    fn operator_mul(&self, rhs: Self) -> Self::Owned;

    fn operator_and(&self, rhs: Self) -> Self::Owned;

    fn operator_or(&self, rhs: Self) -> Self::Owned;

    fn operator_xor(&self, rhs: Self) -> Self::Owned;

    fn subvector(&self, bounds: impl std::ops::RangeBounds<u32>) -> Option<Self::Owned>;
}
