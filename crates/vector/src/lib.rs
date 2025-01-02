pub mod bvect;
pub mod scalar8;
pub mod svect;
pub mod vect;

pub trait VectorOwned: Clone + serde::Serialize + for<'a> serde::Deserialize<'a> + 'static {
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
