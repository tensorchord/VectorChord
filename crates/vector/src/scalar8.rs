use crate::{VectorBorrowed, VectorOwned};
use distance::Distance;
use std::ops::RangeBounds;

#[derive(Debug, Clone)]
pub struct Scalar8Owned {
    sum_of_x2: f32,
    k: f32,
    b: f32,
    sum_of_code: f32,
    code: Vec<u8>,
}

impl Scalar8Owned {
    #[inline(always)]
    pub fn new(sum_of_x2: f32, k: f32, b: f32, sum_of_code: f32, code: Vec<u8>) -> Self {
        Self::new_checked(sum_of_x2, k, b, sum_of_code, code).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        sum_of_x2: f32,
        k: f32,
        b: f32,
        sum_of_code: f32,
        code: Vec<u8>,
    ) -> Option<Self> {
        if !(1..=65535).contains(&code.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(sum_of_x2, k, b, sum_of_code, code) })
    }

    /// # Safety
    ///
    /// * `code.len()` must not be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        sum_of_x2: f32,
        k: f32,
        b: f32,
        sum_of_code: f32,
        code: Vec<u8>,
    ) -> Self {
        Self {
            sum_of_x2,
            k,
            b,
            sum_of_code,
            code,
        }
    }
}

impl VectorOwned for Scalar8Owned {
    type Borrowed<'a> = Scalar8Borrowed<'a>;

    #[inline(always)]
    fn as_borrowed(&self) -> Scalar8Borrowed<'_> {
        Scalar8Borrowed {
            sum_of_x2: self.sum_of_x2,
            k: self.k,
            b: self.b,
            sum_of_code: self.sum_of_code,
            code: self.code.as_slice(),
        }
    }

    #[inline(always)]
    fn zero(dims: u32) -> Self {
        Self {
            sum_of_x2: 0.0,
            k: 0.0,
            b: 0.0,
            sum_of_code: 0.0,
            code: vec![0; dims as usize],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Scalar8Borrowed<'a> {
    sum_of_x2: f32,
    k: f32,
    b: f32,
    sum_of_code: f32,
    code: &'a [u8],
}

impl<'a> Scalar8Borrowed<'a> {
    #[inline(always)]
    pub fn new(sum_of_x2: f32, k: f32, b: f32, sum_of_code: f32, code: &'a [u8]) -> Self {
        Self::new_checked(sum_of_x2, k, b, sum_of_code, code).expect("invalid data")
    }

    #[inline(always)]
    pub fn new_checked(
        sum_of_x2: f32,
        k: f32,
        b: f32,
        sum_of_code: f32,
        code: &'a [u8],
    ) -> Option<Self> {
        if !(1..=65535).contains(&code.len()) {
            return None;
        }
        #[allow(unsafe_code)]
        Some(unsafe { Self::new_unchecked(sum_of_x2, k, b, sum_of_code, code) })
    }

    /// # Safety
    ///
    /// * `code.len()` must not be zero.
    #[allow(unsafe_code)]
    #[inline(always)]
    pub unsafe fn new_unchecked(
        sum_of_x2: f32,
        k: f32,
        b: f32,
        sum_of_code: f32,
        code: &'a [u8],
    ) -> Self {
        Self {
            sum_of_x2,
            k,
            b,
            sum_of_code,
            code,
        }
    }

    #[inline(always)]
    pub fn sum_of_x2(&self) -> f32 {
        self.sum_of_x2
    }

    #[inline(always)]
    pub fn k(&self) -> f32 {
        self.k
    }

    #[inline(always)]
    pub fn b(&self) -> f32 {
        self.b
    }

    #[inline(always)]
    pub fn sum_of_code(&self) -> f32 {
        self.sum_of_code
    }

    #[inline(always)]
    pub fn code(&self) -> &'a [u8] {
        self.code
    }
}

impl VectorBorrowed for Scalar8Borrowed<'_> {
    type Owned = Scalar8Owned;

    #[inline(always)]
    fn dims(&self) -> u32 {
        self.code.len() as u32
    }

    #[inline(always)]
    fn own(&self) -> Scalar8Owned {
        Scalar8Owned {
            sum_of_x2: self.sum_of_x2,
            k: self.k,
            b: self.b,
            sum_of_code: self.sum_of_code,
            code: self.code.to_owned(),
        }
    }

    #[inline(always)]
    fn norm(&self) -> f32 {
        self.sum_of_x2.sqrt()
    }

    #[inline(always)]
    fn operator_dot(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let xy = self.k * rhs.k * simd::u8::reduce_sum_of_xy(self.code, rhs.code) as f32
            + self.b * rhs.b * self.code.len() as f32
            + self.k * rhs.b * self.sum_of_code
            + self.b * rhs.k * rhs.sum_of_code;
        Distance::from(-xy)
    }

    #[inline(always)]
    fn operator_l2(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let xy = self.k * rhs.k * simd::u8::reduce_sum_of_xy(self.code, rhs.code) as f32
            + self.b * rhs.b * self.code.len() as f32
            + self.k * rhs.b * self.sum_of_code
            + self.b * rhs.k * rhs.sum_of_code;
        let x2 = self.sum_of_x2;
        let y2 = rhs.sum_of_x2;
        Distance::from(x2 + y2 - 2.0 * xy)
    }

    #[inline(always)]
    fn operator_cos(self, rhs: Self) -> Distance {
        assert_eq!(self.code.len(), rhs.code.len());
        let xy = self.k * rhs.k * simd::u8::reduce_sum_of_xy(self.code, rhs.code) as f32
            + self.b * rhs.b * self.code.len() as f32
            + self.k * rhs.b * self.sum_of_code
            + self.b * rhs.k * rhs.sum_of_code;
        let x2 = self.sum_of_x2;
        let y2 = rhs.sum_of_x2;
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
    fn function_normalize(&self) -> Scalar8Owned {
        let l = self.sum_of_x2.sqrt();
        Scalar8Owned {
            sum_of_x2: 1.0,
            k: self.k / l,
            b: self.b / l,
            sum_of_code: self.sum_of_code,
            code: self.code.to_owned(),
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

    #[inline(always)]
    fn subvector(&self, bounds: impl RangeBounds<u32>) -> Option<Self::Owned> {
        let start_bound = bounds.start_bound().map(|x| *x as usize);
        let end_bound = bounds.end_bound().map(|x| *x as usize);
        let code = self.code.get((start_bound, end_bound))?;
        if code.is_empty() {
            return None;
        }
        Self::Owned::new_checked(
            {
                // recover it as much as possible
                let mut result = 0.0;
                for &x in code {
                    let y = self.k * (x as f32) + self.b;
                    result += y * y;
                }
                result
            },
            self.k,
            self.b,
            simd::u8::reduce_sum_of_x(code) as f32,
            code.to_owned(),
        )
    }
}
