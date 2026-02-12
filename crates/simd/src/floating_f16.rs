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

use crate::{F16, Floating, f16};

impl Floating for f16 {
    #[inline(always)]
    fn zero() -> Self {
        f16::_ZERO
    }

    #[inline(always)]
    fn infinity() -> Self {
        f16::INFINITY
    }

    #[inline(always)]
    fn mask(self, m: bool) -> Self {
        f16::from_bits(self.to_bits() & (m as u16).wrapping_neg())
    }

    #[inline(always)]
    fn scalar_neg(this: Self) -> Self {
        -this
    }

    #[inline(always)]
    fn scalar_add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline(always)]
    fn scalar_sub(lhs: Self, rhs: Self) -> Self {
        lhs - rhs
    }

    #[inline(always)]
    fn scalar_mul(lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }

    #[inline(always)]
    fn reduce_or_of_is_zero_x(this: &[f16]) -> bool {
        reduce_or_of_is_zero_x::reduce_or_of_is_zero_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x(this: &[f16]) -> f32 {
        reduce_sum_of_x::reduce_sum_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        reduce_sum_of_abs_x::reduce_sum_of_abs_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        reduce_sum_of_x2::reduce_sum_of_x2(this)
    }

    #[inline(always)]
    fn reduce_min_max_of_x(this: &[f16]) -> (f32, f32) {
        reduce_min_max_of_x::reduce_min_max_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_xy::reduce_sum_of_xy(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_d2(lhs: &[f16], rhs: &[f16]) -> f32 {
        reduce_sum_of_d2::reduce_sum_of_d2(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        reduce_sum_of_xy_sparse::reduce_sum_of_xy_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        reduce_sum_of_d2_sparse::reduce_sum_of_d2_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn vector_from_f32(this: &[f32]) -> Vec<Self> {
        vector_from_f32::vector_from_f32(this)
    }

    #[inline(always)]
    fn vector_to_f32(this: &[Self]) -> Vec<f32> {
        vector_to_f32::vector_to_f32(this)
    }

    #[inline(always)]
    fn vector_add(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_add::vector_add(lhs, rhs)
    }

    #[inline(always)]
    fn vector_add_inplace(lhs: &mut [Self], rhs: &[Self]) {
        vector_add_inplace::vector_add_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_sub(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_sub::vector_sub(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        vector_mul::vector_mul(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar(lhs: &[Self], rhs: f32) -> Vec<Self> {
        vector_mul_scalar::vector_mul_scalar(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar_inplace(lhs: &mut [Self], rhs: f32) {
        vector_mul_scalar_inplace::vector_mul_scalar_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]> {
        Self::vector_to_f32(this)
    }

    #[inline(always)]
    fn vector_abs_inplace(this: &mut [Self]) {
        vector_abs_inplace::vector_abs_inplace(this);
    }
}

mod reduce_or_of_is_zero_x {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_or_of_is_zero_x(this: &[f16]) -> bool {
        for &x in this {
            if x == f16::_ZERO {
                return true;
            }
        }
        false
    }
}

mod reduce_sum_of_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_add_ps(sum, x);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_add_ps(sum, x);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, x);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, x);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += this[i]._to_f32();
        }
        x
    }
}

mod reduce_sum_of_abs_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_abs_x_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_add_ps(sum, _mm512_abs_ps(x));
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_add_ps(sum, _mm512_abs_ps(x));
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_abs_x_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_abs_x_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let abs = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, _mm256_and_ps(abs, x));
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_add_ps(sum, _mm256_and_ps(abs, x));
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_abs_x_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += this[i]._to_f32().abs();
        }
        x
    }
}

mod reduce_sum_of_x2 {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x2_v4(this: &[f16]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            sum = _mm512_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            sum = _mm512_fmadd_ps(x, x, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x2_v4(&this) };
            let fallback = fallback(&this);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x2_v3(this: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            sum = _mm256_fmadd_ps(x, x, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x2_v3(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x2 = 0.0f32;
        for i in 0..n {
            x2 += this[i]._to_f32() * this[i]._to_f32();
        }
        x2
    }
}

mod reduce_min_max_of_x {
    use crate::{F16, f16};

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_min_max_of_x_v4(this: &[f16]) -> (f32, f32) {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm512_set1_ps(f32::INFINITY);
        let mut max = _mm512_set1_ps(f32::NEG_INFINITY);
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            min = _mm512_min_ps(x, min);
            max = _mm512_max_ps(x, max);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            min = _mm512_mask_min_ps(min, mask, x, min);
            max = _mm512_mask_max_ps(max, mask, x, max);
        }
        let min = _mm512_reduce_min_ps(min);
        let max = _mm512_reduce_max_ps(max);
        (min, max)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_v4_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v4(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0);
                assert_eq!(specialized.1, fallback.1);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_min_max_of_x_v3(this: &[f16]) -> (f32, f32) {
        use crate::emulate::{
            emulate_mm256_reduce_max_ps, emulate_mm256_reduce_min_ps, partial_load,
        };
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm256_set1_ps(f32::INFINITY);
        let mut max = _mm256_set1_ps(f32::NEG_INFINITY);
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(8, n, a = f16::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
        }
        (
            emulate_mm256_reduce_min_ps(min),
            emulate_mm256_reduce_max_ps(max),
        )
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_min_max_of_x_v3_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f16::NAN, -f16::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v3(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[crate::multiversion(
        @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_min_max_of_x(this: &[f16]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let n = this.len();
        for i in 0..n {
            min = min.min(this[i]._to_f32());
            max = max.max(this[i]._to_f32());
        }
        (min, max)
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    use crate::{F16, f16};

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    fn reduce_sum_of_xy_v4_avx512fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_xy_v4_avx512fp16"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v4_avx512fp16_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512fp16") {
            println!("test {} ... skipped (v4:avx512fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v4_avx512fp16(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            sum = _mm512_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            sum = _mm512_fmadd_ps(x, y, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xy_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        assert!(lhs.len() == rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            sum = _mm256_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            sum = _mm256_fmadd_ps(x, y, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.512")]
    fn reduce_sum_of_xy_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_xy_a3_512"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a3_512_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a3_512(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    fn reduce_sum_of_xy_a2_fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_xy_a2_fp16"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a2_fp16_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("fp16") {
            println!("test {} ... skipped (a2:fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a2_fp16(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4:avx512fp16", @"v4", @"v3", #[cfg(target_endian = "little")] @"a3.512", @"a2:fp16", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += lhs[i]._to_f32() * rhs[i]._to_f32();
        }
        sum
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2 {
    use crate::{F16, f16};

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    fn reduce_sum_of_d2_v4_avx512fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_d2_v4_avx512fp16"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_v4_avx512fp16_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512fp16") {
            println!("test {} ... skipped (v4:avx512fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v4_avx512fp16(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_d2_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v4(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_d2_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(8, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());

            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.512")]
    fn reduce_sum_of_d2_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_d2_a3_512"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_a3_512_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_a3_512(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    fn reduce_sum_of_d2_a2_fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp16_reduce_sum_of_d2_a2_fp16"]
            unsafe fn f(n: usize, a: *const f16, b: *const f16) -> f32;
        }
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr().cast();
        let b = rhs.as_ptr().cast();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_a2_fp16_test() {
        use rand::RngExt;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("fp16") {
            println!("test {} ... skipped (a2:fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_a2_fp16(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4:avx512fp16", @"v4", @"v3", #[cfg(target_endian = "little")] @"a3.512", @"a2:fp16", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_d2(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0_f32;
        for i in 0..n {
            let d = lhs[i]._to_f32() - rhs[i]._to_f32();
            sum += d * d;
        }
        sum
    }
}

mod reduce_sum_of_xy_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    sum += lval[lp]._to_f32() * rval[rp]._to_f32();
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    lp += 1;
                }
                Ordering::Greater => {
                    rp += 1;
                }
            }
        }
        sum
    }
}

mod reduce_sum_of_d2_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    let d = lval[lp]._to_f32() - rval[rp]._to_f32();
                    sum += d * d;
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    sum += lval[lp]._to_f32() * lval[lp]._to_f32();
                    lp += 1;
                }
                Ordering::Greater => {
                    sum += rval[rp]._to_f32() * rval[rp]._to_f32();
                    rp += 1;
                }
            }
        }
        for i in lp..ln {
            sum += lval[i]._to_f32() * lval[i]._to_f32();
        }
        for i in rp..rn {
            sum += rval[i]._to_f32() * rval[i]._to_f32();
        }
        sum
    }
}

mod vector_add {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] + rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_add_inplace {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add_inplace(lhs: &mut [f16], rhs: &[f16]) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        for i in 0..n {
            lhs[i] += rhs[i];
        }
    }
}

mod vector_sub {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_sub(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] - rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul {
    use crate::f16;

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul(lhs: &[f16], rhs: &[f16]) -> Vec<f16> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] * rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul_scalar {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar(lhs: &[f16], rhs: f32) -> Vec<f16> {
        let rhs = f16::_from_f32(rhs);
        let n = lhs.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] * rhs);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_mul_scalar_inplace {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar_inplace(lhs: &mut [f16], rhs: f32) {
        let rhs = f16::_from_f32(rhs);
        let n = lhs.len();
        for i in 0..n {
            lhs[i] *= rhs;
        }
    }
}

mod vector_abs_inplace {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_abs_inplace(this: &mut [f16]) {
        let n = this.len();
        for i in 0..n {
            this[i] = f16::_from_f32(this[i]._to_f32().abs());
        }
    }
}

mod vector_from_f32 {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_from_f32(this: &[f32]) -> Vec<f16> {
        let n = this.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(f16::_from_f32(this[i]));
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_to_f32 {
    use crate::{F16, f16};

    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_to_f32(this: &[f16]) -> Vec<f32> {
        let n = this.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(this[i]._to_f32());
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}
