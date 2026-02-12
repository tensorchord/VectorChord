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

use crate::Floating;

impl Floating for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0f32
    }

    #[inline(always)]
    fn infinity() -> Self {
        f32::INFINITY
    }

    #[inline(always)]
    fn mask(self, m: bool) -> Self {
        f32::from_bits(self.to_bits() & (m as u32).wrapping_neg())
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
    fn reduce_or_of_is_zero_x(this: &[f32]) -> bool {
        reduce_or_of_is_zero_x::reduce_or_of_is_zero_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x(this: &[f32]) -> f32 {
        reduce_sum_of_x::reduce_sum_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_abs_x(this: &[f32]) -> f32 {
        reduce_sum_of_abs_x::reduce_sum_of_abs_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_x2(this: &[f32]) -> f32 {
        reduce_sum_of_x2::reduce_sum_of_x2(this)
    }

    #[inline(always)]
    fn reduce_min_max_of_x(this: &[f32]) -> (f32, f32) {
        reduce_min_max_of_x::reduce_min_max_of_x(this)
    }

    #[inline(always)]
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_xy::reduce_sum_of_xy(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_d2(lhs: &[Self], rhs: &[Self]) -> f32 {
        reduce_sum_of_d2::reduce_sum_of_d2(lhs, rhs)
    }

    #[inline(always)]
    fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f32], ridx: &[u32], rval: &[f32]) -> f32 {
        reduce_sum_of_xy_sparse::reduce_sum_of_xy_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f32], ridx: &[u32], rval: &[f32]) -> f32 {
        reduce_sum_of_d2_sparse::reduce_sum_of_d2_sparse(lidx, lval, ridx, rval)
    }

    #[inline(always)]
    fn vector_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_add::vector_add(lhs, rhs)
    }

    #[inline(always)]
    fn vector_add_inplace(lhs: &mut [f32], rhs: &[f32]) {
        vector_add_inplace::vector_add_inplace(lhs, rhs)
    }

    #[inline(always)]
    fn vector_sub(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_sub::vector_sub(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        vector_mul::vector_mul(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar(lhs: &[f32], rhs: f32) -> Vec<f32> {
        vector_mul_scalar::vector_mul_scalar(lhs, rhs)
    }

    #[inline(always)]
    fn vector_mul_scalar_inplace(lhs: &mut [f32], rhs: f32) {
        vector_mul_scalar_inplace::vector_mul_scalar_inplace(lhs, rhs);
    }

    #[inline(always)]
    fn vector_from_f32(this: &[f32]) -> Vec<f32> {
        this.to_vec()
    }

    #[inline(always)]
    fn vector_to_f32(this: &[f32]) -> Vec<f32> {
        this.to_vec()
    }

    #[inline(always)]
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]> {
        this
    }

    #[inline(always)]
    fn vector_abs_inplace(this: &mut [Self]) {
        vector_abs_inplace::vector_abs_inplace(this);
    }
}

mod reduce_or_of_is_zero_x {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn reduce_or_of_is_zero_x(this: &[f32]) -> bool {
        for &x in this {
            if x == 0.0f32 {
                return true;
            }
        }
        false
    }
}

mod reduce_sum_of_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            sum = _mm512_add_ps(x, sum);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            sum = _mm512_add_ps(x, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v4(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            sum = _mm256_add_ps(x, sum);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n >= 4 {
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            sum = _mm256_add_ps(x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            sum = _mm256_add_ps(x, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_x_v2(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_ps();
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            sum = _mm_add_ps(x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm_loadu_ps(a) };
            sum = _mm_add_ps(x, sum);
        }
        emulate_mm_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v2(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_a2(this: &[f32]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = vdupq_n_f32(0.0);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            sum = vaddq_f32(x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { vld1q_f32(a) };
            sum = vaddq_f32(x, sum);
        }
        vaddvq_f32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_a2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_a2(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x(this: &[f32]) -> f32 {
        let n = this.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += this[i];
        }
        sum
    }
}

mod reduce_sum_of_abs_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_abs_x_v4(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            let abs_x = _mm512_abs_ps(x);
            sum = _mm512_add_ps(abs_x, sum);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            let abs_x = _mm512_abs_ps(x);
            sum = _mm512_add_ps(abs_x, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_abs_x_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.009;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_abs_x_v4(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_abs_x_v3(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let abs = _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_set1_epi32(-1), 1));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            let abs_x = _mm256_and_ps(abs, x);
            sum = _mm256_add_ps(abs_x, sum);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n >= 4 {
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let abs_x = _mm256_and_ps(abs, x);
            sum = _mm256_add_ps(abs_x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let abs_x = _mm256_and_ps(abs, x);
            sum = _mm256_add_ps(abs_x, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_abs_x_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.009;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_abs_x_v2(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let abs = _mm_castsi128_ps(_mm_srli_epi32(_mm_set1_epi32(-1), 1));
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_ps();
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            let abs_x = _mm_and_ps(abs, x);
            sum = _mm_add_ps(abs_x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm_loadu_ps(a) };
            let abs_x = _mm_and_ps(abs, x);
            sum = _mm_add_ps(abs_x, sum);
        }
        emulate_mm_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_abs_x_v2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.009;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_abs_x_v2(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_abs_x_a2(this: &[f32]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = vdupq_n_f32(0.0);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            let abs_x = vabsq_f32(x);
            sum = vaddq_f32(abs_x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { vld1q_f32(a) };
            let abs_x = vabsq_f32(x);
            sum = vaddq_f32(abs_x, sum);
        }
        vaddvq_f32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_abs_x_a2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.009;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_abs_x_a2(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_abs_x(this: &[f32]) -> f32 {
        let n = this.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += this[i].abs();
        }
        sum
    }
}

mod reduce_sum_of_x2 {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x2_v4(this: &[f32]) -> f32 {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            sum = _mm512_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            sum = _mm512_fmadd_ps(x, x, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x2_v4(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x2_v3(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            sum = _mm256_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n >= 4 {
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            sum = _mm256_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            sum = _mm256_fmadd_ps(x, x, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_x2_v2_fma(this: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm_reduce_add_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_ps();
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            sum = _mm_fmadd_ps(x, x, sum);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm_loadu_ps(a) };
            sum = _mm_fmadd_ps(x, x, sum);
        }
        emulate_mm_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x2_v2_fma_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("v2") || !crate::is_feature_detected!("fma") {
            println!("test {} ... skipped (v2:fma)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x2_v2_fma(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x2_a2(this: &[f32]) -> f32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = vdupq_n_f32(0.0);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            sum = vfmaq_f32(sum, x, x);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { vld1q_f32(a) };
            sum = vfmaq_f32(sum, x, x);
        }
        vaddvq_f32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x2_a2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.008;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x2_a2(this) };
                let fallback = fallback(this);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x2(this: &[f32]) -> f32 {
        let n = this.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += this[i] * this[i];
        }
        sum
    }
}

mod reduce_min_max_of_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_min_max_of_x_v4(this: &[f32]) -> (f32, f32) {
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm512_set1_ps(f32::INFINITY);
        let mut max = _mm512_set1_ps(f32::NEG_INFINITY);
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            min = _mm512_min_ps(x, min);
            max = _mm512_max_ps(x, max);
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
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
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f32::NAN, -f32::NAN);
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
    fn reduce_min_max_of_x_v3(this: &[f32]) -> (f32, f32) {
        use crate::emulate::{
            emulate_mm256_reduce_max_ps, emulate_mm256_reduce_min_ps, partial_load,
        };
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm256_set1_ps(f32::INFINITY);
        let mut max = _mm256_set1_ps(f32::NEG_INFINITY);
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
            (n, a) = unsafe { (n - 8, a.add(8)) };
        }
        if n >= 4 {
            let x = _mm256_setr_m128(unsafe { _mm_loadu_ps(a) }, _mm_set1_ps(f32::NAN));
            min = _mm256_min_ps(x, min);
            max = _mm256_max_ps(x, max);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a = f32::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = _mm256_setr_m128(unsafe { _mm_loadu_ps(a) }, _mm_set1_ps(f32::NAN));
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
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f32::NAN, -f32::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v3(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_min_max_of_x_v2(this: &[f32]) -> (f32, f32) {
        use crate::emulate::{emulate_mm_reduce_max_ps, emulate_mm_reduce_min_ps, partial_load};
        use core::arch::x86_64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = _mm_set1_ps(f32::INFINITY);
        let mut max = _mm_set1_ps(f32::NEG_INFINITY);
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            min = _mm_min_ps(x, min);
            max = _mm_max_ps(x, max);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a = f32::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm_loadu_ps(a) };
            min = _mm_min_ps(x, min);
            max = _mm_max_ps(x, max);
        }
        (emulate_mm_reduce_min_ps(min), emulate_mm_reduce_max_ps(max))
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_min_max_of_x_v2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f32::NAN, -f32::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_v2(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_min_max_of_x_a2(this: &[f32]) -> (f32, f32) {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut min = vdupq_n_f32(f32::INFINITY);
        let mut max = vdupq_n_f32(f32::NEG_INFINITY);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            min = vminnmq_f32(x, min);
            max = vmaxnmq_f32(x, max);
            (n, a) = unsafe { (n - 4, a.add(4)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(4, n, a = f32::NAN) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { vld1q_f32(a) };
            min = vminnmq_f32(x, min);
            max = vmaxnmq_f32(x, max);
        }
        (vminnmvq_f32(min), vmaxnmvq_f32(max))
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_min_max_of_x_a2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 200;
            let mut x = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            (x[0], x[1]) = (f32::NAN, -f32::NAN);
            for z in 50..200 {
                let x = &x[..z];
                let specialized = unsafe { reduce_min_max_of_x_a2(x) };
                let fallback = fallback(x);
                assert_eq!(specialized.0, fallback.0,);
                assert_eq!(specialized.1, fallback.1,);
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_min_max_of_x(this: &[f32]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let n = this.len();
        for i in 0..n {
            min = min.min(this[i]);
            max = max.max(this[i]);
        }
        (min, max)
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            let y = unsafe { _mm512_loadu_ps(b) };
            sum = _mm512_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            let y = unsafe { _mm512_maskz_loadu_ps(mask, b) };
            sum = _mm512_fmadd_ps(x, y, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.004;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v4(lhs, rhs) };
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
    fn reduce_sum_of_xy_v3(lhs: &[f32], rhs: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            let y = unsafe { _mm256_loadu_ps(b) };
            sum = _mm256_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n >= 4 {
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let y = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(b)) };
            sum = _mm256_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let y = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(b)) };
            sum = _mm256_fmadd_ps(x, y, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.004;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_xy_v2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm_setzero_ps();
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            let y = unsafe { _mm_loadu_ps(b) };
            sum = _mm_fmadd_ps(x, y, sum);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm_loadu_ps(a) };
            let y = unsafe { _mm_loadu_ps(b) };
            sum = _mm_fmadd_ps(x, y, sum);
        }
        emulate_mm_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v2_fma_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.004;
        if !crate::is_cpu_detected!("v2") || !crate::is_feature_detected!("fma") {
            println!("test {} ... skipped (v2:fma)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v2_fma(lhs, rhs) };
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
    #[crate::target_cpu(enable = "a3.256")]
    fn reduce_sum_of_xy_a3_256(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp32_reduce_sum_of_xy_a3_256"]
            unsafe fn f(n: usize, a: *const f32, b: *const f32) -> f32;
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
    fn reduce_sum_of_xy_a3_256_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.004;
        if !crate::is_cpu_detected!("a3.256") {
            println!("test {} ... skipped (a3.256)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a3_256(lhs, rhs) };
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
    fn reduce_sum_of_xy_a2(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = vdupq_n_f32(0.0);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            let y = unsafe { vld1q_f32(b) };
            sum = vfmaq_f32(sum, x, y);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { vld1q_f32(a) };
            let y = unsafe { vld1q_f32(b) };
            sum = vfmaq_f32(sum, x, y);
        }
        vaddvq_f32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.004;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", #[cfg(target_endian = "little")] @"a3.256", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += lhs[i] * rhs[i];
        }
        sum
    }
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_d2 {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_d2_v4(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_loadu_ps(a) };
            let y = unsafe { _mm512_loadu_ps(b) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_maskz_loadu_ps(mask, a) };
            let y = unsafe { _mm512_maskz_loadu_ps(mask, b) };
            let d = _mm512_sub_ps(x, y);
            sum = _mm512_fmadd_ps(d, d, sum);
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_d2_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.02;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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
    fn reduce_sum_of_d2_v3(lhs: &[f32], rhs: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm256_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_loadu_ps(a) };
            let y = unsafe { _mm256_loadu_ps(b) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 8, a.add(8), b.add(8)) };
        }
        if n >= 4 {
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let y = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(b)) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(a)) };
            let y = unsafe { _mm256_zextps128_ps256(_mm_loadu_ps(b)) };
            let d = _mm256_sub_ps(x, y);
            sum = _mm256_fmadd_ps(d, d, sum);
        }
        emulate_mm256_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_d2_v3_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.02;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
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
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    #[target_feature(enable = "fma")]
    fn reduce_sum_of_d2_v2_fma(lhs: &[f32], rhs: &[f32]) -> f32 {
        use crate::emulate::{emulate_mm_reduce_add_ps, partial_load};
        assert!(lhs.len() == rhs.len());
        use core::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = _mm_setzero_ps();
        while n >= 4 {
            let x = unsafe { _mm_loadu_ps(a) };
            let y = unsafe { _mm_loadu_ps(b) };
            let d = _mm_sub_ps(x, y);
            sum = _mm_fmadd_ps(d, d, sum);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm_loadu_ps(a) };
            let y = unsafe { _mm_loadu_ps(b) };
            let d = _mm_sub_ps(x, y);
            sum = _mm_fmadd_ps(d, d, sum);
        }
        emulate_mm_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_d2_v2_fma_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.02;
        if !crate::is_cpu_detected!("v2") || !crate::is_feature_detected!("fma") {
            println!("test {} ... skipped (v2:fma)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_v2_fma(lhs, rhs) };
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
    #[crate::target_cpu(enable = "a3.256")]
    fn reduce_sum_of_d2_a3_256(lhs: &[f32], rhs: &[f32]) -> f32 {
        unsafe extern "C" {
            #[link_name = "fp32_reduce_sum_of_d2_a3_256"]
            unsafe fn f(n: usize, a: *const f32, b: *const f32) -> f32;
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
    fn reduce_sum_of_d2_a3_256_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.02;
        if !crate::is_cpu_detected!("a3.256") {
            println!("test {} ... skipped (a3.256)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_a3_256(lhs, rhs) };
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
    fn reduce_sum_of_d2_a2(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut sum = vdupq_n_f32(0.0);
        while n >= 4 {
            let x = unsafe { vld1q_f32(a) };
            let y = unsafe { vld1q_f32(b) };
            let d = vsubq_f32(x, y);
            sum = vfmaq_f32(sum, d, d);
            (n, a, b) = unsafe { (n - 4, a.add(4), b.add(4)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(4, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { vld1q_f32(a) };
            let y = unsafe { vld1q_f32(b) };
            let d = vsubq_f32(x, y);
            sum = vfmaq_f32(sum, d, d);
        }
        vaddvq_f32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_a2_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.02;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_d2_a2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    (specialized - fallback).abs() < EPSILON,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2:fma", #[cfg(target_endian = "little")] @"a3.256", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_d2(lhs: &[f32], rhs: &[f32]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut sum = 0.0f32;
        for i in 0..n {
            let d = lhs[i] - rhs[i];
            sum += d * d;
        }
        sum
    }
}

mod reduce_sum_of_xy_sparse {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_sparse_v4(li: &[u32], lv: &[f32], ri: &[u32], rv: &[f32]) -> f32 {
        use crate::emulate::emulate_mm512_2intersect_epi32;
        assert_eq!(li.len(), lv.len());
        assert_eq!(ri.len(), rv.len());
        let (mut lp, ln) = (0, li.len());
        let (mut rp, rn) = (0, ri.len());
        let (li, lv) = (li.as_ptr(), lv.as_ptr());
        let (ri, rv) = (ri.as_ptr(), rv.as_ptr());
        use core::arch::x86_64::*;
        let mut sum = _mm512_setzero_ps();
        while lp + 16 <= ln && rp + 16 <= rn {
            let lx = unsafe { _mm512_loadu_epi32(li.add(lp).cast()) };
            let rx = unsafe { _mm512_loadu_epi32(ri.add(rp).cast()) };
            let (lk, rk) = emulate_mm512_2intersect_epi32(lx, rx);
            let lv = unsafe { _mm512_maskz_compress_ps(lk, _mm512_loadu_ps(lv.add(lp))) };
            let rv = unsafe { _mm512_maskz_compress_ps(rk, _mm512_loadu_ps(rv.add(rp))) };
            sum = _mm512_fmadd_ps(lv, rv, sum);
            let lt = unsafe { li.add(lp + 16 - 1).read() };
            let rt = unsafe { ri.add(rp + 16 - 1).read() };
            lp += (lt <= rt) as usize * 16;
            rp += (lt >= rt) as usize * 16;
        }
        while lp < ln && rp < rn {
            let lw = 16.min(ln - lp);
            let rw = 16.min(rn - rp);
            let lm = _bzhi_u32(0xffff, lw as _) as u16;
            let rm = _bzhi_u32(0xffff, rw as _) as u16;
            let lx =
                unsafe { _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), lm, li.add(lp).cast()) };
            let rx =
                unsafe { _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), rm, ri.add(rp).cast()) };
            let (lk, rk) = emulate_mm512_2intersect_epi32(lx, rx);
            let lv = unsafe { _mm512_maskz_compress_ps(lk, _mm512_maskz_loadu_ps(lm, lv.add(lp))) };
            let rv = unsafe { _mm512_maskz_compress_ps(rk, _mm512_maskz_loadu_ps(rm, rv.add(rp))) };
            sum = _mm512_fmadd_ps(lv, rv, sum);
            let lt = unsafe { li.add(lp + lw - 1).read() };
            let rt = unsafe { ri.add(rp + rw - 1).read() };
            lp += (lt <= rt) as usize * lw;
            rp += (lt >= rt) as usize * rw;
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_sparse_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.000001;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        pub fn sample_u32_sorted(
            rng: &mut (impl RngExt + ?Sized),
            length: u32,
            amount: u32,
        ) -> Vec<u32> {
            let mut x = match rand::seq::index::sample(rng, length as usize, amount as usize) {
                rand::seq::index::IndexVec::U32(x) => x,
                _ => unreachable!(),
            };
            x.sort();
            x
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lm = 300;
            let lidx = sample_u32_sorted(&mut rng, 10000, lm);
            let lval = (0..lm)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rm = 350;
            let ridx = sample_u32_sorted(&mut rng, 10000, rm);
            let rval = (0..rm)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xy_sparse_v4(&lidx, &lval, &ridx, &rval) };
            let fallback = fallback(&lidx, &lval, &ridx, &rval);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[crate::multiversion(@"v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f32], ridx: &[u32], rval: &[f32]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    sum += lval[lp] * rval[rp];
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
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_d2_sparse_v4(li: &[u32], lv: &[f32], ri: &[u32], rv: &[f32]) -> f32 {
        use crate::emulate::emulate_mm512_2intersect_epi32;
        assert_eq!(li.len(), lv.len());
        assert_eq!(ri.len(), rv.len());
        let (mut lp, ln) = (0, li.len());
        let (mut rp, rn) = (0, ri.len());
        let (li, lv) = (li.as_ptr(), lv.as_ptr());
        let (ri, rv) = (ri.as_ptr(), rv.as_ptr());
        use core::arch::x86_64::*;
        let mut sum = _mm512_setzero_ps();
        while lp + 16 <= ln && rp + 16 <= rn {
            let lx = unsafe { _mm512_loadu_epi32(li.add(lp).cast()) };
            let rx = unsafe { _mm512_loadu_epi32(ri.add(rp).cast()) };
            let (lk, rk) = emulate_mm512_2intersect_epi32(lx, rx);
            let lv = unsafe { _mm512_maskz_compress_ps(lk, _mm512_loadu_ps(lv.add(lp))) };
            let rv = unsafe { _mm512_maskz_compress_ps(rk, _mm512_loadu_ps(rv.add(rp))) };
            let d = _mm512_sub_ps(lv, rv);
            sum = _mm512_fmadd_ps(d, d, sum);
            sum = _mm512_sub_ps(sum, _mm512_mul_ps(lv, lv));
            sum = _mm512_sub_ps(sum, _mm512_mul_ps(rv, rv));
            let lt = unsafe { li.add(lp + 16 - 1).read() };
            let rt = unsafe { ri.add(rp + 16 - 1).read() };
            lp += (lt <= rt) as usize * 16;
            rp += (lt >= rt) as usize * 16;
        }
        while lp < ln && rp < rn {
            let lw = 16.min(ln - lp);
            let rw = 16.min(rn - rp);
            let lm = _bzhi_u32(0xffff, lw as _) as u16;
            let rm = _bzhi_u32(0xffff, rw as _) as u16;
            let lx =
                unsafe { _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), lm, li.add(lp).cast()) };
            let rx =
                unsafe { _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), rm, ri.add(rp).cast()) };
            let (lk, rk) = emulate_mm512_2intersect_epi32(lx, rx);
            let lv = unsafe { _mm512_maskz_compress_ps(lk, _mm512_maskz_loadu_ps(lm, lv.add(lp))) };
            let rv = unsafe { _mm512_maskz_compress_ps(rk, _mm512_maskz_loadu_ps(rm, rv.add(rp))) };
            let d = _mm512_sub_ps(lv, rv);
            sum = _mm512_fmadd_ps(d, d, sum);
            sum = _mm512_sub_ps(sum, _mm512_mul_ps(lv, lv));
            sum = _mm512_sub_ps(sum, _mm512_mul_ps(rv, rv));
            let lt = unsafe { li.add(lp + lw - 1).read() };
            let rt = unsafe { ri.add(rp + rw - 1).read() };
            lp += (lt <= rt) as usize * lw;
            rp += (lt >= rt) as usize * rw;
        }
        {
            let mut lp = 0;
            while lp + 16 <= ln {
                let d = unsafe { _mm512_loadu_ps(lv.add(lp)) };
                sum = _mm512_fmadd_ps(d, d, sum);
                lp += 16;
            }
            if lp < ln {
                let lw = ln - lp;
                let lm = _bzhi_u32(0xffff, lw as _) as u16;
                let d = unsafe { _mm512_maskz_loadu_ps(lm, lv.add(lp)) };
                sum = _mm512_fmadd_ps(d, d, sum);
            }
        }
        {
            let mut rp = 0;
            while rp + 16 <= rn {
                let d = unsafe { _mm512_loadu_ps(rv.add(rp)) };
                sum = _mm512_fmadd_ps(d, d, sum);
                rp += 16;
            }
            if rp < rn {
                let rw = rn - rp;
                let rm = _bzhi_u32(0xffff, rw as _) as u16;
                let d = unsafe { _mm512_maskz_loadu_ps(rm, rv.add(rp)) };
                sum = _mm512_fmadd_ps(d, d, sum);
            }
        }
        _mm512_reduce_add_ps(sum)
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_d2_sparse_v4_test() {
        use rand::RngExt;
        const EPSILON: f32 = 0.0004;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        pub fn sample_u32_sorted(
            rng: &mut (impl RngExt + ?Sized),
            length: u32,
            amount: u32,
        ) -> Vec<u32> {
            let mut x = match rand::seq::index::sample(rng, length as usize, amount as usize) {
                rand::seq::index::IndexVec::U32(x) => x,
                _ => unreachable!(),
            };
            x.sort();
            x
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lm = 300;
            let lidx = sample_u32_sorted(&mut rng, 10000, lm);
            let lval = (0..lm)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let rm = 350;
            let ridx = sample_u32_sorted(&mut rng, 10000, rm);
            let rval = (0..rm)
                .map(|_| rng.random_range(-1.0..=1.0))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_d2_sparse_v4(&lidx, &lval, &ridx, &rval) };
            let fallback = fallback(&lidx, &lval, &ridx, &rval);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[crate::multiversion(@"v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f32], ridx: &[u32], rval: &[f32]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut sum = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    let d = lval[lp] - rval[rp];
                    sum += d * d;
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    sum += lval[lp] * lval[lp];
                    lp += 1;
                }
                Ordering::Greater => {
                    sum += rval[rp] * rval[rp];
                    rp += 1;
                }
            }
        }
        for i in lp..ln {
            sum += lval[i] * lval[i];
        }
        for i in rp..rn {
            sum += rval[i] * rval[i];
        }
        sum
    }
}

mod vector_add {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
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
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_add_inplace(lhs: &mut [f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        for i in 0..n {
            lhs[i] += rhs[i];
        }
    }
}

mod vector_sub {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_sub(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
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
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
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
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar(lhs: &[f32], rhs: f32) -> Vec<f32> {
        let n = lhs.len();
        let mut r = Vec::<f32>::with_capacity(n);
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
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_mul_scalar_inplace(lhs: &mut [f32], rhs: f32) {
        let n = lhs.len();
        for i in 0..n {
            lhs[i] *= rhs;
        }
    }
}

mod vector_abs_inplace {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_abs_inplace(this: &mut [f32]) {
        let n = this.len();
        for i in 0..n {
            this[i] = this[i].abs();
        }
    }
}
