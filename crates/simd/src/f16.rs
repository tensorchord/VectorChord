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

use crate::{Floating, f32};
use half::f16;
use zerocopy::FromZeros;

trait AsF32 {
    #[allow(clippy::wrong_self_convention)]
    fn as_f32(self) -> f32;
}

impl AsF32 for f16 {
    fn as_f32(self) -> f32 {
        self.into()
    }
}

impl Floating for f16 {
    #[inline(always)]
    fn zero() -> Self {
        FromZeros::new_zeroed()
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
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_or_of_is_zero_x(this: &[f16]) -> bool {
        for &x in this {
            if x == FromZeros::new_zeroed() {
                return true;
            }
        }
        false
    }
}

mod reduce_sum_of_x {
    // FIXME: add manually-implemented SIMD version

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += this[i].as_f32();
        }
        x
    }
}

mod reduce_sum_of_abs_x {
    // FIXME: add manually-implemented SIMD version

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_abs_x(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x = 0.0f32;
        for i in 0..n {
            x += (this[i].as_f32()).abs();
        }
        x
    }
}

mod reduce_sum_of_x2 {
    // FIXME: add manually-implemented SIMD version

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_x2(this: &[f16]) -> f32 {
        let n = this.len();
        let mut x2 = 0.0f32;
        for i in 0..n {
            x2 += this[i].as_f32() * this[i].as_f32();
        }
        x2
    }
}

mod reduce_min_max_of_x {
    // FIXME: add manually-implemented SIMD version

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_min_max_of_x(this: &[f16]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let n = this.len();
        for i in 0..n {
            min = min.min(this[i].as_f32());
            max = max.max(this[i].as_f32());
        }
        (min, max)
    }
}

mod reduce_sum_of_xy {
    use super::*;

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    pub fn reduce_sum_of_xy_v4_avx512fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_xy_v4_avx512fp16(
                    a: *const (),
                    b: *const (),
                    n: usize,
                ) -> f32;
            }
            fp16_reduce_sum_of_xy_v4_avx512fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_v4_avx512fp16_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512fp16") {
            println!("test {} ... skipped (v4:avx512fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    pub fn reduce_sum_of_xy_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut xy = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            a = unsafe { a.add(16) };
            b = unsafe { b.add(16) };
            n -= 16;
            xy = _mm512_fmadd_ps(x, y, xy);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            xy = _mm512_fmadd_ps(x, y, xy);
        }
        _mm512_reduce_add_ps(xy)
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_v4_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xy_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert!(
                (specialized - fallback).abs() < EPSILON,
                "specialized = {specialized}, fallback = {fallback}."
            );
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    pub fn reduce_sum_of_xy_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::emulate_mm256_reduce_add_ps;
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut xy = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            xy = _mm256_fmadd_ps(x, y, xy);
        }
        let mut xy = emulate_mm256_reduce_add_ps(xy);
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read().as_f32() };
            let y = unsafe { b.read().as_f32() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            xy += x * y;
        }
        xy
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_v3_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    pub fn reduce_sum_of_xy_a2_fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_xy_a2_fp16(
                    a: *const (),
                    b: *const (),
                    n: usize,
                ) -> f32;
            }
            fp16_reduce_sum_of_xy_a2_fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_a2_fp16_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("fp16") {
            println!("test {} ... skipped (a2:fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.512")]
    pub fn reduce_sum_of_xy_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_xy_a3_512(a: *const (), b: *const (), n: usize)
                -> f32;
            }
            fp16_reduce_sum_of_xy_a3_512(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xy_a3_512_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[crate::multiversion(@"v4:avx512fp16", @"v4", @"v3", #[cfg(target_endian = "little")] @"a3.512", @"a2:fp16")]
    pub fn reduce_sum_of_xy(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut xy = 0.0f32;
        for i in 0..n {
            xy += lhs[i].as_f32() * rhs[i].as_f32();
        }
        xy
    }
}

mod reduce_sum_of_d2 {
    use super::*;

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512fp16")]
    pub fn reduce_sum_of_d2_v4_avx512fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_d2_v4_avx512fp16(
                    a: *const (),
                    b: *const (),
                    n: usize,
                ) -> f32;
            }
            fp16_reduce_sum_of_d2_v4_avx512fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_d2_v4_avx512fp16_test() {
        use rand::Rng;
        const EPSILON: f32 = 6.4;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512fp16") {
            println!("test {} ... skipped (v4:avx512fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    pub fn reduce_sum_of_d2_v4(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut d2 = _mm512_setzero_ps();
        while n >= 16 {
            let x = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_loadu_epi16(b.cast())) };
            a = unsafe { a.add(16) };
            b = unsafe { b.add(16) };
            n -= 16;
            let d = _mm512_sub_ps(x, y);
            d2 = _mm512_fmadd_ps(d, d, d2);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xffff, n as u32) as u16;
            let x = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a.cast())) };
            let y = unsafe { _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b.cast())) };
            let d = _mm512_sub_ps(x, y);
            d2 = _mm512_fmadd_ps(d, d, d2);
        }
        _mm512_reduce_add_ps(d2)
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_d2_v4_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    pub fn reduce_sum_of_d2_v3(lhs: &[f16], rhs: &[f16]) -> f32 {
        use crate::emulate::emulate_mm256_reduce_add_ps;
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut d2 = _mm256_setzero_ps();
        while n >= 8 {
            let x = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(a.cast())) };
            let y = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(b.cast())) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            let d = _mm256_sub_ps(x, y);
            d2 = _mm256_fmadd_ps(d, d, d2);
        }
        let mut d2 = emulate_mm256_reduce_add_ps(d2);
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read().as_f32() };
            let y = unsafe { b.read().as_f32() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            let d = x - y;
            d2 += d * d;
        }
        d2
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_d2_v3_test() {
        use rand::Rng;
        const EPSILON: f32 = 2.0;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "fp16")]
    pub fn reduce_sum_of_d2_a2_fp16(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_d2_a2_fp16(
                    a: *const (),
                    b: *const (),
                    n: usize,
                ) -> f32;
            }
            fp16_reduce_sum_of_d2_a2_fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_d2_a2_fp16_test() {
        use rand::Rng;
        const EPSILON: f32 = 6.4;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("fp16") {
            println!("test {} ... skipped (a2:fp16)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[inline]
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[crate::target_cpu(enable = "a3.512")]
    pub fn reduce_sum_of_d2_a3_512(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        unsafe {
            unsafe extern "C" {
                unsafe fn fp16_reduce_sum_of_d2_a3_512(a: *const (), b: *const (), n: usize)
                -> f32;
            }
            fp16_reduce_sum_of_d2_a3_512(lhs.as_ptr().cast(), rhs.as_ptr().cast(), lhs.len())
        }
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little", test, not(miri)))]
    #[test]
    fn reduce_sum_of_d2_a3_512_test() {
        use rand::Rng;
        const EPSILON: f32 = 6.4;
        if !crate::is_cpu_detected!("a3.512") {
            println!("test {} ... skipped (a3.512)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
                .collect::<Vec<_>>();
            let rhs = (0..n)
                .map(|_| f16::from_f32(rng.random_range(-1.0..=1.0)))
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

    #[crate::multiversion(@"v4:avx512fp16", @"v4", @"v3", #[cfg(target_endian = "little")] @"a3.512", @"a2:fp16")]
    pub fn reduce_sum_of_d2(lhs: &[f16], rhs: &[f16]) -> f32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut d2 = 0.0;
        for i in 0..n {
            let d = lhs[i].as_f32() - rhs[i].as_f32();
            d2 += d * d;
        }
        d2
    }
}

mod reduce_sum_of_xy_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut xy = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    xy += lval[lp].as_f32() * rval[rp].as_f32();
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
        xy
    }
}

mod reduce_sum_of_d2_sparse {
    // There is no manually-implemented SIMD version.
    // Add it if `svecf16` is supported.

    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[f16], ridx: &[u32], rval: &[f16]) -> f32 {
        use std::cmp::Ordering;
        assert_eq!(lidx.len(), lval.len());
        assert_eq!(ridx.len(), rval.len());
        let (mut lp, ln) = (0, lidx.len());
        let (mut rp, rn) = (0, ridx.len());
        let mut d2 = 0.0f32;
        while lp < ln && rp < rn {
            match Ord::cmp(&lidx[lp], &ridx[rp]) {
                Ordering::Equal => {
                    let d = lval[lp].as_f32() - rval[rp].as_f32();
                    d2 += d * d;
                    lp += 1;
                    rp += 1;
                }
                Ordering::Less => {
                    d2 += lval[lp].as_f32() * lval[lp].as_f32();
                    lp += 1;
                }
                Ordering::Greater => {
                    d2 += rval[rp].as_f32() * rval[rp].as_f32();
                    rp += 1;
                }
            }
        }
        for i in lp..ln {
            d2 += lval[i].as_f32() * lval[i].as_f32();
        }
        for i in rp..rn {
            d2 += rval[i].as_f32() * rval[i].as_f32();
        }
        d2
    }
}

mod vector_add {
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
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
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_add_inplace(lhs: &mut [f16], rhs: &[f16]) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        for i in 0..n {
            lhs[i] += rhs[i];
        }
    }
}

mod vector_sub {
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
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
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
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
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_mul_scalar(lhs: &[f16], rhs: f32) -> Vec<f16> {
        let rhs = f16::from_f32(rhs);
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
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_mul_scalar_inplace(lhs: &mut [f16], rhs: f32) {
        let rhs = f16::from_f32(rhs);
        let n = lhs.len();
        for i in 0..n {
            lhs[i] *= rhs;
        }
    }
}

mod vector_abs_inplace {
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_abs_inplace(this: &mut [f16]) {
        let n = this.len();
        for i in 0..n {
            this[i] = f16::from_f32(this[i].as_f32().abs());
        }
    }
}

mod vector_from_f32 {
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_from_f32(this: &[f32]) -> Vec<f16> {
        let n = this.len();
        let mut r = Vec::<f16>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(f16::from_f32(this[i]));
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

mod vector_to_f32 {
    use super::*;

    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn vector_to_f32(this: &[f16]) -> Vec<f32> {
        let n = this.len();
        let mut r = Vec::<f32>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(this[i].as_f32());
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}
