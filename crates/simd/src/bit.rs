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

#[inline(always)]
pub fn reduce_sum_of_and(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_and::reduce_sum_of_and(lhs, rhs)
}

mod reduce_sum_of_and {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_and_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut and = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
        }
        _mm512_reduce_add_epi64(and) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_and_v4(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 4] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 4];
        let lut = unsafe { _mm512_loadu_si512((&raw const LUT).cast()) };
        let mask_0 = _mm512_set1_epi8(0x0f);
        let mut sum_and = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            //
            let and = _mm512_and_si512(x, y);
            let and_lo = _mm512_and_si512(and, mask_0);
            let and_hi = _mm512_and_si512(_mm512_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm512_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm512_shuffle_epi8(lut, and_hi);
            let and_res = _mm512_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm512_sad_epu8(and_res, _mm512_setzero_si512());
            sum_and = _mm512_add_epi64(sum_and, and_sad);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            //
            let and = _mm512_and_si512(x, y);
            let and_lo = _mm512_and_si512(and, mask_0);
            let and_hi = _mm512_and_si512(_mm512_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm512_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm512_shuffle_epi8(lut, and_hi);
            let and_res = _mm512_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm512_sad_epu8(and_res, _mm512_setzero_si512());
            sum_and = _mm512_add_epi64(sum_and, and_sad);
        }
        _mm512_reduce_add_epi64(sum_and) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_and_v3(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::emulate_mm256_reduce_add_epi64;
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 2] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 2];
        let lut = unsafe { _mm256_loadu_si256((&raw const LUT).cast()) };
        let mask_0 = _mm256_set1_epi8(0x0f);
        let mut sum_and = _mm256_setzero_si256();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 4 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            a = unsafe { a.add(4) };
            b = unsafe { b.add(4) };
            n -= 4;
            //
            let and = _mm256_and_si256(x, y);
            let and_lo = _mm256_and_si256(and, mask_0);
            let and_hi = _mm256_and_si256(_mm256_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm256_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm256_shuffle_epi8(lut, and_hi);
            let and_res = _mm256_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm256_sad_epu8(and_res, _mm256_setzero_si256());
            sum_and = _mm256_add_epi64(sum_and, and_sad);
        }
        let mut and = emulate_mm256_reduce_add_epi64(sum_and) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            and += (x & y).count_ones();
        }
        and
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_v3(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_and(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut and = 0;
        for i in 0..n {
            and += (lhs[i] & rhs[i]).count_ones();
        }
        and
    }
}

#[inline(always)]
pub fn reduce_sum_of_or(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_or::reduce_sum_of_or(lhs, rhs)
}

mod reduce_sum_of_or {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_or_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        _mm512_reduce_add_epi64(or) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_or_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_or_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_or_v4(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 4] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 4];
        let lut = unsafe { _mm512_loadu_si512((&raw const LUT).cast()) };
        let mask_0 = _mm512_set1_epi8(0x0f);
        let mut sum_or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            //
            let or = _mm512_or_si512(x, y);
            let or_lo = _mm512_and_si512(or, mask_0);
            let or_hi = _mm512_and_si512(_mm512_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm512_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm512_shuffle_epi8(lut, or_hi);
            let or_res = _mm512_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm512_sad_epu8(or_res, _mm512_setzero_si512());
            sum_or = _mm512_add_epi64(sum_or, or_sad);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            //
            let or = _mm512_or_si512(x, y);
            let or_lo = _mm512_and_si512(or, mask_0);
            let or_hi = _mm512_and_si512(_mm512_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm512_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm512_shuffle_epi8(lut, or_hi);
            let or_res = _mm512_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm512_sad_epu8(or_res, _mm512_setzero_si512());
            sum_or = _mm512_add_epi64(sum_or, or_sad);
        }
        _mm512_reduce_add_epi64(sum_or) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_or_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_or_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_or_v3(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::emulate_mm256_reduce_add_epi64;
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 2] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 2];
        let lut = unsafe { _mm256_loadu_si256((&raw const LUT).cast()) };
        let mask_0 = _mm256_set1_epi8(0x0f);
        let mut sum_or = _mm256_setzero_si256();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 4 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            a = unsafe { a.add(4) };
            b = unsafe { b.add(4) };
            n -= 4;
            //
            let or = _mm256_or_si256(x, y);
            let or_lo = _mm256_and_si256(or, mask_0);
            let or_hi = _mm256_and_si256(_mm256_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm256_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm256_shuffle_epi8(lut, or_hi);
            let or_res = _mm256_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm256_sad_epu8(or_res, _mm256_setzero_si256());
            sum_or = _mm256_add_epi64(sum_or, or_sad);
        }
        let mut or = emulate_mm256_reduce_add_epi64(sum_or) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            or += (x | y).count_ones();
        }
        or
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_or_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_or_v3(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_or(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut or = 0;
        for i in 0..n {
            or += (lhs[i] | rhs[i]).count_ones();
        }
        or
    }
}

#[inline(always)]
pub fn reduce_sum_of_xor(lhs: &[u64], rhs: &[u64]) -> u32 {
    reduce_sum_of_xor::reduce_sum_of_xor(lhs, rhs)
}

mod reduce_sum_of_xor {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_xor_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut xor = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            xor = _mm512_add_epi64(xor, _mm512_popcnt_epi64(_mm512_xor_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            xor = _mm512_add_epi64(xor, _mm512_popcnt_epi64(_mm512_xor_si512(x, y)));
        }
        _mm512_reduce_add_epi64(xor) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xor_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xor_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xor_v4(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 4] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 4];
        let lut = unsafe { _mm512_loadu_si512((&raw const LUT).cast()) };
        let mask_0 = _mm512_set1_epi8(0x0f);
        let mut sum_xor = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            //
            let xor = _mm512_xor_si512(x, y);
            let xor_lo = _mm512_and_si512(xor, mask_0);
            let xor_hi = _mm512_and_si512(_mm512_srli_epi16(xor, 4), mask_0);
            let xor_res_lo = _mm512_shuffle_epi8(lut, xor_lo);
            let xor_res_hi = _mm512_shuffle_epi8(lut, xor_hi);
            let xor_res = _mm512_add_epi8(xor_res_lo, xor_res_hi);
            let xor_sad = _mm512_sad_epu8(xor_res, _mm512_setzero_si512());
            sum_xor = _mm512_add_epi64(sum_xor, xor_sad);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            //
            let xor = _mm512_xor_si512(x, y);
            let xor_lo = _mm512_and_si512(xor, mask_0);
            let xor_hi = _mm512_and_si512(_mm512_srli_epi16(xor, 4), mask_0);
            let xor_res_lo = _mm512_shuffle_epi8(lut, xor_lo);
            let xor_res_hi = _mm512_shuffle_epi8(lut, xor_hi);
            let xor_res = _mm512_add_epi8(xor_res_lo, xor_res_hi);
            let xor_sad = _mm512_sad_epu8(xor_res, _mm512_setzero_si512());
            sum_xor = _mm512_add_epi64(sum_xor, xor_sad);
        }
        _mm512_reduce_add_epi64(sum_xor) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xor_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xor_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xor_v3(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::emulate_mm256_reduce_add_epi64;
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 2] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 2];
        let lut = unsafe { _mm256_loadu_si256((&raw const LUT).cast()) };
        let mask_0 = _mm256_set1_epi8(0x0f);
        let mut sum_xor = _mm256_setzero_si256();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 4 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            a = unsafe { a.add(4) };
            b = unsafe { b.add(4) };
            n -= 4;
            //
            let xor = _mm256_xor_si256(x, y);
            let xor_lo = _mm256_and_si256(xor, mask_0);
            let xor_hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), mask_0);
            let xor_res_lo = _mm256_shuffle_epi8(lut, xor_lo);
            let xor_res_hi = _mm256_shuffle_epi8(lut, xor_hi);
            let xor_res = _mm256_add_epi8(xor_res_lo, xor_res_hi);
            let xor_sad = _mm256_sad_epu8(xor_res, _mm256_setzero_si256());
            sum_xor = _mm256_add_epi64(sum_xor, xor_sad);
        }
        let mut xor = emulate_mm256_reduce_add_epi64(sum_xor) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            xor += (x ^ y).count_ones();
        }
        xor
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_xor_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_xor_v3(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xor(lhs: &[u64], rhs: &[u64]) -> u32 {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut xor = 0;
        for i in 0..n {
            xor += (lhs[i] ^ rhs[i]).count_ones();
        }
        xor
    }
}

#[inline(always)]
pub fn reduce_sum_of_and_or(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
    reduce_sum_of_and_or::reduce_sum_of_and_or(lhs, rhs)
}

mod reduce_sum_of_and_or {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_and_or_v4_avx512vpopcntdq(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        let mut and = _mm512_setzero_si512();
        let mut or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            and = _mm512_add_epi64(and, _mm512_popcnt_epi64(_mm512_and_si512(x, y)));
            or = _mm512_add_epi64(or, _mm512_popcnt_epi64(_mm512_or_si512(x, y)));
        }
        (
            _mm512_reduce_add_epi64(and) as u32,
            _mm512_reduce_add_epi64(or) as u32,
        )
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_or_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_or_v4_avx512vpopcntdq(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_and_or_v4(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert!(lhs.len() == rhs.len());
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 4] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 4];
        let lut = unsafe { _mm512_loadu_si512((&raw const LUT).cast()) };
        let mask_0 = _mm512_set1_epi8(0x0f);
        let mut sum_and = _mm512_setzero_si512();
        let mut sum_or = _mm512_setzero_si512();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            let y = unsafe { _mm512_loadu_si512(b.cast()) };
            a = unsafe { a.add(8) };
            b = unsafe { b.add(8) };
            n -= 8;
            //
            let and = _mm512_and_si512(x, y);
            let and_lo = _mm512_and_si512(and, mask_0);
            let and_hi = _mm512_and_si512(_mm512_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm512_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm512_shuffle_epi8(lut, and_hi);
            let and_res = _mm512_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm512_sad_epu8(and_res, _mm512_setzero_si512());
            sum_and = _mm512_add_epi64(sum_and, and_sad);
            //
            let or = _mm512_or_si512(x, y);
            let or_lo = _mm512_and_si512(or, mask_0);
            let or_hi = _mm512_and_si512(_mm512_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm512_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm512_shuffle_epi8(lut, or_hi);
            let or_res = _mm512_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm512_sad_epu8(or_res, _mm512_setzero_si512());
            sum_or = _mm512_add_epi64(sum_or, or_sad);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi64(mask, b.cast()) };
            //
            let and = _mm512_and_si512(x, y);
            let and_lo = _mm512_and_si512(and, mask_0);
            let and_hi = _mm512_and_si512(_mm512_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm512_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm512_shuffle_epi8(lut, and_hi);
            let and_res = _mm512_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm512_sad_epu8(and_res, _mm512_setzero_si512());
            sum_and = _mm512_add_epi64(sum_and, and_sad);
            //
            let or = _mm512_or_si512(x, y);
            let or_lo = _mm512_and_si512(or, mask_0);
            let or_hi = _mm512_and_si512(_mm512_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm512_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm512_shuffle_epi8(lut, or_hi);
            let or_res = _mm512_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm512_sad_epu8(or_res, _mm512_setzero_si512());
            sum_or = _mm512_add_epi64(sum_or, or_sad);
        }
        (
            _mm512_reduce_add_epi64(sum_and) as u32,
            _mm512_reduce_add_epi64(sum_or) as u32,
        )
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_or_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_or_v4(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_and_or_v3(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert!(lhs.len() == rhs.len());
        use crate::emulate::emulate_mm256_reduce_add_epi64;
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 2] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 2];
        let lut = unsafe { _mm256_loadu_si256((&raw const LUT).cast()) };
        let mask_0 = _mm256_set1_epi8(0x0f);
        let mut sum_and = _mm256_setzero_si256();
        let mut sum_or = _mm256_setzero_si256();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut n = lhs.len();
        while n >= 4 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            a = unsafe { a.add(4) };
            b = unsafe { b.add(4) };
            n -= 4;
            //
            let and = _mm256_and_si256(x, y);
            let and_lo = _mm256_and_si256(and, mask_0);
            let and_hi = _mm256_and_si256(_mm256_srli_epi16(and, 4), mask_0);
            let and_res_lo = _mm256_shuffle_epi8(lut, and_lo);
            let and_res_hi = _mm256_shuffle_epi8(lut, and_hi);
            let and_res = _mm256_add_epi8(and_res_lo, and_res_hi);
            let and_sad = _mm256_sad_epu8(and_res, _mm256_setzero_si256());
            sum_and = _mm256_add_epi64(sum_and, and_sad);
            //
            let or = _mm256_or_si256(x, y);
            let or_lo = _mm256_and_si256(or, mask_0);
            let or_hi = _mm256_and_si256(_mm256_srli_epi16(or, 4), mask_0);
            let or_res_lo = _mm256_shuffle_epi8(lut, or_lo);
            let or_res_hi = _mm256_shuffle_epi8(lut, or_hi);
            let or_res = _mm256_add_epi8(or_res_lo, or_res_hi);
            let or_sad = _mm256_sad_epu8(or_res, _mm256_setzero_si256());
            sum_or = _mm256_add_epi64(sum_or, or_sad);
        }
        let mut and = emulate_mm256_reduce_add_epi64(sum_and) as u32;
        let mut or = emulate_mm256_reduce_add_epi64(sum_or) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            and += (x & y).count_ones();
            or += (x | y).count_ones();
        }
        (and, or)
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_and_or_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let lhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let rhs = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_and_or_v3(&lhs, &rhs) };
            let fallback = fallback(&lhs, &rhs);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_and_or(lhs: &[u64], rhs: &[u64]) -> (u32, u32) {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut and = 0;
        let mut or = 0;
        for i in 0..n {
            and += (lhs[i] & rhs[i]).count_ones();
            or += (lhs[i] | rhs[i]).count_ones();
        }
        (and, or)
    }
}

#[inline(always)]
pub fn reduce_sum_of_x(this: &[u64]) -> u32 {
    reduce_sum_of_x::reduce_sum_of_x(this)
}

mod reduce_sum_of_x {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vpopcntdq")]
    fn reduce_sum_of_x_v4_avx512vpopcntdq(this: &[u64]) -> u32 {
        use std::arch::x86_64::*;
        let mut sum = _mm512_setzero_si512();
        let mut a = this.as_ptr();
        let mut n = this.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            a = unsafe { a.add(8) };
            n -= 8;
            sum = _mm512_add_epi64(sum, _mm512_popcnt_epi64(x));
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            sum = _mm512_add_epi64(sum, _mm512_popcnt_epi64(x));
        }
        _mm512_reduce_add_epi64(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_v4_avx512vpopcntdq_test() {
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vpopcntdq") {
            println!("test {} ... skipped (v4:avx512vpopcntdq)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let this = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v4_avx512vpopcntdq(&this) };
            let fallback = fallback(&this);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[u64]) -> u32 {
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 4] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 4];
        let lut = unsafe { _mm512_loadu_si512((&raw const LUT).cast()) };
        let mask_0 = _mm512_set1_epi8(0x0f);
        let mut sum = _mm512_setzero_si512();
        let mut a = this.as_ptr();
        let mut n = this.len();
        while n >= 8 {
            let x = unsafe { _mm512_loadu_si512(a.cast()) };
            a = unsafe { a.add(8) };
            n -= 8;
            let lo = _mm512_and_si512(x, mask_0);
            let hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), mask_0);
            let res_lo = _mm512_shuffle_epi8(lut, lo);
            let res_hi = _mm512_shuffle_epi8(lut, hi);
            let res = _mm512_add_epi8(res_lo, res_hi);
            let sad = _mm512_sad_epu8(res, _mm512_setzero_si512());
            sum = _mm512_add_epi64(sum, sad);
        }
        if n > 0 {
            let mask = _bzhi_u32(0xff, n as u32) as u8;
            let x = unsafe { _mm512_maskz_loadu_epi64(mask, a.cast()) };
            let lo = _mm512_and_si512(x, mask_0);
            let hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), mask_0);
            let res_lo = _mm512_shuffle_epi8(lut, lo);
            let res_hi = _mm512_shuffle_epi8(lut, hi);
            let res = _mm512_add_epi8(res_lo, res_hi);
            let sad = _mm512_sad_epu8(res, _mm512_setzero_si512());
            sum = _mm512_add_epi64(sum, sad);
        }
        _mm512_reduce_add_epi64(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let this = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v4(&this) };
            let fallback = fallback(&this);
            assert_eq!(specialized, fallback);
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[u64]) -> u32 {
        use crate::emulate::emulate_mm256_reduce_add_epi64;
        use std::arch::x86_64::*;
        static LUT: [[i8; 16]; 2] = [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]; 2];
        let lut = unsafe { _mm256_loadu_si256((&raw const LUT).cast()) };
        let mask_0 = _mm256_set1_epi8(0x0f);
        let mut sum = _mm256_setzero_si256();
        let mut a = this.as_ptr();
        let mut n = this.len();
        while n >= 4 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            a = unsafe { a.add(4) };
            n -= 4;
            let lo = _mm256_and_si256(x, mask_0);
            let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask_0);
            let res_lo = _mm256_shuffle_epi8(lut, lo);
            let res_hi = _mm256_shuffle_epi8(lut, hi);
            let res = _mm256_add_epi8(res_lo, res_hi);
            let sad = _mm256_sad_epu8(res, _mm256_setzero_si256());
            sum = _mm256_add_epi64(sum, sad);
        }
        let mut sum = emulate_mm256_reduce_add_epi64(sum) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            a = unsafe { a.add(1) };
            n -= 1;
            let p = x.count_ones();
            sum += p;
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn reduce_sum_of_x_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let this = (0..126).map(|_| rand::random::<u64>()).collect::<Vec<_>>();
            let specialized = unsafe { reduce_sum_of_x_v3(&this) };
            let fallback = fallback(&this);
            assert_eq!(specialized, fallback);
        }
    }

    #[crate::multiversion(@"v4:avx512vpopcntdq", @"v4", @"v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x(this: &[u64]) -> u32 {
        let n = this.len();
        let mut sum = 0;
        for i in 0..n {
            sum += this[i].count_ones();
        }
        sum
    }
}

#[inline(always)]
pub fn vector_and(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_and::vector_and(lhs, rhs)
}

mod vector_and {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_and(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] & rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

#[inline(always)]
pub fn vector_or(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_or::vector_or(lhs, rhs)
}

mod vector_or {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_or(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] | rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}

#[inline(always)]
pub fn vector_xor(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    vector_xor::vector_xor(lhs, rhs)
}

mod vector_xor {
    #[crate::multiversion(
        "v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1"
    )]
    pub fn vector_xor(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let mut r = Vec::<u64>::with_capacity(n);
        for i in 0..n {
            unsafe {
                r.as_mut_ptr().add(i).write(lhs[i] ^ rhs[i]);
            }
        }
        unsafe {
            r.set_len(n);
        }
        r
    }
}
