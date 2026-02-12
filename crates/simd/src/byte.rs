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

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_xy {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    #[target_feature(enable = "avx512vnni")]
    fn reduce_sum_of_xy_v4_avx512vnni(lhs: &[u8], rhs: &[u8]) -> u32 {
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let sign = _mm512_set1_epi8(-128);
        let i8_1 = _mm512_set1_epi8(1);
        let i16_1 = _mm512_set1_epi16(1);
        let mut _0 = _mm512_setzero_si512();
        let mut _1 = _mm512_setzero_si512();
        while n >= 64 {
            let x = unsafe { _mm512_loadu_epi8(a.cast()) };
            let y = unsafe { _mm512_loadu_epi8(b.cast()) };
            _0 = _mm512_dpbusd_epi32(_0, x, _mm512_xor_si512(sign, y));
            _1 = _mm512_add_epi32(_1, _mm512_madd_epi16(_mm512_maddubs_epi16(x, i8_1), i16_1));
            (n, a, b) = unsafe { (n - 64, a.add(64), b.add(64)) };
        }
        if n > 0 {
            let mask = _bzhi_u64(0xffffffffffffffff, n as u32);
            let x = unsafe { _mm512_maskz_loadu_epi8(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi8(mask, b.cast()) };
            _0 = _mm512_dpbusd_epi32(_0, x, _mm512_xor_si512(sign, y));
            _1 = _mm512_add_epi32(_1, _mm512_madd_epi16(_mm512_maddubs_epi16(x, i8_1), i16_1));
        }
        _mm512_reduce_add_epi32(_mm512_add_epi32(_0, _mm512_slli_epi32(_1, 7))) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v4_avx512vnni_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") || !crate::is_feature_detected!("avx512vnni") {
            println!("test {} ... skipped (v4:avx512vnni)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v4_avx512vnni(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_xy_v4(lhs: &[u8], rhs: &[u8]) -> u32 {
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm512_set1_epi16(0x00ff_i16);
        let mut _0 = _mm512_setzero_si512();
        let mut _1 = _mm512_setzero_si512();
        while n >= 64 {
            let x = unsafe { _mm512_loadu_epi8(a.cast()) };
            let y = unsafe { _mm512_loadu_epi8(b.cast()) };
            let x_0 = _mm512_and_si512(x, lo);
            let x_1 = _mm512_srli_epi16(x, 8);
            let y_0 = _mm512_and_si512(y, lo);
            let y_1 = _mm512_srli_epi16(y, 8);
            let z_0 = _mm512_madd_epi16(x_0, y_0);
            let z_1 = _mm512_madd_epi16(x_1, y_1);
            _0 = _mm512_add_epi32(_0, z_0);
            _1 = _mm512_add_epi32(_1, z_1);
            (n, a, b) = unsafe { (n - 64, a.add(64), b.add(64)) };
        }
        if n > 0 {
            let mask = _bzhi_u64(0xffffffffffffffff, n as u32);
            let x = unsafe { _mm512_maskz_loadu_epi8(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi8(mask, b.cast()) };
            let x_0 = _mm512_and_si512(x, lo);
            let x_1 = _mm512_srli_epi16(x, 8);
            let y_0 = _mm512_and_si512(y, lo);
            let y_1 = _mm512_srli_epi16(y, 8);
            let z_0 = _mm512_madd_epi16(x_0, y_0);
            let z_1 = _mm512_madd_epi16(x_1, y_1);
            _0 = _mm512_add_epi32(_0, z_0);
            _1 = _mm512_add_epi32(_1, z_1);
        }
        _mm512_reduce_add_epi32(_mm512_add_epi32(_0, _1)) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_v4_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v4(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_xy_v3(lhs: &[u8], rhs: &[u8]) -> u32 {
        use crate::emulate::{emulate_mm256_reduce_add_epi32, partial_load};
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm256_set1_epi16(0x00ff_i16);
        let mut _0 = _mm256_setzero_si256();
        let mut _1 = _mm256_setzero_si256();
        while n >= 32 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            let x_0 = _mm256_and_si256(x, lo);
            let x_1 = _mm256_srli_epi16(x, 8);
            let y_0 = _mm256_and_si256(y, lo);
            let y_1 = _mm256_srli_epi16(y, 8);
            let z_0 = _mm256_madd_epi16(x_0, y_0);
            let z_1 = _mm256_madd_epi16(x_1, y_1);
            _0 = _mm256_add_epi32(_0, z_0);
            _1 = _mm256_add_epi32(_1, z_1);
            (n, a, b) = unsafe { (n - 32, a.add(32), b.add(32)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(32, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            let x_0 = _mm256_and_si256(x, lo);
            let x_1 = _mm256_srli_epi16(x, 8);
            let y_0 = _mm256_and_si256(y, lo);
            let y_1 = _mm256_srli_epi16(y, 8);
            let z_0 = _mm256_madd_epi16(x_0, y_0);
            let z_1 = _mm256_madd_epi16(x_1, y_1);
            _0 = _mm256_add_epi32(_0, z_0);
            _1 = _mm256_add_epi32(_1, z_1);
        }
        emulate_mm256_reduce_add_epi32(_mm256_add_epi32(_0, _1)) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v3_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_xy_v2(lhs: &[u8], rhs: &[u8]) -> u32 {
        use crate::emulate::{emulate_mm_reduce_add_epi32, partial_load};
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm_set1_epi16(0x00ff_i16);
        let mut _0 = _mm_setzero_si128();
        let mut _1 = _mm_setzero_si128();
        while n >= 16 {
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            let y = unsafe { _mm_loadu_si128(b.cast()) };
            let x_0 = _mm_and_si128(x, lo);
            let x_1 = _mm_srli_epi16(x, 8);
            let y_0 = _mm_and_si128(y, lo);
            let y_1 = _mm_srli_epi16(y, 8);
            let z_0 = _mm_madd_epi16(x_0, y_0);
            let z_1 = _mm_madd_epi16(x_1, y_1);
            _0 = _mm_add_epi32(_0, z_0);
            _1 = _mm_add_epi32(_1, z_1);
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(16, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            let y = unsafe { _mm_loadu_si128(b.cast()) };
            let x_0 = _mm_and_si128(x, lo);
            let x_1 = _mm_srli_epi16(x, 8);
            let y_0 = _mm_and_si128(y, lo);
            let y_1 = _mm_srli_epi16(y, 8);
            let z_0 = _mm_madd_epi16(x_0, y_0);
            let z_1 = _mm_madd_epi16(x_1, y_1);
            _0 = _mm_add_epi32(_0, z_0);
            _1 = _mm_add_epi32(_1, z_1);
        }
        emulate_mm_reduce_add_epi32(_mm_add_epi32(_0, _1)) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_xy_v2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_v2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    #[target_feature(enable = "dotprod")]
    fn reduce_sum_of_xy_a2_dotprod(lhs: &[u8], rhs: &[u8]) -> u32 {
        unsafe extern "C" {
            #[link_name = "byte_reduce_sum_of_xy_a2_dotprod"]
            unsafe fn f(n: usize, a: *const u8, b: *const u8) -> u32;
        }
        assert_eq!(lhs.len(), rhs.len());
        let n = lhs.len();
        let a = lhs.as_ptr();
        let b = rhs.as_ptr();
        unsafe { f(n, a, b) }
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a2_dotprod_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") || !crate::is_feature_detected!("dotprod") {
            println!("test {} ... skipped (a2:dotprod)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a2_dotprod(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_xy_a2(lhs: &[u8], rhs: &[u8]) -> u32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let mut _0 = vdupq_n_u32(0);
        let mut _1 = vdupq_n_u32(0);
        while n >= 16 {
            let x = unsafe { vld1q_u8(a.cast()) };
            let y = unsafe { vld1q_u8(b.cast()) };
            let lo = vmull_u8(vget_low_u8(x), vget_low_u8(y));
            let hi = vmull_u8(vget_high_u8(x), vget_high_u8(y));
            _0 = vaddq_u32(_0, vpaddlq_u16(lo));
            _1 = vaddq_u32(_1, vpaddlq_u16(hi));
            (n, a, b) = unsafe { (n - 16, a.add(16), b.add(16)) };
        }
        if n > 0 {
            let (_a, _b) = unsafe { partial_load!(16, n, a, b) };
            (a, b) = (_a.as_ptr(), _b.as_ptr());
            let x = unsafe { vld1q_u8(a.cast()) };
            let y = unsafe { vld1q_u8(b.cast()) };
            let lo = vmull_u8(vget_low_u8(x), vget_low_u8(y));
            let hi = vmull_u8(vget_high_u8(x), vget_high_u8(y));
            _0 = vaddq_u32(_0, vpaddlq_u16(lo));
            _1 = vaddq_u32(_1, vpaddlq_u16(hi));
        }
        vaddvq_u32(vaddq_u32(_0, _1))
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_xy_a2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let lhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            let rhs = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let lhs = &lhs[..z];
                let rhs = &rhs[..z];
                let specialized = unsafe { reduce_sum_of_xy_a2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4:avx512vnni", @"v4", @"v3", @"v2", @"a2:dotprod", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let mut result = 0;
        for i in 0..n {
            result += (s[i] as u32) * (t[i] as u32);
        }
        result
    }
}

#[inline(always)]
pub fn reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
    reduce_sum_of_xy::reduce_sum_of_xy(s, t)
}

#[cfg_attr(feature = "internal", simd_macros::public)]
mod reduce_sum_of_x {
    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_v4(this: &[u8]) -> u32 {
        use core::arch::x86_64::*;
        let i8_1 = _mm512_set1_epi8(1);
        let i16_1 = _mm512_set1_epi16(1);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm512_setzero_si512();
        while n >= 64 {
            let x = unsafe { _mm512_loadu_epi8(a.cast()) };
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_maddubs_epi16(x, i8_1), i16_1));
            (n, a) = unsafe { (n - 64, a.add(64)) };
        }
        if n > 0 {
            let mask = _bzhi_u64(0xffffffffffffffff, n as u32);
            let x = unsafe { _mm512_maskz_loadu_epi8(mask, a.cast()) };
            sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_maddubs_epi16(x, i8_1), i16_1));
        }
        _mm512_reduce_add_epi32(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_v4_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v4(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_v3(this: &[u8]) -> u32 {
        use crate::emulate::{emulate_mm256_reduce_add_epi32, partial_load};
        use core::arch::x86_64::*;
        let i8_1 = _mm256_set1_epi8(1);
        let i16_1 = _mm256_set1_epi16(1);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm256_setzero_si256();
        while n >= 32 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(x, i8_1), i16_1));
            (n, a) = unsafe { (n - 32, a.add(32)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(32, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(x, i8_1), i16_1));
        }
        emulate_mm256_reduce_add_epi32(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v3_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v3(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_x_v2(this: &[u8]) -> u32 {
        use crate::emulate::{emulate_mm_reduce_add_epi32, partial_load};
        use core::arch::x86_64::*;
        let i8_1 = _mm_set1_epi8(1);
        let i16_1 = _mm_set1_epi16(1);
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = _mm_setzero_si128();
        while n >= 16 {
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(x, i8_1), i16_1));
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(16, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(x, i8_1), i16_1));
        }
        emulate_mm_reduce_add_epi32(sum) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_v2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_v2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[cfg_attr(feature = "internal", simd_macros::public)]
    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_a2(this: &[u8]) -> u32 {
        use crate::emulate::partial_load;
        use core::arch::aarch64::*;
        let mut n = this.len();
        let mut a = this.as_ptr();
        let mut sum = vdupq_n_u32(0);
        while n >= 16 {
            let x = unsafe { vld1q_u8(a.cast()) };
            sum = vaddq_u32(sum, vpaddlq_u16(vpaddlq_u8(x)));
            (n, a) = unsafe { (n - 16, a.add(16)) };
        }
        if n > 0 {
            let (_a,) = unsafe { partial_load!(16, n, a) };
            (a,) = (_a.as_ptr(),);
            let x = unsafe { vld1q_u8(a.cast()) };
            sum = vaddq_u32(sum, vpaddlq_u16(vpaddlq_u8(x)));
        }
        vaddvq_u32(sum)
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn reduce_sum_of_x_a2_test() {
        use rand::RngExt;
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        let mut rng = rand::rng();
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            let n = 4016;
            let this = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
            for z in 3984..4016 {
                let this = &this[..z];
                let specialized = unsafe { reduce_sum_of_x_a2(this) };
                let fallback = fallback(this);
                assert_eq!(specialized, fallback);
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x(this: &[u8]) -> u32 {
        let n = this.len();
        let mut sum = 0;
        for i in 0..n {
            sum += this[i] as u32;
        }
        sum
    }
}

#[inline(always)]
pub fn reduce_sum_of_x(vector: &[u8]) -> u32 {
    reduce_sum_of_x::reduce_sum_of_x(vector)
}
