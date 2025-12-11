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

mod reduce_sum_of_x_as_u32_y_as_u32 {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn reduce_sum_of_x_as_u32_y_as_u32_v4(lhs: &[u8], rhs: &[u8]) -> u32 {
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm512_set1_epi16(0x000f_i16);
        let mut _0 = _mm512_setzero_si512();
        let mut _1 = _mm512_setzero_si512();
        let mut _2 = _mm512_setzero_si512();
        let mut _3 = _mm512_setzero_si512();
        let mut _4 = _mm512_setzero_si512();
        let mut _5 = _mm512_setzero_si512();
        let mut _6 = _mm512_setzero_si512();
        let mut _7 = _mm512_setzero_si512();
        while n >= 64 {
            let x = unsafe { _mm512_loadu_epi8(a.cast()) };
            let y = unsafe { _mm512_loadu_epi8(b.cast()) };
            let x_0 = _mm512_and_si512(x, lo);
            let x_1 = _mm512_and_si512(_mm512_srli_epi16(x, 4), lo);
            let x_2 = _mm512_and_si512(_mm512_srli_epi16(x, 8), lo);
            let x_3 = _mm512_srli_epi16(x, 12);
            let y_0 = _mm512_and_si512(y, lo);
            let y_1 = _mm512_and_si512(_mm512_srli_epi16(y, 4), lo);
            let y_2 = _mm512_and_si512(_mm512_srli_epi16(y, 8), lo);
            let y_3 = _mm512_srli_epi16(y, 12);
            let z_0 = _mm512_mullo_epi16(x_0, y_0);
            let z_1 = _mm512_mullo_epi16(x_1, y_1);
            let z_2 = _mm512_mullo_epi16(x_2, y_2);
            let z_3 = _mm512_mullo_epi16(x_3, y_3);
            a = unsafe { a.add(64) };
            b = unsafe { b.add(64) };
            n -= 64;
            _0 = _mm512_add_epi32(_0, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_0, 0)));
            _1 = _mm512_add_epi32(_1, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_0, 1)));
            _2 = _mm512_add_epi32(_2, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_1, 0)));
            _3 = _mm512_add_epi32(_3, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_1, 1)));
            _4 = _mm512_add_epi32(_4, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_2, 0)));
            _5 = _mm512_add_epi32(_5, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_2, 1)));
            _6 = _mm512_add_epi32(_6, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_3, 0)));
            _7 = _mm512_add_epi32(_7, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_3, 1)));
        }
        if n > 0 {
            let mask = _bzhi_u64(0xffffffffffffffff, n as u32);
            let x = unsafe { _mm512_maskz_loadu_epi8(mask, a.cast()) };
            let y = unsafe { _mm512_maskz_loadu_epi8(mask, b.cast()) };
            let x_0 = _mm512_and_si512(x, lo);
            let x_1 = _mm512_and_si512(_mm512_srli_epi16(x, 4), lo);
            let x_2 = _mm512_and_si512(_mm512_srli_epi16(x, 8), lo);
            let x_3 = _mm512_srli_epi16(x, 12);
            let y_0 = _mm512_and_si512(y, lo);
            let y_1 = _mm512_and_si512(_mm512_srli_epi16(y, 4), lo);
            let y_2 = _mm512_and_si512(_mm512_srli_epi16(y, 8), lo);
            let y_3 = _mm512_srli_epi16(y, 12);
            let z_0 = _mm512_mullo_epi16(x_0, y_0);
            let z_1 = _mm512_mullo_epi16(x_1, y_1);
            let z_2 = _mm512_mullo_epi16(x_2, y_2);
            let z_3 = _mm512_mullo_epi16(x_3, y_3);
            _0 = _mm512_add_epi32(_0, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_0, 0)));
            _1 = _mm512_add_epi32(_1, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_0, 1)));
            _2 = _mm512_add_epi32(_2, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_1, 0)));
            _3 = _mm512_add_epi32(_3, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_1, 1)));
            _4 = _mm512_add_epi32(_4, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_2, 0)));
            _5 = _mm512_add_epi32(_5, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_2, 1)));
            _6 = _mm512_add_epi32(_6, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_3, 0)));
            _7 = _mm512_add_epi32(_7, _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(z_3, 1)));
        }
        let r_0 = _mm512_add_epi32(_0, _4);
        let r_1 = _mm512_add_epi32(_1, _5);
        let r_2 = _mm512_add_epi32(_2, _6);
        let r_3 = _mm512_add_epi32(_3, _7);
        let r_4 = _mm512_add_epi32(r_0, r_2);
        let r_5 = _mm512_add_epi32(r_1, r_3);
        let r_6 = _mm512_add_epi32(r_4, r_5);
        _mm512_reduce_add_epi32(r_6) as u32
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_as_u32_y_as_u32_v4_test() {
        use rand::Rng;
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
                let specialized = unsafe { reduce_sum_of_x_as_u32_y_as_u32_v4(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn reduce_sum_of_x_as_u32_y_as_u32_v3(lhs: &[u8], rhs: &[u8]) -> u32 {
        use crate::emulate::emulate_mm256_reduce_add_epi32;
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm256_set1_epi16(0x000f_i16);
        let mut _0 = _mm256_setzero_si256();
        let mut _1 = _mm256_setzero_si256();
        let mut _2 = _mm256_setzero_si256();
        let mut _3 = _mm256_setzero_si256();
        let mut _4 = _mm256_setzero_si256();
        let mut _5 = _mm256_setzero_si256();
        let mut _6 = _mm256_setzero_si256();
        let mut _7 = _mm256_setzero_si256();
        while n >= 32 {
            let x = unsafe { _mm256_loadu_si256(a.cast()) };
            let y = unsafe { _mm256_loadu_si256(b.cast()) };
            let x_0 = _mm256_and_si256(x, lo);
            let x_1 = _mm256_and_si256(_mm256_srli_epi16(x, 4), lo);
            let x_2 = _mm256_and_si256(_mm256_srli_epi16(x, 8), lo);
            let x_3 = _mm256_srli_epi16(x, 12);
            let y_0 = _mm256_and_si256(y, lo);
            let y_1 = _mm256_and_si256(_mm256_srli_epi16(y, 4), lo);
            let y_2 = _mm256_and_si256(_mm256_srli_epi16(y, 8), lo);
            let y_3 = _mm256_srli_epi16(y, 12);
            let z_0 = _mm256_mullo_epi16(x_0, y_0);
            let z_1 = _mm256_mullo_epi16(x_1, y_1);
            let z_2 = _mm256_mullo_epi16(x_2, y_2);
            let z_3 = _mm256_mullo_epi16(x_3, y_3);
            a = unsafe { a.add(32) };
            b = unsafe { b.add(32) };
            n -= 32;
            _0 = _mm256_add_epi32(_0, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_0, 0)));
            _1 = _mm256_add_epi32(_1, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_0, 1)));
            _2 = _mm256_add_epi32(_2, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_1, 0)));
            _3 = _mm256_add_epi32(_3, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_1, 1)));
            _4 = _mm256_add_epi32(_4, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_2, 0)));
            _5 = _mm256_add_epi32(_5, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_2, 1)));
            _6 = _mm256_add_epi32(_6, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_3, 0)));
            _7 = _mm256_add_epi32(_7, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(z_3, 1)));
        }
        let mut sum = emulate_mm256_reduce_add_epi32(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(_0, _4), _mm256_add_epi32(_1, _5)),
            _mm256_add_epi32(_mm256_add_epi32(_2, _6), _mm256_add_epi32(_3, _7)),
        )) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            sum += (x as u32 & 0xf) * (y as u32 & 0xf);
            sum += (x as u32 >> 4) * (y as u32 >> 4);
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_as_u32_y_as_u32_v3_test() {
        use rand::Rng;
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
                let specialized = unsafe { reduce_sum_of_x_as_u32_y_as_u32_v3(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn reduce_sum_of_x_as_u32_y_as_u32_v2(lhs: &[u8], rhs: &[u8]) -> u32 {
        use crate::emulate::emulate_mm_reduce_add_epi32;
        use core::arch::x86_64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = _mm_set1_epi16(0x000f_i16);
        let mut _0 = _mm_setzero_si128();
        let mut _1 = _mm_setzero_si128();
        let mut _2 = _mm_setzero_si128();
        let mut _3 = _mm_setzero_si128();
        let mut _4 = _mm_setzero_si128();
        let mut _5 = _mm_setzero_si128();
        let mut _6 = _mm_setzero_si128();
        let mut _7 = _mm_setzero_si128();
        while n >= 16 {
            let x = unsafe { _mm_loadu_si128(a.cast()) };
            let y = unsafe { _mm_loadu_si128(b.cast()) };
            let x_0 = _mm_and_si128(x, lo);
            let x_1 = _mm_and_si128(_mm_srli_epi16(x, 4), lo);
            let x_2 = _mm_and_si128(_mm_srli_epi16(x, 8), lo);
            let x_3 = _mm_srli_epi16(x, 12);
            let y_0 = _mm_and_si128(y, lo);
            let y_1 = _mm_and_si128(_mm_srli_epi16(y, 4), lo);
            let y_2 = _mm_and_si128(_mm_srli_epi16(y, 8), lo);
            let y_3 = _mm_srli_epi16(y, 12);
            let z_0 = _mm_mullo_epi16(x_0, y_0);
            let z_1 = _mm_mullo_epi16(x_1, y_1);
            let z_2 = _mm_mullo_epi16(x_2, y_2);
            let z_3 = _mm_mullo_epi16(x_3, y_3);
            a = unsafe { a.add(16) };
            b = unsafe { b.add(16) };
            n -= 16;
            _0 = _mm_add_epi32(_0, _mm_cvtepu16_epi32(z_0));
            _1 = _mm_add_epi32(_1, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(z_0, z_0)));
            _2 = _mm_add_epi32(_2, _mm_cvtepu16_epi32(z_1));
            _3 = _mm_add_epi32(_3, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(z_1, z_1)));
            _4 = _mm_add_epi32(_4, _mm_cvtepu16_epi32(z_2));
            _5 = _mm_add_epi32(_5, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(z_2, z_2)));
            _6 = _mm_add_epi32(_6, _mm_cvtepu16_epi32(z_3));
            _7 = _mm_add_epi32(_7, _mm_cvtepu16_epi32(_mm_unpackhi_epi64(z_3, z_3)));
        }
        let mut sum = emulate_mm_reduce_add_epi32(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(_0, _4), _mm_add_epi32(_1, _5)),
            _mm_add_epi32(_mm_add_epi32(_2, _6), _mm_add_epi32(_3, _7)),
        )) as u32;
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            sum += (x as u32 & 0xf) * (y as u32 & 0xf);
            sum += (x as u32 >> 4) * (y as u32 >> 4);
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", test))]
    #[test]
    fn reduce_sum_of_x_as_u32_y_as_u32_v2_test() {
        use rand::Rng;
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
                let specialized = unsafe { reduce_sum_of_x_as_u32_y_as_u32_v2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn reduce_sum_of_x_as_u32_y_as_u32_a2(lhs: &[u8], rhs: &[u8]) -> u32 {
        use core::arch::aarch64::*;
        assert_eq!(lhs.len(), rhs.len());
        let mut n = lhs.len();
        let mut a = lhs.as_ptr();
        let mut b = rhs.as_ptr();
        let lo = vdupq_n_u16(0x000f_u16);
        let mut _0 = vdupq_n_u32(0);
        let mut _1 = vdupq_n_u32(0);
        let mut _2 = vdupq_n_u32(0);
        let mut _3 = vdupq_n_u32(0);
        let mut _4 = vdupq_n_u32(0);
        let mut _5 = vdupq_n_u32(0);
        let mut _6 = vdupq_n_u32(0);
        let mut _7 = vdupq_n_u32(0);
        while n >= 16 {
            let x = unsafe { vld1q_u16(a.cast()) };
            let y = unsafe { vld1q_u16(b.cast()) };
            let x_0 = vandq_u16(x, lo);
            let x_1 = vandq_u16(vshrq_n_u16(x, 4), lo);
            let x_2 = vandq_u16(vshrq_n_u16(x, 8), lo);
            let x_3 = vshrq_n_u16(x, 12);
            let y_0 = vandq_u16(y, lo);
            let y_1 = vandq_u16(vshrq_n_u16(y, 4), lo);
            let y_2 = vandq_u16(vshrq_n_u16(y, 8), lo);
            let y_3 = vshrq_n_u16(y, 12);
            let z_0 = vmulq_u16(x_0, y_0);
            let z_1 = vmulq_u16(x_1, y_1);
            let z_2 = vmulq_u16(x_2, y_2);
            let z_3 = vmulq_u16(x_3, y_3);
            a = unsafe { a.add(16) };
            b = unsafe { b.add(16) };
            n -= 16;
            _0 = vaddq_u32(_0, vmovl_u16(vget_low_u16(z_0)));
            _1 = vaddq_u32(_1, vmovl_u16(vget_high_u16(z_0)));
            _2 = vaddq_u32(_2, vmovl_u16(vget_low_u16(z_1)));
            _3 = vaddq_u32(_3, vmovl_u16(vget_high_u16(z_1)));
            _4 = vaddq_u32(_4, vmovl_u16(vget_low_u16(z_2)));
            _5 = vaddq_u32(_5, vmovl_u16(vget_high_u16(z_2)));
            _6 = vaddq_u32(_6, vmovl_u16(vget_low_u16(z_3)));
            _7 = vaddq_u32(_7, vmovl_u16(vget_high_u16(z_3)));
        }
        let mut sum = vaddvq_u32(vaddq_u32(
            vaddq_u32(vaddq_u32(_0, _4), vaddq_u32(_1, _5)),
            vaddq_u32(vaddq_u32(_2, _6), vaddq_u32(_3, _7)),
        ));
        // this hint is used to disable loop unrolling
        while std::hint::black_box(n) > 0 {
            let x = unsafe { a.read() };
            let y = unsafe { b.read() };
            a = unsafe { a.add(1) };
            b = unsafe { b.add(1) };
            n -= 1;
            sum += (x as u32 & 0xf) * (y as u32 & 0xf);
            sum += (x as u32 >> 4) * (y as u32 >> 4);
        }
        sum
    }

    #[cfg(all(target_arch = "aarch64", test))]
    #[test]
    fn reduce_sum_of_x_as_u32_y_as_u32_a2_test() {
        use rand::Rng;
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
                let specialized = unsafe { reduce_sum_of_x_as_u32_y_as_u32_a2(lhs, rhs) };
                let fallback = fallback(lhs, rhs);
                assert!(
                    specialized == fallback,
                    "specialized = {specialized}, fallback = {fallback}."
                );
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
    pub fn reduce_sum_of_x_as_u32_y_as_u32(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let mut result = 0;
        for i in 0..n {
            result += (s[i] as u32 & 0xf) * (t[i] as u32 & 0xf);
            result += (s[i] as u32 >> 4) * (t[i] as u32 >> 4);
        }
        result
    }
}

#[inline(always)]
pub fn reduce_sum_of_x_as_u32_y_as_u32(s: &[u8], t: &[u8]) -> u32 {
    reduce_sum_of_x_as_u32_y_as_u32::reduce_sum_of_x_as_u32_y_as_u32(s, t)
}
