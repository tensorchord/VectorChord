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

/*

## code layout for 4-bit quantizer

group i = | vector i | (total bytes = n/2)

byte:      | 0      | 1      | 2      | ... | n/2 - 1  |
bits 0..3: | code 0 | code 2 | code 4 | ... | code n-2 |
bits 4..7: | code 1 | code 3 | code 5 | ... | code n-1 |

## packed_code layout for 4-bit quantizer

group i = | vector 32i | vector 32i+1 | vector 32i+2 | ... | vector 32i+31 | (total bytes = n * 16)

byte      | 0                | 1                | 2                | ... | 14               | 15               |
bits 0..3 | code 0,vector 0  | code 0,vector 8  | code 0,vector 1  | ... | code 0,vector 14 | code 0,vector 15 |
bits 4..7 | code 0,vector 16 | code 0,vector 24 | code 0,vector 17 | ... | code 0,vector 30 | code 0,vector 31 |

byte      | 16               | 17               | 18               | ... | 30               | 31               |
bits 0..3 | code 1,vector 0  | code 1,vector 8  | code 1,vector 1  | ... | code 1,vector 14 | code 1,vector 15 |
bits 4..7 | code 1,vector 16 | code 1,vector 24 | code 1,vector 17 | ... | code 1,vector 30 | code 1,vector 31 |

byte      | 32               | 33               | 34               | ... | 46               | 47               |
bits 0..3 | code 2,vector 0  | code 2,vector 8  | code 2,vector 1  | ... | code 2,vector 14 | code 2,vector 15 |
bits 4..7 | code 2,vector 16 | code 2,vector 24 | code 2,vector 17 | ... | code 2,vector 30 | code 2,vector 31 |

...

byte      | n*32-32              | n*32-31              | ... | n*32-1               |
bits 0..3 | code (n-1),vector 0  | code (n-1),vector 8  | ... | code (n-1),vector 15 |
bits 4..7 | code (n-1),vector 16 | code (n-1),vector 24 | ... | code (n-1),vector 31 |

*/

mod scan {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4")]
    fn scan_v4(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        use std::arch::x86_64::*;

        #[inline]
        #[crate::target_cpu(enable = "v4")]
        fn combine2x2(x0x1: __m256i, y0y1: __m256i) -> __m256i {
            let x1y0 = _mm256_permute2f128_si256(x0x1, y0y1, 0x21);
            let x0y1 = _mm256_blend_epi32(x0x1, y0y1, 0xf0);
            _mm256_add_epi16(x1y0, x0y1)
        }

        #[inline]
        #[crate::target_cpu(enable = "v4")]
        fn combine4x2(x0x1x2x3: __m512i, y0y1y2y3: __m512i) -> __m256i {
            let x0x1 = _mm512_castsi512_si256(x0x1x2x3);
            let x2x3 = _mm512_extracti64x4_epi64(x0x1x2x3, 1);
            let y0y1 = _mm512_castsi512_si256(y0y1y2y3);
            let y2y3 = _mm512_extracti64x4_epi64(y0y1y2y3, 1);
            let x01y01 = combine2x2(x0x1, y0y1);
            let x23y23 = combine2x2(x2x3, y2y3);
            _mm256_add_epi16(x01y01, x23y23)
        }

        let mut accu_0 = _mm512_setzero_si512();
        let mut accu_1 = _mm512_setzero_si512();
        let mut accu_2 = _mm512_setzero_si512();
        let mut accu_3 = _mm512_setzero_si512();

        let mut i = 0_usize;
        while i + 4 <= n {
            let code = unsafe { _mm512_loadu_si512(code.as_ptr().add(i).cast()) };

            let mask = _mm512_set1_epi8(0xf);
            let clo = _mm512_and_si512(code, mask);
            let chi = _mm512_and_si512(_mm512_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm512_loadu_si512(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm512_shuffle_epi8(lut, clo);
            accu_0 = _mm512_add_epi16(accu_0, res_lo);
            accu_1 = _mm512_add_epi16(accu_1, _mm512_srli_epi16(res_lo, 8));
            let res_hi = _mm512_shuffle_epi8(lut, chi);
            accu_2 = _mm512_add_epi16(accu_2, res_hi);
            accu_3 = _mm512_add_epi16(accu_3, _mm512_srli_epi16(res_hi, 8));

            i += 4;
        }
        if i + 2 <= n {
            let code = unsafe { _mm256_loadu_si256(code.as_ptr().add(i).cast()) };

            let mask = _mm256_set1_epi8(0xf);
            let clo = _mm256_and_si256(code, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm256_loadu_si256(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm512_zextsi256_si512(_mm256_shuffle_epi8(lut, clo));
            accu_0 = _mm512_add_epi16(accu_0, res_lo);
            accu_1 = _mm512_add_epi16(accu_1, _mm512_srli_epi16(res_lo, 8));
            let res_hi = _mm512_zextsi256_si512(_mm256_shuffle_epi8(lut, chi));
            accu_2 = _mm512_add_epi16(accu_2, res_hi);
            accu_3 = _mm512_add_epi16(accu_3, _mm512_srli_epi16(res_hi, 8));

            i += 2;
        }
        if i < n {
            let code = unsafe { _mm_loadu_si128(code.as_ptr().add(i).cast()) };

            let mask = _mm_set1_epi8(0xf);
            let clo = _mm_and_si128(code, mask);
            let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm_loadu_si128(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm512_zextsi128_si512(_mm_shuffle_epi8(lut, clo));
            accu_0 = _mm512_add_epi16(accu_0, res_lo);
            accu_1 = _mm512_add_epi16(accu_1, _mm512_srli_epi16(res_lo, 8));
            let res_hi = _mm512_zextsi128_si512(_mm_shuffle_epi8(lut, chi));
            accu_2 = _mm512_add_epi16(accu_2, res_hi);
            accu_3 = _mm512_add_epi16(accu_3, _mm512_srli_epi16(res_hi, 8));

            i += 1;
        }
        debug_assert_eq!(i, n);

        let mut result = [0_u16; 32];

        accu_0 = _mm512_sub_epi16(accu_0, _mm512_slli_epi16(accu_1, 8));
        unsafe {
            _mm256_storeu_si256(
                result.as_mut_ptr().add(0).cast(),
                combine4x2(accu_0, accu_1),
            );
        }

        accu_2 = _mm512_sub_epi16(accu_2, _mm512_slli_epi16(accu_3, 8));
        unsafe {
            _mm256_storeu_si256(
                result.as_mut_ptr().add(16).cast(),
                combine4x2(accu_2, accu_3),
            );
        }

        result
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn scan_v4_test() {
        if !crate::is_cpu_detected!("v4") {
            println!("test {} ... skipped (v4)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            for n in 90..110 {
                let code = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                let lut = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                unsafe {
                    assert_eq!(scan_v4(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn scan_v3(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        use std::arch::x86_64::*;

        #[inline]
        #[crate::target_cpu(enable = "v3")]
        fn combine2x2(x0x1: __m256i, y0y1: __m256i) -> __m256i {
            let x1y0 = _mm256_permute2f128_si256(x0x1, y0y1, 0x21);
            let x0y1 = _mm256_blend_epi32(x0x1, y0y1, 0xf0);
            _mm256_add_epi16(x1y0, x0y1)
        }

        let mut accu_0 = _mm256_setzero_si256();
        let mut accu_1 = _mm256_setzero_si256();
        let mut accu_2 = _mm256_setzero_si256();
        let mut accu_3 = _mm256_setzero_si256();

        let mut i = 0_usize;
        while i + 2 <= n {
            let code = unsafe { _mm256_loadu_si256(code.as_ptr().add(i).cast()) };

            let mask = _mm256_set1_epi8(0xf);
            let clo = _mm256_and_si256(code, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm256_loadu_si256(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm256_shuffle_epi8(lut, clo);
            accu_0 = _mm256_add_epi16(accu_0, res_lo);
            accu_1 = _mm256_add_epi16(accu_1, _mm256_srli_epi16(res_lo, 8));
            let res_hi = _mm256_shuffle_epi8(lut, chi);
            accu_2 = _mm256_add_epi16(accu_2, res_hi);
            accu_3 = _mm256_add_epi16(accu_3, _mm256_srli_epi16(res_hi, 8));

            i += 2;
        }
        if i < n {
            let code = unsafe { _mm_loadu_si128(code.as_ptr().add(i).cast()) };

            let mask = _mm_set1_epi8(0xf);
            let clo = _mm_and_si128(code, mask);
            let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm_loadu_si128(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm256_zextsi128_si256(_mm_shuffle_epi8(lut, clo));
            accu_0 = _mm256_add_epi16(accu_0, res_lo);
            accu_1 = _mm256_add_epi16(accu_1, _mm256_srli_epi16(res_lo, 8));
            let res_hi = _mm256_zextsi128_si256(_mm_shuffle_epi8(lut, chi));
            accu_2 = _mm256_add_epi16(accu_2, res_hi);
            accu_3 = _mm256_add_epi16(accu_3, _mm256_srli_epi16(res_hi, 8));

            i += 1;
        }
        debug_assert_eq!(i, n);

        let mut result = [0_u16; 32];

        accu_0 = _mm256_sub_epi16(accu_0, _mm256_slli_epi16(accu_1, 8));
        unsafe {
            _mm256_storeu_si256(
                result.as_mut_ptr().add(0).cast(),
                combine2x2(accu_0, accu_1),
            );
        }

        accu_2 = _mm256_sub_epi16(accu_2, _mm256_slli_epi16(accu_3, 8));
        unsafe {
            _mm256_storeu_si256(
                result.as_mut_ptr().add(16).cast(),
                combine2x2(accu_2, accu_3),
            );
        }

        result
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn scan_v3_test() {
        if !crate::is_cpu_detected!("v3") {
            println!("test {} ... skipped (v3)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            for n in 90..110 {
                let code = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                let lut = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                unsafe {
                    assert_eq!(scan_v3(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn scan_v2(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        use std::arch::x86_64::*;

        let mut accu_0 = _mm_setzero_si128();
        let mut accu_1 = _mm_setzero_si128();
        let mut accu_2 = _mm_setzero_si128();
        let mut accu_3 = _mm_setzero_si128();

        let mut i = 0_usize;
        while i < n {
            let code = unsafe { _mm_loadu_si128(code.as_ptr().add(i).cast()) };

            let mask = _mm_set1_epi8(0xf);
            let clo = _mm_and_si128(code, mask);
            let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

            let lut = unsafe { _mm_loadu_si128(lut.as_ptr().add(i).cast()) };
            let res_lo = _mm_shuffle_epi8(lut, clo);
            accu_0 = _mm_add_epi16(accu_0, res_lo);
            accu_1 = _mm_add_epi16(accu_1, _mm_srli_epi16(res_lo, 8));
            let res_hi = _mm_shuffle_epi8(lut, chi);
            accu_2 = _mm_add_epi16(accu_2, res_hi);
            accu_3 = _mm_add_epi16(accu_3, _mm_srli_epi16(res_hi, 8));

            i += 1;
        }
        debug_assert_eq!(i, n);

        let mut result = [0_u16; 32];

        accu_0 = _mm_sub_epi16(accu_0, _mm_slli_epi16(accu_1, 8));
        unsafe {
            _mm_storeu_si128(result.as_mut_ptr().add(0).cast(), accu_0);
            _mm_storeu_si128(result.as_mut_ptr().add(8).cast(), accu_1);
        }

        accu_2 = _mm_sub_epi16(accu_2, _mm_slli_epi16(accu_3, 8));
        unsafe {
            _mm_storeu_si128(result.as_mut_ptr().add(16).cast(), accu_2);
            _mm_storeu_si128(result.as_mut_ptr().add(24).cast(), accu_3);
        }

        result
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn scan_v2_test() {
        if !crate::is_cpu_detected!("v2") {
            println!("test {} ... skipped (v2)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            for n in 90..110 {
                let code = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                let lut = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                unsafe {
                    assert_eq!(scan_v2(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn scan_a2(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        use std::arch::aarch64::*;

        let mut accu_0 = vdupq_n_u16(0);
        let mut accu_1 = vdupq_n_u16(0);
        let mut accu_2 = vdupq_n_u16(0);
        let mut accu_3 = vdupq_n_u16(0);

        let mut i = 0_usize;
        while i < n {
            let code = unsafe { vld1q_u8(code.as_ptr().add(i).cast()) };

            let clo = vandq_u8(code, vdupq_n_u8(0xf));
            let chi = vshrq_n_u8(code, 4);

            let lut = unsafe { vld1q_u8(lut.as_ptr().add(i).cast()) };
            let res_lo = vreinterpretq_u16_u8(vqtbl1q_u8(lut, clo));
            accu_0 = vaddq_u16(accu_0, res_lo);
            accu_1 = vaddq_u16(accu_1, vshrq_n_u16(res_lo, 8));
            let res_hi = vreinterpretq_u16_u8(vqtbl1q_u8(lut, chi));
            accu_2 = vaddq_u16(accu_2, res_hi);
            accu_3 = vaddq_u16(accu_3, vshrq_n_u16(res_hi, 8));

            i += 1;
        }
        debug_assert_eq!(i, n);

        let mut result = [0_u16; 32];

        accu_0 = vsubq_u16(accu_0, vshlq_n_u16(accu_1, 8));
        unsafe {
            vst1q_u16(result.as_mut_ptr().add(0).cast(), accu_0);
            vst1q_u16(result.as_mut_ptr().add(8).cast(), accu_1);
        }

        accu_2 = vsubq_u16(accu_2, vshlq_n_u16(accu_3, 8));
        unsafe {
            vst1q_u16(result.as_mut_ptr().add(16).cast(), accu_2);
            vst1q_u16(result.as_mut_ptr().add(24).cast(), accu_3);
        }

        result
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn scan_a2_test() {
        if !crate::is_cpu_detected!("a2") {
            println!("test {} ... skipped (a2)", module_path!());
            return;
        }
        for _ in 0..if cfg!(not(miri)) { 256 } else { 1 } {
            for n in 90..110 {
                let code = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                let lut = (0..n)
                    .map(|_| std::array::from_fn(|_| rand::random()))
                    .collect::<Vec<[u8; 16]>>();
                unsafe {
                    assert_eq!(scan_a2(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[crate::multiversion(@"v4", @"v3", @"v2", @"a2")]
    pub fn scan(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        fn binary(op: impl Fn(u16, u16) -> u16, a: [u16; 8], b: [u16; 8]) -> [u16; 8] {
            std::array::from_fn(|i| op(a[i], b[i]))
        }
        fn shuffle(a: [u8; 16], b: [u8; 16]) -> [u8; 16] {
            std::array::from_fn(|i| a[b[i] as usize])
        }

        assert_eq!(code.len(), lut.len());
        let n = code.len();

        let mut a_0 = [0u16; 8];
        let mut a_1 = [0u16; 8];
        let mut a_2 = [0u16; 8];
        let mut a_3 = [0u16; 8];

        for i in 0..n {
            let code = code[i];

            let clo = code.map(|x| x & 0xf);
            let chi = code.map(|x| x >> 4);

            let lut = lut[i];
            let res_lo = zerocopy::transmute!(shuffle(lut, clo));
            a_0 = binary(u16::wrapping_add, a_0, res_lo);
            a_1 = binary(u16::wrapping_add, a_1, res_lo.map(|x| x >> 8));
            let res_hi = zerocopy::transmute!(shuffle(lut, chi));
            a_2 = binary(u16::wrapping_add, a_2, res_hi);
            a_3 = binary(u16::wrapping_add, a_3, res_hi.map(|x| x >> 8));
        }

        a_0 = binary(u16::wrapping_sub, a_0, a_1.map(|x| x << 8));
        a_2 = binary(u16::wrapping_sub, a_2, a_3.map(|x| x << 8));

        zerocopy::transmute!([a_0, a_1, a_2, a_3])
    }
}

#[inline(always)]
pub fn fast_scan(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
    scan::scan(code, lut)
}
