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

pub fn pack(x: [&[u8]; 32]) -> Vec<[u8; 16]> {
    let n = {
        let l = x.each_ref().map(|i| i.len());
        for i in 1..32 {
            assert!(l[0] == l[i]);
        }
        l[0]
    };
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push([
            x[0][i] | (x[16][i] << 4),
            x[8][i] | (x[24][i] << 4),
            x[1][i] | (x[17][i] << 4),
            x[9][i] | (x[25][i] << 4),
            x[2][i] | (x[18][i] << 4),
            x[10][i] | (x[26][i] << 4),
            x[3][i] | (x[19][i] << 4),
            x[11][i] | (x[27][i] << 4),
            x[4][i] | (x[20][i] << 4),
            x[12][i] | (x[28][i] << 4),
            x[5][i] | (x[21][i] << 4),
            x[13][i] | (x[29][i] << 4),
            x[6][i] | (x[22][i] << 4),
            x[14][i] | (x[30][i] << 4),
            x[7][i] | (x[23][i] << 4),
            x[15][i] | (x[31][i] << 4),
        ]);
    }
    result
}

pub fn unpack(x: &[[u8; 16]]) -> [Vec<u8>; 32] {
    let n = x.len();
    let mut result = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        result[0].push(x[i][0] & 0xf);
        result[1].push(x[i][2] & 0xf);
        result[2].push(x[i][4] & 0xf);
        result[3].push(x[i][6] & 0xf);
        result[4].push(x[i][8] & 0xf);
        result[5].push(x[i][10] & 0xf);
        result[6].push(x[i][12] & 0xf);
        result[7].push(x[i][14] & 0xf);
        result[8].push(x[i][1] & 0xf);
        result[9].push(x[i][3] & 0xf);
        result[10].push(x[i][5] & 0xf);
        result[11].push(x[i][7] & 0xf);
        result[12].push(x[i][9] & 0xf);
        result[13].push(x[i][11] & 0xf);
        result[14].push(x[i][13] & 0xf);
        result[15].push(x[i][15] & 0xf);
        result[16].push(x[i][0] >> 4);
        result[17].push(x[i][2] >> 4);
        result[18].push(x[i][4] >> 4);
        result[19].push(x[i][6] >> 4);
        result[20].push(x[i][8] >> 4);
        result[21].push(x[i][10] >> 4);
        result[22].push(x[i][12] >> 4);
        result[23].push(x[i][14] >> 4);
        result[24].push(x[i][1] >> 4);
        result[25].push(x[i][3] >> 4);
        result[26].push(x[i][5] >> 4);
        result[27].push(x[i][7] >> 4);
        result[28].push(x[i][9] >> 4);
        result[29].push(x[i][11] >> 4);
        result[30].push(x[i][13] >> 4);
        result[31].push(x[i][15] >> 4);
    }
    result
}

pub fn padding_pack(x: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Vec<[u8; 16]> {
    let x = x.into_iter().collect::<Vec<_>>();
    let x = x.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    if x.is_empty() || x.len() > 32 {
        panic!("too few or too many slices");
    }
    let n = x[0].len();
    let t = vec![0; n];
    pack(std::array::from_fn(|i| {
        if i < x.len() { x[i] } else { t.as_slice() }
    }))
}

pub fn any_pack<T: Default>(mut x: impl Iterator<Item = T>) -> [T; 32] {
    std::array::from_fn(|_| x.next()).map(|x| x.unwrap_or_default())
}

#[allow(clippy::module_inception)]
mod fast_scan {
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v4.512")]
    fn fast_scan_v4_512(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        unsafe {
            use std::arch::x86_64::*;

            #[inline]
            #[crate::target_cpu(enable = "v4.512")]
            fn combine2x2(x0x1: __m256i, y0y1: __m256i) -> __m256i {
                unsafe {
                    let x1y0 = _mm256_permute2f128_si256(x0x1, y0y1, 0x21);
                    let x0y1 = _mm256_blend_epi32(x0x1, y0y1, 0xf0);
                    _mm256_add_epi16(x1y0, x0y1)
                }
            }

            #[inline]
            #[crate::target_cpu(enable = "v4.512")]
            fn combine4x2(x0x1x2x3: __m512i, y0y1y2y3: __m512i) -> __m256i {
                unsafe {
                    let x0x1 = _mm512_castsi512_si256(x0x1x2x3);
                    let x2x3 = _mm512_extracti64x4_epi64(x0x1x2x3, 1);
                    let y0y1 = _mm512_castsi512_si256(y0y1y2y3);
                    let y2y3 = _mm512_extracti64x4_epi64(y0y1y2y3, 1);
                    let x01y01 = combine2x2(x0x1, y0y1);
                    let x23y23 = combine2x2(x2x3, y2y3);
                    _mm256_add_epi16(x01y01, x23y23)
                }
            }

            let mut accu_0 = _mm512_setzero_si512();
            let mut accu_1 = _mm512_setzero_si512();
            let mut accu_2 = _mm512_setzero_si512();
            let mut accu_3 = _mm512_setzero_si512();

            let mut i = 0_usize;
            while i + 4 <= n {
                let code = _mm512_loadu_si512(code.as_ptr().add(i).cast());

                let mask = _mm512_set1_epi8(0xf);
                let clo = _mm512_and_si512(code, mask);
                let chi = _mm512_and_si512(_mm512_srli_epi16(code, 4), mask);

                let lut = _mm512_loadu_si512(lut.as_ptr().add(i).cast());
                let res_lo = _mm512_shuffle_epi8(lut, clo);
                accu_0 = _mm512_add_epi16(accu_0, res_lo);
                accu_1 = _mm512_add_epi16(accu_1, _mm512_srli_epi16(res_lo, 8));
                let res_hi = _mm512_shuffle_epi8(lut, chi);
                accu_2 = _mm512_add_epi16(accu_2, res_hi);
                accu_3 = _mm512_add_epi16(accu_3, _mm512_srli_epi16(res_hi, 8));

                i += 4;
            }
            if i + 2 <= n {
                let code = _mm256_loadu_si256(code.as_ptr().add(i).cast());

                let mask = _mm256_set1_epi8(0xf);
                let clo = _mm256_and_si256(code, mask);
                let chi = _mm256_and_si256(_mm256_srli_epi16(code, 4), mask);

                let lut = _mm256_loadu_si256(lut.as_ptr().add(i).cast());
                let res_lo = _mm512_zextsi256_si512(_mm256_shuffle_epi8(lut, clo));
                accu_0 = _mm512_add_epi16(accu_0, res_lo);
                accu_1 = _mm512_add_epi16(accu_1, _mm512_srli_epi16(res_lo, 8));
                let res_hi = _mm512_zextsi256_si512(_mm256_shuffle_epi8(lut, chi));
                accu_2 = _mm512_add_epi16(accu_2, res_hi);
                accu_3 = _mm512_add_epi16(accu_3, _mm512_srli_epi16(res_hi, 8));

                i += 2;
            }
            if i < n {
                let code = _mm_loadu_si128(code.as_ptr().add(i).cast());

                let mask = _mm_set1_epi8(0xf);
                let clo = _mm_and_si128(code, mask);
                let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

                let lut = _mm_loadu_si128(lut.as_ptr().add(i).cast());
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
            _mm256_storeu_si256(
                result.as_mut_ptr().add(0).cast(),
                combine4x2(accu_0, accu_1),
            );

            accu_2 = _mm512_sub_epi16(accu_2, _mm512_slli_epi16(accu_3, 8));
            _mm256_storeu_si256(
                result.as_mut_ptr().add(16).cast(),
                combine4x2(accu_2, accu_3),
            );

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn fast_scan_v4_test() {
        if !crate::is_cpu_detected!("v4.512") {
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
                    assert_eq!(fast_scan_v4_512(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v3")]
    fn fast_scan_v3(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        unsafe {
            use std::arch::x86_64::*;

            #[inline]
            #[crate::target_cpu(enable = "v3")]
            fn combine2x2(x0x1: __m256i, y0y1: __m256i) -> __m256i {
                unsafe {
                    let x1y0 = _mm256_permute2f128_si256(x0x1, y0y1, 0x21);
                    let x0y1 = _mm256_blend_epi32(x0x1, y0y1, 0xf0);
                    _mm256_add_epi16(x1y0, x0y1)
                }
            }

            let mut accu_0 = _mm256_setzero_si256();
            let mut accu_1 = _mm256_setzero_si256();
            let mut accu_2 = _mm256_setzero_si256();
            let mut accu_3 = _mm256_setzero_si256();

            let mut i = 0_usize;
            while i + 2 <= n {
                let code = _mm256_loadu_si256(code.as_ptr().add(i).cast());

                let mask = _mm256_set1_epi8(0xf);
                let clo = _mm256_and_si256(code, mask);
                let chi = _mm256_and_si256(_mm256_srli_epi16(code, 4), mask);

                let lut = _mm256_loadu_si256(lut.as_ptr().add(i).cast());
                let res_lo = _mm256_shuffle_epi8(lut, clo);
                accu_0 = _mm256_add_epi16(accu_0, res_lo);
                accu_1 = _mm256_add_epi16(accu_1, _mm256_srli_epi16(res_lo, 8));
                let res_hi = _mm256_shuffle_epi8(lut, chi);
                accu_2 = _mm256_add_epi16(accu_2, res_hi);
                accu_3 = _mm256_add_epi16(accu_3, _mm256_srli_epi16(res_hi, 8));

                i += 2;
            }
            if i < n {
                let code = _mm_loadu_si128(code.as_ptr().add(i).cast());

                let mask = _mm_set1_epi8(0xf);
                let clo = _mm_and_si128(code, mask);
                let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

                let lut = _mm_loadu_si128(lut.as_ptr().add(i).cast());
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
            _mm256_storeu_si256(
                result.as_mut_ptr().add(0).cast(),
                combine2x2(accu_0, accu_1),
            );

            accu_2 = _mm256_sub_epi16(accu_2, _mm256_slli_epi16(accu_3, 8));
            _mm256_storeu_si256(
                result.as_mut_ptr().add(16).cast(),
                combine2x2(accu_2, accu_3),
            );

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn fast_scan_v3_test() {
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
                    assert_eq!(fast_scan_v3(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[crate::target_cpu(enable = "v2")]
    fn fast_scan_v2(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        unsafe {
            use std::arch::x86_64::*;

            let mut accu_0 = _mm_setzero_si128();
            let mut accu_1 = _mm_setzero_si128();
            let mut accu_2 = _mm_setzero_si128();
            let mut accu_3 = _mm_setzero_si128();

            let mut i = 0_usize;
            while i < n {
                let code = _mm_loadu_si128(code.as_ptr().add(i).cast());

                let mask = _mm_set1_epi8(0xf);
                let clo = _mm_and_si128(code, mask);
                let chi = _mm_and_si128(_mm_srli_epi16(code, 4), mask);

                let lut = _mm_loadu_si128(lut.as_ptr().add(i).cast());
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
            _mm_storeu_si128(result.as_mut_ptr().add(0).cast(), accu_0);
            _mm_storeu_si128(result.as_mut_ptr().add(8).cast(), accu_1);

            accu_2 = _mm_sub_epi16(accu_2, _mm_slli_epi16(accu_3, 8));
            _mm_storeu_si128(result.as_mut_ptr().add(16).cast(), accu_2);
            _mm_storeu_si128(result.as_mut_ptr().add(24).cast(), accu_3);

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", test, not(miri)))]
    #[test]
    fn fast_scan_v2_test() {
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
                    assert_eq!(fast_scan_v2(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[crate::target_cpu(enable = "a2")]
    fn fast_scan_a2(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
        // bounds checking is not enforced by compiler, so check it manually
        assert_eq!(code.len(), lut.len());
        let n = code.len();

        unsafe {
            use std::arch::aarch64::*;

            let mut accu_0 = vdupq_n_u16(0);
            let mut accu_1 = vdupq_n_u16(0);
            let mut accu_2 = vdupq_n_u16(0);
            let mut accu_3 = vdupq_n_u16(0);

            let mut i = 0_usize;
            while i < n {
                let code = vld1q_u8(code.as_ptr().add(i).cast());

                let clo = vandq_u8(code, vdupq_n_u8(0xf));
                let chi = vshrq_n_u8(code, 4);

                let lut = vld1q_u8(lut.as_ptr().add(i).cast());
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
            vst1q_u16(result.as_mut_ptr().add(0).cast(), accu_0);
            vst1q_u16(result.as_mut_ptr().add(8).cast(), accu_1);

            accu_2 = vsubq_u16(accu_2, vshlq_n_u16(accu_3, 8));
            vst1q_u16(result.as_mut_ptr().add(16).cast(), accu_2);
            vst1q_u16(result.as_mut_ptr().add(24).cast(), accu_3);

            result
        }
    }

    #[cfg(all(target_arch = "aarch64", test, not(miri)))]
    #[test]
    fn fast_scan_a2_test() {
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
                    assert_eq!(fast_scan_a2(&code, &lut), fallback(&code, &lut));
                }
            }
        }
    }

    #[crate::multiversion(@"v4.512", @"v3", @"v2", @"a2")]
    pub fn fast_scan(code: &[[u8; 16]], lut: &[[u8; 16]]) -> [u16; 32] {
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
    fast_scan::fast_scan(code, lut)
}
