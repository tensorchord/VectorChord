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

#if defined(__clang__)
#if !(__clang_major__ >= 16)
#error "Clang version must be at least 16."
#endif
#elif defined(__GNUC__)
#if !(__GNUC__ >= 12)
#error "GCC version must be at least 12."
#endif
#else
#error "This file requires Clang or GCC."
#endif

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__clang__) && defined(_MSC_VER) && (__clang_major__ <= 19)
// https://github.com/llvm/llvm-project/issues/53520
// clang-format off
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <lzcntintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <f16cintrin.h>
#include <fmaintrin.h>
#include <bmiintrin.h>
#include <bmi2intrin.h>
#include <lzcntintrin.h>
#include <avx512fintrin.h>
#include <avx512bwintrin.h>
#include <avx512cdintrin.h>
#include <avx512dqintrin.h>
#include <avx512vlintrin.h>
#include <avx512vlbwintrin.h>
#include <avx512vlcdintrin.h>
#include <avx512vldqintrin.h>
#include <avx512fp16intrin.h>
#include <avx512vlfp16intrin.h>
// clang-format on
#endif

typedef _Float16 f16;
typedef float f32;

__attribute__((target("arch=x86-64-v4,avx512fp16"))) float
fp16_reduce_sum_of_xy_v4_avx512fp16(f16 *restrict a, f16 *restrict b, size_t n) {
  __m512h xy = _mm512_setzero_ph();
  while (n >= 32) {
    __m512h x = _mm512_loadu_ph(a);
    __m512h y = _mm512_loadu_ph(b);
    a += 32;
    b += 32;
    n -= 32;
    xy = _mm512_fmadd_ph(x, y, xy);
  }
  if (n > 0) {
    unsigned int mask = _bzhi_u32(0xffffffff, n);
    __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
    __m512h y = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
    xy = _mm512_fmadd_ph(x, y, xy);
  }
  return _mm512_reduce_add_ph(xy);
}

__attribute__((target("arch=x86-64-v4,avx512fp16"))) float
fp16_reduce_sum_of_d2_v4_avx512fp16(f16 *restrict a, f16 *restrict b, size_t n) {
  __m512h d2 = _mm512_setzero_ph();
  while (n >= 32) {
    __m512h x = _mm512_loadu_ph(a);
    __m512h y = _mm512_loadu_ph(b);
    a += 32;
    b += 32;
    n -= 32;
    __m512h d = _mm512_sub_ph(x, y);
    d2 = _mm512_fmadd_ph(d, d, d2);
  }
  if (n > 0) {
    unsigned int mask = _bzhi_u32(0xffffffff, n);
    __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
    __m512h y = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
    __m512h d = _mm512_sub_ph(x, y);
    d2 = _mm512_fmadd_ph(d, d, d2);
  }
  return _mm512_reduce_add_ph(d2);
}
