#if defined(__clang__)
#if !(__clang_major__ >= 16)
#error "Clang version must be at least 16."
#endif
#else
#error "This file requires Clang."
#endif

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

typedef _Float16 f16;
typedef float f32;

__attribute__((target("arch=x86-64-v4,avx512fp16"))) float
fp16_reduce_sum_of_xy_v4fp16(f16 *restrict a, f16 *restrict b, size_t n) {
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
fp16_reduce_sum_of_d2_v4fp16(f16 *restrict a, f16 *restrict b, size_t n) {
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
