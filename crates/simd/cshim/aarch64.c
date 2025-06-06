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
#if !(__GNUC__ >= 14)
#error "GCC version must be at least 14."
#endif
#else
#error "This file requires Clang or GCC."
#endif

#include <arm_neon.h>
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;
typedef float f32;

__attribute__((target("+fp16"))) float
fp16_reduce_sum_of_xy_a2_fp16(f16 *restrict a, f16 *restrict b, size_t n) {
  float16x8_t xy_0 = vdupq_n_f16(0.0);
  float16x8_t xy_1 = vdupq_n_f16(0.0);
  float16x8_t xy_2 = vdupq_n_f16(0.0);
  float16x8_t xy_3 = vdupq_n_f16(0.0);
  while (n >= 32) {
    float16x8_t x_0 = vld1q_f16(a + 0);
    float16x8_t x_1 = vld1q_f16(a + 8);
    float16x8_t x_2 = vld1q_f16(a + 16);
    float16x8_t x_3 = vld1q_f16(a + 24);
    float16x8_t y_0 = vld1q_f16(b + 0);
    float16x8_t y_1 = vld1q_f16(b + 8);
    float16x8_t y_2 = vld1q_f16(b + 16);
    float16x8_t y_3 = vld1q_f16(b + 24);
    a += 32;
    b += 32;
    n -= 32;
    xy_0 = vfmaq_f16(xy_0, x_0, y_0);
    xy_1 = vfmaq_f16(xy_1, x_1, y_1);
    xy_2 = vfmaq_f16(xy_2, x_2, y_2);
    xy_3 = vfmaq_f16(xy_3, x_3, y_3);
  }
  if (n > 0) {
    f16 A[32] = {};
    f16 B[32] = {};
    for (size_t i = 0; i < n; i += 1) {
      A[i] = a[i];
      B[i] = b[i];
    }
    float16x8_t x_0 = vld1q_f16(A + 0);
    float16x8_t x_1 = vld1q_f16(A + 8);
    float16x8_t x_2 = vld1q_f16(A + 16);
    float16x8_t x_3 = vld1q_f16(A + 24);
    float16x8_t y_0 = vld1q_f16(B + 0);
    float16x8_t y_1 = vld1q_f16(B + 8);
    float16x8_t y_2 = vld1q_f16(B + 16);
    float16x8_t y_3 = vld1q_f16(B + 24);
    xy_0 = vfmaq_f16(xy_0, x_0, y_0);
    xy_1 = vfmaq_f16(xy_1, x_1, y_1);
    xy_2 = vfmaq_f16(xy_2, x_2, y_2);
    xy_3 = vfmaq_f16(xy_3, x_3, y_3);
  }
  float16x8_t xy = vaddq_f16(vaddq_f16(xy_0, xy_1), vaddq_f16(xy_2, xy_3));
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(xy));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(xy));
  return vaddvq_f32(lo) + vaddvq_f32(hi);
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_xy_a3_512(f16 *restrict a, f16 *restrict b, size_t n) {
  svfloat16_t xy = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    xy = svmla_f16_x(mask, xy, x, y);
  }
  return svaddv_f16(svptrue_b16(), xy);
}

__attribute__((target("+fp16"))) float
fp16_reduce_sum_of_d2_a2_fp16(f16 *restrict a, f16 *restrict b, size_t n) {
  float16x8_t d2_0 = vdupq_n_f16(0.0);
  float16x8_t d2_1 = vdupq_n_f16(0.0);
  float16x8_t d2_2 = vdupq_n_f16(0.0);
  float16x8_t d2_3 = vdupq_n_f16(0.0);
  while (n >= 32) {
    float16x8_t x_0 = vld1q_f16(a + 0);
    float16x8_t x_1 = vld1q_f16(a + 8);
    float16x8_t x_2 = vld1q_f16(a + 16);
    float16x8_t x_3 = vld1q_f16(a + 24);
    float16x8_t y_0 = vld1q_f16(b + 0);
    float16x8_t y_1 = vld1q_f16(b + 8);
    float16x8_t y_2 = vld1q_f16(b + 16);
    float16x8_t y_3 = vld1q_f16(b + 24);
    a += 32;
    b += 32;
    n -= 32;
    float16x8_t d_0 = vsubq_f16(x_0, y_0);
    float16x8_t d_1 = vsubq_f16(x_1, y_1);
    float16x8_t d_2 = vsubq_f16(x_2, y_2);
    float16x8_t d_3 = vsubq_f16(x_3, y_3);
    d2_0 = vfmaq_f16(d2_0, d_0, d_0);
    d2_1 = vfmaq_f16(d2_1, d_1, d_1);
    d2_2 = vfmaq_f16(d2_2, d_2, d_2);
    d2_3 = vfmaq_f16(d2_3, d_3, d_3);
  }
  if (n > 0) {
    f16 A[32] = {};
    f16 B[32] = {};
    for (size_t i = 0; i < n; i += 1) {
      A[i] = a[i];
      B[i] = b[i];
    }
    float16x8_t x_0 = vld1q_f16(A + 0);
    float16x8_t x_1 = vld1q_f16(A + 8);
    float16x8_t x_2 = vld1q_f16(A + 16);
    float16x8_t x_3 = vld1q_f16(A + 24);
    float16x8_t y_0 = vld1q_f16(B + 0);
    float16x8_t y_1 = vld1q_f16(B + 8);
    float16x8_t y_2 = vld1q_f16(B + 16);
    float16x8_t y_3 = vld1q_f16(B + 24);
    float16x8_t d_0 = vsubq_f16(x_0, y_0);
    float16x8_t d_1 = vsubq_f16(x_1, y_1);
    float16x8_t d_2 = vsubq_f16(x_2, y_2);
    float16x8_t d_3 = vsubq_f16(x_3, y_3);
    d2_0 = vfmaq_f16(d2_0, d_0, d_0);
    d2_1 = vfmaq_f16(d2_1, d_1, d_1);
    d2_2 = vfmaq_f16(d2_2, d_2, d_2);
    d2_3 = vfmaq_f16(d2_3, d_3, d_3);
  }
  float16x8_t d2 = vaddq_f16(vaddq_f16(d2_0, d2_1), vaddq_f16(d2_2, d2_3));
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(d2));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(d2));
  return vaddvq_f32(lo) + vaddvq_f32(hi);
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_d2_a3_512(f16 *restrict a, f16 *restrict b, size_t n) {
  svfloat16_t d2 = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    svfloat16_t d = svsub_f16_x(mask, x, y);
    d2 = svmla_f16_x(mask, d2, d, d);
  }
  return svaddv_f16(svptrue_b16(), d2);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_x_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svadd_f32_x(mask, sum, x);
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_abs_x_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svadd_f32_x(mask, sum, svabs_f32_x(mask, x));
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_x2_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svmla_f32_x(mask, sum, x, x);
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) void
fp32_reduce_min_max_of_x_a3_256(float *restrict this, size_t n, float *out_min,
                                float *out_max) {
  svfloat32_t min = svdup_f32(1.0 / 0.0);
  svfloat32_t max = svdup_f32(-1.0 / 0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    min = svmin_f32_x(mask, min, x);
    max = svmax_f32_x(mask, max, x);
  }
  *out_min = svminv_f32(svptrue_b32(), min);
  *out_max = svmaxv_f32(svptrue_b32(), max);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_xy_a3_256(float *restrict lhs, float *restrict rhs,
                             size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, lhs + i);
    svfloat32_t y = svld1_f32(mask, rhs + i);
    sum = svmla_f32_x(mask, sum, x, y);
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_d2_a3_256(float *restrict lhs, float *restrict rhs,
                             size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, lhs + i);
    svfloat32_t y = svld1_f32(mask, rhs + i);
    svfloat32_t d = svsub_f32_x(mask, x, y);
    sum = svmla_f32_x(mask, sum, d, d);
  }
  return svaddv_f32(svptrue_b32(), sum);
}
