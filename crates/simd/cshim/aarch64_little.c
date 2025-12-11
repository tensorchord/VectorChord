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

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_xy_a3_512(f16 *restrict a, f16 *restrict b, size_t n) {
  svfloat16_t xy = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    xy = svmla_f16_m(mask, xy, x, y);
  }
  return svaddv_f16(svptrue_b16(), xy);
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_d2_a3_512(f16 *restrict a, f16 *restrict b, size_t n) {
  svfloat16_t d2 = svdup_f16(0.0);
  for (size_t i = 0; i < n; i += svcnth()) {
    svbool_t mask = svwhilelt_b16((int64_t)i, (int64_t)n);
    svfloat16_t x = svld1_f16(mask, a + i);
    svfloat16_t y = svld1_f16(mask, b + i);
    svfloat16_t d = svsub_f16_z(mask, x, y);
    d2 = svmla_f16_m(mask, d2, d, d);
  }
  return svaddv_f16(svptrue_b16(), d2);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_x_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svadd_f32_m(mask, sum, x);
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_abs_x_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svadd_f32_m(mask, sum, svabs_f32_z(mask, x));
  }
  return svaddv_f32(svptrue_b32(), sum);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_x2_a3_256(float *restrict this, size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, this + i);
    sum = svmla_f32_m(mask, sum, x, x);
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
    min = svminnm_f32_m(mask, min, x);
    max = svmaxnm_f32_m(mask, max, x);
  }
  *out_min = svminnmv_f32(svptrue_b32(), min);
  *out_max = svmaxnmv_f32(svptrue_b32(), max);
}

__attribute__((target("+sve"))) float
fp32_reduce_sum_of_xy_a3_256(float *restrict lhs, float *restrict rhs,
                             size_t n) {
  svfloat32_t sum = svdup_f32(0.0);
  for (size_t i = 0; i < n; i += svcntw()) {
    svbool_t mask = svwhilelt_b32((int64_t)i, (int64_t)n);
    svfloat32_t x = svld1_f32(mask, lhs + i);
    svfloat32_t y = svld1_f32(mask, rhs + i);
    sum = svmla_f32_m(mask, sum, x, y);
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
    svfloat32_t d = svsub_f32_z(mask, x, y);
    sum = svmla_f32_m(mask, sum, d, d);
  }
  return svaddv_f32(svptrue_b32(), sum);
}
