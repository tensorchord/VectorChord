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

#![allow(unsafe_code)]

use criterion::{Criterion, criterion_group, criterion_main};

fn floating_f32_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let x = (0..4095)
        .map(|_| rng.random_range(-1.0..=1.0f32))
        .collect::<Vec<_>>();
    let y = (0..4095)
        .map(|_| rng.random_range(-1.0..=1.0f32))
        .collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("floating_f32::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("floating_f32::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") && simd::is_feature_detected!("fma") {
        c.bench_function("floating_f32::reduce_sum_of_xy::v2_fma", |b| {
            b.iter(|| unsafe {
                simd::floating_f32::reduce_sum_of_xy::reduce_sum_of_xy_v2_fma(&x, &y)
            })
        });
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if simd::is_cpu_detected!("a3.256") {
        c.bench_function("floating_f32::reduce_sum_of_xy::a3_256", |b| {
            b.iter(|| unsafe {
                simd::floating_f32::reduce_sum_of_xy::reduce_sum_of_xy_a3_256(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") {
        c.bench_function("floating_f32::reduce_sum_of_xy::a2", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_xy::reduce_sum_of_xy_a2(&x, &y) })
        });
    }
}

fn floating_f32_reduce_sum_of_d2(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let x = (0..4095)
        .map(|_| rng.random_range(-1.0..=1.0f32))
        .collect::<Vec<_>>();
    let y = (0..4095)
        .map(|_| rng.random_range(-1.0..=1.0f32))
        .collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("floating_f32::reduce_sum_of_d2::v4", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_d2::reduce_sum_of_d2_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("floating_f32::reduce_sum_of_d2::v3", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_d2::reduce_sum_of_d2_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") && simd::is_feature_detected!("fma") {
        c.bench_function("floating_f32::reduce_sum_of_d2::v2_fma", |b| {
            b.iter(|| unsafe {
                simd::floating_f32::reduce_sum_of_d2::reduce_sum_of_d2_v2_fma(&x, &y)
            })
        });
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if simd::is_cpu_detected!("a3.256") {
        c.bench_function("floating_f32::reduce_sum_of_d2::a3_256", |b| {
            b.iter(|| unsafe {
                simd::floating_f32::reduce_sum_of_d2::reduce_sum_of_d2_a3_256(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") {
        c.bench_function("floating_f32::reduce_sum_of_d2::a2", |b| {
            b.iter(|| unsafe { simd::floating_f32::reduce_sum_of_d2::reduce_sum_of_d2_a2(&x, &y) })
        });
    }
}

fn floating_f16_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::RngExt;
    use simd::{F16, f16};
    let mut rng = rand::rng();
    let x = (0..4095)
        .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
        .collect::<Vec<_>>();
    let y = (0..4095)
        .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
        .collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512fp16") {
        c.bench_function("floating_f16::reduce_sum_of_xy::v4_avx512fp16", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_xy::reduce_sum_of_xy_v4_avx512fp16(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("floating_f16::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::floating_f16::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("floating_f16::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::floating_f16::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a3.512") {
        c.bench_function("floating_f16::reduce_sum_of_xy::a3_512", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_xy::reduce_sum_of_xy_a3_512(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") && simd::is_feature_detected!("fp16") {
        c.bench_function("floating_f16::reduce_sum_of_xy::a2_fp16", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_xy::reduce_sum_of_xy_a2_fp16(&x, &y)
            })
        });
    }
}

fn floating_f16_reduce_sum_of_d2(c: &mut Criterion) {
    use rand::RngExt;
    use simd::{F16, f16};
    let mut rng = rand::rng();
    let x = (0..4095)
        .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
        .collect::<Vec<_>>();
    let y = (0..4095)
        .map(|_| f16::_from_f32(rng.random_range(-1.0..=1.0)))
        .collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512fp16") {
        c.bench_function("floating_f16::reduce_sum_of_d2::v4_avx512fp16", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_d2::reduce_sum_of_d2_v4_avx512fp16(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("floating_f16::reduce_sum_of_d2::v4", |b| {
            b.iter(|| unsafe { simd::floating_f16::reduce_sum_of_d2::reduce_sum_of_d2_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("floating_f16::reduce_sum_of_d2::v3", |b| {
            b.iter(|| unsafe { simd::floating_f16::reduce_sum_of_d2::reduce_sum_of_d2_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a3.512") {
        c.bench_function("floating_f16::reduce_sum_of_d2::a3_512", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_d2::reduce_sum_of_d2_a3_512(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") && simd::is_feature_detected!("fp16") {
        c.bench_function("floating_f16::reduce_sum_of_d2::a2_fp16", |b| {
            b.iter(|| unsafe {
                simd::floating_f16::reduce_sum_of_d2::reduce_sum_of_d2_a2_fp16(&x, &y)
            })
        });
    }
}

fn byte_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let x = (0..4095).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    let y = (0..4095).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512vnni") {
        c.bench_function("byte::reduce_sum_of_xy::v4_avx512vnni", |b| {
            b.iter(|| unsafe {
                simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v4_avx512vnni(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("byte::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("byte::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("byte::reduce_sum_of_xy::v2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_v2(&x, &y) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") && simd::is_feature_detected!("dotprod") {
        c.bench_function("byte::reduce_sum_of_xy::a2_prod", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_a2_dotprod(&x, &y) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") {
        c.bench_function("byte::reduce_sum_of_xy::a2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_xy::reduce_sum_of_xy_a2(&x, &y) })
        });
    }
}

fn byte_reduce_sum_of_x(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let this = (0..4095).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("byte::reduce_sum_of_x::v4", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v4(&this) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("byte::reduce_sum_of_x::v3", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v3(&this) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("byte::reduce_sum_of_x::v2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_v2(&this) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") {
        c.bench_function("byte::reduce_sum_of_x::a2", |b| {
            b.iter(|| unsafe { simd::byte::reduce_sum_of_x::reduce_sum_of_x_a2(&this) })
        });
    }
}

fn halfbyte_reduce_sum_of_xy(c: &mut Criterion) {
    use rand::RngExt;
    let mut rng = rand::rng();
    let x = (0..2047).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    let y = (0..2047).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") && simd::is_feature_detected!("avx512vnni") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v4_avx512vnni", |b| {
            b.iter(|| unsafe {
                simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v4_avx512vnni(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v4") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v4", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v4(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v3") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v3", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v3(&x, &y) })
        });
    }
    #[cfg(target_arch = "x86_64")]
    if simd::is_cpu_detected!("v2") {
        c.bench_function("halfbyte::reduce_sum_of_xy::v2", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_v2(&x, &y) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") && simd::is_feature_detected!("dotprod") {
        c.bench_function("halfbyte::reduce_sum_of_xy::a2_prod", |b| {
            b.iter(|| unsafe {
                simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_a2_dotprod(&x, &y)
            })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if simd::is_cpu_detected!("a2") {
        c.bench_function("halfbyte::reduce_sum_of_xy::a2", |b| {
            b.iter(|| unsafe { simd::halfbyte::reduce_sum_of_xy::reduce_sum_of_xy_a2(&x, &y) })
        });
    }
}

criterion_group!(
    benches,
    floating_f32_reduce_sum_of_xy,
    floating_f32_reduce_sum_of_d2,
    floating_f16_reduce_sum_of_xy,
    floating_f16_reduce_sum_of_d2,
    byte_reduce_sum_of_xy,
    byte_reduce_sum_of_x,
    halfbyte_reduce_sum_of_xy,
);
criterion_main!(benches);
