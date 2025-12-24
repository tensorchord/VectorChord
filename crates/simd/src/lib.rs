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
#![cfg_attr(feature = "nightly_f16", feature(f16))]
#![cfg_attr(target_arch = "s390x", feature(s390x_target_feature))]
#![cfg_attr(target_arch = "s390x", feature(stdarch_s390x))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "powerpc64", feature(powerpc_target_feature))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc))]
#![cfg_attr(target_arch = "riscv64", feature(stdarch_riscv_feature_detection))]
#![cfg_attr(target_arch = "riscv64", feature(riscv_target_feature))]

mod aligned;
mod emulate;
mod floating_f16;
mod floating_f32;

pub mod bit;
pub mod byte;
pub mod fast_scan;
pub mod fht;
pub mod halfbyte;
pub mod quantize;
pub mod rotate;

#[cfg(not(feature = "nightly_f16"))]
pub use half::f16;

#[cfg(feature = "nightly_f16")]
pub use f16;

pub trait F16: Sized {
    const _ZERO: Self;

    fn _from_f32(x: f32) -> Self;

    fn _to_f32(self) -> f32;
}

#[cfg(not(feature = "nightly_f16"))]
impl F16 for f16 {
    const _ZERO: Self = f16::ZERO;

    fn _from_f32(x: f32) -> Self {
        f16::from_f32(x)
    }

    fn _to_f32(self) -> f32 {
        self.to_f32()
    }
}

#[cfg(feature = "nightly_f16")]
impl F16 for f16 {
    const _ZERO: Self = 0.0;

    fn _from_f32(x: f32) -> Self {
        x as _
    }

    fn _to_f32(self) -> f32 {
        self as _
    }
}

pub trait Floating:
    Copy + Send + Sync + std::fmt::Debug + Default + 'static + PartialEq + PartialOrd
{
    fn zero() -> Self;
    fn infinity() -> Self;
    fn mask(self, m: bool) -> Self;

    fn scalar_neg(this: Self) -> Self;
    fn scalar_add(lhs: Self, rhs: Self) -> Self;
    fn scalar_sub(lhs: Self, rhs: Self) -> Self;
    fn scalar_mul(lhs: Self, rhs: Self) -> Self;

    fn reduce_or_of_is_zero_x(this: &[Self]) -> bool;
    fn reduce_sum_of_x(this: &[Self]) -> f32;
    fn reduce_sum_of_abs_x(this: &[Self]) -> f32;
    fn reduce_sum_of_x2(this: &[Self]) -> f32;
    fn reduce_min_max_of_x(this: &[Self]) -> (f32, f32);
    fn reduce_sum_of_xy(lhs: &[Self], rhs: &[Self]) -> f32;
    fn reduce_sum_of_d2(lhs: &[Self], rhs: &[Self]) -> f32;
    fn reduce_sum_of_xy_sparse(lidx: &[u32], lval: &[Self], ridx: &[u32], rval: &[Self]) -> f32;
    fn reduce_sum_of_d2_sparse(lidx: &[u32], lval: &[Self], ridx: &[u32], rval: &[Self]) -> f32;

    fn vector_from_f32(this: &[f32]) -> Vec<Self>;
    fn vector_to_f32(this: &[Self]) -> Vec<f32>;
    fn vector_to_f32_borrowed(this: &[Self]) -> impl AsRef<[f32]>;
    fn vector_add(lhs: &[Self], rhs: &[Self]) -> Vec<Self>;
    fn vector_add_inplace(lhs: &mut [Self], rhs: &[Self]);
    fn vector_sub(lhs: &[Self], rhs: &[Self]) -> Vec<Self>;
    fn vector_mul(lhs: &[Self], rhs: &[Self]) -> Vec<Self>;
    fn vector_mul_scalar(lhs: &[Self], rhs: f32) -> Vec<Self>;
    fn vector_mul_scalar_inplace(lhs: &mut [Self], rhs: f32);
    fn vector_abs_inplace(this: &mut [Self]);
}

mod internal {
    #[cfg(target_arch = "x86_64")]
    simd_macros::define_is_cpu_detected!("x86_64");

    #[cfg(target_arch = "aarch64")]
    simd_macros::define_is_cpu_detected!("aarch64");

    #[cfg(target_arch = "s390x")]
    simd_macros::define_is_cpu_detected!("s390x");

    #[cfg(target_arch = "powerpc64")]
    simd_macros::define_is_cpu_detected!("powerpc64");

    #[cfg(target_arch = "riscv64")]
    simd_macros::define_is_cpu_detected!("riscv64");

    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    pub use is_x86_64_cpu_detected;

    #[cfg(target_arch = "aarch64")]
    #[allow(unused_imports)]
    pub use is_aarch64_cpu_detected;

    #[cfg(target_arch = "s390x")]
    #[allow(unused_imports)]
    pub use is_s390x_cpu_detected;

    #[cfg(target_arch = "powerpc64")]
    #[allow(unused_imports)]
    pub use is_powerpc64_cpu_detected;

    #[cfg(target_arch = "riscv64")]
    #[allow(unused_imports)]
    pub use is_riscv64_cpu_detected;

    #[cfg(target_arch = "x86_64")]
    pub fn is_v4_detected() -> bool {
        std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512cd")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "x86_64")]
    pub fn is_v3_detected() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("lzcnt")
            && std::arch::is_x86_feature_detected!("movbe")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "x86_64")]
    pub fn is_v2_detected() -> bool {
        std::arch::is_x86_feature_detected!("sse4.2")
            && std::arch::is_x86_feature_detected!("popcnt")
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(target_endian = "big", expect(dead_code))]
    pub fn is_a3_512_detected() -> bool {
        #[target_feature(enable = "sve")]
        fn is_512_detected() -> bool {
            let vl: u64;
            unsafe {
                core::arch::asm!(
                    "rdvl {0}, #8",
                    out(reg) vl
                );
            }
            vl >= 512
        }
        std::arch::is_aarch64_feature_detected!("sve") && unsafe { is_512_detected() }
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(target_endian = "big", expect(dead_code))]
    pub fn is_a3_256_detected() -> bool {
        #[target_feature(enable = "sve")]
        fn is_256_detected() -> bool {
            let vl: u64;
            unsafe {
                core::arch::asm!(
                    "rdvl {0}, #8",
                    out(reg) vl
                );
            }
            vl >= 256
        }
        std::arch::is_aarch64_feature_detected!("sve") && unsafe { is_256_detected() }
    }

    #[cfg(target_arch = "aarch64")]
    #[expect(dead_code)]
    pub fn is_a3_128_detected() -> bool {
        std::arch::is_aarch64_feature_detected!("sve")
    }

    #[cfg(target_arch = "aarch64")]
    pub fn is_a2_detected() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z17_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-3")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-3")
            && std::arch::is_s390x_feature_detected!("nnp-assist")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-4")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z16_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement-2")
            && std::arch::is_s390x_feature_detected!("nnp-assist")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z15_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-2")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal-enhancement")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-3")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z14_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
            && std::arch::is_s390x_feature_detected!("vector-enhancements-1")
            && std::arch::is_s390x_feature_detected!("vector-packed-decimal")
            && std::arch::is_s390x_feature_detected!("miscellaneous-extensions-2")
    }

    #[cfg(target_arch = "s390x")]
    pub fn is_z13_detected() -> bool {
        std::arch::is_s390x_feature_detected!("vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p9_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
            && std::arch::is_powerpc64_feature_detected!("power8-altivec")
            && std::arch::is_powerpc64_feature_detected!("power8-crypto")
            && std::arch::is_powerpc64_feature_detected!("power8-vector")
            && std::arch::is_powerpc64_feature_detected!("power9-altivec")
            && std::arch::is_powerpc64_feature_detected!("power9-vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p8_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
            && std::arch::is_powerpc64_feature_detected!("power8-altivec")
            && std::arch::is_powerpc64_feature_detected!("power8-crypto")
            && std::arch::is_powerpc64_feature_detected!("power8-vector")
    }

    #[cfg(target_arch = "powerpc64")]
    pub fn is_p7_detected() -> bool {
        std::arch::is_powerpc64_feature_detected!("altivec")
            && std::arch::is_powerpc64_feature_detected!("vsx")
    }

    #[cfg(target_arch = "riscv64")]
    pub fn is_r1_detected() -> bool {
        std::arch::is_riscv_feature_detected!("v")
    }
}

pub use simd_macros::{multiversion, target_cpu};

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use std::arch::is_x86_feature_detected as is_feature_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use std::arch::is_aarch64_feature_detected as is_feature_detected;

#[cfg(target_arch = "s390x")]
#[allow(unused_imports)]
pub use std::arch::is_s390x_feature_detected as is_feature_detected;

#[cfg(target_arch = "powerpc64")]
#[allow(unused_imports)]
pub use std::arch::is_powerpc64_feature_detected as is_feature_detected;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use internal::is_x86_64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use internal::is_aarch64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "s390x")]
#[allow(unused_imports)]
pub use internal::is_s390x_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "powerpc64")]
#[allow(unused_imports)]
pub use internal::is_powerpc64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "riscv64")]
#[allow(unused_imports)]
pub use internal::is_riscv64_cpu_detected as is_cpu_detected;
