#![feature(target_feature_11)]
#![feature(avx512_target_feature)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512_f16))]

mod aligned;
mod emulate;
mod f16;
mod f32;

pub mod bit;
pub mod fast_scan;
pub mod packed_u4;
pub mod quantize;
pub mod u8;

pub trait Floating:
    Copy
    + Send
    + Sync
    + std::fmt::Debug
    + serde::Serialize
    + for<'a> serde::Deserialize<'a>
    + Default
    + 'static
    + PartialEq
    + PartialOrd
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
}

mod internal {
    #[cfg(target_arch = "x86_64")]
    simd_macros::define_is_cpu_detected!("x86_64");

    #[cfg(target_arch = "aarch64")]
    simd_macros::define_is_cpu_detected!("aarch64");

    #[cfg(target_arch = "riscv64")]
    simd_macros::define_is_cpu_detected!("riscv64");

    #[cfg(target_arch = "x86_64")]
    #[allow(unused_imports)]
    pub use is_x86_64_cpu_detected;

    #[cfg(target_arch = "aarch64")]
    #[allow(unused_imports)]
    pub use is_aarch64_cpu_detected;

    #[cfg(target_arch = "riscv64")]
    #[allow(unused_imports)]
    pub use is_riscv64_cpu_detected;
}

pub use simd_macros::multiversion;
pub use simd_macros::target_cpu;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use std::arch::is_x86_feature_detected as is_feature_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use std::arch::is_aarch64_feature_detected as is_feature_detected;

#[cfg(target_arch = "riscv64")]
#[allow(unused_imports)]
pub use std::arch::is_riscv_feature_detected as is_feature_detected;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use internal::is_x86_64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
pub use internal::is_aarch64_cpu_detected as is_cpu_detected;

#[cfg(target_arch = "riscv64")]
#[allow(unused_imports)]
pub use internal::is_riscv64_cpu_detected as is_cpu_detected;
