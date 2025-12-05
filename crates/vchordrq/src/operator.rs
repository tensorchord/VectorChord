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

use distance::Distance;
use index::accessor::{
    Accessor1, Accessor2, ByteDistanceAccessor, DefaultWithDimension, DistanceAccessor, Dot,
    HalfbyteDistanceAccessor, L2S, RAccess,
};
use rabitq::bit::CodeMetadata;
use rabitq::bit::binary::BinaryLut;
use rabitq::bit::block::{BlockLut, STEP};
use simd::{Floating, f16};
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::rabitq4::{Rabitq4Borrowed, Rabitq4Owned};
use vector::rabitq8::{Rabitq8Borrowed, Rabitq8Owned};
use vector::vect::{VectBorrowed, VectOwned};
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[derive(Debug)]
pub struct BlockAccessor<F>([u32; 32], F);

impl<F: Call<u32, CodeMetadata, f32>>
    Accessor2<[u8; 16], [u8; 16], (&[[f32; 32]; 4], &[f32; 32]), ()> for BlockAccessor<F>
{
    type Output = [F::Output; 32];

    #[inline(always)]
    fn push(&mut self, input: &[[u8; 16]], target: &[[u8; 16]]) {
        use std::iter::zip;

        for (input, target) in zip(input.chunks(STEP), target.chunks(STEP)) {
            let delta = simd::fast_scan::scan(input, target);
            simd::fast_scan::accu(&mut self.0, &delta);
        }
    }

    #[inline(always)]
    fn finish(mut self, (metadata, delta): (&[[f32; 32]; 4], &[f32; 32]), (): ()) -> Self::Output {
        std::array::from_fn(|i| {
            (self.1).call(
                self.0[i],
                CodeMetadata {
                    dis_u_2: metadata[0][i],
                    factor_cnt: metadata[1][i],
                    factor_ip: metadata[2][i],
                    factor_err: metadata[3][i],
                },
                delta[i],
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct CloneAccessor<V: Vector>(Vec<V::Element>);

impl<V: Vector> Default for CloneAccessor<V> {
    #[inline(always)]
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<V: Vector> Accessor1<V::Element, (V::Metadata, u32)> for CloneAccessor<V> {
    type Output = V;

    #[inline(always)]
    fn push(&mut self, input: &[V::Element]) {
        self.0.extend(input);
    }

    #[inline(always)]
    fn finish(self, (metadata, dim): (V::Metadata, u32)) -> Self::Output {
        V::pack(dim, self.0, metadata)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[Self::Element]>, Self::Metadata);

    fn count(n: usize) -> usize;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);

    fn pack(dim: u32, elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut;

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut);

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code;

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[f32]>, ()) {
        let vector = vector.slice();
        (
            match vector.len() {
                0 => unreachable!(),
                1..=960 => vec![vector],
                961..=1280 => vec![&vector[..640], &vector[640..]],
                1281.. => vector.chunks(1920).collect(),
            },
            (),
        )
    }

    fn count(n: usize) -> usize {
        match n {
            0 => unreachable!(),
            1..=960 => 1,
            961..=1280 => 2,
            1281.. => n.div_ceil(1920),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::bit::block::preprocess(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::bit::preprocess(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        rabitq::bit::code(vector.slice())
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        f32::reduce_sum_of_x2(vector.slice())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[f16]>, ()) {
        let vector = vector.slice();
        (
            match vector.len() {
                0 => unreachable!(),
                1..=1920 => vec![vector],
                1921..=2560 => vec![&vector[..1280], &vector[1280..]],
                2561.. => vector.chunks(3840).collect(),
            },
            (),
        )
    }

    fn count(n: usize) -> usize {
        match n {
            0 => unreachable!(),
            1..=1920 => 1,
            1921..=2560 => 2,
            2561.. => n.div_ceil(3840),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::bit::block::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::bit::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        rabitq::bit::code(&f16::vector_to_f32(vector.slice()))
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        f16::reduce_sum_of_x2(vector.slice())
    }
}

impl Vector for Rabitq8Owned {
    type Metadata = [f32; 4];

    type Element = u8;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[u8]>, [f32; 4]) {
        (
            match vector.packed_code().len() {
                0 => unreachable!(),
                1..=3840 => vec![vector.packed_code()],
                3841..=5120 => vec![&vector.packed_code()[..1280], &vector.packed_code()[1280..]],
                5121.. => vector.packed_code().chunks(7680).collect(),
            },
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn count(n: usize) -> usize {
        match n {
            0 => unreachable!(),
            1..=3840 => 1,
            3841..=5120 => 2,
            5121.. => n.div_ceil(7680),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (
            vector.packed_code(),
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn pack(dim: u32, elements: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq8Owned::new(dim, _0, _1, _2, _3, elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::block::preprocess(&result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::preprocess(&result)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        let n = vector.dim();
        let sum_of_abs_x = vector.sum_of_abs_x();
        let sum_of_x_2 = vector.sum_of_x2();
        (
            CodeMetadata {
                dis_u_2: sum_of_x_2,
                factor_cnt: {
                    let cnt_pos = vector.unpacked_code().filter(|&x| x >= 128).count();
                    let cnt_neg = vector.unpacked_code().filter(|&x| x <= 127).count();
                    cnt_pos as f32 - cnt_neg as f32
                },
                factor_ip: sum_of_x_2 / sum_of_abs_x,
                factor_err: {
                    let dis_u = sum_of_x_2.sqrt();
                    let x_0 = sum_of_abs_x / dis_u / (n as f32).sqrt();
                    dis_u * (1.0 / (x_0 * x_0) - 1.0).sqrt() / (n as f32 - 1.0).sqrt()
                },
            },
            {
                let vector = vector.unpacked_code();
                let mut signs = Vec::new();
                for x in vector {
                    signs.push(x >= 128);
                }
                signs
            },
        )
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        vector.sum_of_x2()
    }
}

impl Vector for Rabitq4Owned {
    type Metadata = [f32; 4];

    type Element = u8;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[u8]>, [f32; 4]) {
        (
            match vector.packed_code().len() {
                0 => unreachable!(),
                1..=3840 => vec![vector.packed_code()],
                3841..=5120 => vec![&vector.packed_code()[..1280], &vector.packed_code()[1280..]],
                5121.. => vector.packed_code().chunks(7680).collect(),
            },
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn count(n: usize) -> usize {
        match n {
            0 => unreachable!(),
            1..=3840 => 1,
            3841..=5120 => 2,
            5121.. => n.div_ceil(7680),
        }
    }

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (
            vector.packed_code(),
            [
                vector.sum_of_x2(),
                vector.norm_of_lattice(),
                vector.sum_of_code(),
                vector.sum_of_abs_x(),
            ],
        )
    }

    fn pack(dim: u32, elements: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq4Owned::new(dim, _0, _1, _2, _3, elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::block::preprocess(&result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bit::preprocess(&result)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::bit::Code {
        let n = vector.dim();
        let sum_of_abs_x = vector.sum_of_abs_x();
        let sum_of_x_2 = vector.sum_of_x2();
        (
            CodeMetadata {
                dis_u_2: sum_of_x_2,
                factor_cnt: {
                    let cnt_pos = vector.unpacked_code().filter(|&x| x >= 8).count();
                    let cnt_neg = vector.unpacked_code().filter(|&x| x <= 7).count();
                    cnt_pos as f32 - cnt_neg as f32
                },
                factor_ip: sum_of_x_2 / sum_of_abs_x,
                factor_err: {
                    let dis_u = sum_of_x_2.sqrt();
                    let x_0 = sum_of_abs_x / dis_u / (n as f32).sqrt();
                    dis_u * (1.0 / (x_0 * x_0) - 1.0).sqrt() / (n as f32 - 1.0).sqrt()
                },
            },
            {
                let vector = vector.unpacked_code();
                let mut signs = Vec::new();
                for x in vector {
                    signs.push(x >= 8);
                }
                signs
            },
        )
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        vector.sum_of_x2()
    }
}

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type DistanceAccessor: DefaultWithDimension
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>;

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32);

    fn build(
        vector: <Self::Vector as VectorOwned>::Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32);
}

#[derive(Debug)]
pub struct Op<V, D>(PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Clone for Op<V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, D> Copy for Op<V, D> {}

impl Operator for Op<VectOwned<f32>, L2S> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, L2S>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| {
                if !is_residual {
                    rabitq::bit::block::half_process_l2(value, code, lut.0)
                } else {
                    rabitq::bit::block::half_process_l2_residual(value, code, lut.0, dis_f, delta)
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            if !is_residual {
                rabitq::bit::binary::half_process_l2(value, code, lut.0)
            } else {
                rabitq::bit::binary::half_process_l2_residual(value, code, lut.0, dis_f, delta)
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                2.0 * dis_u * t / x_0
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, Dot>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| {
                if !is_residual {
                    rabitq::bit::block::half_process_dot(value, code, lut.0)
                } else {
                    rabitq::bit::block::half_process_dot_residual(
                        value, code, lut.0, dis_f, delta, norm,
                    )
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            if !is_residual {
                rabitq::bit::binary::half_process_dot(value, code, lut.0)
            } else {
                rabitq::bit::binary::half_process_dot_residual(
                    value, code, lut.0, dis_f, delta, norm,
                )
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                dis_u * t / x_0 - f32::reduce_sum_of_xy(residual.slice(), centroid.slice())
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f16>, L2S> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, L2S>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| {
                if !is_residual {
                    rabitq::bit::block::half_process_l2(value, code, lut.0)
                } else {
                    rabitq::bit::block::half_process_l2_residual(value, code, lut.0, dis_f, delta)
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            if !is_residual {
                rabitq::bit::binary::half_process_l2(value, code, lut.0)
            } else {
                rabitq::bit::binary::half_process_l2_residual(value, code, lut.0, dis_f, delta)
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .map(simd::F16::_to_f32)
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                2.0 * dis_u * t / x_0
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, Dot>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, delta| {
                if !is_residual {
                    rabitq::bit::block::half_process_dot(value, code, lut.0)
                } else {
                    rabitq::bit::block::half_process_dot_residual(
                        value, code, lut.0, dis_f, delta, norm,
                    )
                }
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        dis_f: f32,
        norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            if !is_residual {
                rabitq::bit::binary::half_process_dot(value, code, lut.0)
            } else {
                rabitq::bit::binary::half_process_dot_residual(
                    value, code, lut.0, dis_f, delta, norm,
                )
            }
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dim = vector.dim();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .map(simd::F16::_to_f32)
                    .sum::<f32>()
                    / (dim as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dim as f32).sqrt();
                dis_u * t / x_0 - f16::reduce_sum_of_xy(residual.slice(), centroid.slice())
            };
            (code, delta)
        } else {
            let code = Self::Vector::code(vector);
            let delta = 0.0;
            (code, delta)
        }
    }
}

impl Operator for Op<Rabitq8Owned, L2S> {
    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, L2S>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(!is_residual);
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_l2(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(!is_residual);
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_l2(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq8Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq8Owned, Dot> {
    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, Dot>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(!is_residual);
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_dot(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(!is_residual);
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_dot(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq8Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq4Owned, L2S> {
    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, L2S>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(!is_residual);
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_l2(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(!is_residual);
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_l2(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq4Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

impl Operator for Op<Rabitq4Owned, Dot> {
    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, Dot>;

    fn block_access(
        lut: &BlockLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [(f32, f32); 32]>
    {
        assert!(!is_residual);
        RAccess::new(
            (&lut.1, ()),
            BlockAccessor([0_u32; 32], move |value, code, _delta| {
                rabitq::bit::block::half_process_dot(value, code, lut.0)
            }),
        )
    }

    fn binary_access(
        lut: &BinaryLut,
        is_residual: bool,
        _dis_f: f32,
        _norm: f32,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> (f32, f32) {
        assert!(!is_residual);
        move |metadata: [f32; 4], elements: &[u64], _delta: f32| {
            let value = rabitq::bit::binary::accumulate(elements, &lut.1);
            let code = CodeMetadata {
                dis_u_2: metadata[0],
                factor_cnt: metadata[1],
                factor_ip: metadata[2],
                factor_err: metadata[3],
            };
            rabitq::bit::binary::half_process_dot(value, code, lut.0)
        }
    }

    fn build(
        vector: Rabitq4Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::bit::Code, f32) {
        if centroid.is_some() {
            unimplemented!();
        }
        (Self::Vector::code(vector), 0.0)
    }
}

pub trait Call<A, B, C> {
    type Output;

    fn call(&mut self, a: A, b: B, c: C) -> Self::Output;
}

impl<A, B, C, F: Fn(A, B, C) -> R, R> Call<A, B, C> for F {
    type Output = R;

    #[inline(always)]
    fn call(&mut self, a: A, b: B, c: C) -> R {
        (self)(a, b, c)
    }
}
