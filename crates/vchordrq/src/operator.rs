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
use half::f16;
use rabitq::original::CodeMetadata;
use rabitq::original::binary::{BinaryLut, BinaryLutMetadata};
use rabitq::original::block::{BlockLut, BlockLutMetadata, STEP};
use simd::Floating;
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::vect::{VectBorrowed, VectOwned};
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub trait Accessor2<E0, E1, M0, M1> {
    type Output;
    fn push(&mut self, input: &[E0], target: &[E1]);
    fn finish(self, input: M0, target: M1) -> Self::Output;
}

impl<E0, E1, M0: Copy, M1: Copy> Accessor2<E0, E1, M0, M1> for () {
    type Output = ();

    fn push(&mut self, _: &[E0], _: &[E1]) {}

    fn finish(self, _: M0, _: M1) -> Self::Output {}
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>> Accessor2<E0, E1, M0, M1> for (A,) {
    type Output = (A::Output,);

    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
    }

    fn finish(self, input: M0, target: M1) -> Self::Output {
        (self.0.finish(input, target),)
    }
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>, B: Accessor2<E0, E1, M0, M1>>
    Accessor2<E0, E1, M0, M1> for (A, B)
{
    type Output = (A::Output, B::Output);

    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
        self.1.push(input, target);
    }

    fn finish(self, input: M0, target: M1) -> Self::Output {
        (self.0.finish(input, target), self.1.finish(input, target))
    }
}

pub trait Accessor1<E, M> {
    type Output;
    fn push(&mut self, input: &[E]);
    fn finish(self, input: M) -> Self::Output;
}

impl<E, M: Copy> Accessor1<E, M> for () {
    type Output = ();

    fn push(&mut self, _: &[E]) {}

    fn finish(self, _: M) -> Self::Output {}
}

impl<E, M: Copy, A> Accessor1<E, M> for (A,)
where
    A: Accessor1<E, M>,
{
    type Output = (A::Output,);

    fn push(&mut self, input: &[E]) {
        self.0.push(input);
    }

    fn finish(self, input: M) -> Self::Output {
        (self.0.finish(input),)
    }
}

impl<E, M: Copy, A, B> Accessor1<E, M> for (A, B)
where
    A: Accessor1<E, M>,
    B: Accessor1<E, M>,
{
    type Output = (A::Output, B::Output);

    fn push(&mut self, input: &[E]) {
        self.0.push(input);
        self.1.push(input);
    }

    fn finish(self, input: M) -> Self::Output {
        (self.0.finish(input), self.1.finish(input))
    }
}

pub struct FunctionalAccessor<T, P, F> {
    data: T,
    p: P,
    f: F,
}

impl<T, P, F> FunctionalAccessor<T, P, F> {
    pub fn new(data: T, p: P, f: F) -> Self {
        Self { data, p, f }
    }
}

impl<E, M, T, P, F, R> Accessor1<E, M> for FunctionalAccessor<T, P, F>
where
    P: for<'a> FnMut(&'a mut T, &'a [E]),
    F: FnOnce(T, M) -> R,
{
    type Output = R;

    fn push(&mut self, input: &[E]) {
        (self.p)(&mut self.data, input);
    }

    fn finish(self, input: M) -> Self::Output {
        (self.f)(self.data, input)
    }
}

pub struct LAccess<'a, E, M, A> {
    elements: &'a [E],
    metadata: M,
    accessor: A,
}

impl<'a, E, M, A> LAccess<'a, E, M, A> {
    pub fn new((elements, metadata): (&'a [E], M), accessor: A) -> Self {
        Self {
            elements,
            metadata,
            accessor,
        }
    }
}

impl<E0, E1, M0, M1, A: Accessor2<E0, E1, M0, M1>> Accessor1<E1, M1> for LAccess<'_, E0, M0, A> {
    type Output = A::Output;

    fn push(&mut self, rhs: &[E1]) {
        let (lhs, elements) = self.elements.split_at(rhs.len());
        self.accessor.push(lhs, rhs);
        self.elements = elements;
    }

    fn finish(self, rhs: M1) -> Self::Output {
        assert!(self.elements.is_empty(), "goal is shorter than expected");
        self.accessor.finish(self.metadata, rhs)
    }
}

pub struct RAccess<'a, E, M, A> {
    elements: &'a [E],
    metadata: M,
    accessor: A,
}

impl<'a, E, M, A> RAccess<'a, E, M, A> {
    pub fn new((elements, metadata): (&'a [E], M), accessor: A) -> Self {
        Self {
            elements,
            metadata,
            accessor,
        }
    }
}

impl<E0, E1, M0, M1, A: Accessor2<E0, E1, M0, M1>> Accessor1<E0, M0> for RAccess<'_, E1, M1, A> {
    type Output = A::Output;

    fn push(&mut self, lhs: &[E0]) {
        let (rhs, elements) = self.elements.split_at(lhs.len());
        self.accessor.push(lhs, rhs);
        self.elements = elements;
    }

    fn finish(self, lhs: M0) -> Self::Output {
        assert!(self.elements.is_empty(), "goal is shorter than expected");
        self.accessor.finish(lhs, self.metadata)
    }
}

pub trait TryAccessor1<E, M>: Sized {
    type Output;
    #[must_use]
    fn push(&mut self, input: &[E]) -> Option<()>;
    #[must_use]
    fn finish(self, input: M) -> Option<Self::Output>;
}

impl<E, M: Copy> TryAccessor1<E, M> for () {
    type Output = ();

    fn push(&mut self, _: &[E]) -> Option<()> {
        Some(())
    }

    fn finish(self, _: M) -> Option<Self::Output> {
        Some(())
    }
}

impl<E, M: Copy, A> TryAccessor1<E, M> for (A,)
where
    A: TryAccessor1<E, M>,
{
    type Output = (A::Output,);

    fn push(&mut self, input: &[E]) -> Option<()> {
        self.0.push(input)?;
        Some(())
    }

    fn finish(self, input: M) -> Option<Self::Output> {
        Some((self.0.finish(input)?,))
    }
}

impl<E, M: Copy, A, B> TryAccessor1<E, M> for (A, B)
where
    A: TryAccessor1<E, M>,
    B: TryAccessor1<E, M>,
{
    type Output = (A::Output, B::Output);

    fn push(&mut self, input: &[E]) -> Option<()> {
        self.0.push(input)?;
        self.1.push(input)?;
        Some(())
    }

    fn finish(self, input: M) -> Option<Self::Output> {
        Some((self.0.finish(input)?, self.1.finish(input)?))
    }
}

pub struct LTryAccess<'a, E, M, A> {
    elements: &'a [E],
    metadata: M,
    accessor: A,
}

impl<'a, E, M, A> LTryAccess<'a, E, M, A> {
    pub fn new((elements, metadata): (&'a [E], M), accessor: A) -> Self {
        Self {
            elements,
            metadata,
            accessor,
        }
    }
}

impl<E0, E1, M0, M1, A: Accessor2<E0, E1, M0, M1>> TryAccessor1<E1, M1>
    for LTryAccess<'_, E0, M0, A>
{
    type Output = A::Output;

    fn push(&mut self, rhs: &[E1]) -> Option<()> {
        let (lhs, elements) = self.elements.split_at_checked(rhs.len())?;
        self.accessor.push(lhs, rhs);
        self.elements = elements;
        Some(())
    }

    fn finish(self, rhs: M1) -> Option<Self::Output> {
        if !self.elements.is_empty() {
            return None;
        }
        Some(self.accessor.finish(self.metadata, rhs))
    }
}

#[derive(Debug)]
pub struct BlockAccessor<F>([u32; 32], F);

impl<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>
    Accessor2<[u8; 16], [u8; 16], (&[[f32; 32]; 4], &[f32; 32]), BlockLutMetadata>
    for BlockAccessor<F>
{
    type Output = [F::Output; 32];

    fn push(&mut self, input: &[[u8; 16]], target: &[[u8; 16]]) {
        use std::iter::zip;

        for (input, target) in zip(input.chunks(STEP), target.chunks(STEP)) {
            let delta = simd::fast_scan::scan(input, target);
            simd::fast_scan::accu(&mut self.0, &delta);
        }
    }

    fn finish(
        mut self,
        (metadata, delta): (&[[f32; 32]; 4], &[f32; 32]),
        lut: BlockLutMetadata,
    ) -> Self::Output {
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
                lut,
            )
        })
    }
}

#[derive(Debug)]
pub struct DistanceAccessor<V, D>(f32, PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Default for DistanceAccessor<V, D> {
    fn default() -> Self {
        Self(0.0, PhantomData, PhantomData)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, L2S> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, Dot> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, L2S> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, Dot> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

#[derive(Debug, Clone)]
pub struct CloneAccessor<V: Vector>(Vec<V::Element>);

impl<V: Vector> Default for CloneAccessor<V> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<V: Vector> Accessor1<V::Element, V::Metadata> for CloneAccessor<V> {
    type Output = V;

    fn push(&mut self, input: &[V::Element]) {
        self.0.extend(input);
    }

    fn finish(self, metadata: V::Metadata) -> Self::Output {
        V::pack(self.0, metadata)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[Self::Element]>, Self::Metadata);

    fn count(n: usize) -> usize;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);

    fn pack(elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut;

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut);

    fn code(vector: Self::Borrowed<'_>) -> rabitq::original::Code;

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

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::original::block::preprocess(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::original::preprocess(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::original::Code {
        rabitq::original::code(vector.slice())
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

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::original::block::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::original::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::original::Code {
        rabitq::original::code(&f16::vector_to_f32(vector.slice()))
    }

    fn squared_norm(vector: Self::Borrowed<'_>) -> f32 {
        f16::reduce_sum_of_x2(vector.slice())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct L2S;

#[derive(Debug, Clone, Copy)]
pub struct Dot;

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    fn block_access<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>(
        lut: &BlockLut,
        f: F,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [F::Output; 32]>;

    fn binary_access<F: Call<u32, CodeMetadata, f32, BinaryLutMetadata>>(
        lut: &BinaryLut,
        f: F,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> F::Output;

    fn block_process(
        value: u32,
        code: CodeMetadata,
        lut: BlockLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32);

    fn binary_process(
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32);

    fn build(
        vector: <Self::Vector as VectorOwned>::Borrowed<'_>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::original::Code, f32);
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

    fn block_access<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>(
        lut: &BlockLut,
        f: F,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [F::Output; 32]>
    {
        RAccess::new((&lut.1, lut.0), BlockAccessor([0u32; 32], f))
    }

    fn binary_access<F: Call<u32, CodeMetadata, f32, BinaryLutMetadata>>(
        lut: &BinaryLut,
        mut f: F,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> F::Output {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::original::binary::accumulate(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    factor_cnt: metadata[1],
                    factor_ip: metadata[2],
                    factor_err: metadata[3],
                },
                delta,
                lut.0,
            )
        }
    }

    fn block_process(
        value: u32,
        code: CodeMetadata,
        lut: BlockLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        _: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::block::half_process_l2(value, code, lut)
        } else {
            rabitq::original::block::half_process_l2_residual(value, code, lut, dis_f, delta)
        }
    }

    fn binary_process(
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        _: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::binary::half_process_l2(value, code, lut)
        } else {
            rabitq::original::binary::half_process_l2_residual(value, code, lut, dis_f, delta)
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::original::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dims = vector.dims();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dims as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dims as f32).sqrt();
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

    fn block_access<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>(
        lut: &BlockLut,
        f: F,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [F::Output; 32]>
    {
        RAccess::new((&lut.1, lut.0), BlockAccessor([0u32; 32], f))
    }

    fn binary_access<F: Call<u32, CodeMetadata, f32, BinaryLutMetadata>>(
        lut: &BinaryLut,
        mut f: F,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> F::Output {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::original::binary::accumulate(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    factor_cnt: metadata[1],
                    factor_ip: metadata[2],
                    factor_err: metadata[3],
                },
                delta,
                lut.0,
            )
        }
    }

    fn block_process(
        value: u32,
        code: CodeMetadata,
        lut: BlockLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::block::half_process_dot(value, code, lut)
        } else {
            rabitq::original::block::half_process_dot_residual(value, code, lut, dis_f, delta, norm)
        }
    }

    fn binary_process(
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::binary::half_process_dot(value, code, lut)
        } else {
            rabitq::original::binary::half_process_dot_residual(
                value, code, lut, dis_f, delta, norm,
            )
        }
    }

    fn build(
        vector: VectBorrowed<'_, f32>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::original::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f32::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dims = vector.dims();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num))
                    .sum::<f32>()
                    / (dims as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dims as f32).sqrt();
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

    fn block_access<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>(
        lut: &BlockLut,
        f: F,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [F::Output; 32]>
    {
        RAccess::new((&lut.1, lut.0), BlockAccessor([0u32; 32], f))
    }

    fn binary_access<F: Call<u32, CodeMetadata, f32, BinaryLutMetadata>>(
        lut: &BinaryLut,
        mut f: F,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> F::Output {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::original::binary::accumulate(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    factor_cnt: metadata[1],
                    factor_ip: metadata[2],
                    factor_err: metadata[3],
                },
                delta,
                lut.0,
            )
        }
    }

    fn block_process(
        value: u32,
        code: CodeMetadata,
        lut: BlockLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        _: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::block::half_process_l2(value, code, lut)
        } else {
            rabitq::original::block::half_process_l2_residual(value, code, lut, dis_f, delta)
        }
    }

    fn binary_process(
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        _: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::binary::half_process_l2(value, code, lut)
        } else {
            rabitq::original::binary::half_process_l2_residual(value, code, lut, dis_f, delta)
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::original::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dims = vector.dims();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num).to_f32())
                    .sum::<f32>()
                    / (dims as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dims as f32).sqrt();
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

    fn block_access<F: Call<u32, CodeMetadata, f32, BlockLutMetadata>>(
        lut: &BlockLut,
        f: F,
    ) -> impl for<'x> Accessor1<[u8; 16], (&'x [[f32; 32]; 4], &'x [f32; 32]), Output = [F::Output; 32]>
    {
        RAccess::new((&lut.1, lut.0), BlockAccessor([0u32; 32], f))
    }

    fn binary_access<F: Call<u32, CodeMetadata, f32, BinaryLutMetadata>>(
        lut: &BinaryLut,
        mut f: F,
    ) -> impl FnMut([f32; 4], &[u64], f32) -> F::Output {
        move |metadata: [f32; 4], elements: &[u64], delta: f32| {
            let value = rabitq::original::binary::accumulate(elements, &lut.1);
            f.call(
                value,
                CodeMetadata {
                    dis_u_2: metadata[0],
                    factor_cnt: metadata[1],
                    factor_ip: metadata[2],
                    factor_err: metadata[3],
                },
                delta,
                lut.0,
            )
        }
    }

    fn block_process(
        value: u32,
        code: CodeMetadata,
        lut: BlockLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::block::half_process_dot(value, code, lut)
        } else {
            rabitq::original::block::half_process_dot_residual(value, code, lut, dis_f, delta, norm)
        }
    }

    fn binary_process(
        value: u32,
        code: CodeMetadata,
        lut: BinaryLutMetadata,
        is_residual: bool,
        dis_f: f32,
        delta: f32,
        norm: f32,
    ) -> (f32, f32) {
        if !is_residual {
            rabitq::original::binary::half_process_dot(value, code, lut)
        } else {
            rabitq::original::binary::half_process_dot_residual(
                value, code, lut, dis_f, delta, norm,
            )
        }
    }

    fn build(
        vector: VectBorrowed<'_, f16>,
        centroid: Option<Self::Vector>,
    ) -> (rabitq::original::Code, f32) {
        if let Some(centroid) = centroid {
            let residual = VectOwned::new(f16::vector_sub(vector.slice(), centroid.slice()));
            let code = Self::Vector::code(residual.as_borrowed());
            let delta = {
                use std::iter::zip;
                let dims = vector.dims();
                let t = zip(&code.1, centroid.slice())
                    .map(|(&sign, &num)| std::hint::select_unpredictable(sign, num, -num).to_f32())
                    .sum::<f32>()
                    / (dims as f32).sqrt();
                let sum_of_x_2 = code.0.dis_u_2;
                let sum_of_abs_x = sum_of_x_2 / code.0.factor_ip;
                let dis_u = sum_of_x_2.sqrt();
                let x_0 = sum_of_abs_x / dis_u / (dims as f32).sqrt();
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

pub trait Call<A, B, C, D> {
    type Output;

    fn call(&mut self, a: A, b: B, c: C, d: D) -> Self::Output;
}

impl<A, B, C, D, F: Fn(A, B, C, D) -> R, R> Call<A, B, C, D> for F {
    type Output = R;

    fn call(&mut self, a: A, b: B, c: C, d: D) -> R {
        (self)(a, b, c, d)
    }
}
