use distance::Distance;
use half::f16;
use rabitq::binary::{BinaryCode, BinaryLut};
use rabitq::block::BlockLut;
use simd::Floating;
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::vect::VectOwned;
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
        assert!(!self.elements.is_empty(), "goal is shorter than expected");
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
        assert!(!self.elements.is_empty(), "goal is shorter than expected");
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
pub struct BlockAccessor<D>([u16; 32], PhantomData<fn(D) -> D>);

impl<D> Default for BlockAccessor<D> {
    fn default() -> Self {
        Self([0u16; 32], PhantomData)
    }
}

impl
    Accessor2<
        [u8; 16],
        [u8; 16],
        (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (f32, f32, f32, f32),
    > for BlockAccessor<L2>
{
    type Output = [(f32, f32); 32];

    fn push(&mut self, input: &[[u8; 16]], target: &[[u8; 16]]) {
        let t = simd::fast_scan::scan(input, target);
        for i in 0..32 {
            self.0[i] += t[i];
        }
    }

    fn finish(
        self,
        (dis_u_2, factor_ppc, factor_ip, factor_err): (
            &[f32; 32],
            &[f32; 32],
            &[f32; 32],
            &[f32; 32],
        ),
        (dis_v_2, b, k, qvector_sum): (f32, f32, f32, f32),
    ) -> Self::Output {
        std::array::from_fn(|i| {
            let rough = dis_u_2[i]
                + dis_v_2
                + b * factor_ppc[i]
                + ((2.0 * self.0[i] as f32) - qvector_sum) * factor_ip[i] * k;
            let err = factor_err[i] * dis_v_2.sqrt();
            (rough, err)
        })
    }
}

impl
    Accessor2<
        [u8; 16],
        [u8; 16],
        (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (f32, f32, f32, f32),
    > for BlockAccessor<Dot>
{
    type Output = [(f32, f32); 32];

    fn push(&mut self, input: &[[u8; 16]], target: &[[u8; 16]]) {
        let t = simd::fast_scan::scan(input, target);
        for i in 0..32 {
            self.0[i] += t[i];
        }
    }

    fn finish(
        self,
        (_, factor_ppc, factor_ip, factor_err): (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (dis_v_2, b, k, qvector_sum): (f32, f32, f32, f32),
    ) -> Self::Output {
        std::array::from_fn(|i| {
            let rough = 0.5 * b * factor_ppc[i]
                + 0.5 * ((2.0 * self.0[i] as f32) - qvector_sum) * factor_ip[i] * k;
            let err = 0.5 * factor_err[i] * dis_v_2.sqrt();
            (rough, err)
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

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, L2> {
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

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, L2> {
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
pub struct ResidualAccessor<V: Vector>(Vec<V::Element>);

impl<V: Vector> Default for ResidualAccessor<V> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl Accessor2<f32, f32, (), ()> for ResidualAccessor<VectOwned<f32>> {
    type Output = VectOwned<f32>;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0.extend(f32::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for ResidualAccessor<VectOwned<f16>> {
    type Output = VectOwned<f16>;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0.extend(f16::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn split(vector: Self::Borrowed<'_>) -> (Vec<&[Self::Element]>, Self::Metadata);

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut;

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut);

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code;
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

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::block::preprocess(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::preprocess(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code {
        rabitq::code(vector.dims(), vector.slice())
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

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn block_preprocess(vector: Self::Borrowed<'_>) -> BlockLut {
        rabitq::block::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> (BlockLut, BinaryLut) {
        rabitq::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code {
        rabitq::code(vector.dims(), &f16::vector_to_f32(vector.slice()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct L2;

#[derive(Debug, Clone, Copy)]
pub struct Dot;

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type BlockAccessor: Default
        + for<'a> Accessor2<
            [u8; 16],
            [u8; 16],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            (f32, f32, f32, f32),
            Output = [(f32, f32); 32],
        >;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    type ResidualAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Self::Vector,
        >;

    const SUPPORTS_RESIDUAL: bool;

    fn binary_process(lut: &BinaryLut, code: BinaryCode<'_>) -> (f32, f32);
}

#[derive(Debug)]
pub struct Op<V, D>(PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Clone for Op<V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, D> Copy for Op<V, D> {}

impl Operator for Op<VectOwned<f32>, L2> {
    type Vector = VectOwned<f32>;

    type BlockAccessor = BlockAccessor<L2>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, L2>;

    type ResidualAccessor = ResidualAccessor<VectOwned<f32>>;

    const SUPPORTS_RESIDUAL: bool = true;

    fn binary_process(lut: &BinaryLut, code: BinaryCode<'_>) -> (f32, f32) {
        rabitq::binary::process_l2(lut, code)
    }
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    type BlockAccessor = BlockAccessor<Dot>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, Dot>;

    type ResidualAccessor = ResidualAccessor<VectOwned<f32>>;

    const SUPPORTS_RESIDUAL: bool = false;

    fn binary_process(lut: &BinaryLut, code: BinaryCode<'_>) -> (f32, f32) {
        rabitq::binary::process_dot(lut, code)
    }
}

impl Operator for Op<VectOwned<f16>, L2> {
    type Vector = VectOwned<f16>;

    type BlockAccessor = BlockAccessor<L2>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, L2>;

    type ResidualAccessor = ResidualAccessor<VectOwned<f16>>;

    const SUPPORTS_RESIDUAL: bool = true;

    fn binary_process(lut: &BinaryLut, code: BinaryCode<'_>) -> (f32, f32) {
        rabitq::binary::process_l2(lut, code)
    }
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    type BlockAccessor = BlockAccessor<Dot>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, Dot>;

    type ResidualAccessor = ResidualAccessor<VectOwned<f16>>;

    const SUPPORTS_RESIDUAL: bool = false;

    fn binary_process(lut: &BinaryLut, code: BinaryCode<'_>) -> (f32, f32) {
        rabitq::binary::process_dot(lut, code)
    }
}
