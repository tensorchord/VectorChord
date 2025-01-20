use crate::types::{DistanceKind, OwnedVector};
use distance::Distance;
use half::f16;
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

#[derive(Debug)]
pub struct Sum<O>(f32, PhantomData<fn(O) -> O>);

impl<O> Default for Sum<O> {
    fn default() -> Self {
        Self(0.0, PhantomData)
    }
}

impl Accessor2<f32, f32, (), ()> for Sum<Op<VectOwned<f32>, L2>> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f32, f32, (), ()> for Sum<Op<VectOwned<f32>, Dot>> {
    type Output = Distance;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for Sum<Op<VectOwned<f16>, L2>> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_d2(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for Sum<Op<VectOwned<f16>, Dot>> {
    type Output = Distance;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_xy(target, input)
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

#[derive(Debug, Clone)]
pub struct Diff<O: Operator>(Vec<<O::Vector as Vector>::Element>);

impl<O: Operator> Default for Diff<O> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl Accessor2<f32, f32, (), ()> for Diff<Op<VectOwned<f32>, L2>> {
    type Output = VectOwned<f32>;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0.extend(f32::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

impl Accessor2<f32, f32, (), ()> for Diff<Op<VectOwned<f32>, Dot>> {
    type Output = VectOwned<f32>;

    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0.extend(f32::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for Diff<Op<VectOwned<f16>, L2>> {
    type Output = VectOwned<f16>;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0.extend(f16::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for Diff<Op<VectOwned<f16>, Dot>> {
    type Output = VectOwned<f16>;

    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0.extend(f16::vector_sub(target, input));
    }

    fn finish(self, (): (), (): ()) -> Self::Output {
        VectOwned::new(self.0)
    }
}

#[derive(Debug)]
pub struct Block<D: OperatorDistance>([u16; 32], PhantomData<fn(D) -> D>);

impl<D: OperatorDistance> Default for Block<D> {
    fn default() -> Self {
        Self([0u16; 32], PhantomData)
    }
}

impl
    Accessor2<
        [u64; 2],
        [u64; 2],
        (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (f32, f32, f32, f32, f32),
    > for Block<L2>
{
    type Output = [Distance; 32];

    fn push(&mut self, input: &[[u64; 2]], target: &[[u64; 2]]) {
        let t = simd::fast_scan::fast_scan(input, target);
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
        (dis_v_2, b, k, qvector_sum, epsilon): (f32, f32, f32, f32, f32),
    ) -> Self::Output {
        std::array::from_fn(|i| {
            let rough = dis_u_2[i]
                + dis_v_2
                + b * factor_ppc[i]
                + ((2.0 * self.0[i] as f32) - qvector_sum) * factor_ip[i] * k;
            let err = factor_err[i] * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        })
    }
}

impl
    Accessor2<
        [u64; 2],
        [u64; 2],
        (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (f32, f32, f32, f32, f32),
    > for Block<Dot>
{
    type Output = [Distance; 32];

    fn push(&mut self, input: &[[u64; 2]], target: &[[u64; 2]]) {
        let t = simd::fast_scan::fast_scan(input, target);
        for i in 0..32 {
            self.0[i] += t[i];
        }
    }

    fn finish(
        self,
        (_, factor_ppc, factor_ip, factor_err): (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32]),
        (dis_v_2, b, k, qvector_sum, epsilon): (f32, f32, f32, f32, f32),
    ) -> Self::Output {
        std::array::from_fn(|i| {
            let rough = 0.5 * b * factor_ppc[i]
                + 0.5 * ((2.0 * self.0[i] as f32) - qvector_sum) * factor_ip[i] * k;
            let err = 0.5 * factor_err[i] * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        })
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
        self.accessor.finish(self.metadata, rhs)
    }
}

pub struct RAccess<'a, E, M, A> {
    elements: &'a [E],
    metadata: M,
    accessor: A,
}

impl<'a, E, M, A> RAccess<'a, E, M, A> {
    #[allow(dead_code)]
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
        self.accessor.finish(lhs, self.metadata)
    }
}

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;
    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn vector_split(vector: Self::Borrowed<'_>) -> (Self::Metadata, Vec<&[Self::Element]>);
    fn elements_and_metadata(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);
    fn from_owned(vector: OwnedVector) -> Self;

    fn random_projection(vector: Self::Borrowed<'_>) -> Self;

    fn compute_lut_block(vector: Self::Borrowed<'_>) -> (f32, f32, f32, f32, Vec<[u64; 2]>);

    fn compute_lut(
        vector: Self::Borrowed<'_>,
    ) -> (
        (f32, f32, f32, f32, Vec<[u64; 2]>),
        (f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
    );

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code;

    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32>;

    fn build_from_vecf32(x: &[f32]) -> Self;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn vector_split(vector: Self::Borrowed<'_>) -> ((), Vec<&[f32]>) {
        let vector = vector.slice();
        ((), match vector.len() {
            0..=960 => vec![vector],
            961..=1280 => vec![&vector[..640], &vector[640..]],
            1281.. => vector.chunks(1920).collect(),
        })
    }

    fn elements_and_metadata(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn from_owned(vector: OwnedVector) -> Self {
        match vector {
            OwnedVector::Vecf32(x) => x,
            _ => unreachable!(),
        }
    }

    fn random_projection(vector: Self::Borrowed<'_>) -> Self {
        Self::new(crate::projection::project(vector.slice()))
    }

    fn compute_lut_block(vector: Self::Borrowed<'_>) -> (f32, f32, f32, f32, Vec<[u64; 2]>) {
        rabitq::block::preprocess(vector.slice())
    }

    fn compute_lut(
        vector: Self::Borrowed<'_>,
    ) -> (
        (f32, f32, f32, f32, Vec<[u64; 2]>),
        (f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
    ) {
        rabitq::compute_lut(vector.slice())
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code {
        rabitq::code(vector.dims(), vector.slice())
    }

    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        vector.slice().to_vec()
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(x.to_vec())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn vector_split(vector: Self::Borrowed<'_>) -> ((), Vec<&[f16]>) {
        let vector = vector.slice();
        ((), match vector.len() {
            0..=1920 => vec![vector],
            1921..=2560 => vec![&vector[..1280], &vector[1280..]],
            2561.. => vector.chunks(3840).collect(),
        })
    }

    fn elements_and_metadata(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn from_owned(vector: OwnedVector) -> Self {
        match vector {
            OwnedVector::Vecf16(x) => x,
            _ => unreachable!(),
        }
    }

    fn random_projection(vector: Self::Borrowed<'_>) -> Self {
        Self::new(f16::vector_from_f32(&crate::projection::project(
            &f16::vector_to_f32(vector.slice()),
        )))
    }

    fn compute_lut_block(vector: Self::Borrowed<'_>) -> (f32, f32, f32, f32, Vec<[u64; 2]>) {
        rabitq::block::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn compute_lut(
        vector: Self::Borrowed<'_>,
    ) -> (
        (f32, f32, f32, f32, Vec<[u64; 2]>),
        (f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
    ) {
        rabitq::compute_lut(&f16::vector_to_f32(vector.slice()))
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::Code {
        rabitq::code(vector.dims(), &f16::vector_to_f32(vector.slice()))
    }

    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        f16::vector_to_f32(vector.slice())
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(f16::vector_from_f32(x))
    }
}

pub trait OperatorDistance: 'static + Debug + Copy {
    const KIND: DistanceKind;

    fn compute_lowerbound_binary(
        lut: &(f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
        code: (f32, f32, f32, f32, &[u64]),
        epsilon: f32,
    ) -> Distance;

    type BlockAccessor: for<'a> Accessor2<
            [u64; 2],
            [u64; 2],
            (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]),
            (f32, f32, f32, f32, f32),
            Output = [Distance; 32],
        > + Default;

    fn block_accessor() -> Self::BlockAccessor {
        Default::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct L2;

impl OperatorDistance for L2 {
    const KIND: DistanceKind = DistanceKind::L2;

    fn compute_lowerbound_binary(
        lut: &(f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
        code: (f32, f32, f32, f32, &[u64]),
        epsilon: f32,
    ) -> Distance {
        rabitq::binary::process_lowerbound_l2(lut, code, epsilon)
    }

    type BlockAccessor = Block<L2>;
}

#[derive(Debug, Clone, Copy)]
pub struct Dot;

impl OperatorDistance for Dot {
    const KIND: DistanceKind = DistanceKind::Dot;

    fn compute_lowerbound_binary(
        lut: &(f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)),
        code: (f32, f32, f32, f32, &[u64]),
        epsilon: f32,
    ) -> Distance {
        rabitq::binary::process_lowerbound_dot(lut, code, epsilon)
    }

    type BlockAccessor = Block<Dot>;
}

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    type Distance: OperatorDistance;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    const SUPPORTS_RESIDUAL: bool;

    type ResidualAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Self::Vector,
        >;
}

#[derive(Debug)]
pub struct Op<V, D>(PhantomData<(fn(V) -> V, fn(D) -> D)>);

impl<V, D> Clone for Op<V, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, D> Copy for Op<V, D> {}

impl Operator for Op<VectOwned<f32>, L2> {
    type Vector = VectOwned<f32>;

    type Distance = L2;

    type DistanceAccessor = Sum<Op<VectOwned<f32>, L2>>;

    const SUPPORTS_RESIDUAL: bool = true;

    type ResidualAccessor = Diff<Op<VectOwned<f32>, L2>>;
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    type Distance = Dot;

    type DistanceAccessor = Sum<Op<VectOwned<f32>, Dot>>;

    const SUPPORTS_RESIDUAL: bool = false;

    type ResidualAccessor = Diff<Op<VectOwned<f32>, Dot>>;
}

impl Operator for Op<VectOwned<f16>, L2> {
    type Vector = VectOwned<f16>;

    type Distance = L2;

    type DistanceAccessor = Sum<Op<VectOwned<f16>, L2>>;

    const SUPPORTS_RESIDUAL: bool = true;

    type ResidualAccessor = Diff<Op<VectOwned<f16>, L2>>;
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    type Distance = Dot;

    type DistanceAccessor = Sum<Op<VectOwned<f16>, Dot>>;

    const SUPPORTS_RESIDUAL: bool = false;

    type ResidualAccessor = Diff<Op<VectOwned<f16>, Dot>>;
}
