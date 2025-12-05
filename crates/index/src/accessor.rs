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
use rabitq::byte::CodeMetadata;
use simd::{Floating, f16};
use std::marker::PhantomData;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;

#[derive(Debug, Clone, Copy)]
pub struct L2S;

#[derive(Debug, Clone, Copy)]
pub struct Dot;

pub trait Accessor2<E0, E1, M0, M1> {
    type Output;
    fn push(&mut self, input: &[E0], target: &[E1]);
    fn finish(self, input: M0, target: M1) -> Self::Output;
}

impl<E0, E1, M0: Copy, M1: Copy> Accessor2<E0, E1, M0, M1> for () {
    type Output = ();

    #[inline(always)]
    fn push(&mut self, _: &[E0], _: &[E1]) {}

    #[inline(always)]
    fn finish(self, _: M0, _: M1) -> Self::Output {}
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>> Accessor2<E0, E1, M0, M1> for (A,) {
    type Output = (A::Output,);

    #[inline(always)]
    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
    }

    #[inline(always)]
    fn finish(self, input: M0, target: M1) -> Self::Output {
        (self.0.finish(input, target),)
    }
}

impl<E0, E1, M0: Copy, M1: Copy, A: Accessor2<E0, E1, M0, M1>, B: Accessor2<E0, E1, M0, M1>>
    Accessor2<E0, E1, M0, M1> for (A, B)
{
    type Output = (A::Output, B::Output);

    #[inline(always)]
    fn push(&mut self, input: &[E0], target: &[E1]) {
        self.0.push(input, target);
        self.1.push(input, target);
    }

    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, _: &[E]) {}

    #[inline(always)]
    fn finish(self, _: M) -> Self::Output {}
}

impl<E, M: Copy, A> Accessor1<E, M> for (A,)
where
    A: Accessor1<E, M>,
{
    type Output = (A::Output,);

    #[inline(always)]
    fn push(&mut self, input: &[E]) {
        self.0.push(input);
    }

    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, input: &[E]) {
        self.0.push(input);
        self.1.push(input);
    }

    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, input: &[E]) {
        (self.p)(&mut self.data, input);
    }

    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, rhs: &[E1]) {
        let (lhs, elements) = self.elements.split_at(rhs.len());
        self.accessor.push(lhs, rhs);
        self.elements = elements;
    }

    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, lhs: &[E0]) {
        let (rhs, elements) = self.elements.split_at(lhs.len());
        self.accessor.push(lhs, rhs);
        self.elements = elements;
    }

    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, _: &[E]) -> Option<()> {
        Some(())
    }

    #[inline(always)]
    fn finish(self, _: M) -> Option<Self::Output> {
        Some(())
    }
}

impl<E, M: Copy, A> TryAccessor1<E, M> for (A,)
where
    A: TryAccessor1<E, M>,
{
    type Output = (A::Output,);

    #[inline(always)]
    fn push(&mut self, input: &[E]) -> Option<()> {
        self.0.push(input)?;
        Some(())
    }

    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, input: &[E]) -> Option<()> {
        self.0.push(input)?;
        self.1.push(input)?;
        Some(())
    }

    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
    fn push(&mut self, rhs: &[E1]) -> Option<()> {
        let (lhs, elements) = self.elements.split_at_checked(rhs.len())?;
        self.accessor.push(lhs, rhs);
        self.elements = elements;
        Some(())
    }

    #[inline(always)]
    fn finish(self, rhs: M1) -> Option<Self::Output> {
        if !self.elements.is_empty() {
            return None;
        }
        Some(self.accessor.finish(self.metadata, rhs))
    }
}

#[derive(Debug)]
pub struct DistanceAccessor<V, D>(f32, PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> Default for DistanceAccessor<V, D> {
    #[inline(always)]
    fn default() -> Self {
        Self(0.0, PhantomData, PhantomData)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, L2S> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_d2(target, input)
    }

    #[inline(always)]
    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f32, f32, (), ()> for DistanceAccessor<VectOwned<f32>, Dot> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[f32], input: &[f32]) {
        self.0 += f32::reduce_sum_of_xy(target, input)
    }

    #[inline(always)]
    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, L2S> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_d2(target, input)
    }

    #[inline(always)]
    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(self.0)
    }
}

impl Accessor2<f16, f16, (), ()> for DistanceAccessor<VectOwned<f16>, Dot> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[f16], input: &[f16]) {
        self.0 += f16::reduce_sum_of_xy(target, input)
    }

    #[inline(always)]
    fn finish(self, (): (), (): ()) -> Self::Output {
        Distance::from_f32(-self.0)
    }
}

pub trait DefaultWithDimension {
    fn default_with_dimension(dim: u32) -> Self;
}

impl<T: Default> DefaultWithDimension for T {
    fn default_with_dimension(_dim: u32) -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct ByteDistanceAccessor<V, D>(u32, u32, PhantomData<fn(V) -> V>, PhantomData<fn(D) -> D>);

impl<V, D> DefaultWithDimension for ByteDistanceAccessor<V, D> {
    #[inline(always)]
    fn default_with_dimension(dim: u32) -> Self {
        Self(dim, 0, PhantomData, PhantomData)
    }
}

impl Accessor2<u8, u8, [f32; 4], [f32; 4]> for ByteDistanceAccessor<Rabitq8Owned, L2S> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[u8], input: &[u8]) {
        self.1 += rabitq::byte::binary::accumulate(target, input);
    }

    #[inline(always)]
    fn finish(self, target: [f32; 4], input: [f32; 4]) -> Self::Output {
        Distance::from_f32(
            rabitq::byte::binary::half_process_l2(
                self.0,
                self.1,
                CodeMetadata::from_array(std::array::from_fn(|i| target[i])),
                CodeMetadata::from_array(std::array::from_fn(|i| input[i])),
            )
            .0,
        )
    }
}

impl Accessor2<u8, u8, [f32; 4], [f32; 4]> for ByteDistanceAccessor<Rabitq8Owned, Dot> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[u8], input: &[u8]) {
        self.1 += rabitq::byte::binary::accumulate(target, input);
    }

    #[inline(always)]
    fn finish(self, target: [f32; 4], input: [f32; 4]) -> Self::Output {
        Distance::from_f32(
            rabitq::byte::binary::half_process_dot(
                self.0,
                self.1,
                CodeMetadata::from_array(std::array::from_fn(|i| target[i])),
                CodeMetadata::from_array(std::array::from_fn(|i| input[i])),
            )
            .0,
        )
    }
}

#[derive(Debug)]
pub struct HalfbyteDistanceAccessor<V, D>(
    u32,
    u32,
    PhantomData<fn(V) -> V>,
    PhantomData<fn(D) -> D>,
);

impl<V, D> DefaultWithDimension for HalfbyteDistanceAccessor<V, D> {
    #[inline(always)]
    fn default_with_dimension(dim: u32) -> Self {
        Self(dim, 0, PhantomData, PhantomData)
    }
}

impl Accessor2<u8, u8, [f32; 4], [f32; 4]> for HalfbyteDistanceAccessor<Rabitq4Owned, L2S> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[u8], input: &[u8]) {
        self.1 += rabitq::halfbyte::binary::accumulate(target, input);
    }

    #[inline(always)]
    fn finish(self, target: [f32; 4], input: [f32; 4]) -> Self::Output {
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_l2(
                self.0,
                self.1,
                CodeMetadata::from_array(std::array::from_fn(|i| target[i])),
                CodeMetadata::from_array(std::array::from_fn(|i| input[i])),
            )
            .0,
        )
    }
}

impl Accessor2<u8, u8, [f32; 4], [f32; 4]> for HalfbyteDistanceAccessor<Rabitq4Owned, Dot> {
    type Output = Distance;

    #[inline(always)]
    fn push(&mut self, target: &[u8], input: &[u8]) {
        self.1 += rabitq::halfbyte::binary::accumulate(target, input);
    }

    #[inline(always)]
    fn finish(self, target: [f32; 4], input: [f32; 4]) -> Self::Output {
        Distance::from_f32(
            rabitq::halfbyte::binary::half_process_dot(
                self.0,
                self.1,
                CodeMetadata::from_array(std::array::from_fn(|i| target[i])),
                CodeMetadata::from_array(std::array::from_fn(|i| input[i])),
            )
            .0,
        )
    }
}
