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
use simd::Floating;
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::vect::VectOwned;
use vector::{VectorBorrowed, VectorOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub trait Vector: VectorOwned {
    type Element: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;
    type Metadata: Debug + Copy + FromBytes + IntoBytes + Immutable + KnownLayout;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata);
    fn pack(elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code;
    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code {
        rabitq::b1::code(vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut {
        rabitq::b1::binary::preprocess(vector.slice())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn code(vector: Self::Borrowed<'_>) -> rabitq::b1::Code {
        rabitq::b1::code(&f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::b1::binary::BinaryLut {
        rabitq::b1::binary::preprocess(&f16::vector_to_f32(vector.slice()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct L2S;

#[derive(Debug, Clone, Copy)]
pub struct Dot;

pub trait Operator: 'static + Debug + Copy {
    type Vector: Vector;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance;
    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance;
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

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_l2(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance {
        lhs.operator_l2s(rhs)
    }
}

impl Operator for Op<VectOwned<f32>, Dot> {
    type Vector = VectOwned<f32>;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_dot(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance {
        lhs.operator_dot(rhs)
    }
}

impl Operator for Op<VectOwned<f16>, L2S> {
    type Vector = VectOwned<f16>;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_l2(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance {
        lhs.operator_l2s(rhs)
    }
}

impl Operator for Op<VectOwned<f16>, Dot> {
    type Vector = VectOwned<f16>;

    fn process(code: ([f32; 3], &[u64]), lut: &rabitq::b1::binary::BinaryLut) -> Distance {
        use rabitq::b1::CodeMetadata;
        let value = rabitq::b1::binary::accumulate(code.1, &lut.1);
        let (distance,) =
            rabitq::b1::binary::half_process_dot(value, CodeMetadata::from_array(code.0), lut.0);
        Distance::from_f32(distance)
    }

    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance {
        lhs.operator_dot(rhs)
    }
}
