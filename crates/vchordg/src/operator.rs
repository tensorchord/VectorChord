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

use crate::types::DistanceKind;
use algo::accessor::{Accessor1, Accessor2, DistanceAccessor, Dot, L2S};
use distance::Distance;
use half::f16;
use rabitq::bits::Bits;
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
    fn split(
        vector: Self::Borrowed<'_>,
        m: usize,
    ) -> (Vec<&[Self::Element]>, (&[Self::Element], Self::Metadata));
    fn pack(elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code;
    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn split(
        vector: Self::Borrowed<'_>,
        m: usize,
    ) -> (Vec<&[Self::Element]>, (&[Self::Element], Self::Metadata)) {
        let slice = vector.slice();
        let tailing = (size_of::<crate::tuples::OptionNeighbour>() * m)
            .next_multiple_of(crate::tuples::ALIGN);
        assert!(tailing <= 8000);
        if slice.len() <= (8000 - tailing) / size_of::<f32>() {
            return (vec![], (slice, ()));
        }
        let (l, r) = slice.split_at(slice.len() - (8000 - tailing) / size_of::<f32>());
        (
            l.chunks(8000 / size_of::<f32>()).collect::<Vec<_>>(),
            (r, ()),
        )
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code {
        rabitq::bits::code(bits, vector.slice())
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut {
        rabitq::bits::binary::preprocess(vector.slice())
    }
}

impl Vector for VectOwned<f16> {
    type Metadata = ();

    type Element = f16;

    fn unpack(vector: Self::Borrowed<'_>) -> (&[Self::Element], Self::Metadata) {
        (vector.slice(), ())
    }

    fn split(
        vector: Self::Borrowed<'_>,
        m: usize,
    ) -> (Vec<&[Self::Element]>, (&[Self::Element], Self::Metadata)) {
        let slice = vector.slice();
        let tailing = (size_of::<crate::tuples::OptionNeighbour>() * m)
            .next_multiple_of(crate::tuples::ALIGN);
        assert!(tailing <= 8000);
        if slice.len() <= (8000 - tailing) / size_of::<f16>() {
            return (vec![], (slice, ()));
        }
        let (l, r) = slice.split_at(slice.len() - (8000 - tailing) / size_of::<f16>());
        (
            l.chunks(8000 / size_of::<f16>()).collect::<Vec<_>>(),
            (r, ()),
        )
    }

    fn pack(elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code {
        rabitq::bits::code(bits, &f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut {
        rabitq::bits::binary::preprocess(&f16::vector_to_f32(vector.slice()))
    }
}

pub trait Operator: 'static + Debug + Copy {
    const DISTANCE: DistanceKind;

    type Vector: Vector;

    type DistanceAccessor: Default
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    fn process(
        bits: Bits,
        n: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance;
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
    const DISTANCE: DistanceKind = DistanceKind::L2S;

    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, L2S>;

    fn process(
        bits: Bits,
        n: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let value = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2(
            bits,
            n,
            value,
            CodeMetadata::from_array(code.0),
            lut.0,
        );
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
    const DISTANCE: DistanceKind = DistanceKind::Dot;

    type Vector = VectOwned<f32>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f32>, Dot>;

    fn process(
        bits: Bits,
        n: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let value = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            n,
            value,
            CodeMetadata::from_array(code.0),
            lut.0,
        );
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
    const DISTANCE: DistanceKind = DistanceKind::L2S;

    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, L2S>;

    fn process(
        bits: Bits,
        n: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let value = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2(
            bits,
            n,
            value,
            CodeMetadata::from_array(code.0),
            lut.0,
        );
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
    const DISTANCE: DistanceKind = DistanceKind::Dot;

    type Vector = VectOwned<f16>;

    type DistanceAccessor = DistanceAccessor<VectOwned<f16>, Dot>;

    fn process(
        bits: Bits,
        n: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let value = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            n,
            value,
            CodeMetadata::from_array(code.0),
            lut.0,
        );
        Distance::from_f32(distance)
    }

    fn distance(
        lhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
        rhs: <Self::Vector as VectorOwned>::Borrowed<'_>,
    ) -> Distance {
        lhs.operator_dot(rhs)
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
