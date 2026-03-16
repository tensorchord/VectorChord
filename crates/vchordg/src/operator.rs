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
use distance::Distance;
use index_accessor::{
    Accessor1, Accessor2, ByteDistanceAccessor, DefaultWithDimension, DistanceAccessor, Dot,
    HalfbyteDistanceAccessor, L2S,
};
use rabitq::bits::Bits;
use simd::{Floating, f16};
use std::fmt::Debug;
use std::marker::PhantomData;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
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
    fn pack(dim: u32, elements: Vec<Self::Element>, metadata: Self::Metadata) -> Self;

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

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
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

    fn pack(_: u32, elements: Vec<Self::Element>, (): Self::Metadata) -> Self {
        VectOwned::new(elements)
    }

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code {
        rabitq::bits::code(bits, &f16::vector_to_f32(vector.slice()))
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut {
        rabitq::bits::binary::preprocess(&f16::vector_to_f32(vector.slice()))
    }
}

impl Vector for Rabitq8Owned {
    type Metadata = [f32; 4];

    type Element = u8;

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

    fn split(
        vector: Self::Borrowed<'_>,
        m: usize,
    ) -> (Vec<&[Self::Element]>, (&[Self::Element], Self::Metadata)) {
        let metadata = [
            vector.sum_of_x2(),
            vector.norm_of_lattice(),
            vector.sum_of_code(),
            vector.sum_of_abs_x(),
        ];
        let slice = vector.packed_code();
        let tailing = (size_of::<crate::tuples::OptionNeighbour>() * m)
            .next_multiple_of(crate::tuples::ALIGN);
        assert!(tailing <= 8000);
        if slice.len() <= 8000 - tailing {
            return (vec![], (slice, metadata));
        }
        let (l, r) = slice.split_at(slice.len() - (8000 - tailing));
        (l.chunks(8000).collect::<Vec<_>>(), (r, metadata))
    }

    fn pack(dim: u32, code: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq8Owned::new(dim, _0, _1, _2, _3, code)
    }

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bits::code(bits, &result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 8) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bits::binary::preprocess(&result)
    }
}

impl Vector for Rabitq4Owned {
    type Metadata = [f32; 4];

    type Element = u8;

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

    fn split(
        vector: Self::Borrowed<'_>,
        m: usize,
    ) -> (Vec<&[Self::Element]>, (&[Self::Element], Self::Metadata)) {
        let metadata = [
            vector.sum_of_x2(),
            vector.norm_of_lattice(),
            vector.sum_of_code(),
            vector.sum_of_abs_x(),
        ];
        let slice = vector.packed_code();
        let tailing = (size_of::<crate::tuples::OptionNeighbour>() * m)
            .next_multiple_of(crate::tuples::ALIGN);
        assert!(tailing <= 8000);
        if slice.len() <= 8000 - tailing {
            return (vec![], (slice, metadata));
        }
        let (l, r) = slice.split_at(slice.len() - (8000 - tailing));
        (l.chunks(8000).collect::<Vec<_>>(), (r, metadata))
    }

    fn pack(dim: u32, code: Vec<Self::Element>, [_0, _1, _2, _3]: Self::Metadata) -> Self {
        Rabitq4Owned::new(dim, _0, _1, _2, _3, code)
    }

    fn code(bits: Bits, vector: Self::Borrowed<'_>) -> rabitq::bits::Code {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bits::code(bits, &result)
    }

    fn preprocess(vector: Self::Borrowed<'_>) -> rabitq::bits::binary::BinaryLut {
        let scale = vector.sum_of_x2().sqrt() / vector.norm_of_lattice();
        let mut result = Vec::with_capacity(vector.dim() as _);
        for c in vector.unpacked_code() {
            let base = -0.5 * ((1 << 4) - 1) as f32;
            result.push((base + c as f32) * scale);
        }
        rabitq::bits::binary::preprocess(&result)
    }
}

pub trait Operator: 'static + Debug + Copy {
    const DISTANCE: DistanceKind;

    type Vector: Vector;

    type DistanceAccessor: DefaultWithDimension
        + Accessor2<
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Element,
            <Self::Vector as Vector>::Metadata,
            <Self::Vector as Vector>::Metadata,
            Output = Distance,
        >;

    fn process(
        bits: Bits,
        dim: u32,
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
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2s(
            bits,
            dim,
            sum,
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
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            dim,
            sum,
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
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2s(
            bits,
            dim,
            sum,
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
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            dim,
            sum,
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

impl Operator for Op<Rabitq8Owned, L2S> {
    const DISTANCE: DistanceKind = DistanceKind::L2S;

    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, L2S>;

    fn process(
        bits: Bits,
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2s(
            bits,
            dim,
            sum,
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

impl Operator for Op<Rabitq8Owned, Dot> {
    const DISTANCE: DistanceKind = DistanceKind::Dot;

    type Vector = Rabitq8Owned;

    type DistanceAccessor = ByteDistanceAccessor<Rabitq8Owned, Dot>;

    fn process(
        bits: Bits,
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            dim,
            sum,
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

impl Operator for Op<Rabitq4Owned, L2S> {
    const DISTANCE: DistanceKind = DistanceKind::L2S;

    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, L2S>;

    fn process(
        bits: Bits,
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_l2s(
            bits,
            dim,
            sum,
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

impl Operator for Op<Rabitq4Owned, Dot> {
    const DISTANCE: DistanceKind = DistanceKind::Dot;

    type Vector = Rabitq4Owned;

    type DistanceAccessor = HalfbyteDistanceAccessor<Rabitq4Owned, Dot>;

    fn process(
        bits: Bits,
        dim: u32,
        code: ([f32; 3], &[u64]),
        lut: &rabitq::bits::binary::BinaryLut,
    ) -> Distance {
        use rabitq::bits::CodeMetadata;
        let sum = rabitq::bits::binary::accumulate(bits, code.1, &lut.1);
        let (distance,) = rabitq::bits::binary::half_process_dot(
            bits,
            dim,
            sum,
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
pub struct CloneAccessor<V: Vector>(u32, Vec<V::Element>);

impl<V: Vector> DefaultWithDimension for CloneAccessor<V> {
    fn default_with_dimension(dim: u32) -> Self {
        Self(dim, Vec::new())
    }
}

impl<V: Vector> Accessor1<V::Element, V::Metadata> for CloneAccessor<V> {
    type Output = V;

    fn push(&mut self, input: &[V::Element]) {
        self.1.extend(input);
    }

    fn finish(self, metadata: V::Metadata) -> Self::Output {
        V::pack(self.0, self.1, metadata)
    }
}
