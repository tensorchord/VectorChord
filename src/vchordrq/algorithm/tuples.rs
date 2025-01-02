use std::num::NonZeroU64;

use super::rabitq::{self, Code, Lut};
use crate::vchordrq::types::DistanceKind;
use crate::vchordrq::types::OwnedVector;
use half::f16;
use rkyv::{Archive, ArchiveUnsized, CheckBytes, Deserialize, Serialize};
use simd::Floating;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub trait Vector: VectorOwned {
    type Metadata: Copy
        + Serialize<
            rkyv::ser::serializers::CompositeSerializer<
                rkyv::ser::serializers::AlignedSerializer<rkyv::AlignedVec>,
                rkyv::ser::serializers::FallbackScratch<
                    rkyv::ser::serializers::HeapScratch<8192>,
                    rkyv::ser::serializers::AllocScratch,
                >,
                rkyv::ser::serializers::SharedSerializeMap,
            >,
        > + for<'a> CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>;
    type Element: Copy
        + Serialize<
            rkyv::ser::serializers::CompositeSerializer<
                rkyv::ser::serializers::AlignedSerializer<rkyv::AlignedVec>,
                rkyv::ser::serializers::FallbackScratch<
                    rkyv::ser::serializers::HeapScratch<8192>,
                    rkyv::ser::serializers::AllocScratch,
                >,
                rkyv::ser::serializers::SharedSerializeMap,
            >,
        > + for<'a> CheckBytes<rkyv::validation::validators::DefaultValidator<'a>>
        + Archive<Archived = Self::Element>;

    fn metadata_from_archived(
        archived: &<Self::Metadata as ArchiveUnsized>::Archived,
    ) -> Self::Metadata;

    fn vector_split(vector: Self::Borrowed<'_>) -> (Self::Metadata, Vec<&[Self::Element]>);
    fn vector_merge(metadata: Self::Metadata, slice: &[Self::Element]) -> Self;
    fn from_owned(vector: OwnedVector) -> Self;

    type DistanceAccumulator;
    fn distance_begin(distance_kind: DistanceKind) -> Self::DistanceAccumulator;
    fn distance_next(
        accumulator: &mut Self::DistanceAccumulator,
        left: &[Self::Element],
        right: &[Self::Element],
    );
    fn distance_end(
        accumulator: Self::DistanceAccumulator,
        left: Self::Metadata,
        right: Self::Metadata,
    ) -> f32;

    fn random_projection(vector: Self::Borrowed<'_>) -> Self;

    fn residual(vector: Self::Borrowed<'_>, center: Self::Borrowed<'_>) -> Self;

    fn rabitq_fscan_preprocess(vector: Self::Borrowed<'_>) -> Lut;

    fn rabitq_code(dims: u32, vector: Self::Borrowed<'_>) -> Code;

    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32>;

    fn build_from_vecf32(x: &[f32]) -> Self;
}

impl Vector for VectOwned<f32> {
    type Metadata = ();

    type Element = f32;

    fn metadata_from_archived(_: &<Self::Metadata as ArchiveUnsized>::Archived) -> Self::Metadata {}

    fn vector_split(vector: Self::Borrowed<'_>) -> ((), Vec<&[f32]>) {
        let vector = vector.slice();
        ((), match vector.len() {
            0..=960 => vec![vector],
            961..=1280 => vec![&vector[..640], &vector[640..]],
            1281.. => vector.chunks(1920).collect(),
        })
    }

    fn vector_merge((): Self::Metadata, slice: &[Self::Element]) -> Self {
        VectOwned::new(slice.to_vec())
    }

    fn from_owned(vector: OwnedVector) -> Self {
        match vector {
            OwnedVector::Vecf32(x) => x,
            _ => unreachable!(),
        }
    }

    type DistanceAccumulator = (DistanceKind, f32);
    fn distance_begin(distance_kind: DistanceKind) -> Self::DistanceAccumulator {
        (distance_kind, 0.0)
    }
    fn distance_next(
        accumulator: &mut Self::DistanceAccumulator,
        left: &[Self::Element],
        right: &[Self::Element],
    ) {
        match accumulator.0 {
            DistanceKind::L2 => accumulator.1 += f32::reduce_sum_of_d2(left, right),
            DistanceKind::Dot => accumulator.1 += -f32::reduce_sum_of_xy(left, right),
        }
    }
    fn distance_end(
        accumulator: Self::DistanceAccumulator,
        (): Self::Metadata,
        (): Self::Metadata,
    ) -> f32 {
        accumulator.1
    }

    fn random_projection(vector: Self::Borrowed<'_>) -> Self {
        Self::new(crate::projection::project(vector.slice()))
    }

    fn residual(vector: Self::Borrowed<'_>, center: Self::Borrowed<'_>) -> Self {
        Self::new(Floating::vector_sub(vector.slice(), center.slice()))
    }

    fn rabitq_fscan_preprocess(vector: Self::Borrowed<'_>) -> Lut {
        rabitq::preprocess(vector.slice())
    }

    fn rabitq_code(dims: u32, vector: Self::Borrowed<'_>) -> Code {
        rabitq::code(dims, vector.slice())
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

    fn metadata_from_archived(_: &<Self::Metadata as ArchiveUnsized>::Archived) -> Self::Metadata {}

    fn vector_split(vector: Self::Borrowed<'_>) -> ((), Vec<&[f16]>) {
        let vector = vector.slice();
        ((), match vector.len() {
            0..=1920 => vec![vector],
            1921..=2560 => vec![&vector[..1280], &vector[1280..]],
            2561.. => vector.chunks(3840).collect(),
        })
    }

    fn vector_merge((): Self::Metadata, slice: &[Self::Element]) -> Self {
        VectOwned::new(slice.to_vec())
    }

    fn from_owned(vector: OwnedVector) -> Self {
        match vector {
            OwnedVector::Vecf16(x) => x,
            _ => unreachable!(),
        }
    }

    type DistanceAccumulator = (DistanceKind, f32);
    fn distance_begin(distance_kind: DistanceKind) -> Self::DistanceAccumulator {
        (distance_kind, 0.0)
    }
    fn distance_next(
        accumulator: &mut Self::DistanceAccumulator,
        left: &[Self::Element],
        right: &[Self::Element],
    ) {
        match accumulator.0 {
            DistanceKind::L2 => accumulator.1 += f16::reduce_sum_of_d2(left, right),
            DistanceKind::Dot => accumulator.1 += -f16::reduce_sum_of_xy(left, right),
        }
    }
    fn distance_end(
        accumulator: Self::DistanceAccumulator,
        (): Self::Metadata,
        (): Self::Metadata,
    ) -> f32 {
        accumulator.1
    }

    fn random_projection(vector: Self::Borrowed<'_>) -> Self {
        Self::new(f16::vector_from_f32(&crate::projection::project(
            &f16::vector_to_f32(vector.slice()),
        )))
    }

    fn residual(vector: Self::Borrowed<'_>, center: Self::Borrowed<'_>) -> Self {
        Self::new(Floating::vector_sub(vector.slice(), center.slice()))
    }

    fn rabitq_fscan_preprocess(vector: Self::Borrowed<'_>) -> Lut {
        rabitq::preprocess(&f16::vector_to_f32(vector.slice()))
    }

    fn rabitq_code(dims: u32, vector: Self::Borrowed<'_>) -> Code {
        rabitq::code(dims, &f16::vector_to_f32(vector.slice()))
    }

    fn build_to_vecf32(vector: Self::Borrowed<'_>) -> Vec<f32> {
        f16::vector_to_f32(vector.slice())
    }

    fn build_from_vecf32(x: &[f32]) -> Self {
        Self::new(f16::vector_from_f32(x))
    }
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct MetaTuple {
    pub dims: u32,
    pub height_of_root: u32,
    pub is_residual: bool,
    pub vectors_first: u32,
    // raw vector
    pub mean: (u32, u16),
    // for meta tuple, it's pointers to next level
    pub first: u32,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct VectorTuple<V: Vector> {
    pub slice: Vec<V::Element>,
    pub payload: Option<NonZeroU64>,
    pub chain: Result<(u32, u16), V::Metadata>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height1Tuple {
    // raw vector
    pub mean: (u32, u16),
    // for height 1 tuple, it's pointers to next level
    pub first: u32,
    // RaBitQ algorithm
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub t: Vec<u64>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height0Tuple {
    // raw vector
    pub mean: (u32, u16),
    // for height 0 tuple, it's pointers to heap relation
    pub payload: NonZeroU64,
    // RaBitQ algorithm
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub t: Vec<u64>,
}
