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

use crate::index::vchordg::opclass::Opfamily;
use algo::accessor::{Dot, L2S};
use algo::prefetcher::*;
use algo::*;
use simd::f16;
use std::num::NonZero;
use vchordg::Opaque;
use vchordg::operator::Op;
use vchordg::types::*;
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

pub fn prewarm<R>(opfamily: Opfamily, index: &R) -> String
where
    R: RelationRead,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordg::prewarm::<_, Op<VectOwned<f32>, L2S>>(index)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordg::prewarm::<_, Op<VectOwned<f32>, Dot>>(index)
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordg::prewarm::<_, Op<VectOwned<f16>, L2S>>(index)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordg::prewarm::<_, Op<VectOwned<f16>, Dot>>(index)
        }
    }
}

pub fn bulkdelete<R>(
    opfamily: Opfamily,
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R: RelationRead + RelationWrite + RelationLength,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordg::bulkdelete::<_, Op<VectOwned<f32>, L2S>>(index, &check, &callback);
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordg::bulkdelete::<_, Op<VectOwned<f16>, L2S>>(index, &check, &callback);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordg::bulkdelete::<_, Op<VectOwned<f32>, Dot>>(index, &check, &callback);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordg::bulkdelete::<_, Op<VectOwned<f16>, Dot>>(index, &check, &callback);
        }
    }
}

pub fn maintain<R>(opfamily: Opfamily, index: &R, check: impl Fn())
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordg::maintain::<_, Op<VectOwned<f32>, L2S>>(index, &check);
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordg::maintain::<_, Op<VectOwned<f16>, L2S>>(index, &check);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordg::maintain::<_, Op<VectOwned<f32>, Dot>>(index, &check);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordg::maintain::<_, Op<VectOwned<f16>, Dot>>(index, &check);
        }
    }
}

pub fn build<R>(vector_options: VectorOptions, vchordg_options: VchordgIndexOptions, index: &R)
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordg::build::<_, Op<VectOwned<f32>, L2S>>(vector_options, vchordg_options, index)
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordg::build::<_, Op<VectOwned<f16>, L2S>>(vector_options, vchordg_options, index)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordg::build::<_, Op<VectOwned<f32>, Dot>>(vector_options, vchordg_options, index)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordg::build::<_, Op<VectOwned<f16>, Dot>>(vector_options, vchordg_options, index)
        }
    }
}

pub fn insert<R>(opfamily: Opfamily, index: &R, payload: NonZero<u64>, vector: OwnedVector)
where
    R: RelationRead + RelationWrite + RelationReadStream,
    R::Page: Page<Opaque = Opaque>,
{
    let bump = bumpalo::Bump::new();
    let make_vertex_plain_prefetcher = MakePlainPrefetcher { index };
    let make_vector_plain_prefetcher = MakePlainPrefetcher { index };
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(unprojected), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vchordg::insert::<_, Op<VectOwned<f32>, L2S>>(
                index,
                projected.as_borrowed(),
                payload,
                &bump,
                make_vertex_plain_prefetcher,
                make_vector_plain_prefetcher,
            )
        }
        (OwnedVector::Vecf16(unprojected), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vchordg::insert::<_, Op<VectOwned<f16>, L2S>>(
                index,
                projected.as_borrowed(),
                payload,
                &bump,
                make_vertex_plain_prefetcher,
                make_vector_plain_prefetcher,
            )
        }
        (OwnedVector::Vecf32(unprojected), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vchordg::insert::<_, Op<VectOwned<f32>, Dot>>(
                index,
                projected.as_borrowed(),
                payload,
                &bump,
                make_vertex_plain_prefetcher,
                make_vector_plain_prefetcher,
            )
        }
        (OwnedVector::Vecf16(unprojected), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(unprojected.as_borrowed());
            vchordg::insert::<_, Op<VectOwned<f16>, Dot>>(
                index,
                projected.as_borrowed(),
                payload,
                &bump,
                make_vertex_plain_prefetcher,
                make_vector_plain_prefetcher,
            )
        }
    }
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use rabitq::rotate::rotate;
        let input = self.slice();
        VectOwned::new(rotate(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use rabitq::rotate::rotate;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&rotate(&input)))
    }
}

#[derive(Debug)]
pub struct MakePlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakePlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherSequenceFamily<'b, R> for MakePlainPrefetcher<'b, R> {
    type P<S: Sequence>
        = PlainPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, seq)
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeSimplePrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeSimplePrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead + RelationPrefetch> PrefetcherSequenceFamily<'b, R>
    for MakeSimplePrefetcher<'b, R>
{
    type P<S: Sequence>
        = SimplePrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        SimplePrefetcher::new(self.index, seq)
    }

    fn is_not_plain(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct MakeStreamPrefetcher<'b, R> {
    pub index: &'b R,
    pub hints: Hints,
}

impl<'b, R> Clone for MakeStreamPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints.clone(),
        }
    }
}

impl<'b, R: RelationRead + RelationReadStream> PrefetcherSequenceFamily<'b, R>
    for MakeStreamPrefetcher<'b, R>
{
    type P<S: Sequence>
        = StreamPrefetcher<'b, R, S>
    where
        S::Item: Fetch<'b>;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch<'b>,
    {
        StreamPrefetcher::new(self.index, seq, self.hints.clone())
    }

    fn is_not_plain(&self) -> bool {
        true
    }
}
