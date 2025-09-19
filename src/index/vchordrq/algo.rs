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

use crate::index::vchordrq::am::am_build::InternalBuild;
use crate::index::vchordrq::opclass::Opfamily;
use algo::accessor::{Dot, L2S};
use algo::prefetcher::*;
use algo::*;
use simd::f16;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vchordrq::operator::Op;
use vchordrq::types::*;
use vchordrq::{Chooser, FastHeap, Opaque};
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

pub fn prewarm<R>(opfamily: Opfamily, index: &R, height: i32) -> String
where
    R: RelationRead,
    R::Page: Page<Opaque = Opaque>,
{
    let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<VectOwned<f32>, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<VectOwned<f32>, Dot>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordrq::prewarm::<_, Op<VectOwned<f16>, L2S>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordrq::prewarm::<_, Op<VectOwned<f16>, Dot>>(index, height, make_h0_plain_prefetcher)
        }
    }
}

pub fn bulkdelete<R>(
    opfamily: Opfamily,
    index: &R,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordrq::bulkdelete::<_, Op<VectOwned<f32>, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f32>, L2S>>(index, &check, &callback);
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordrq::bulkdelete::<_, Op<VectOwned<f32>, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f32>, Dot>>(index, &check, &callback);
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordrq::bulkdelete::<_, Op<VectOwned<f16>, L2S>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f16>, L2S>>(index, &check, &callback);
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordrq::bulkdelete::<_, Op<VectOwned<f16>, Dot>>(index, &check, &callback);
            vchordrq::bulkdelete_vectors::<_, Op<VectOwned<f16>, Dot>>(index, &check, &callback);
        }
    }
}

pub fn maintain<R>(opfamily: Opfamily, index: &R, check: impl Fn())
where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
    let maintain = match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2S) => {
            vchordrq::maintain::<_, Op<VectOwned<f32>, L2S>>(index, make_h0_plain_prefetcher, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            vchordrq::maintain::<_, Op<VectOwned<f32>, Dot>>(index, make_h0_plain_prefetcher, check)
        }
        (VectorKind::Vecf16, DistanceKind::L2S) => {
            vchordrq::maintain::<_, Op<VectOwned<f16>, L2S>>(index, make_h0_plain_prefetcher, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            vchordrq::maintain::<_, Op<VectOwned<f16>, Dot>>(index, make_h0_plain_prefetcher, check)
        }
    };
    pgrx::info!(
        "maintain: number_of_formerly_allocated_pages = {}",
        maintain.number_of_formerly_allocated_pages
    );
    pgrx::info!(
        "maintain: number_of_freshly_allocated_pages = {}",
        maintain.number_of_freshly_allocated_pages
    );
    pgrx::info!(
        "maintain: number_of_freed_pages = {}",
        maintain.number_of_freed_pages
    );
}

pub fn build<R>(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: &R,
    structures: Vec<Structure<Vec<f32>>>,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2S) => vchordrq::build::<_, Op<VectOwned<f32>, L2S>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => vchordrq::build::<_, Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::L2S) => vchordrq::build::<_, Op<VectOwned<f16>, L2S>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => vchordrq::build::<_, Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
    }
}

pub fn insert<R>(
    opfamily: Opfamily,
    index: &R,
    payload: NonZero<u64>,
    vector: OwnedVector,
    skip_freespaces: bool,
    skip_search: bool,
    chooser: &mut impl Chooser,
    bump: &impl Bump,
) where
    R: RelationRead + RelationWrite,
    R::Page: Page<Opaque = Opaque>,
{
    let make_h1_plain_prefetcher = MakeH1PlainPrefetcherForInsertion { index };
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(vector.as_borrowed());
            let key = vchordrq::insert_vector::<_, Op<VectOwned<f32>, L2S>>(
                index,
                payload,
                vector.as_borrowed(),
                chooser,
                skip_search,
            );
            vchordrq::insert::<_, Op<VectOwned<f32>, L2S>>(
                index,
                payload,
                projected.as_borrowed(),
                key,
                bump,
                make_h1_plain_prefetcher,
                skip_freespaces,
            )
        }
        (OwnedVector::Vecf32(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let projected = RandomProject::project(vector.as_borrowed());
            let key = vchordrq::insert_vector::<_, Op<VectOwned<f32>, Dot>>(
                index,
                payload,
                vector.as_borrowed(),
                chooser,
                skip_search,
            );
            vchordrq::insert::<_, Op<VectOwned<f32>, Dot>>(
                index,
                payload,
                projected.as_borrowed(),
                key,
                bump,
                make_h1_plain_prefetcher,
                skip_freespaces,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::L2S) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(vector.as_borrowed());
            let key = vchordrq::insert_vector::<_, Op<VectOwned<f16>, L2S>>(
                index,
                payload,
                vector.as_borrowed(),
                chooser,
                skip_search,
            );
            vchordrq::insert::<_, Op<VectOwned<f16>, L2S>>(
                index,
                payload,
                projected.as_borrowed(),
                key,
                bump,
                make_h1_plain_prefetcher,
                skip_freespaces,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let projected = RandomProject::project(vector.as_borrowed());
            let key = vchordrq::insert_vector::<_, Op<VectOwned<f16>, Dot>>(
                index,
                payload,
                vector.as_borrowed(),
                chooser,
                skip_search,
            );
            vchordrq::insert::<_, Op<VectOwned<f16>, Dot>>(
                index,
                payload,
                projected.as_borrowed(),
                key,
                bump,
                make_h1_plain_prefetcher,
                skip_freespaces,
            )
        }
    }
}

fn map_structures<T, U>(x: Vec<Structure<T>>, f: impl Fn(T) -> U + Copy) -> Vec<Structure<U>> {
    x.into_iter()
        .map(
            |Structure {
                 centroids,
                 children,
             }| Structure {
                centroids: centroids.into_iter().map(f).collect(),
                children,
            },
        )
        .collect()
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
pub struct MakeH1PlainPrefetcherForInsertion<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH1PlainPrefetcherForInsertion<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherHeapFamily<'b, R> for MakeH1PlainPrefetcherForInsertion<'b, R> {
    type P<T>
        = PlainPrefetcher<'b, R, FastHeap<T>>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, FastHeap::from(seq))
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeH1PlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH1PlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherHeapFamily<'b, R> for MakeH1PlainPrefetcher<'b, R> {
    type P<T>
        = PlainPrefetcher<'b, R, BinaryHeap<T>>
    where
        T: Ord + Fetch<'b>;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch<'b>,
    {
        PlainPrefetcher::new(self.index, BinaryHeap::from(seq))
    }

    fn is_not_plain(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MakeH0PlainPrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH0PlainPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead> PrefetcherSequenceFamily<'b, R> for MakeH0PlainPrefetcher<'b, R> {
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
pub struct MakeH0SimplePrefetcher<'b, R> {
    pub index: &'b R,
}

impl<'b, R> Clone for MakeH0SimplePrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'b, R: RelationRead + RelationPrefetch> PrefetcherSequenceFamily<'b, R>
    for MakeH0SimplePrefetcher<'b, R>
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
pub struct MakeH0StreamPrefetcher<'b, R> {
    pub index: &'b R,
    pub hints: Hints,
}

impl<'b, R> Clone for MakeH0StreamPrefetcher<'b, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints.clone(),
        }
    }
}

impl<'b, R: RelationRead + RelationReadStream> PrefetcherSequenceFamily<'b, R>
    for MakeH0StreamPrefetcher<'b, R>
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
