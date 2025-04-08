use super::opclass::Opfamily;
use crate::index::am::am_build::InternalBuild;
use algorithm::operator::{Dot, L2, Op};
use algorithm::types::*;
use algorithm::{RelationRead, RelationWrite};
use half::f16;
use std::num::NonZeroU64;
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

pub fn prewarm(
    opfamily: Opfamily,
    index: impl RelationRead,
    height: i32,
    check: impl Fn(),
) -> String {
    let message = match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::prewarm::<Op<VectOwned<f32>, L2>>(index, height, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::prewarm::<Op<VectOwned<f32>, Dot>>(index, height, check)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::prewarm::<Op<VectOwned<f16>, L2>>(index, height, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::prewarm::<Op<VectOwned<f16>, Dot>>(index, height, check)
        }
    };
    match message {
        Ok(message) => message,
        Err(e) => panic!("{e}"),
    }
}

pub fn bulkdelete(
    opfamily: Opfamily,
    index: impl RelationWrite,
    check: impl Fn(),
    callback: impl Fn(NonZeroU64) -> bool,
) {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, Dot>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, Dot>>(index, check, callback)
        }
    }
}

pub fn maintain(opfamily: Opfamily, index: impl RelationWrite, check: impl Fn()) {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f32>, L2>>(index, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f32>, Dot>>(index, check)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f16>, L2>>(index, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f16>, Dot>>(index, check)
        }
    }
}

pub fn build(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: impl RelationWrite,
    structures: Vec<Structure<Vec<f32>>>,
) {
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f32>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f16>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
    }
}

pub fn insert(
    opfamily: Opfamily,
    index: impl RelationWrite,
    payload: NonZeroU64,
    vector: OwnedVector,
) {
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            algorithm::insert::<Op<VectOwned<f32>, L2>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
            )
        }
        (OwnedVector::Vecf32(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            algorithm::insert::<Op<VectOwned<f32>, Dot>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            algorithm::insert::<Op<VectOwned<f16>, L2>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            algorithm::insert::<Op<VectOwned<f16>, Dot>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
            )
        }
    }
}

fn map_structures<T, U>(x: Vec<Structure<T>>, f: impl Fn(T) -> U + Copy) -> Vec<Structure<U>> {
    x.into_iter()
        .map(|Structure { means, children }| Structure {
            means: means.into_iter().map(f).collect(),
            children,
        })
        .collect()
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use crate::index::projection::project;
        let input = self.slice();
        VectOwned::new(project(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use crate::index::projection::project;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&project(&input)))
    }
}
