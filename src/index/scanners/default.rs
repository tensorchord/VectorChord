use super::{SearchBuilder, SearchFetcher, SearchOptions};
use crate::index::algorithm::RandomProject;
use crate::index::am::pointer_to_kv;
use crate::index::opclass::{Opfamily, Sphere};
use algorithm::operator::{Dot, L2, Op};
use algorithm::types::{DistanceKind, OwnedVector, VectorKind};
use algorithm::{RelationReadStream, RerankMethod};
use half::f16;
use std::num::NonZero;
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecIp
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorIp
                | Opfamily::VectorL2
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
            spheres: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            1 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vector(x)) };
                self.orderbys.push(x);
            }
            2 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_sphere(x)) };
                self.spheres.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'a>(
        self,
        relation: impl RelationReadStream + 'a,
        options: SearchOptions,
        mut fetcher: impl SearchFetcher + 'a,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'a> {
        let mut vector = None;
        let mut threshold = None;
        let mut recheck = false;
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(orderby_vector);
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for Sphere { center, radius } in self.spheres.into_iter().flatten() {
            if vector.is_none() {
                (vector, threshold) = (Some(center), Some(radius));
            } else {
                recheck = true;
            }
        }
        let opfamily = self.opfamily;
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2) => {
                    let vector = RandomProject::project(
                        if let OwnedVector::Vecf32(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                        .as_borrowed(),
                    );
                    let results = algorithm::default_search::<Op<VectOwned<f32>, L2>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes,
                        options.epsilon,
                    );
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let (datums, is_nulls) = fetcher.fetch(key)?;
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(RandomProject::project(raw.as_borrowed()))
                    };
                    let method = algorithm::how(relation.clone());
                    match method {
                        RerankMethod::Index => Box::new(
                            algorithm::rerank_index::<Op<VectOwned<f32>, L2>, _>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                        RerankMethod::Heap => Box::new(
                            algorithm::rerank_heap::<Op<VectOwned<f32>, L2>, _>(
                                vector, results, fetch,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                    }
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    let vector = RandomProject::project(
                        if let OwnedVector::Vecf32(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                        .as_borrowed(),
                    );
                    let results = algorithm::default_search::<Op<VectOwned<f32>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes,
                        options.epsilon,
                    );
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let (datums, is_nulls) = fetcher.fetch(key)?;
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(RandomProject::project(raw.as_borrowed()))
                    };
                    let method = algorithm::how(relation.clone());
                    match method {
                        RerankMethod::Index => Box::new(
                            algorithm::rerank_index::<Op<VectOwned<f32>, Dot>, _>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                        RerankMethod::Heap => Box::new(
                            algorithm::rerank_heap::<Op<VectOwned<f32>, Dot>, _>(
                                vector, results, fetch,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                    }
                }
                (VectorKind::Vecf16, DistanceKind::L2) => {
                    let vector = RandomProject::project(
                        if let OwnedVector::Vecf16(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                        .as_borrowed(),
                    );
                    let results = algorithm::default_search::<Op<VectOwned<f16>, L2>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes,
                        options.epsilon,
                    );
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let (datums, is_nulls) = fetcher.fetch(key)?;
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(RandomProject::project(raw.as_borrowed()))
                    };
                    let method = algorithm::how(relation.clone());
                    match method {
                        RerankMethod::Index => Box::new(
                            algorithm::rerank_index::<Op<VectOwned<f16>, L2>, _>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                        RerankMethod::Heap => Box::new(
                            algorithm::rerank_heap::<Op<VectOwned<f16>, L2>, _>(
                                vector, results, fetch,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                    }
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    let vector = RandomProject::project(
                        if let OwnedVector::Vecf16(vector) = vector {
                            vector
                        } else {
                            unreachable!()
                        }
                        .as_borrowed(),
                    );
                    let results = algorithm::default_search::<Op<VectOwned<f16>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes,
                        options.epsilon,
                    );
                    let fetch = move |payload| {
                        let (key, _) = pointer_to_kv(payload);
                        let (datums, is_nulls) = fetcher.fetch(key)?;
                        let datum = (!is_nulls[0]).then_some(datums[0]);
                        let maybe_vector = unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                        let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap() {
                            vector
                        } else {
                            unreachable!()
                        };
                        Some(RandomProject::project(raw.as_borrowed()))
                    };
                    let method = algorithm::how(relation.clone());
                    match method {
                        RerankMethod::Index => Box::new(
                            algorithm::rerank_index::<Op<VectOwned<f16>, Dot>, _>(
                                relation, vector, results,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                        RerankMethod::Heap => Box::new(
                            algorithm::rerank_heap::<Op<VectOwned<f16>, Dot>, _>(
                                vector, results, fetch,
                            )
                            .map(move |(distance, payload)| (opfamily.output(distance), payload)),
                        ),
                    }
                }
            };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(x, _)| *x < threshold))
        } else {
            iter
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        Box::new(iter.map(move |(distance, pointer)| {
            let (key, _) = pointer_to_kv(pointer);
            (distance, key, recheck)
        }))
    }
}
