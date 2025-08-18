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

use crate::index::collector::{CollectorSender, UniVec};
use crate::index::fetcher::{Fetcher, pointer_to_kv};
use crate::index::opclass::Sphere;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordg::algo::*;
use crate::index::vchordg::opclass::Opfamily;
use crate::index::vchordg::scanners::SearchOptions;
use algo::accessor::{Dot, L2S};
use algo::*;
use distance::Distance;
use half::f16;
use std::num::NonZero;
use vchordg::operator::{self};
use vchordg::types::{DistanceKind, OwnedVector, VectorKind};
use vchordg::*;
use vector::vect::{VectBorrowed, VectOwned};

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

impl SearchBuilder for DefaultBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordg::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorL2
                | Opfamily::VectorIp
                | Opfamily::HalfvecIp
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

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        _fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        sender: impl CollectorSender,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordg::Opaque>,
    {
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
        sender.send(UniVec::G(vector.clone()));
        let make_vertex_plain_prefetcher = MakePlainPrefetcher { index };
        let make_vertex_simple_prefetcher = MakeSimplePrefetcher { index };
        let make_vertex_stream_prefetcher = MakeStreamPrefetcher {
            index,
            hints: Hints::default().full(true),
        };
        let make_vector_plain_prefetcher = MakePlainPrefetcher { index };
        let make_vector_simple_prefetcher = MakeSimplePrefetcher { index };
        let make_vector_stream_prefetcher = MakeStreamPrefetcher {
            index,
            hints: Hints::default().full(true),
        };
        let iter: Box<dyn Iterator<Item = (Distance, NonZero<u64>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2S) => {
                    type Op = operator::Op<VectOwned<f32>, L2S>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector {
                        VectBorrowed::new(bump.alloc_slice(vector.slice()))
                    } else {
                        unreachable!()
                    };
                    let projected = {
                        let projected = RandomProject::project(unprojected);
                        VectBorrowed::new(bump.alloc_slice(projected.slice()))
                    };
                    match (options.io_search, options.io_rerank) {
                        (Io::Plain, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Simple, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Stream, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                    }
                }
                (VectorKind::Vecf16, DistanceKind::L2S) => {
                    type Op = operator::Op<VectOwned<f16>, L2S>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector {
                        VectBorrowed::new(bump.alloc_slice(vector.slice()))
                    } else {
                        unreachable!()
                    };
                    let projected = {
                        let projected = RandomProject::project(unprojected);
                        VectBorrowed::new(bump.alloc_slice(projected.slice()))
                    };
                    match (options.io_search, options.io_rerank) {
                        (Io::Plain, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Simple, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Stream, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                    }
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    type Op = operator::Op<VectOwned<f32>, Dot>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector {
                        VectBorrowed::new(bump.alloc_slice(vector.slice()))
                    } else {
                        unreachable!()
                    };
                    let projected = {
                        let projected = RandomProject::project(unprojected);
                        VectBorrowed::new(bump.alloc_slice(projected.slice()))
                    };
                    match (options.io_search, options.io_rerank) {
                        (Io::Plain, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Simple, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Stream, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                    }
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    type Op = operator::Op<VectOwned<f16>, Dot>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector {
                        VectBorrowed::new(bump.alloc_slice(vector.slice()))
                    } else {
                        unreachable!()
                    };
                    let projected = {
                        let projected = RandomProject::project(unprojected);
                        VectBorrowed::new(bump.alloc_slice(projected.slice()))
                    };
                    match (options.io_search, options.io_rerank) {
                        (Io::Plain, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Plain, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_plain_prefetcher,
                        ),
                        (Io::Simple, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Simple, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_simple_prefetcher,
                        ),
                        (Io::Stream, Io::Plain) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_plain_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Simple) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_simple_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                        (Io::Stream, Io::Stream) => search::<_, Op>(
                            index,
                            projected,
                            options.ef_search,
                            options.beam_search,
                            bump,
                            make_vertex_stream_prefetcher,
                            make_vector_stream_prefetcher,
                        ),
                    }
                }
            };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(distance, _)| distance.to_f32() < threshold))
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
            (opfamily.output(distance), key, recheck)
        }))
    }
}
