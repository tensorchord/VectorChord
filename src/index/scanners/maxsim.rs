use super::{SearchBuilder, SearchFetcher, SearchOptions};
use crate::index::algorithm::RandomProject;
use crate::index::am::pointer_to_ctid;
use crate::index::opclass::Opfamily;
use algorithm::operator::{Dot, Op, Vector};
use algorithm::types::{DistanceKind, OwnedVector, VectorKind};
use algorithm::{RelationRead, RerankMethod};
use always_equal::AlwaysEqual;
use distance::Distance;
use half::f16;
use pgrx::pg_sys::{BlockIdData, ItemPointerData};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use vector::VectorOwned;
use vector::vect::VectOwned;

pub struct MaxsimBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<Vec<OwnedVector>>>,
}

impl SearchBuilder for MaxsimBuilder {
    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            3 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vectors(x)) };
                self.orderbys.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'a>(
        self,
        relation: impl RelationRead + 'a,
        options: SearchOptions,
        _: impl SearchFetcher + 'a,
    ) -> Box<dyn Iterator<Item = (f32, ItemPointerData, bool)> + 'a> {
        let mut vectors = None;
        for orderby_vectors in self.orderbys.into_iter().flatten() {
            if vectors.is_none() {
                vectors = Some(orderby_vectors);
            } else {
                pgrx::error!("maxsim search with multiple vectors is not supported");
            }
        }
        if let Some(_max_scan_tuples) = options.max_scan_tuples {
            pgrx::error!("maxsim search with max_scan_tuples is not supported");
        }
        let max_maxsim_tuples = options
            .max_maxsim_tuples
            .expect("max_maxsim_tuples must be set for maxsim search");
        let maxsim_threshold = options.maxsim_threshold;
        let opfamily = self.opfamily;
        let Some(vectors) = vectors else {
            return Box::new(std::iter::empty())
                as Box<dyn Iterator<Item = (f32, ItemPointerData, bool)>>;
        };
        let method = algorithm::how(relation.clone());
        if !matches!(method, RerankMethod::Index) {
            pgrx::error!("maxsim search with rerank_in_table is not supported");
        }
        assert!(matches!(opfamily.distance_kind(), DistanceKind::Dot));
        let (c, estimations) = match opfamily.vector_kind() {
            VectorKind::Vecf32 => {
                let vectors = vectors
                    .into_iter()
                    .map(|vector| {
                        RandomProject::project(
                            VectOwned::<f32>::from_owned(vector.clone()).as_borrowed(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut estimations = Vec::new();
                let mut c = HashMap::<[u16; 3], States>::new();
                for (query_id, vector) in vectors.iter().enumerate() {
                    let (results, est) = algorithm::search_and_estimate::<Op<VectOwned<f32>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes.clone(),
                        options.epsilon,
                        maxsim_threshold,
                    );
                    if results.is_empty() {
                        estimations.push(0.0);
                        continue;
                    }
                    let returning = if options.epsilon == 0.0 && options.allows_skipping_rerank {
                        algorithm::skip(results)
                            .take(max_maxsim_tuples as usize)
                            .collect::<Vec<_>>()
                    } else {
                        algorithm::rerank_index::<Op<VectOwned<f32>, Dot>>(
                            relation.clone(),
                            vector.clone(),
                            results,
                        )
                        .take(max_maxsim_tuples as usize)
                        .collect::<Vec<_>>()
                    };
                    let mut max = f32::NEG_INFINITY;
                    for (distance, payload) in returning {
                        max = max.max(distance.to_f32());
                        use std::collections::hash_map::Entry::{Occupied, Vacant};
                        let (ctid, _) = pointer_to_ctid(payload);
                        let key = [ctid.ip_blkid.bi_hi, ctid.ip_blkid.bi_lo, ctid.ip_posid];
                        match c.entry(key) {
                            Vacant(e) => {
                                let states = e.insert(States::new(vectors.len()));
                                states.update(query_id, distance);
                            }
                            Occupied(mut e) => {
                                let states = e.get_mut();
                                states.update(query_id, distance);
                            }
                        }
                    }
                    estimations.push(max.max(est.to_f32()));
                }
                (c, estimations)
            }
            VectorKind::Vecf16 => {
                let vectors = vectors
                    .into_iter()
                    .map(|vector| {
                        RandomProject::project(
                            VectOwned::<f16>::from_owned(vector.clone()).as_borrowed(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut estimations = Vec::new();
                let mut c = HashMap::<[u16; 3], States>::new();
                for (query_id, vector) in vectors.iter().enumerate() {
                    let (results, est) = algorithm::search_and_estimate::<Op<VectOwned<f16>, Dot>>(
                        relation.clone(),
                        vector.clone(),
                        options.probes.clone(),
                        options.epsilon,
                        maxsim_threshold,
                    );
                    if results.is_empty() {
                        estimations.push(0.0);
                        continue;
                    }
                    let returning = if options.epsilon == 0.0 && options.allows_skipping_rerank {
                        algorithm::skip(results)
                            .take(max_maxsim_tuples as usize)
                            .collect::<Vec<_>>()
                    } else {
                        algorithm::rerank_index::<Op<VectOwned<f16>, Dot>>(
                            relation.clone(),
                            vector.clone(),
                            results,
                        )
                        .take(max_maxsim_tuples as usize)
                        .collect::<Vec<_>>()
                    };
                    let mut max = f32::NEG_INFINITY;
                    for (distance, payload) in returning {
                        max = max.max(distance.to_f32());
                        use std::collections::hash_map::Entry::{Occupied, Vacant};
                        let (ctid, _) = pointer_to_ctid(payload);
                        let key = [ctid.ip_blkid.bi_hi, ctid.ip_blkid.bi_lo, ctid.ip_posid];
                        match c.entry(key) {
                            Vacant(e) => {
                                let states = e.insert(States::new(vectors.len()));
                                states.update(query_id, distance);
                            }
                            Occupied(mut e) => {
                                let states = e.get_mut();
                                states.update(query_id, distance);
                            }
                        }
                    }
                    estimations.push(max.max(est.to_f32()));
                }
                (c, estimations)
            }
        };
        let c = c
            .into_iter()
            .map(|(key, states)| {
                let mut maxsim = 0.0f32;
                for (query_id, distance) in states.into_iter() {
                    let d = distance.unwrap_or(Distance::from_f32(estimations[query_id]));
                    maxsim += opfamily.output(d);
                }
                (Reverse(Distance::from_f32(maxsim)), AlwaysEqual(key))
            })
            .collect::<Vec<_>>();
        let mut c = BinaryHeap::from(c);
        Box::new(std::iter::from_fn(move || {
            let (Reverse(distance), AlwaysEqual(key)) = c.pop()?;
            let distance = distance.to_f32();
            let ctid = ItemPointerData {
                ip_blkid: BlockIdData {
                    bi_hi: key[0],
                    bi_lo: key[1],
                },
                ip_posid: key[2],
            };
            let recheck = false;
            Some((distance, ctid, recheck))
        }))
    }
}

struct States {
    inner: Vec<Option<Distance>>,
}

impl States {
    fn new(len: usize) -> Self {
        Self {
            inner: vec![None; len],
        }
    }
    fn update(&mut self, query_id: usize, distance: Distance) {
        let this = self.inner[query_id].get_or_insert(Distance::INFINITY);
        *this = std::cmp::min(*this, distance);
    }
    fn into_iter(self) -> impl Iterator<Item = (usize, Option<Distance>)> {
        self.inner.into_iter().enumerate()
    }
}
