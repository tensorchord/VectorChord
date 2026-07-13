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
// Copyright (c) 2025-2026 TensorChord Inc.

use super::candidate::{HeapKey, PageCandidate};
use crate::index::fetcher::{Fetcher, FilterableTuple, Tuple};
use crate::index::vchordrq::opclass::Opfamily;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vchordrq::types::OwnedVector;

pub(super) struct CandidateTensor {
    pub candidate: PageCandidate,
    pub vectors: Vec<OwnedVector>,
}

pub(super) trait CandidateTensorSource {
    fn fetch(&mut self, candidate: PageCandidate) -> Result<Option<CandidateTensor>, RerankError>;
}

pub(super) struct HeapArrayTensorSource<'a, F> {
    fetcher: &'a mut F,
    opfamily: Opfamily,
}

impl<'a, F> HeapArrayTensorSource<'a, F> {
    pub fn new(fetcher: &'a mut F, opfamily: Opfamily) -> Self {
        Self { fetcher, opfamily }
    }
}

impl<F: Fetcher> CandidateTensorSource for HeapArrayTensorSource<'_, F> {
    fn fetch(&mut self, candidate: PageCandidate) -> Result<Option<CandidateTensor>, RerankError> {
        let Some(mut tuple) = self.fetcher.fetch(candidate.heap_key) else {
            return Ok(None);
        };
        // Exact sources are the last boundary before a tensor may leave the
        // executor process. Re-evaluate the active base-relation scan qual
        // here even if token reranking already prefiltered some hits. This
        // keeps same-relation quals ahead of CPU/GPU tensor access and
        // also covers configurations with token-level refine disabled.
        if !tuple.filter() {
            return Ok(None);
        }
        let (values, is_nulls) = tuple.build();
        if is_nulls[0] {
            return Err(RerankError::TensorMismatch);
        }
        let vectors =
            unsafe { self.opfamily.input_vectors(values[0]) }.ok_or(RerankError::TensorMismatch)?;
        Ok(Some(CandidateTensor { candidate, vectors }))
    }
}

pub(super) trait ExactMaxsimBackend {
    type Results: Iterator<Item = RerankedPage>;

    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<Self::Results, RerankError>;
}

#[derive(Debug)]
pub(super) enum RerankError {
    TensorMismatch,
    ModelContractMismatch,
    InvalidDescriptor(&'static str),
    Registry(String),
    Configuration(&'static str),
    UnsupportedTensorKind,
    RequestTooLarge,
    Transport(String),
    Protocol(String),
    Remote(String),
}

impl std::fmt::Display for RerankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorMismatch => write!(f, "MaxSim tensor kind or dimension is not matched"),
            Self::ModelContractMismatch => write!(f, "MaxSim model contract is not matched"),
            Self::InvalidDescriptor(message) => {
                write!(f, "invalid external MaxSim tensor descriptor: {message}")
            }
            Self::Registry(message) => {
                write!(f, "MaxSim tensor-source registry error: {message}")
            }
            Self::Configuration(message) => write!(f, "MaxSim configuration error: {message}"),
            Self::UnsupportedTensorKind => {
                write!(f, "GPU MaxSim supports only vector and halfvec tensors")
            }
            Self::RequestTooLarge => write!(f, "GPU MaxSim request exceeds configured limits"),
            Self::Transport(message) => write!(f, "GPU MaxSim transport error: {message}"),
            Self::Protocol(message) => write!(f, "GPU MaxSim protocol error: {message}"),
            Self::Remote(message) => write!(f, "GPU MaxSim sidecar error: {message}"),
        }
    }
}

impl std::error::Error for RerankError {}

#[derive(Default)]
pub(super) struct CpuExactMaxsimBackend;

impl ExactMaxsimBackend for CpuExactMaxsimBackend {
    type Results = RerankResults;

    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<Self::Results, RerankError> {
        let mut results = BinaryHeap::new();
        for candidate in candidates {
            let Some(tensor) = source.fetch(candidate)? else {
                continue;
            };
            let Some(distance) = exact_maxsim_distance(query, &tensor.vectors) else {
                return Err(RerankError::TensorMismatch);
            };
            results.push((Reverse(distance), Reverse(tensor.candidate.heap_key)));
        }
        Ok(RerankResults { inner: results })
    }
}

fn exact_maxsim_distance(query: &[OwnedVector], document: &[OwnedVector]) -> Option<Distance> {
    if query.is_empty() || document.is_empty() {
        return None;
    }
    let mut maxsim = 0.0f32;
    for query_vector in query {
        let mut best = Distance::INFINITY;
        for document_vector in document {
            let distance = document_vector.operator_dot(query_vector)?;
            best = std::cmp::min(best, distance);
        }
        maxsim += best.to_f32();
    }
    Some(Distance::from_f32(maxsim))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RerankedPage {
    pub distance: Distance,
    pub heap_key: HeapKey,
}

pub(super) struct RerankResults {
    pub(super) inner: BinaryHeap<(Reverse<Distance>, Reverse<HeapKey>)>,
}

impl Iterator for RerankResults {
    type Item = RerankedPage;

    fn next(&mut self) -> Option<Self::Item> {
        let (Reverse(distance), Reverse(heap_key)) = self.inner.pop()?;
        Some(RerankedPage { distance, heap_key })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len();
        (exact, Some(exact))
    }
}

impl ExactSizeIterator for RerankResults {}
impl std::iter::FusedIterator for RerankResults {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vector::vect::VectOwned;

    struct MockTensorSource(BTreeMap<HeapKey, Vec<OwnedVector>>);

    impl CandidateTensorSource for MockTensorSource {
        fn fetch(
            &mut self,
            candidate: PageCandidate,
        ) -> Result<Option<CandidateTensor>, RerankError> {
            Ok(Some(CandidateTensor {
                candidate,
                vectors: self
                    .0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            }))
        }
    }

    fn vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf32(VectOwned::new(values.to_vec()))
    }

    struct RejectingFetcher;

    struct RejectingTuple;

    impl Tuple for RejectingTuple {
        fn build(&mut self) -> (&[pgrx::pg_sys::Datum; 32], &[bool; 32]) {
            panic!("a rejected tuple must not materialize its tensor")
        }

        fn attribute(&mut self, _attnum: i16) -> Option<crate::index::fetcher::TupleAttribute> {
            panic!("a rejected tuple must not expose heap attributes")
        }
    }

    impl FilterableTuple for RejectingTuple {
        fn filter(&mut self) -> bool {
            false
        }
    }

    impl Fetcher for RejectingFetcher {
        type Tuple<'a> = RejectingTuple;

        fn fetch(&mut self, _key: HeapKey) -> Option<Self::Tuple<'_>> {
            Some(RejectingTuple)
        }
    }

    #[test]
    fn heap_source_applies_scan_qual_before_materializing_tensor() {
        let mut fetcher = RejectingFetcher;
        let mut source = HeapArrayTensorSource::new(&mut fetcher, Opfamily::VectorMaxsim);
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 1],
        };

        assert!(source.fetch(candidate).unwrap().is_none());
    }

    #[test]
    fn cpu_backend_reorders_candidates_by_exact_page_maxsim() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let query = vec![vector(&[1.0, 0.0]), vector(&[0.0, 1.0])];
        let mut candidates = vec![
            PageCandidate {
                approximate_distance: Distance::from_f32(-2.0),
                heap_key: page_2,
            },
            PageCandidate {
                approximate_distance: Distance::from_f32(-1.0),
                heap_key: page_1,
            },
        ]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([
            (page_1, vec![vector(&[1.0, 0.0]), vector(&[0.0, 1.0])]),
            (page_2, vec![vector(&[0.5, 0.5])]),
        ]));
        let results = CpuExactMaxsimBackend
            .rerank(&query, &mut candidates, &mut source)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].heap_key, page_1);
        assert_eq!(results[0].distance.to_f32(), -2.0);
        assert_eq!(results[1].heap_key, page_2);
        assert_eq!(results[1].distance.to_f32(), -1.0);
    }

    #[test]
    fn exact_ties_are_ordered_by_heap_key() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![
            PageCandidate {
                approximate_distance: Distance::from_f32(0.0),
                heap_key: page_2,
            },
            PageCandidate {
                approximate_distance: Distance::from_f32(0.0),
                heap_key: page_1,
            },
        ]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([
            (page_1, vec![vector(&[1.0, 0.0])]),
            (page_2, vec![vector(&[1.0, 0.0])]),
        ]));

        let results = CpuExactMaxsimBackend
            .rerank(&query, &mut candidates, &mut source)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(
            results.iter().map(|page| page.heap_key).collect::<Vec<_>>(),
            vec![page_1, page_2]
        );
    }
}
