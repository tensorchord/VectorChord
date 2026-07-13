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

use crate::index::fetcher::pointer_to_kv;
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;

pub(super) type HeapKey = [u16; 3];
pub(super) type TokenCandidate = (Distance, NonZero<u64>);
pub(super) type TokenSearchResult = (Vec<TokenCandidate>, Vec<TokenCandidate>, Distance);

#[derive(Clone, Copy, Debug)]
pub(super) struct PageCandidate {
    pub approximate_distance: Distance,
    pub heap_key: HeapKey,
}

pub(super) trait PageCandidateGenerator {
    type Candidates: Iterator<Item = PageCandidate>;

    fn generate(
        &mut self,
        query_count: usize,
        token_searches: &mut dyn Iterator<Item = TokenSearchResult>,
        candidate_limit: usize,
    ) -> Self::Candidates;
}

#[derive(Default)]
pub(super) struct LegacyPageCandidateGenerator;

impl PageCandidateGenerator for LegacyPageCandidateGenerator {
    type Candidates = LegacyPageCandidates;

    fn generate(
        &mut self,
        query_count: usize,
        token_searches: &mut dyn Iterator<Item = TokenSearchResult>,
        candidate_limit: usize,
    ) -> Self::Candidates {
        let mut updates = Vec::new();
        let mut estimations = Vec::with_capacity(query_count);
        for (query_id, (accu_set, rough_set, estimation_by_threshold)) in token_searches.enumerate()
        {
            updates.reserve(accu_set.len() + rough_set.len());
            let is_empty = accu_set.is_empty() && rough_set.is_empty();
            let mut estimation_by_scope = Distance::NEG_INFINITY;
            for (distance, payload) in accu_set {
                estimation_by_scope = std::cmp::max(estimation_by_scope, distance);
                let (key, _) = pointer_to_kv(payload);
                updates.push((key, query_id, distance));
            }
            for (distance, payload) in rough_set {
                let (key, _) = pointer_to_kv(payload);
                updates.push((key, query_id, distance));
            }
            estimations.push(if !is_empty {
                std::cmp::max(estimation_by_scope, estimation_by_threshold)
            } else {
                Distance::ZERO
            });
        }
        debug_assert_eq!(estimations.len(), query_count);
        updates.sort_unstable_by_key(|&(key, ..)| key);
        let inner = updates
            .chunk_by(|(kl, ..), (kr, ..)| kl == kr)
            .map(|chunk| {
                let key = chunk[0].0;
                let mut value = vec![None; query_count];
                for &(_, query_id, distance) in chunk {
                    let this = value[query_id].get_or_insert(Distance::INFINITY);
                    *this = std::cmp::min(*this, distance);
                }
                let mut maxsim = 0.0f32;
                for (query_id, distance) in value.into_iter().enumerate() {
                    let distance = distance.unwrap_or(estimations[query_id]);
                    maxsim += distance.to_f32();
                }
                (Reverse(Distance::from_f32(maxsim)), AlwaysEqual(key))
            })
            .collect::<BinaryHeap<_>>();
        LegacyPageCandidates {
            inner,
            remaining: candidate_limit,
        }
    }
}

pub(super) struct LegacyPageCandidates {
    inner: BinaryHeap<(Reverse<Distance>, AlwaysEqual<HeapKey>)>,
    remaining: usize,
}

impl Iterator for LegacyPageCandidates {
    type Item = PageCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let (Reverse(approximate_distance), AlwaysEqual(heap_key)) = self.inner.pop()?;
        self.remaining -= 1;
        Some(PageCandidate {
            approximate_distance,
            heap_key,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.inner.len().min(self.remaining);
        (exact, Some(exact))
    }
}

impl ExactSizeIterator for LegacyPageCandidates {}
impl std::iter::FusedIterator for LegacyPageCandidates {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetcher::kv_to_pointer;

    fn distance(value: f32) -> Distance {
        Distance::from_f32(value)
    }

    #[test]
    fn aggregates_tokens_by_page_and_preserves_distance_order() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let mut token_searches = vec![
            (
                vec![
                    (distance(-1.0), kv_to_pointer((page_1, 0))),
                    (distance(-0.5), kv_to_pointer((page_2, 0))),
                ],
                vec![(distance(-0.4), kv_to_pointer((page_2, 1)))],
                distance(-0.75),
            ),
            (
                vec![(distance(-2.0), kv_to_pointer((page_1, 1)))],
                vec![],
                distance(-1.0),
            ),
        ]
        .into_iter();
        let candidates = LegacyPageCandidateGenerator
            .generate(2, &mut token_searches, usize::MAX)
            .collect::<Vec<_>>();

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].heap_key, page_1);
        assert_eq!(candidates[0].approximate_distance.to_f32(), -3.0);
        assert_eq!(candidates[1].heap_key, page_2);
        assert_eq!(candidates[1].approximate_distance.to_f32(), -1.5);
    }

    #[test]
    fn candidate_limit_is_applied_after_global_ordering() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let mut token_searches = vec![(
            vec![
                (distance(-1.0), kv_to_pointer((page_1, 0))),
                (distance(-2.0), kv_to_pointer((page_2, 0))),
            ],
            vec![],
            Distance::ZERO,
        )]
        .into_iter();
        let candidates = LegacyPageCandidateGenerator
            .generate(1, &mut token_searches, 1)
            .collect::<Vec<_>>();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].heap_key, page_2);
    }
}
