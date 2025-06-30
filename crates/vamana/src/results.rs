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

use always_equal::AlwaysEqual;
use distance::Distance;
use min_max_heap::MinMaxHeap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct Results<T> {
    ef: usize,
    front: MinMaxHeap<(Distance, AlwaysEqual<T>)>,
    back: BinaryHeap<(Reverse<Distance>, AlwaysEqual<T>)>,
    // len(front) < ef, then len(back) == 0
    // len(front) == ef, then max(front) < min(back)
}

impl<T> Results<T> {
    pub fn new(ef: usize) -> Self {
        assert!(ef > 0, "ef must be positive integer");
        Self {
            ef,
            front: MinMaxHeap::with_capacity(ef),
            back: BinaryHeap::new(),
        }
    }
    pub fn push(&mut self, item: (Reverse<Distance>, AlwaysEqual<T>)) {
        if self.front.len() < self.ef {
            self.front.push((item.0.0, item.1));
        } else {
            let item = self.front.push_pop_max((item.0.0, item.1));
            self.back.push((Reverse(item.0), item.1));
        }
    }
    pub fn peek_ef_th(&mut self) -> Option<Distance> {
        if self.front.len() < self.ef {
            None
        } else {
            self.front.peek_max().map(|&(d, _)| d)
        }
    }
    #[allow(clippy::collapsible_else_if)]
    pub fn pop_min(&mut self) -> Option<(Distance, AlwaysEqual<T>)> {
        if self.front.len() < self.ef {
            self.front.pop_min()
        } else {
            if let Some(item) = self.back.pop() {
                Some(self.front.push_pop_min((item.0.0, item.1)))
            } else {
                self.front.pop_min()
            }
        }
    }
    pub fn into_inner(
        self,
    ) -> (
        Vec<(Distance, AlwaysEqual<T>)>,
        Vec<(Reverse<Distance>, AlwaysEqual<T>)>,
    ) {
        (self.front.into_vec(), self.back.into_vec())
    }
}
