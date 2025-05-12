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

pub mod algorithm;
pub mod am;
pub mod functions;
pub mod gucs;
pub mod hook;
pub mod lazy_cell;
pub mod opclass;
pub mod projection;
pub mod scanners;
pub mod storage;
pub mod types;

pub fn init() {
    am::init();
    hook::init();
    gucs::init();
    for x in gucs::prewarm_dim() {
        projection::prewarm(x as _);
    }
}
