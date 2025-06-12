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

pub mod allocator;
pub mod fetcher;
pub mod functions;
pub mod storage;
pub mod vamana;
pub mod vchordrq;

pub fn init() {
    vchordrq::am::init();
    vchordrq::hook::init();
    vchordrq::gucs::init();
    vamana::am::init();
    vamana::hook::init();
    vamana::gucs::init();
}
