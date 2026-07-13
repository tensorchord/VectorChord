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

mod default;
mod maxsim;

use crate::index::gucs::PostgresMaxsimBackend;
use crate::index::scanners::Io;
use std::ffi::CString;

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub maxsim_candidate_limit: Option<u32>,
    pub maxsim_backend: PostgresMaxsimBackend,
    pub maxsim_gpu_endpoint: Option<CString>,
    pub maxsim_gpu_timeout_ms: u32,
    pub maxsim_gpu_max_batch_tokens: u32,
    pub maxsim_gpu_max_batch_bytes: u32,
    pub io_search: Io,
    pub io_rerank: Io,
    pub prefilter: bool,
}
