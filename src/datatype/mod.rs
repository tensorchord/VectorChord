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

mod binary_rabitq4;
mod binary_rabitq8;
mod functions_rabitq4;
mod functions_rabitq8;
pub mod memory_halfvec;
pub mod memory_rabitq4;
pub mod memory_rabitq8;
pub mod memory_vector;
mod operators_halfvec;
mod operators_rabitq4;
mod operators_rabitq8;
mod operators_vector;
mod text_rabitq4;
mod text_rabitq8;
pub mod typmod;
mod typmod_rabitq4;
mod typmod_rabitq8;

pub(crate) const MAX_MAXSIM_VECTORS: usize = u16::MAX as usize + 1;

pub(crate) fn validate_maxsim_array_len(len: usize) {
    if len == 0 {
        pgrx::error!("MaxSim arrays must contain at least one vector");
    }
    if len > MAX_MAXSIM_VECTORS {
        pgrx::error!(
            "MaxSim arrays cannot contain more than {} vectors",
            MAX_MAXSIM_VECTORS
        );
    }
}
