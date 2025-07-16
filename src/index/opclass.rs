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

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_l2_ops() -> String {
    "vchordg_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_cosine_ops() -> String {
    "vchordg_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_vector_ip_ops() -> String {
    "vchordg_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_l2_ops() -> String {
    "vchordg_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_cosine_ops() -> String {
    "vchordg_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordg_support_halfvec_ip_ops() -> String {
    "vchordg_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_l2_ops() -> String {
    "vchordrq_vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_ip_ops() -> String {
    "vchordrq_vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_cosine_ops() -> String {
    "vchordrq_vector_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_l2_ops() -> String {
    "vchordrq_halfvec_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_ip_ops() -> String {
    "vchordrq_halfvec_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_cosine_ops() -> String {
    "vchordrq_halfvec_cosine_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_vector_maxsim_ops() -> String {
    "vchordrq_vector_maxsim_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchordrq_support_halfvec_maxsim_ops() -> String {
    "vchordrq_halfvec_maxsim_ops".to_string()
}

pub struct Sphere<T> {
    pub center: T,
    pub radius: f32,
}
