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

use half::f16;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};
use vector::vect::{VectBorrowed, VectOwned};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VamanaIndexOptions {
    #[serde(default = "VamanaIndexOptions::default_m")]
    #[validate(range(min = 1, max = 512))]
    pub m: u32,
    #[serde(default = "VamanaIndexOptions::default_max_alpha")]
    #[validate(range(min = 1.0, max = 2.0))]
    pub max_alpha: f32,
    #[serde(default = "VamanaIndexOptions::default_ef_construction")]
    #[validate(range(min = 1, max = 65535))]
    pub ef_construction: u32,
    #[serde(default = "VamanaIndexOptions::default_beam_construction")]
    #[validate(range(min = 1, max = 65535))]
    pub beam_construction: u32,
}

impl VamanaIndexOptions {
    fn default_m() -> u32 {
        32
    }
    fn default_max_alpha() -> f32 {
        1.0
    }
    fn default_ef_construction() -> u32 {
        64
    }
    fn default_beam_construction() -> u32 {
        1
    }
}

impl Default for VamanaIndexOptions {
    fn default() -> Self {
        Self {
            m: Self::default_m(),
            max_alpha: Self::default_max_alpha(),
            ef_construction: Self::default_ef_construction(),
            beam_construction: Self::default_beam_construction(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OwnedVector {
    Vecf32(VectOwned<f32>),
    Vecf16(VectOwned<f16>),
}

#[derive(Debug, Clone, Copy)]
pub enum BorrowedVector<'a> {
    Vecf32(VectBorrowed<'a, f32>),
    Vecf16(VectBorrowed<'a, f16>),
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DistanceKind {
    L2S,
    Dot,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VectorKind {
    Vecf32,
    Vecf16,
}

impl VectorKind {
    pub fn element_size(self) -> u32 {
        match self {
            VectorKind::Vecf32 => size_of::<f32>() as _,
            VectorKind::Vecf16 => size_of::<f16>() as _,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "Self::validate_self"))]
pub struct VectorOptions {
    #[validate(range(min = 1))]
    #[serde(rename = "dimensions")]
    pub dims: u32,
    #[serde(rename = "vector")]
    pub v: VectorKind,
    #[serde(rename = "distance")]
    pub d: DistanceKind,
}

impl VectorOptions {
    pub fn validate_self(&self) -> Result<(), ValidationError> {
        match (self.v, self.d, self.dims) {
            (_, _, 1..=60000) => Ok(()),
            _ => Err(ValidationError::new("invalid vector options")),
        }
    }
}

pub struct Structure<V> {
    pub centroids: Vec<V>,
    pub children: Vec<Vec<u32>>,
}

impl<V> Structure<V> {
    pub fn len(&self) -> usize {
        self.children.len()
    }
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}
