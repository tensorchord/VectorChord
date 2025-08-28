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

use serde::{Deserialize, Serialize};
use simd::f16;
use validator::{Validate, ValidationError};
use vector::vect::{VectBorrowed, VectOwned};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqIndexOptions {
    #[serde(default = "VchordrqIndexOptions::default_residual_quantization")]
    pub residual_quantization: bool,
    #[serde(default = "VchordrqIndexOptions::default_rerank_in_table")]
    pub rerank_in_table: bool,
}

impl VchordrqIndexOptions {
    fn default_residual_quantization() -> bool {
        false
    }
    fn default_rerank_in_table() -> bool {
        false
    }
}

impl Default for VchordrqIndexOptions {
    fn default() -> Self {
        Self {
            residual_quantization: Self::default_residual_quantization(),
            rerank_in_table: Self::default_rerank_in_table(),
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
