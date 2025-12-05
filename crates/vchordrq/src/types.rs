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
use vector::rabitq4::{Rabitq4Borrowed, Rabitq4Owned};
use vector::rabitq8::{Rabitq8Borrowed, Rabitq8Owned};
use vector::vect::{VectBorrowed, VectOwned};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqIndexOptions {
    #[serde(default = "VchordrqIndexOptions::default_residual_quantization")]
    pub residual_quantization: bool,
    #[serde(default = "VchordrqIndexOptions::default_rerank_in_table")]
    pub rerank_in_table: bool,
    #[serde(default = "VchordrqIndexOptions::default_degree_of_parallelism")]
    #[validate(range(min = 1, max = 256))]
    pub degree_of_parallelism: u32,
}

impl VchordrqIndexOptions {
    fn default_residual_quantization() -> bool {
        false
    }
    fn default_rerank_in_table() -> bool {
        false
    }
    fn default_degree_of_parallelism() -> u32 {
        32
    }
}

impl Default for VchordrqIndexOptions {
    fn default() -> Self {
        Self {
            residual_quantization: Self::default_residual_quantization(),
            rerank_in_table: Self::default_rerank_in_table(),
            degree_of_parallelism: Self::default_degree_of_parallelism(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OwnedVector {
    Vecf32(VectOwned<f32>),
    Vecf16(VectOwned<f16>),
    Rabitq8(Rabitq8Owned),
    Rabitq4(Rabitq4Owned),
}

#[derive(Debug, Clone, Copy)]
pub enum BorrowedVector<'a> {
    Vecf32(VectBorrowed<'a, f32>),
    Vecf16(VectBorrowed<'a, f16>),
    Rabitq8(Rabitq8Borrowed<'a>),
    Rabitq4(Rabitq4Borrowed<'a>),
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DistanceKind {
    L2S,
    Dot,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VectorKind {
    Vecf32,
    Vecf16,
    Rabitq8,
    Rabitq4,
}

impl VectorKind {
    pub fn number_of_bits_of_an_elements(self) -> u32 {
        match self {
            VectorKind::Vecf32 => 32,
            VectorKind::Vecf16 => 16,
            VectorKind::Rabitq8 => 8,
            VectorKind::Rabitq4 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, Validate)]
#[validate(schema(function = "Self::validate_self"))]
pub struct VectorOptions {
    #[validate(range(min = 1))]
    pub dim: u32,
    pub v: VectorKind,
    pub d: DistanceKind,
}

impl VectorOptions {
    pub fn validate_self(&self) -> Result<(), ValidationError> {
        match (self.v, self.d, self.dim) {
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
