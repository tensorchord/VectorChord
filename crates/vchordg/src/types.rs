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

use serde::{Deserialize, Serialize};
use simd::f16;
use validator::{Validate, ValidationError};
use vector::rabitq4::{Rabitq4Borrowed, Rabitq4Owned};
use vector::rabitq8::{Rabitq8Borrowed, Rabitq8Owned};
use vector::vect::{VectBorrowed, VectOwned};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordgIndexOptions {
    #[serde(default = "VchordgIndexOptions::default_bits")]
    #[validate(range(min = 1, max = 2))]
    pub bits: u8,
    #[serde(default = "VchordgIndexOptions::default_m")]
    #[validate(range(min = 1, max = 512))]
    pub m: u32,
    #[serde(default = "VchordgIndexOptions::default_alpha")]
    #[validate(length(min = 1, max = 8), custom(function = VchordgIndexOptions::validate_alpha))]
    pub alpha: Vec<f32>,
    #[serde(default = "VchordgIndexOptions::default_ef_construction")]
    #[validate(range(min = 1, max = 65535))]
    pub ef_construction: u32,
    #[serde(default = "VchordgIndexOptions::default_beam_construction")]
    #[validate(range(min = 1, max = 65535))]
    pub beam_construction: u32,
}

impl VchordgIndexOptions {
    fn default_bits() -> u8 {
        2
    }
    fn default_m() -> u32 {
        32
    }
    fn default_alpha() -> Vec<f32> {
        vec![1.0, 1.2]
    }
    fn validate_alpha(alpha: &[f32]) -> Result<(), ValidationError> {
        if !alpha.is_sorted() {
            return Err(ValidationError::new("`alpha` should be in ascending order"));
        }
        if !alpha.iter().all(|x| (1.0..2.0).contains(x)) {
            return Err(ValidationError::new("alpha is too large or too small"));
        }
        if alpha.first() != Some(&1.0) {
            return Err(ValidationError::new("`alpha` should contain `1.0`"));
        }
        Ok(())
    }
    fn default_ef_construction() -> u32 {
        64
    }
    fn default_beam_construction() -> u32 {
        1
    }
}

impl Default for VchordgIndexOptions {
    fn default() -> Self {
        Self {
            bits: Self::default_bits(),
            m: Self::default_m(),
            alpha: Self::default_alpha(),
            ef_construction: Self::default_ef_construction(),
            beam_construction: Self::default_beam_construction(),
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

#[derive(Debug, Clone, Validate)]
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
            (_, _, 1..=30000) => Ok(()),
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

#[cfg(test)]
mod tests {
    use super::VchordgIndexOptions;

    #[test]
    fn validate_alpha_handles_empty_without_panicking() {
        let empty: &[f32] = &[];
        assert!(VchordgIndexOptions::validate_alpha(empty).is_err());
    }

    #[test]
    fn validate_alpha_accepts_default() {
        assert!(VchordgIndexOptions::validate_alpha(&[1.0, 1.2]).is_ok());
    }

    #[test]
    fn validate_alpha_rejects_unsorted_or_out_of_range() {
        assert!(VchordgIndexOptions::validate_alpha(&[1.2, 1.0]).is_err());
        assert!(VchordgIndexOptions::validate_alpha(&[2.0]).is_err());
    }
}
