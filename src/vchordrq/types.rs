use half::f16;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};
use vector::vect::{VectBorrowed, VectOwned};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqInternalBuildOptions {
    #[serde(default = "VchordrqInternalBuildOptions::default_lists")]
    #[validate(length(min = 1, max = 8), custom(function = VchordrqInternalBuildOptions::validate_lists))]
    pub lists: Vec<u32>,
    #[serde(default = "VchordrqInternalBuildOptions::default_spherical_centroids")]
    pub spherical_centroids: bool,
    #[serde(default = "VchordrqInternalBuildOptions::default_sampling_factor")]
    #[validate(range(min = 1, max = 1024))]
    pub sampling_factor: u32,
    #[serde(default = "VchordrqInternalBuildOptions::default_build_threads")]
    #[validate(range(min = 1, max = 255))]
    pub build_threads: u16,
}

impl VchordrqInternalBuildOptions {
    fn default_lists() -> Vec<u32> {
        vec![1000]
    }
    fn validate_lists(lists: &[u32]) -> Result<(), ValidationError> {
        if !lists.is_sorted() {
            return Err(ValidationError::new("`lists` should be in ascending order"));
        }
        if !lists.iter().all(|x| (1..=1 << 24).contains(x)) {
            return Err(ValidationError::new("list is too long or too short"));
        }
        Ok(())
    }
    fn default_spherical_centroids() -> bool {
        false
    }
    fn default_sampling_factor() -> u32 {
        256
    }
    fn default_build_threads() -> u16 {
        1
    }
}

impl Default for VchordrqInternalBuildOptions {
    fn default() -> Self {
        Self {
            lists: Self::default_lists(),
            spherical_centroids: Self::default_spherical_centroids(),
            sampling_factor: Self::default_sampling_factor(),
            build_threads: Self::default_build_threads(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqExternalBuildOptions {
    pub table: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum VchordrqBuildOptions {
    Internal(VchordrqInternalBuildOptions),
    External(VchordrqExternalBuildOptions),
}

impl Default for VchordrqBuildOptions {
    fn default() -> Self {
        Self::Internal(Default::default())
    }
}

impl Validate for VchordrqBuildOptions {
    fn validate(&self) -> Result<(), ValidationErrors> {
        use VchordrqBuildOptions::*;
        match self {
            Internal(internal_build) => internal_build.validate(),
            External(external_build) => external_build.validate(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqIndexingOptions {
    #[serde(default = "VchordrqIndexingOptions::default_residual_quantization")]
    pub residual_quantization: bool,
    pub build: VchordrqBuildOptions,
}

impl VchordrqIndexingOptions {
    fn default_residual_quantization() -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    L2,
    Dot,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VectorKind {
    Vecf32,
    Vecf16,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "Self::validate_self"))]
pub struct VectorOptions {
    #[validate(range(min = 1, max = 1_048_575))]
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
            (VectorKind::Vecf32, DistanceKind::L2, 1..65536) => Ok(()),
            (VectorKind::Vecf32, DistanceKind::Dot, 1..65536) => Ok(()),
            (VectorKind::Vecf16, DistanceKind::L2, 1..65536) => Ok(()),
            (VectorKind::Vecf16, DistanceKind::Dot, 1..65536) => Ok(()),
            _ => Err(ValidationError::new("not valid vector options")),
        }
    }
}
