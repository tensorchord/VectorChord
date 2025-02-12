use algorithm::types::VchordrqIndexOptions;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqInternalBuildOptions {
    #[serde(default = "VchordrqInternalBuildOptions::default_lists")]
    #[validate(length(min = 0, max = 8), custom(function = VchordrqInternalBuildOptions::validate_lists))]
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
pub enum VchordrqBuildSourceOptions {
    Internal(VchordrqInternalBuildOptions),
    External(VchordrqExternalBuildOptions),
}

impl Default for VchordrqBuildSourceOptions {
    fn default() -> Self {
        Self::Internal(Default::default())
    }
}

impl Validate for VchordrqBuildSourceOptions {
    fn validate(&self) -> Result<(), ValidationErrors> {
        use VchordrqBuildSourceOptions::*;
        match self {
            Internal(internal_build) => internal_build.validate(),
            External(external_build) => external_build.validate(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub struct VchordrqBuildOptions {
    #[serde(flatten)]
    pub source: VchordrqBuildSourceOptions,
    #[serde(default = "VchordrqBuildOptions::default_pin")]
    pub pin: bool,
}

impl VchordrqBuildOptions {
    pub fn default_pin() -> bool {
        false
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqIndexingOptions {
    #[serde(flatten)]
    pub index: VchordrqIndexOptions,
    pub build: VchordrqBuildOptions,
}
