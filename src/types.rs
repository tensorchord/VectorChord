use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct RabbitholeInternalBuildOptions {
    #[serde(default = "RabbitholeInternalBuildOptions::default_lists")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub lists: u32,
    #[serde(default = "RabbitholeInternalBuildOptions::default_spherical_centroids")]
    pub spherical_centroids: bool,
    #[serde(default = "RabbitholeInternalBuildOptions::default_build_threads")]
    #[validate(range(min = 1, max = 255))]
    pub build_threads: u16,
}

impl RabbitholeInternalBuildOptions {
    fn default_lists() -> u32 {
        1000
    }
    fn default_spherical_centroids() -> bool {
        false
    }
    fn default_build_threads() -> u16 {
        1
    }
}

impl Default for RabbitholeInternalBuildOptions {
    fn default() -> Self {
        Self {
            lists: Self::default_lists(),
            spherical_centroids: Self::default_spherical_centroids(),
            build_threads: Self::default_build_threads(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct RabbitholeExternalBuildOptions {
    pub table: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum RabbitholeBuildOptions {
    Internal(RabbitholeInternalBuildOptions),
    External(RabbitholeExternalBuildOptions),
}

impl Default for RabbitholeBuildOptions {
    fn default() -> Self {
        Self::Internal(Default::default())
    }
}

impl Validate for RabbitholeBuildOptions {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        use RabbitholeBuildOptions::*;
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
    pub build: RabbitholeBuildOptions,
}

impl VchordrqIndexingOptions {
    fn default_residual_quantization() -> bool {
        false
    }
}
