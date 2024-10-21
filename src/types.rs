use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCentroids {
    pub table: String,
    pub h1_means_column: String,
    pub h1_children_column: Option<String>,
    pub h2_mean_column: Option<String>,
    pub h2_children_column: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct RabbitholeIndexingOptions {
    #[serde(default = "RabbitholeIndexingOptions::default_nlist")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub nlist: u32,
    #[serde(default = "RabbitholeIndexingOptions::default_spherical_centroids")]
    pub spherical_centroids: bool,
    #[serde(default = "RabbitholeIndexingOptions::default_residual_quantization")]
    pub residual_quantization: bool,
    #[serde(default = "RabbitholeIndexingOptions::default_build_threads")]
    #[validate(range(min = 1, max = 255))]
    pub build_threads: u16,
    pub external_centroids: Option<ExternalCentroids>,
}

impl RabbitholeIndexingOptions {
    fn default_nlist() -> u32 {
        1000
    }
    fn default_spherical_centroids() -> bool {
        false
    }
    fn default_residual_quantization() -> bool {
        false
    }
    fn default_build_threads() -> u16 {
        1
    }
}

impl Default for RabbitholeIndexingOptions {
    fn default() -> Self {
        Self {
            nlist: Self::default_nlist(),
            spherical_centroids: Self::default_spherical_centroids(),
            residual_quantization: Self::default_residual_quantization(),
            build_threads: Self::default_build_threads(),
            external_centroids: None,
        }
    }
}
