use crate::vchordrqfscan::types::DistanceKind;
use distance::Distance;

pub use rabitq::block::Code;
pub use rabitq::block::code;
pub use rabitq::block::dummy_code;
pub use rabitq::block::fscan_preprocess;
pub use rabitq::block::pack_codes;
pub use rabitq::block::{fscan_process_lowerbound_dot, fscan_process_lowerbound_l2};

pub fn fscan_process_lowerbound(
    distance_kind: DistanceKind,
    dims: u32,
    lut: &(f32, f32, f32, f32, Vec<u8>),
    code: (&[f32; 32], &[f32; 32], &[f32; 32], &[f32; 32], &[u8]),
    epsilon: f32,
) -> [Distance; 32] {
    match distance_kind {
        DistanceKind::L2 => fscan_process_lowerbound_l2(dims, lut, code, epsilon),
        DistanceKind::Dot => fscan_process_lowerbound_dot(dims, lut, code, epsilon),
    }
}
