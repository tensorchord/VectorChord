use crate::vchordrq::types::DistanceKind;
use distance::Distance;

pub use rabitq::binary::Code;
pub use rabitq::binary::Lut;
pub use rabitq::binary::code;
pub use rabitq::binary::preprocess;
pub use rabitq::binary::{process_lowerbound_dot, process_lowerbound_l2};

pub fn process_lowerbound(
    distance_kind: DistanceKind,
    dims: u32,
    lut: &Lut,
    code: (f32, f32, f32, f32, &[u64]),
    epsilon: f32,
) -> Distance {
    match distance_kind {
        DistanceKind::L2 => process_lowerbound_l2(dims, lut, code, epsilon),
        DistanceKind::Dot => process_lowerbound_dot(dims, lut, code, epsilon),
    }
}
