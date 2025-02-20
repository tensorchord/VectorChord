pub mod am;
pub mod functions;
pub mod gucs;
pub mod opclass;
pub mod projection;
pub mod storage;
pub mod types;

pub fn init() {
    am::init();
    gucs::init();
    for x in gucs::prewarm_dim() {
        projection::prewarm(x as _);
    }
}
