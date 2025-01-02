#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Aligned16<T>(pub T);
