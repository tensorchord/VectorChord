use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct MetaTuple {
    pub dims: u32,
    pub height_of_root: u32,
    pub is_residual: bool,
    pub vectors_first: u32,
    // raw vector
    pub mean: (u32, u16),
    // for meta tuple, it's pointers to next level
    pub first: u32,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct VectorTuple {
    pub slice: Vec<f32>,
    pub payload: Option<u64>,
    pub chain: Option<(u32, u16)>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height1Tuple {
    // raw vector
    pub mean: (u32, u16),
    // for height 1 tuple, it's pointers to next level
    pub first: u32,
    // RaBitQ algorithm
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub t: Vec<u64>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height0Tuple {
    // raw vector
    pub mean: (u32, u16),
    // for height 0 tuple, it's pointers to heap relation
    pub payload: u64,
    // RaBitQ algorithm
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub t: Vec<u64>,
}
