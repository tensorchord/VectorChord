use crate::algorithm::rabitq;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct MetaTuple {
    pub dims: u32,
    pub is_residual: bool,
    pub vectors_first: u32,
    pub forwards_first: u32,
    // raw vector
    pub mean: (u32, u16),
    // for meta tuple, it's pointers to next level
    pub first: u32,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct VectorTuple {
    pub vector: Vec<f32>,
    // this field is saved only for vacuum
    pub payload: Option<u64>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height2Tuple {
    pub mask: [bool; 32],
    // raw vector
    pub mean: [(u32, u16); 32],
    // for height 2 tuple, it's pointers to next level
    pub first: [u32; 32],
    // RaBitQ algorithm
    pub dis_u_2: [f32; 32],
    pub factor_ppc: [f32; 32],
    pub factor_ip: [f32; 32],
    pub factor_err: [f32; 32],
    pub t: Vec<u8>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height1Tuple {
    pub mask: [bool; 32],
    // raw vector
    pub mean: [(u32, u16); 32],
    // for height 1 tuple, it's pointers to next level
    pub first: [u32; 32],
    // RaBitQ algorithm
    pub dis_u_2: [f32; 32],
    pub factor_ppc: [f32; 32],
    pub factor_ip: [f32; 32],
    pub factor_err: [f32; 32],
    pub t: Vec<u8>,
}

#[derive(Clone, PartialEq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Height0Tuple {
    pub mask: [bool; 32],
    // raw vector
    pub mean: [(u32, u16); 32],
    // for height 0 tuple, it's pointers to heap relation
    pub payload: [u64; 32],
    // RaBitQ algorithm
    pub dis_u_2: [f32; 32],
    pub factor_ppc: [f32; 32],
    pub factor_ip: [f32; 32],
    pub factor_err: [f32; 32],
    pub t: Vec<u8>,
}

pub fn put(
    bytes: &mut [u8],
    dims: u32,
    code: &rabitq::Code,
    vector: (u32, u16),
    payload: u64,
) -> bool {
    // todo: use mutable api
    let mut x = rkyv::from_bytes::<Height0Tuple>(bytes).expect("data corruption");
    for j in 0..32 {
        if !x.mask[j] {
            x.mean[j] = vector;
            x.payload[j] = payload;
            x.mask[j] = true;
            x.dis_u_2[j] = code.dis_u_2;
            x.factor_ppc[j] = code.factor_ppc;
            x.factor_ip[j] = code.factor_ip;
            x.factor_err[j] = code.factor_err;
            let width = dims.div_ceil(4) as usize;
            let table = [
                (0, 0),
                (2, 0),
                (4, 0),
                (6, 0),
                (8, 0),
                (10, 0),
                (12, 0),
                (14, 0),
                (1, 0),
                (3, 0),
                (5, 0),
                (7, 0),
                (9, 0),
                (11, 0),
                (13, 0),
                (15, 0),
                (0, 1),
                (2, 1),
                (4, 1),
                (6, 1),
                (8, 1),
                (10, 1),
                (12, 1),
                (14, 1),
                (1, 1),
                (3, 1),
                (5, 1),
                (7, 1),
                (9, 1),
                (11, 1),
                (13, 1),
                (15, 1),
            ];
            let pos = table[j].0;
            let mask = match table[j].1 {
                0 => 0xf0,
                1 => 0x0f,
                _ => unreachable!(),
            };
            let shift = match table[j].1 {
                0 => 0,
                1 => 4,
                _ => unreachable!(),
            };
            let mut buffer = vec![0u8; width];
            for j in 0..width {
                let b0 = code.signs.get(4 * j + 0).copied().unwrap_or_default();
                let b1 = code.signs.get(4 * j + 1).copied().unwrap_or_default();
                let b2 = code.signs.get(4 * j + 2).copied().unwrap_or_default();
                let b3 = code.signs.get(4 * j + 3).copied().unwrap_or_default();
                buffer[j] = b0 | b1 << 1 | b2 << 2 | b3 << 3;
            }
            for j in 0..width {
                x.t[16 * j + pos] &= mask;
                x.t[16 * j + pos] |= buffer[j] << shift;
            }
            bytes.copy_from_slice(&rkyv::to_bytes::<_, 8192>(&x).unwrap());
            return true;
        }
    }
    false
}
