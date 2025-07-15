// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use simd::Floating;

#[derive(Debug, Clone, Copy)]
pub struct CodeMetadata {
    pub dis_u_2: f32,
    pub norm_of_lattice: f32,
    pub sum_of_code: f32,
}

impl CodeMetadata {
    pub fn into_tuple(self) -> (f32, f32, f32) {
        (self.dis_u_2, self.norm_of_lattice, self.sum_of_code)
    }
    pub fn into_array(self) -> [f32; 3] {
        [self.dis_u_2, self.norm_of_lattice, self.sum_of_code]
    }
    pub fn from_tuple((dis_u_2, norm_of_lattice, sum_of_code): (f32, f32, f32)) -> Self {
        Self {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        }
    }
    pub fn from_array([dis_u_2, norm_of_lattice, sum_of_code]: [f32; 3]) -> Self {
        Self {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        }
    }
}

pub type Code = (CodeMetadata, Vec<u8>);

pub fn code<const BITS: usize>(vector: &[f32]) -> Code {
    assert!((1..=8).contains(&BITS));

    let n = vector.len();
    let dis_u_2 = f32::reduce_sum_of_x2(vector);
    let code = {
        let min = -(1 << (BITS - 1));
        let max = (1 << (BITS - 1)) - 1;
        let normalized_vector = {
            let mut vector = vector.to_vec();
            f32::vector_mul_scalar_inplace(&mut vector, 1.0 / dis_u_2.sqrt());
            vector
        };
        let scale = {
            let mut o = normalized_vector.clone();
            f32::vector_abs_inplace(&mut o);
            find_scale::<BITS>(&o) as f32 + f32::EPSILON
        };
        let mut code = Vec::with_capacity(n as _);
        for i in 0..n {
            let v = scale * normalized_vector[i];
            let c = v.floor().clamp(min as f32, max as f32) as i32;
            code.push((c + (1 << (BITS - 1))) as _);
        }
        code
    };
    let norm_of_lattice = {
        let base = -0.5 * ((1 << BITS) - 1) as f32;
        let mut y = 0.0;
        for i in 0..n {
            let x = base + code[i] as f32;
            y += x * x;
        }
        y.sqrt()
    };
    let sum_of_code = {
        let mut y = 0;
        for i in 0..n {
            let x = code[i] as u32;
            y += x;
        }
        y as f32
    };
    (
        CodeMetadata {
            dis_u_2,
            norm_of_lattice,
            sum_of_code,
        },
        code,
    )
}

pub fn half_process_l2<const X: usize, const Y: usize>(
    n: u32,
    value: u32,
    lhs: CodeMetadata,
    rhs: CodeMetadata,
) -> f32 {
    assert!((1..=8).contains(&X));
    assert!((1..=8).contains(&Y));

    let c_x = ((1 << X) - 1) as f32 * 0.5;
    let c_y = ((1 << Y) - 1) as f32 * 0.5;

    let ip = value as f32 - (c_y * lhs.sum_of_code + c_x * rhs.sum_of_code) + n as f32 * c_x * c_y;
    lhs.dis_u_2 + rhs.dis_u_2
        - 2.0
            * ip
            * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice)
            * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice)
}

pub fn half_process_dot<const X: usize, const Y: usize>(
    n: u32,
    value: u32,
    lhs: CodeMetadata,
    rhs: CodeMetadata,
) -> f32 {
    assert!((1..=8).contains(&X));
    assert!((1..=8).contains(&Y));

    let c_x = ((1 << X) - 1) as f32 * 0.5;
    let c_y = ((1 << Y) - 1) as f32 * 0.5;

    let ip = value as f32 - (c_y * lhs.sum_of_code + c_x * rhs.sum_of_code) + n as f32 * c_x * c_y;
    -ip * (lhs.dis_u_2.sqrt() / lhs.norm_of_lattice) * (rhs.dis_u_2.sqrt() / rhs.norm_of_lattice)
}

fn find_scale<const B: usize>(o: &[f32]) -> f64 {
    assert!((1..=8).contains(&B));

    let mask = (1_u32 << (B - 1)) - 1;
    let dims = o.len();

    let mut code = Vec::<u8>::with_capacity(dims);
    let mut numerator = 0.0f64;
    let mut sqr_denominator = 0.0f64;

    let (mut y_m, mut x_m);

    for i in 0..dims {
        code.push(0);
        numerator += 0.5 * o[i] as f64;
        sqr_denominator += 0.5 * 0.5;
    }
    {
        let x = 0.0;
        let y = numerator / sqr_denominator.sqrt();
        (y_m, x_m) = (y, x);
    }

    let mut events = Vec::<(f64, usize)>::new();
    for i in 0..dims {
        for c in 1..=mask {
            let x = (c as f64) / o[i] as f64;
            events.push((x, i));
        }
    }
    events.sort_unstable_by(|(x, _), (y, _)| f64::total_cmp(x, y));
    for (x, i) in events.into_iter() {
        code[i] += 1;
        numerator += o[i] as f64;
        sqr_denominator += 2.0 * code[i] as f64;

        let y = numerator / sqr_denominator.sqrt();
        if y > y_m {
            (y_m, x_m) = (y, x);
        }
    }

    x_m
}

pub fn pack_code<const BITS: usize>(input: &[u8]) -> [Vec<u64>; BITS] {
    #[inline(always)]
    fn f(array: &[u8; 64], bit: usize) -> u64 {
        let mut result = 0_u64;
        for i in 0..64 {
            result |= ((array[i] as u64 >> bit) & 1) << i;
        }
        result
    }
    let (arrays, remainder) = input.as_chunks::<64>();
    let mut buffer = [0_u8; 64];
    let tailing = if !remainder.is_empty() {
        buffer[..remainder.len()].copy_from_slice(remainder);
        Some(&buffer)
    } else {
        None
    };
    std::array::from_fn(|bit| arrays.iter().chain(tailing).map(|t| f(t, bit)).collect())
}

pub fn accumulate<const X: usize, const Y: usize>(lhs: &[u64], rhs: &[Vec<u64>; Y]) -> u32 {
    assert!(lhs.len().is_multiple_of(X));
    let d = lhs.len() / X;
    let mut result = 0_u32;
    for i in 0..X {
        for j in 0..Y {
            result += simd::bit::reduce_sum_of_and(&lhs[i * d..][..d], &rhs[j]) << (i + j);
        }
    }
    result
}
