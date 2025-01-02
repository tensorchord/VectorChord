use distance::Distance;
use simd::Floating;

#[derive(Debug, Clone)]
pub struct Code {
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<u8>,
}

pub fn code(dims: u32, vector: &[f32]) -> Code {
    let sum_of_abs_x = f32::reduce_sum_of_abs_x(vector);
    let sum_of_x_2 = f32::reduce_sum_of_x2(vector);
    let dis_u = sum_of_x_2.sqrt();
    let x0 = sum_of_abs_x / (sum_of_x_2 * (dims as f32)).sqrt();
    let x_x0 = dis_u / x0;
    let fac_norm = (dims as f32).sqrt();
    let max_x1 = 1.0f32 / (dims as f32 - 1.0).sqrt();
    let factor_err = 2.0f32 * max_x1 * (x_x0 * x_x0 - dis_u * dis_u).sqrt();
    let factor_ip = -2.0f32 / fac_norm * x_x0;
    let cnt_pos = vector
        .iter()
        .map(|x| x.is_sign_positive() as i32)
        .sum::<i32>();
    let cnt_neg = vector
        .iter()
        .map(|x| x.is_sign_negative() as i32)
        .sum::<i32>();
    let factor_ppc = factor_ip * (cnt_pos - cnt_neg) as f32;
    let mut signs = Vec::new();
    for i in 0..dims {
        signs.push(vector[i as usize].is_sign_positive() as u8);
    }
    Code {
        dis_u_2: sum_of_x_2,
        factor_ppc,
        factor_ip,
        factor_err,
        signs,
    }
}

pub fn dummy_code(dims: u32) -> Code {
    Code {
        dis_u_2: 0.0,
        factor_ppc: 0.0,
        factor_ip: 0.0,
        factor_err: 0.0,
        signs: vec![0; dims as _],
    }
}

pub struct PackedCodes {
    pub dis_u_2: [f32; 32],
    pub factor_ppc: [f32; 32],
    pub factor_ip: [f32; 32],
    pub factor_err: [f32; 32],
    pub t: Vec<u8>,
}

pub fn pack_codes(dims: u32, codes: [Code; 32]) -> PackedCodes {
    use crate::utils::InfiniteByteChunks;
    PackedCodes {
        dis_u_2: std::array::from_fn(|i| codes[i].dis_u_2),
        factor_ppc: std::array::from_fn(|i| codes[i].factor_ppc),
        factor_ip: std::array::from_fn(|i| codes[i].factor_ip),
        factor_err: std::array::from_fn(|i| codes[i].factor_err),
        t: {
            let signs = codes.map(|code| {
                InfiniteByteChunks::new(code.signs.into_iter())
                    .map(|[b0, b1, b2, b3]| b0 | b1 << 1 | b2 << 2 | b3 << 3)
                    .take(dims.div_ceil(4) as usize)
                    .collect::<Vec<_>>()
            });
            simd::fast_scan::pack(dims.div_ceil(4), signs).collect()
        },
    }
}

pub fn fscan_preprocess(vector: &[f32]) -> (f32, f32, f32, f32, Vec<u8>) {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = simd::quantize::quantize(vector, 15.0);
    let qvector_sum = if vector.len() <= 4369 {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x(&qvector) as f32
    };
    (dis_v_2, b, k, qvector_sum, compress(qvector))
}

pub fn fscan_process_lowerbound_l2(
    dims: u32,
    lut: &(f32, f32, f32, f32, Vec<u8>),
    (dis_u_2, factor_ppc, factor_ip, factor_err, t): (
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[u8],
    ),
    epsilon: f32,
) -> [Distance; 32] {
    let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
    let r = simd::fast_scan::fast_scan(dims.div_ceil(4), t, s);
    std::array::from_fn(|i| {
        let rough = dis_u_2[i]
            + dis_v_2
            + b * factor_ppc[i]
            + ((2.0 * r[i] as f32) - qvector_sum) * factor_ip[i] * k;
        let err = factor_err[i] * dis_v_2.sqrt();
        Distance::from_f32(rough - epsilon * err)
    })
}

pub fn fscan_process_lowerbound_dot(
    dims: u32,
    lut: &(f32, f32, f32, f32, Vec<u8>),
    (_, factor_ppc, factor_ip, factor_err, t): (
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[u8],
    ),
    epsilon: f32,
) -> [Distance; 32] {
    let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
    let r = simd::fast_scan::fast_scan(dims.div_ceil(4), t, s);
    std::array::from_fn(|i| {
        let rough =
            0.5 * b * factor_ppc[i] + 0.5 * ((2.0 * r[i] as f32) - qvector_sum) * factor_ip[i] * k;
        let err = 0.5 * factor_err[i] * dis_v_2.sqrt();
        Distance::from_f32(rough - epsilon * err)
    })
}

fn compress(mut qvector: Vec<u8>) -> Vec<u8> {
    let dims = qvector.len() as u32;
    let width = dims.div_ceil(4);
    qvector.resize(qvector.len().next_multiple_of(4), 0);
    let mut t = vec![0u8; width as usize * 16];
    for i in 0..width as usize {
        unsafe {
            // this hint is used to skip bound checks
            std::hint::assert_unchecked(4 * i + 3 < qvector.len());
            std::hint::assert_unchecked(16 * i + 15 < t.len());
        }
        let t0 = qvector[4 * i + 0];
        let t1 = qvector[4 * i + 1];
        let t2 = qvector[4 * i + 2];
        let t3 = qvector[4 * i + 3];
        t[16 * i + 0b0000] = 0;
        t[16 * i + 0b0001] = t0;
        t[16 * i + 0b0010] = t1;
        t[16 * i + 0b0011] = t1 + t0;
        t[16 * i + 0b0100] = t2;
        t[16 * i + 0b0101] = t2 + t0;
        t[16 * i + 0b0110] = t2 + t1;
        t[16 * i + 0b0111] = t2 + t1 + t0;
        t[16 * i + 0b1000] = t3;
        t[16 * i + 0b1001] = t3 + t0;
        t[16 * i + 0b1010] = t3 + t1;
        t[16 * i + 0b1011] = t3 + t1 + t0;
        t[16 * i + 0b1100] = t3 + t2;
        t[16 * i + 0b1101] = t3 + t2 + t0;
        t[16 * i + 0b1110] = t3 + t2 + t1;
        t[16 * i + 0b1111] = t3 + t2 + t1 + t0;
    }
    t
}
