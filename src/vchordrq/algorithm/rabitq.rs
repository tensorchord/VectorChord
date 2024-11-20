use base::distance::{Distance, DistanceKind};
use base::scalar::ScalarLike;

#[derive(Debug, Clone)]
pub struct Code {
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub signs: Vec<u8>,
}

impl Code {
    pub fn t(&self) -> Vec<u64> {
        use quantization::utils::InfiniteByteChunks;
        let mut result = Vec::new();
        for x in InfiniteByteChunks::<_, 64>::new(self.signs.iter().copied())
            .take(self.signs.len().div_ceil(64))
        {
            let mut r = 0_u64;
            for i in 0..64 {
                r |= (x[i] as u64) << i;
            }
            result.push(r);
        }
        result
    }
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

pub type Lut = (f32, f32, f32, f32, (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>));

pub fn fscan_preprocess(vector: &[f32]) -> Lut {
    use quantization::quantize;
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = quantize::quantize::<15>(vector);
    let qvector_sum = if vector.len() <= 4369 {
        quantize::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        quantize::reduce_sum_of_x_as_u32(&qvector) as f32
    };
    (dis_v_2, b, k, qvector_sum, binarize(&qvector))
}

pub fn fscan_process_lowerbound(
    distance_kind: DistanceKind,
    _dims: u32,
    lut: &Lut,
    (dis_u_2, factor_ppc, factor_ip, factor_err, t): (f32, f32, f32, f32, &[u64]),
    epsilon: f32,
) -> Distance {
    match distance_kind {
        DistanceKind::L2 => {
            let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
            let value = asymmetric_binary_dot_product(t, s) as u16;
            let rough = dis_u_2
                + dis_v_2
                + b * factor_ppc
                + ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
            let err = factor_err * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        }
        DistanceKind::Dot => {
            let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
            let value = asymmetric_binary_dot_product(t, s) as u16;
            let rough =
                0.5 * b * factor_ppc + 0.5 * ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
            let err = 0.5 * factor_err * dis_v_2.sqrt();
            Distance::from_f32(rough - epsilon * err)
        }
        DistanceKind::Hamming => unimplemented!(),
        DistanceKind::Jaccard => unimplemented!(),
    }
}

fn binarize(vector: &[u8]) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let n = vector.len();
    let mut t0 = vec![0u64; n.div_ceil(64)];
    let mut t1 = vec![0u64; n.div_ceil(64)];
    let mut t2 = vec![0u64; n.div_ceil(64)];
    let mut t3 = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        t0[i / 64] |= (((vector[i] >> 0) & 1) as u64) << (i % 64);
        t1[i / 64] |= (((vector[i] >> 1) & 1) as u64) << (i % 64);
        t2[i / 64] |= (((vector[i] >> 2) & 1) as u64) << (i % 64);
        t3[i / 64] |= (((vector[i] >> 3) & 1) as u64) << (i % 64);
    }
    (t0, t1, t2, t3)
}

#[detect::multiversion(v2, fallback)]
fn asymmetric_binary_dot_product(x: &[u64], y: &(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)) -> u32 {
    assert_eq!(x.len(), y.0.len());
    assert_eq!(x.len(), y.1.len());
    assert_eq!(x.len(), y.2.len());
    assert_eq!(x.len(), y.3.len());
    let n = x.len();
    let (mut t0, mut t1, mut t2, mut t3) = (0, 0, 0, 0);
    for i in 0..n {
        t0 += (x[i] & y.0[i]).count_ones();
    }
    for i in 0..n {
        t1 += (x[i] & y.1[i]).count_ones();
    }
    for i in 0..n {
        t2 += (x[i] & y.2[i]).count_ones();
    }
    for i in 0..n {
        t3 += (x[i] & y.3[i]).count_ones();
    }
    (t0 << 0) + (t1 << 1) + (t2 << 2) + (t3 << 3)
}
