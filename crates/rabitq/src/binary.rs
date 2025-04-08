use simd::Floating;

pub type BinaryLut = (
    (f32, f32, f32, f32),
    (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>),
);
pub type BinaryCode<'a> = (f32, f32, f32, f32, &'a [u64]);

pub fn preprocess(vector: &[f32]) -> BinaryLut {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = simd::quantize::quantize(vector, 15.0);
    let qvector_sum = if vector.len() <= 4369 {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x(&qvector) as f32
    };
    ((dis_v_2, b, k, qvector_sum), binarize(&qvector))
}

pub fn process_l2(
    lut: &BinaryLut,
    (dis_u_2, factor_ppc, factor_ip, factor_err, t): BinaryCode<'_>,
) -> (f32, f32) {
    let &((dis_v_2, b, k, qvector_sum), ref s) = lut;
    let value = asymmetric_binary_dot_product(t, s) as u16;
    let rough =
        dis_u_2 + dis_v_2 + b * factor_ppc + ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
    let err = factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub fn process_dot(
    lut: &BinaryLut,
    (_, factor_ppc, factor_ip, factor_err, t): BinaryCode<'_>,
) -> (f32, f32) {
    let &((dis_v_2, b, k, qvector_sum), ref s) = lut;
    let value = asymmetric_binary_dot_product(t, s) as u16;
    let rough = 0.5 * b * factor_ppc + 0.5 * ((2.0 * value as f32) - qvector_sum) * factor_ip * k;
    let err = 0.5 * factor_err * dis_v_2.sqrt();
    (rough, err)
}

pub(crate) fn binarize(vector: &[u8]) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
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

pub(crate) fn asymmetric_binary_dot_product(
    x: &[u64],
    y: &(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>),
) -> u32 {
    let t0 = simd::bit::sum_of_and(x, &y.0);
    let t1 = simd::bit::sum_of_and(x, &y.1);
    let t2 = simd::bit::sum_of_and(x, &y.2);
    let t3 = simd::bit::sum_of_and(x, &y.3);
    (t0 << 0) + (t1 << 1) + (t2 << 2) + (t3 << 3)
}
