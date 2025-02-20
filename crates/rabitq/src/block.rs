use distance::Distance;
use simd::Floating;

pub fn preprocess(vector: &[f32]) -> (f32, f32, f32, f32, Vec<[u8; 16]>) {
    let dis_v_2 = f32::reduce_sum_of_x2(vector);
    let (k, b, qvector) = simd::quantize::quantize(vector, 15.0);
    let qvector_sum = if vector.len() <= 4369 {
        simd::u8::reduce_sum_of_x_as_u16(&qvector) as f32
    } else {
        simd::u8::reduce_sum_of_x(&qvector) as f32
    };
    (dis_v_2, b, k, qvector_sum, compress(qvector))
}

pub fn process_lowerbound_l2(
    lut: &(f32, f32, f32, f32, Vec<[u8; 16]>),
    (dis_u_2, factor_ppc, factor_ip, factor_err, t): (
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[[u8; 16]],
    ),
    epsilon: f32,
) -> [Distance; 32] {
    let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
    let r = simd::fast_scan::fast_scan(t, s);
    std::array::from_fn(|i| {
        let rough = dis_u_2[i]
            + dis_v_2
            + b * factor_ppc[i]
            + ((2.0 * r[i] as f32) - qvector_sum) * factor_ip[i] * k;
        let err = factor_err[i] * dis_v_2.sqrt();
        Distance::from_f32(rough - epsilon * err)
    })
}

pub fn process_lowerbound_dot(
    lut: &(f32, f32, f32, f32, Vec<[u8; 16]>),
    (_, factor_ppc, factor_ip, factor_err, t): (
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[f32; 32],
        &[[u8; 16]],
    ),
    epsilon: f32,
) -> [Distance; 32] {
    let &(dis_v_2, b, k, qvector_sum, ref s) = lut;
    let r = simd::fast_scan::fast_scan(t, s);
    std::array::from_fn(|i| {
        let rough =
            0.5 * b * factor_ppc[i] + 0.5 * ((2.0 * r[i] as f32) - qvector_sum) * factor_ip[i] * k;
        let err = 0.5 * factor_err[i] * dis_v_2.sqrt();
        Distance::from_f32(rough - epsilon * err)
    })
}

pub fn compress(mut vector: Vec<u8>) -> Vec<[u8; 16]> {
    let n = vector.len().div_ceil(4);
    vector.resize(n * 4, 0);
    let mut result = vec![[0u8; 16]; n];
    for i in 0..n {
        #[allow(unsafe_code)]
        unsafe {
            // this hint is used to skip bound checks
            std::hint::assert_unchecked(4 * i + 3 < vector.len());
        }
        let t_0 = vector[4 * i + 0];
        let t_1 = vector[4 * i + 1];
        let t_2 = vector[4 * i + 2];
        let t_3 = vector[4 * i + 3];
        result[i] = [
            0,
            t_0,
            t_1,
            t_1 + t_0,
            t_2,
            t_2 + t_0,
            t_2 + t_1,
            t_2 + t_1 + t_0,
            t_3,
            t_3 + t_0,
            t_3 + t_1,
            t_3 + t_1 + t_0,
            t_3 + t_2,
            t_3 + t_2 + t_0,
            t_3 + t_2 + t_1,
            t_3 + t_2 + t_1 + t_0,
        ];
    }
    result
}
