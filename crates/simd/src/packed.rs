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

pub fn u2_u2_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
    u2_u2_reduce_sum_of_xy::u2_u2_reduce_sum_of_xy(s, t)
}

mod u2_u2_reduce_sum_of_xy {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn u2_u2_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let mut result_0 = 0_u32;
        let mut result_1 = 0_u32;
        let mut result_2 = 0_u32;
        let mut result_3 = 0_u32;
        for i in 0..n {
            let (s, t) = (s[i], t[i]);
            result_0 += (((s >> 0) & 3) * ((t >> 0) & 3)) as u32;
            result_1 += (((s >> 2) & 3) * ((t >> 2) & 3)) as u32;
            result_2 += (((s >> 4) & 3) * ((t >> 4) & 3)) as u32;
            result_3 += (((s >> 6) & 3) * ((t >> 6) & 3)) as u32;
        }
        result_0 + result_1 + result_2 + result_3
    }
}

pub fn u4_u4_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
    u4_u4_reduce_sum_of_xy::u4_u4_reduce_sum_of_xy(s, t)
}

mod u4_u4_reduce_sum_of_xy {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn u4_u4_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len());
        let n = s.len();
        let mut result_0 = 0;
        let mut result_1 = 0;
        for i in 0..n {
            let (s, t) = (s[i], t[i]);
            result_0 += (((s >> 0) & 15) * ((t >> 0) & 15)) as u32;
            result_1 += (((s >> 4) & 15) * ((t >> 4) & 15)) as u32;
        }
        result_0 + result_1
    }
}

pub fn u2_u8_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
    u2_u8_reduce_sum_of_xy::u2_u8_reduce_sum_of_xy(s, t)
}

mod u2_u8_reduce_sum_of_xy {
    #[crate::multiversion("v4", "v3", "v2", "a2")]
    pub fn u2_u8_reduce_sum_of_xy(s: &[u8], t: &[u8]) -> u32 {
        assert_eq!(s.len(), t.len().div_ceil(4));
        let mut result_0 = 0_u32;
        let mut result_1 = 0_u32;
        let mut result_2 = 0_u32;
        let mut result_3 = 0_u32;
        let (arrays, remainder) = t.as_chunks::<4>();
        assert_eq!(s.len(), arrays.len());
        let n = arrays.len();
        for i in 0..n {
            let (s, t) = (s[i], arrays[i]);
            result_0 += ((s >> 0) & 3) as u32 * t[0] as u32;
            result_1 += ((s >> 2) & 3) as u32 * t[1] as u32;
            result_2 += ((s >> 4) & 3) as u32 * t[2] as u32;
            result_3 += ((s >> 6) & 3) as u32 * t[3] as u32;
        }
        let mut buffer = [0u8; 4];
        if !remainder.is_empty() {
            buffer[..remainder.len()].copy_from_slice(remainder);
            let (s, t) = (s[n], buffer);
            result_0 += ((s >> 0) & 3) as u32 * t[0] as u32;
            result_1 += ((s >> 2) & 3) as u32 * t[1] as u32;
            result_2 += ((s >> 4) & 3) as u32 * t[2] as u32;
            result_3 += ((s >> 6) & 3) as u32 * t[3] as u32;
        }
        result_0 + result_1 + result_2 + result_3
    }
}
