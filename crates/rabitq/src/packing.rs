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

pub fn pack(x: [&[u8]; 32]) -> Vec<[u8; 16]> {
    let n = {
        let l = x.each_ref().map(|i| i.len());
        for i in 1..32 {
            assert!(l[0] == l[i]);
        }
        l[0]
    };
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push([
            x[0][i] | (x[16][i] << 4),
            x[8][i] | (x[24][i] << 4),
            x[1][i] | (x[17][i] << 4),
            x[9][i] | (x[25][i] << 4),
            x[2][i] | (x[18][i] << 4),
            x[10][i] | (x[26][i] << 4),
            x[3][i] | (x[19][i] << 4),
            x[11][i] | (x[27][i] << 4),
            x[4][i] | (x[20][i] << 4),
            x[12][i] | (x[28][i] << 4),
            x[5][i] | (x[21][i] << 4),
            x[13][i] | (x[29][i] << 4),
            x[6][i] | (x[22][i] << 4),
            x[14][i] | (x[30][i] << 4),
            x[7][i] | (x[23][i] << 4),
            x[15][i] | (x[31][i] << 4),
        ]);
    }
    result
}

pub fn unpack(x: &[[u8; 16]]) -> [Vec<u8>; 32] {
    let n = x.len();
    let mut result = std::array::from_fn(|_| Vec::with_capacity(n));
    for i in 0..n {
        result[0].push(x[i][0] & 0xf);
        result[1].push(x[i][2] & 0xf);
        result[2].push(x[i][4] & 0xf);
        result[3].push(x[i][6] & 0xf);
        result[4].push(x[i][8] & 0xf);
        result[5].push(x[i][10] & 0xf);
        result[6].push(x[i][12] & 0xf);
        result[7].push(x[i][14] & 0xf);
        result[8].push(x[i][1] & 0xf);
        result[9].push(x[i][3] & 0xf);
        result[10].push(x[i][5] & 0xf);
        result[11].push(x[i][7] & 0xf);
        result[12].push(x[i][9] & 0xf);
        result[13].push(x[i][11] & 0xf);
        result[14].push(x[i][13] & 0xf);
        result[15].push(x[i][15] & 0xf);
        result[16].push(x[i][0] >> 4);
        result[17].push(x[i][2] >> 4);
        result[18].push(x[i][4] >> 4);
        result[19].push(x[i][6] >> 4);
        result[20].push(x[i][8] >> 4);
        result[21].push(x[i][10] >> 4);
        result[22].push(x[i][12] >> 4);
        result[23].push(x[i][14] >> 4);
        result[24].push(x[i][1] >> 4);
        result[25].push(x[i][3] >> 4);
        result[26].push(x[i][5] >> 4);
        result[27].push(x[i][7] >> 4);
        result[28].push(x[i][9] >> 4);
        result[29].push(x[i][11] >> 4);
        result[30].push(x[i][13] >> 4);
        result[31].push(x[i][15] >> 4);
    }
    result
}

pub fn padding_pack(x: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Vec<[u8; 16]> {
    let x = x.into_iter().collect::<Vec<_>>();
    let x = x.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    if x.is_empty() || x.len() > 32 {
        panic!("too few or too many slices");
    }
    let n = x[0].len();
    let t = vec![0; n];
    pack(std::array::from_fn(|i| {
        if i < x.len() { x[i] } else { t.as_slice() }
    }))
}

pub fn any_pack<T: Default>(mut x: impl Iterator<Item = T>) -> [T; 32] {
    std::array::from_fn(|_| x.next()).map(|x| x.unwrap_or_default())
}
