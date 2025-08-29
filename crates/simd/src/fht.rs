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

#[inline(always)]
fn basic_1<const Q: usize>(x: &mut [f32]) {
    assert!(x.len() == (1 << (8 + 1)));
    for i in 0..1 << (8 - Q) {
        basic_2::<Q>(&mut x[i << (Q + 1)..][..1 << (Q + 1)]);
    }
}

#[inline(always)]
fn basic_2<const Q: usize>(x: &mut [f32]) {
    assert!(x.len() == (1 << (Q + 1)));
    for j in 0..1 << Q {
        let (l, r) = (x[j], x[j + (1 << Q)]);
        (x[j], x[j + (1 << Q)]) = (l + r, l - r);
    }
}

mod step_1 {
    seq_macro::seq!(
        Q in 0..16 {
            mod dispatch_~Q {
                #[crate::multiversion("v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
                pub fn f(x: &mut [f32]) {
                    crate::fht::basic_1::<Q>(x);
                }
            }
            #[allow(unused_imports)]
            pub use dispatch_~Q::f as dispatch_~Q;
        }
    );
}

mod step_2 {
    seq_macro::seq!(
        Q in 0..16 {
            mod dispatch_~Q {
                #[crate::multiversion("v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
                pub fn f(x: &mut [f32]) {
                    crate::fht::basic_2::<Q>(x);
                }
            }
            #[allow(unused_imports)]
            pub use dispatch_~Q::f as dispatch_~Q;
        }
    );
}

macro_rules! fht {
    ($p:literal, 0) => {
        {
            #[crate::multiversion("v4", "v3", "v2", "a2", "z17", "z16", "z15", "z14", "z13", "p9", "p8", "p7", "r1")]
            fn walk(x: &mut [f32]) {
                assert!(x.len() == (1 << $p));
                seq_macro::seq!(
                    Q in 0..$p {
                        for i in 0..1 << ($p - Q - 1) {
                            basic_2::<Q>(&mut x[i << (Q + 1)..][..1 << (Q + 1)]);
                        }
                    }
                );
            }
            walk as fn(&mut [f32])
        }
    };
    ($p:literal, 1) => {
        {
            fn walk(x: &mut [f32]) {
                assert!(x.len() == (1 << $p));
                seq_macro::seq!(
                    Q in 0..8 {
                        for i in (0..(1 << ($p - Q - 1))).step_by(1 << (8 - Q)) {
                            step_1::dispatch_~Q(&mut x[i << (Q + 1)..][..1 << (8 + 1)]);
                        }
                    }
                );
                seq_macro::seq!(
                    Q in 8..$p {
                        for i in 0..1 << ($p - Q - 1) {
                            step_2::dispatch_~Q(&mut x[i << (Q + 1)..][..1 << (Q + 1)]);
                        }
                    }
                );
            }
            walk as fn(&mut [f32])
        }
    };
}

pub fn fht(x: &mut [f32]) {
    const FHT: [fn(&mut [f32]); 1 + 16] = [
        |_| (),
        fht!(1, 0),
        fht!(2, 0),
        fht!(3, 0),
        fht!(4, 0),
        fht!(5, 0),
        fht!(6, 0),
        fht!(7, 0),
        fht!(8, 0),
        fht!(9, 1),
        fht!(10, 1),
        fht!(11, 1),
        fht!(12, 1),
        fht!(13, 1),
        fht!(14, 1),
        fht!(15, 1),
        fht!(16, 1),
    ];
    let n = x.len();
    let Some(i) = n.checked_ilog2() else {
        panic!("the dimension of the vector is 0")
    };
    if n != (1 << i) {
        panic!("the dimension of the vector is not a power of 2");
    }
    if i > 16 {
        panic!("the dimension of the vector is too large");
    }
    FHT[i as usize](x)
}

#[cfg(test)]
mod tests {
    fn native(x: &mut [f32]) {
        let n = x.len();
        assert!(n.is_power_of_two());
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(h * 2) {
                for j in i..i + h {
                    (x[j], x[j + h]) = (x[j] + x[j + h], x[j] - x[j + h]);
                }
            }
            h *= 2;
        }
    }

    #[test]
    fn fht() {
        use rand::Rng;
        use std::iter::zip;
        const EPSILON: f32 = 1e-6;
        let mut rng = rand::rng();
        let mut n = 1_usize;
        while n <= 65536 {
            let x = (0..n)
                .map(|_| rng.random_range(-1.0_f32..=1.0_f32))
                .collect::<Vec<_>>();
            let x_expected = {
                let mut x = x.clone();
                native(x.as_mut_slice());
                x
            };
            let x_got = {
                let mut x = x.clone();
                crate::fht::fht(x.as_mut_slice());
                x
            };
            let mse = zip(x_expected, x_got)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                / n as f32;
            eprintln!("n = {n}, mse = {mse:.12}");
            assert!(mse <= EPSILON);
            n *= 2;
        }
    }
}
