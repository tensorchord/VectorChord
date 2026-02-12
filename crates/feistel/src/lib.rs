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

pub fn feistel<I>(width: u32, x: I, round: u32, secret: impl Fn(u32, I) -> I) -> I
where
    I: Copy,
    I: Ord,
    I: std::ops::Add<Output = I>,
    I: std::ops::Sub<Output = I>,
    I: std::ops::BitAnd<Output = I>,
    I: std::ops::BitOr<Output = I>,
    I: std::ops::BitXor<Output = I>,
    I: std::ops::Shl<u32, Output = I>,
    I: std::ops::Shr<u32, Output = I>,
    I: Zero,
    I: One,
    I: std::fmt::Debug,
{
    assert_eq!(width % 2, 0);
    assert_eq!(x >> width, I::zero());

    let half_width = width >> 1;
    let half_mask = (I::one() << half_width) - I::one();

    let mut left = (x >> half_width) & half_mask;
    let mut right = x & half_mask;

    for i in 0..round {
        (left, right) = (right, left ^ (secret(i, right) & half_mask));
    }

    (left << half_width) | right
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

macro_rules! impl_traits {
    ($t:ty) => {
        impl Zero for $t {
            fn zero() -> Self {
                0
            }
        }

        impl One for $t {
            fn one() -> Self {
                1
            }
        }
    };
}

impl_traits!(u8);
impl_traits!(u16);
impl_traits!(u32);
impl_traits!(u64);
impl_traits!(u128);

// This is a standalone crate, simply because we want to run tests for it.

#[test]
fn is_a_permutation() {
    let key_0 = [7u8; _];
    let key_1 = [8u8; _];
    let secret = move |round: u32, x: u32| {
        let buffer = [round.to_le_bytes(), x.to_le_bytes(), key_0, key_1];
        wyhash::wyhash(buffer.as_flattened(), 0) as u32
    };
    for width in (0..if cfg!(not(miri)) { 20 } else { 10 }).step_by(2) {
        let mut y = Vec::new();
        for x in 0..1 << width {
            y.push(feistel::<u32>(width, x, 8, secret));
        }
        if width <= 8 {
            eprintln!("feistel({width}, _, 8, *) = {y:?}");
        }
        y.sort_unstable();
        for x in 0..1 << width {
            assert_eq!(y[x as usize], x);
        }
    }
}

#[test]
fn sample() {
    let n = if cfg!(not(miri)) { 6370_u32 } else { 637_u32 };
    let width = (n.ilog2() + 1).next_multiple_of(2);
    let key_0 = rand::RngExt::random(&mut rand::rng());
    let key_1 = rand::RngExt::random(&mut rand::rng());
    let secret = move |round: u32, x: u32| {
        let buffer = [round.to_le_bytes(), x.to_le_bytes(), key_0, key_1];
        wyhash::wyhash(buffer.as_flattened(), 0) as u32
    };
    let mut permutation = (0..1 << width)
        .map(move |i| feistel(width, i, 8, secret))
        .filter(move |&x| x < n);
    for _ in 0..n {
        assert!(permutation.next().is_some());
    }
    assert_eq!(permutation.next(), None);
}
