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

use std::marker::PhantomData;
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;

pub struct RefChecker<'a> {
    bytes: &'a [u8],
}

impl<'a> RefChecker<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    #[inline]
    pub fn prefix<T: FromBytes + IntoBytes + KnownLayout + Immutable + Sized>(
        &self,
        s: impl Into<usize> + Copy,
    ) -> &'a T {
        let start = Into::<usize>::into(s);
        let end = Into::<usize>::into(s) + size_of::<T>();
        let bytes = &self.bytes[start..end];
        FromBytes::ref_from_bytes(bytes).expect("deserialization: bad bytes")
    }
    #[inline]
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + Immutable + ?Sized>(
        &self,
        s: impl Into<usize> + Copy,
        e: impl Into<usize> + Copy,
    ) -> &'a T {
        let start = Into::<usize>::into(s);
        let end = Into::<usize>::into(e);
        let bytes = &self.bytes[start..end];
        FromBytes::ref_from_bytes(bytes).expect("deserialization: bad bytes")
    }
    /// # Safety
    ///
    /// * `FromBytes` could be implemented for `T`.
    #[inline]
    #[allow(unsafe_code)]
    pub unsafe fn bytes_slice_unchecked<T>(
        &self,
        s: impl Into<usize> + Copy,
        e: impl Into<usize> + Copy,
    ) -> &'a [T] {
        let start = Into::<usize>::into(s);
        let end = Into::<usize>::into(e);
        let bytes = &self.bytes[start..end];
        if size_of::<T>() == 0 || bytes.len() % size_of::<T>() == 0 {
            let ptr = bytes as *const [u8] as *const T;
            if ptr.is_aligned() {
                unsafe { std::slice::from_raw_parts(ptr, bytes.len() / size_of::<T>()) }
            } else {
                panic!("deserialization: bad bytes")
            }
        } else {
            panic!("deserialization: bad bytes")
        }
    }
}

pub struct MutChecker<'a> {
    flag: usize,
    bytes: *mut [u8],
    phantom: PhantomData<&'a mut [u8]>,
}

impl<'a> MutChecker<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            flag: 0,
            bytes,
            phantom: PhantomData,
        }
    }
    #[inline]
    pub fn prefix<T: FromBytes + IntoBytes + KnownLayout + Sized>(
        &mut self,
        s: impl Into<usize> + Copy,
    ) -> &'a mut T {
        let start = Into::<usize>::into(s);
        let end = Into::<usize>::into(s) + size_of::<T>();
        if !(start <= end && end <= self.bytes.len()) {
            panic!("deserialization: bad bytes");
        }
        if !(self.flag <= start) {
            panic!("deserialization: bad bytes");
        } else {
            self.flag = end;
        }
        #[allow(unsafe_code)]
        let bytes = unsafe {
            std::slice::from_raw_parts_mut((self.bytes as *mut u8).add(start), end - start)
        };
        FromBytes::mut_from_bytes(bytes).expect("deserialization: bad bytes")
    }
    #[inline]
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + ?Sized>(
        &mut self,
        s: impl Into<usize> + Copy,
        e: impl Into<usize> + Copy,
    ) -> &'a mut T {
        let start = Into::<usize>::into(s);
        let end = Into::<usize>::into(e);
        if !(start <= end && end <= self.bytes.len()) {
            panic!("deserialization: bad bytes");
        }
        if !(self.flag <= start) {
            panic!("deserialization: bad bytes");
        } else {
            self.flag = end;
        }
        #[allow(unsafe_code)]
        let bytes = unsafe {
            std::slice::from_raw_parts_mut((self.bytes as *mut u8).add(start), end - start)
        };
        FromBytes::mut_from_bytes(bytes).expect("deserialization: bad bytes")
    }
}

#[test]
fn aliasing_test() {
    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct ExampleHeader {
        elements_s: u16,
        elements_e: u16,
        _padding_0: [Padding; 4],
    }
    let serialized = {
        let elements = (0u32..1111).collect::<Vec<u32>>();
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<ExampleHeader>()));
        let elements_s = buffer.len() as u16;
        buffer.extend(elements.as_bytes());
        let elements_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        buffer[..size_of::<ExampleHeader>()].copy_from_slice(
            ExampleHeader {
                elements_s,
                elements_e,
                _padding_0: Default::default(),
            }
            .as_bytes(),
        );
        buffer
    };
    let mut source = vec![0u64; serialized.len().next_multiple_of(8)];
    source.as_mut_bytes()[..serialized.len()].copy_from_slice(&serialized);
    let deserialized = {
        let mut checker = MutChecker::new(source.as_mut_bytes());
        let header: &mut ExampleHeader = checker.prefix(0_u16);
        let elements: &mut [u32] = checker.bytes(header.elements_s, header.elements_e);
        (header, elements)
    };
    assert_eq!(
        deserialized.1,
        (0u32..1111).collect::<Vec<u32>>().as_slice()
    );
}

#[repr(transparent)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct Padding(Option<NonZero<u8>>);

impl Padding {
    pub const ZERO: Self = Self(None);
}

#[repr(transparent)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    IntoBytes,
    FromBytes,
    Immutable,
    KnownLayout,
)]
pub struct Bool(u8);

impl Bool {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);
}

impl From<Bool> for bool {
    fn from(value: Bool) -> Self {
        value != Bool::FALSE
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        std::hint::select_unpredictable(value, Self::TRUE, Self::FALSE)
    }
}
