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

use crate::operator::Vector;
use algo::tuples::{Bool, MutChecker, Padding, RefChecker};
use distance::Distance;
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: Tag = Tag::from_ne_bytes(*b"vcvamana");
const VERSION: u64 = 10;

pub trait Tuple: 'static {
    fn serialize(&self) -> Vec<u8>;
}

pub trait WithReader: Tuple {
    type Reader<'a>;
    fn deserialize_ref(source: &[u8]) -> Self::Reader<'_>;
}

pub trait WithWriter: Tuple {
    type Writer<'a>;
    fn deserialize_mut(source: &mut [u8]) -> Self::Writer<'_>;
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct MetaTupleHeader {
    version: u64,
    dims: u32,
    m: u32,
    alpha: f32,
    ef_construction: u32,
    beam_construction: u32,
    _padding_0: [Padding; 4],
    start: OptionPointer,
}

pub struct MetaTuple {
    pub dims: u32,
    pub m: u32,
    pub alpha: f32,
    pub ef_construction: u32,
    pub beam_construction: u32,
    pub start: OptionPointer,
}

impl Tuple for MetaTuple {
    #[allow(clippy::match_single_binding)]
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            MetaTuple {
                dims,
                m,
                alpha,
                ef_construction,
                beam_construction,
                start,
            } => {
                buffer.extend((MAGIC as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<MetaTupleHeader>()));
                // header
                buffer[size_of::<Tag>()..][..size_of::<MetaTupleHeader>()].copy_from_slice(
                    MetaTupleHeader {
                        version: VERSION,
                        dims: *dims,
                        m: *m,
                        alpha: *alpha,
                        ef_construction: *ef_construction,
                        beam_construction: *beam_construction,
                        start: *start,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl WithReader for MetaTuple {
    type Reader<'a> = MetaTupleReader<'a>;
    fn deserialize_ref(source: &[u8]) -> MetaTupleReader<'_> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            MAGIC => {
                let checker = RefChecker::new(source);
                if VERSION != *checker.prefix::<u64>(size_of::<Tag>()) {
                    panic!("deserialization: bad version number");
                }
                let header: &MetaTupleHeader = checker.prefix(size_of::<Tag>());
                MetaTupleReader { header }
            }
            _ => panic!("deserialization: bad magic number"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaTupleReader<'a> {
    header: &'a MetaTupleHeader,
}

impl<'a> MetaTupleReader<'a> {
    pub fn dims(self) -> u32 {
        self.header.dims
    }
    pub fn m(self) -> u32 {
        self.header.m
    }
    pub fn alpha(self) -> f32 {
        self.header.alpha
    }
    pub fn ef_construction(self) -> u32 {
        self.header.ef_construction
    }
    pub fn beam_construction(self) -> u32 {
        self.header.beam_construction
    }
    pub fn start(self) -> OptionPointer {
        self.header.start
    }
}

impl WithWriter for MetaTuple {
    type Writer<'a> = MetaTupleWriter<'a>;
    fn deserialize_mut(source: &mut [u8]) -> MetaTupleWriter<'_> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            MAGIC => {
                let checker = RefChecker::new(source);
                if VERSION != *checker.prefix::<u64>(size_of::<Tag>()) {
                    panic!("deserialization: bad version number");
                }
                let mut checker = MutChecker::new(source);
                let header: &mut MetaTupleHeader = checker.prefix(size_of::<Tag>());
                MetaTupleWriter { header }
            }
            _ => panic!("deserialization: bad magic number"),
        }
    }
}

#[derive(Debug)]
pub struct MetaTupleWriter<'a> {
    header: &'a mut MetaTupleHeader,
}

impl<'a> MetaTupleWriter<'a> {
    pub fn start(&mut self) -> &mut OptionPointer {
        &mut self.header.start
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VertexTupleHeader {
    metadata: [f32; 3],
    elements_s: u16,
    elements_e: u16,
    payload: Option<NonZero<u64>>,
    pointer: Pointer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VertexTuple {
    pub metadata: [f32; 3],
    pub elements: Vec<u64>,
    pub payload: Option<NonZero<u64>>,
    pub pointer: Pointer,
}

impl Tuple for VertexTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<VertexTupleHeader>()));
        // elements
        let elements_s = buffer.len() as u16;
        buffer.extend(self.elements.as_bytes());
        let elements_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // header
        buffer[..size_of::<VertexTupleHeader>()].copy_from_slice(
            VertexTupleHeader {
                metadata: self.metadata,
                elements_s,
                elements_e,
                payload: self.payload,
                pointer: self.pointer,
            }
            .as_bytes(),
        );
        buffer
    }
}

impl WithReader for VertexTuple {
    type Reader<'a> = VertexTupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> VertexTupleReader<'_> {
        let checker = RefChecker::new(source);
        let header: &VertexTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        VertexTupleReader { header, elements }
    }
}

impl WithWriter for VertexTuple {
    type Writer<'a> = VertexTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> VertexTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut VertexTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        VertexTupleWriter { header, elements }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexTupleReader<'a> {
    header: &'a VertexTupleHeader,
    elements: &'a [u64],
}

impl<'a> VertexTupleReader<'a> {
    pub fn metadata(self) -> [f32; 3] {
        self.header.metadata
    }
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
    }
    pub fn elements(self) -> &'a [u64] {
        self.elements
    }
    pub fn pointer(self) -> Pointer {
        self.header.pointer
    }
}

#[derive(Debug)]
pub struct VertexTupleWriter<'a> {
    header: &'a mut VertexTupleHeader,
    elements: &'a mut [u64],
}

impl VertexTupleWriter<'_> {
    #[expect(dead_code)]
    pub fn metadata(&mut self) -> &mut [f32; 3] {
        &mut self.header.metadata
    }
    pub fn payload(&mut self) -> &mut Option<NonZero<u64>> {
        &mut self.header.payload
    }
    #[expect(dead_code)]
    pub fn elements(&mut self) -> &mut [u64] {
        self.elements
    }
    pub fn pointer(&mut self) -> &mut Pointer {
        &mut self.header.pointer
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader {
    payload: Option<NonZero<u64>>,
    elements_s: u16,
    elements_e: u16,
    metadata_s: u16,
    neighbours_s: u16,
    neighbours_e: u16,
    _padding_0: [Padding; 6],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorTuple<V: Vector> {
    pub payload: Option<NonZero<u64>>,
    pub metadata: V::Metadata,
    pub elements: Vec<V::Element>,
    pub neighbours: Vec<OptionNeighbour>,
}

impl<V: Vector> Tuple for VectorTuple<V> {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader>()));
        // elements
        let elements_s = buffer.len() as u16;
        buffer.extend(self.elements.as_bytes());
        let elements_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // metadata
        let metadata_s = buffer.len() as u16;
        buffer.extend(self.metadata.as_bytes());
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // neighbours
        let neighbours_s = buffer.len() as u16;
        buffer.extend(self.neighbours.as_bytes());
        let neighbours_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // header
        buffer[..size_of::<VectorTupleHeader>()].copy_from_slice(
            VectorTupleHeader {
                metadata_s,
                elements_s,
                elements_e,
                neighbours_s,
                neighbours_e,
                payload: self.payload,
                _padding_0: Default::default(),
            }
            .as_bytes(),
        );
        buffer
    }
}

impl<V: Vector> WithReader for VectorTuple<V> {
    type Reader<'a> = VectorTupleReader<'a, V>;

    fn deserialize_ref(source: &[u8]) -> VectorTupleReader<'_, V> {
        let checker = RefChecker::new(source);
        let header: &VectorTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        let metadata = checker.prefix(header.metadata_s);
        let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
        VectorTupleReader {
            header,
            elements,
            metadata,
            neighbours,
        }
    }
}

impl<V: Vector> WithWriter for VectorTuple<V> {
    type Writer<'a> = VectorTupleWriter<'a, V>;

    fn deserialize_mut(source: &mut [u8]) -> VectorTupleWriter<'_, V> {
        let mut checker = MutChecker::new(source);
        let header: &mut VectorTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        let metadata = checker.prefix(header.metadata_s);
        let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
        VectorTupleWriter {
            header,
            elements,
            metadata,
            neighbours,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct VectorTupleReader<'a, V: Vector> {
    header: &'a VectorTupleHeader,
    elements: &'a [V::Element],
    metadata: &'a V::Metadata,
    neighbours: &'a [OptionNeighbour],
}

impl<'a, V: Vector> Copy for VectorTupleReader<'a, V> {}

impl<'a, V: Vector> Clone for VectorTupleReader<'a, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: Vector> VectorTupleReader<'a, V> {
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
    }
    pub fn elements(self) -> &'a [V::Element] {
        self.elements
    }
    pub fn metadata(self) -> &'a V::Metadata {
        self.metadata
    }
    pub fn neighbours(self) -> &'a [OptionNeighbour] {
        self.neighbours
    }
}

#[derive(Debug)]
pub struct VectorTupleWriter<'a, V: Vector> {
    header: &'a mut VectorTupleHeader,
    elements: &'a mut [V::Element],
    metadata: &'a mut V::Metadata,
    neighbours: &'a mut [OptionNeighbour],
}

impl<V: Vector> VectorTupleWriter<'_, V> {
    pub fn payload(&mut self) -> &mut Option<NonZero<u64>> {
        &mut self.header.payload
    }
    #[expect(dead_code)]
    pub fn elements(&mut self) -> &mut [V::Element] {
        self.elements
    }
    #[expect(dead_code)]
    pub fn metadata(&mut self) -> &mut V::Metadata {
        self.metadata
    }
    pub fn neighbours(&mut self) -> &mut [OptionNeighbour] {
        self.neighbours
    }
}

#[repr(C)]
#[derive(
    Debug,
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
pub struct Pointer {
    x: u32,
    y: u16,
    _padding_0: [Padding; 2],
}

impl Pointer {
    pub fn new((x, y): (u32, u16)) -> Self {
        Self {
            x,
            y,
            _padding_0: Default::default(),
        }
    }
    pub fn into_inner(self) -> (u32, u16) {
        (self.x, self.y)
    }
}

#[repr(C)]
#[derive(
    Debug,
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
pub struct OptionPointer {
    x: u32,
    y: u16,
    _padding_0: [Padding; 1],
    validity: Bool,
}

impl OptionPointer {
    pub const NONE: Self = Self {
        x: 0,
        y: 0,
        _padding_0: [Padding::ZERO; 1],
        validity: Bool::FALSE,
    };
    pub fn some((x, y): (u32, u16)) -> Self {
        Self {
            x,
            y,
            _padding_0: Default::default(),
            validity: Bool::TRUE,
        }
    }
    pub fn into_inner(self) -> Option<(u32, u16)> {
        if self.validity.into() {
            Some((self.x, self.y))
        } else {
            None
        }
    }
}

#[repr(C)]
#[derive(
    Debug,
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
pub struct OptionNeighbour {
    pointer: OptionPointer,
    distance: Distance,
}

impl OptionNeighbour {
    pub const NONE: Self = Self {
        pointer: OptionPointer::NONE,
        distance: Distance::ZERO,
    };
    pub fn some(pointer: (u32, u16), distance: Distance) -> Self {
        Self {
            pointer: OptionPointer::some(pointer),
            distance,
        }
    }
    pub fn into_inner(self) -> Option<((u32, u16), Distance)> {
        let inner = self.pointer.into_inner()?;
        Some((inner, self.distance))
    }
}
