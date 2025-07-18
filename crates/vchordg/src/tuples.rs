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
use std::num::{NonZero, Wrapping};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: Tag = Tag::from_ne_bytes(*b"vchordg\0");
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
    bits: u8,
    _padding: [Padding; 7],
    m: u32,
    max_alpha: f32,
    ef_construction: u32,
    beam_construction: u32,
    start: OptionPointer,
    skip: u32,
}

pub struct MetaTuple {
    pub dims: u32,
    pub bits: u8,
    pub m: u32,
    pub max_alpha: f32,
    pub ef_construction: u32,
    pub beam_construction: u32,
    pub start: OptionPointer,
    pub skip: u32,
}

impl Tuple for MetaTuple {
    #[allow(clippy::match_single_binding)]
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            MetaTuple {
                dims,
                bits,
                m,
                max_alpha,
                ef_construction,
                beam_construction,
                start,
                skip,
            } => {
                buffer.extend((MAGIC as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<MetaTupleHeader>()));
                // header
                buffer[size_of::<Tag>()..][..size_of::<MetaTupleHeader>()].copy_from_slice(
                    MetaTupleHeader {
                        version: VERSION,
                        dims: *dims,
                        bits: *bits,
                        m: *m,
                        max_alpha: *max_alpha,
                        ef_construction: *ef_construction,
                        beam_construction: *beam_construction,
                        start: *start,
                        skip: *skip,
                        _padding: Default::default(),
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
    pub fn bits(self) -> u8 {
        self.header.bits
    }
    pub fn m(self) -> u32 {
        self.header.m
    }
    pub fn max_alpha(self) -> f32 {
        self.header.max_alpha
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
    pub fn skip(self) -> u32 {
        self.header.skip
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
    pub fn skip(&mut self) -> &mut u32 {
        &mut self.header.skip
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VertexTupleHeader {
    metadata: [f32; 3],
    elements_s: u16,
    elements_e: u16,
    payload: Option<NonZero<u64>>,
    pointers_s: u16,
    pointers_e: u16,
    _padding_0: [Padding; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VertexTuple {
    pub metadata: [f32; 3],
    pub elements: Vec<u64>,
    pub payload: Option<NonZero<u64>>,
    pub pointers: Vec<Pointer>,
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
        // pointers
        let pointers_s = buffer.len() as u16;
        buffer.extend(self.pointers.as_bytes());
        let pointers_e = buffer.len() as u16;
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
                pointers_s,
                pointers_e,
                _padding_0: Default::default(),
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
        let pointers = checker.bytes(header.pointers_s, header.pointers_e);
        VertexTupleReader {
            header,
            elements,
            pointers,
        }
    }
}

impl WithWriter for VertexTuple {
    type Writer<'a> = VertexTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> VertexTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut VertexTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        let pointers = checker.bytes(header.pointers_s, header.pointers_e);
        VertexTupleWriter {
            header,
            elements,
            pointers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexTupleReader<'a> {
    header: &'a VertexTupleHeader,
    elements: &'a [u64],
    pointers: &'a [Pointer],
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
    pub fn pointers(self) -> &'a [Pointer] {
        self.pointers
    }
}

#[derive(Debug)]
pub struct VertexTupleWriter<'a> {
    header: &'a mut VertexTupleHeader,
    elements: &'a mut [u64],
    pointers: &'a mut [Pointer],
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
    pub fn pointers(&mut self) -> &mut [Pointer] {
        self.pointers
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader0 {
    payload: Option<NonZero<u64>>,
    elements_s: u16,
    elements_e: u16,
    metadata_s: u16,
    neighbours_s: u16,
    neighbours_e: u16,
    _padding_0: [Padding; 2],
    version: Wrapping<u32>,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader1 {
    payload: Option<NonZero<u64>>,
    elements_s: u16,
    elements_e: u16,
    index: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorTuple<V: Vector> {
    _0 {
        payload: Option<NonZero<u64>>,
        metadata: V::Metadata,
        elements: Vec<V::Element>,
        neighbours: Vec<OptionNeighbour>,
        version: Wrapping<u32>,
    },
    _1 {
        payload: Option<NonZero<u64>>,
        elements: Vec<V::Element>,
        index: u32,
    },
}

impl<V: Vector> Tuple for VectorTuple<V> {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            VectorTuple::_0 {
                payload,
                metadata,
                elements,
                neighbours,
                version,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader0>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // metadata
                let metadata_s = buffer.len() as u16;
                buffer.extend(metadata.as_bytes());
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // neighbours
                let neighbours_s = buffer.len() as u16;
                buffer.extend(neighbours.as_bytes());
                let neighbours_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<VectorTupleHeader0>()].copy_from_slice(
                    VectorTupleHeader0 {
                        metadata_s,
                        elements_s,
                        elements_e,
                        neighbours_s,
                        neighbours_e,
                        payload: *payload,
                        version: *version,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            VectorTuple::_1 {
                payload,
                elements,
                index,
            } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader0>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<VectorTupleHeader1>()].copy_from_slice(
                    VectorTupleHeader1 {
                        elements_s,
                        elements_e,
                        payload: *payload,
                        index: *index,
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl<V: Vector> WithReader for VectorTuple<V> {
    type Reader<'a> = VectorTupleReader<'a, V>;

    fn deserialize_ref(source: &[u8]) -> VectorTupleReader<'_, V> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &VectorTupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                let metadata = checker.prefix(header.metadata_s);
                let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
                VectorTupleReader::_0(VectorTupleReader0 {
                    header,
                    elements,
                    metadata,
                    neighbours,
                })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &VectorTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                VectorTupleReader::_1(VectorTupleReader1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

impl<V: Vector> WithWriter for VectorTuple<V> {
    type Writer<'a> = VectorTupleWriter<'a, V>;

    fn deserialize_mut(source: &mut [u8]) -> VectorTupleWriter<'_, V> {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let mut checker = MutChecker::new(source);
                let header: &mut VectorTupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                let metadata = checker.prefix(header.metadata_s);
                let neighbours = checker.bytes(header.neighbours_s, header.neighbours_e);
                VectorTupleWriter::_0(VectorTupleWriter0 {
                    header,
                    elements,
                    metadata,
                    neighbours,
                })
            }
            1 => {
                let mut checker = MutChecker::new(source);
                let header: &mut VectorTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                VectorTupleWriter::_1(VectorTupleWriter1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

#[derive(Debug)]
pub enum VectorTupleReader<'a, V: Vector> {
    _0(VectorTupleReader0<'a, V>),
    _1(VectorTupleReader1<'a, V>),
}

impl<'a, V: Vector> Copy for VectorTupleReader<'a, V> {}

impl<'a, V: Vector> Clone for VectorTupleReader<'a, V> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Debug, PartialEq)]
pub struct VectorTupleReader0<'a, V: Vector> {
    header: &'a VectorTupleHeader0,
    elements: &'a [V::Element],
    metadata: &'a V::Metadata,
    neighbours: &'a [OptionNeighbour],
}

impl<'a, V: Vector> Copy for VectorTupleReader0<'a, V> {}

impl<'a, V: Vector> Clone for VectorTupleReader0<'a, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: Vector> VectorTupleReader0<'a, V> {
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
    pub fn version(self) -> Wrapping<u32> {
        self.header.version
    }
}

#[derive(Debug, PartialEq)]
pub struct VectorTupleReader1<'a, V: Vector> {
    header: &'a VectorTupleHeader1,
    elements: &'a [V::Element],
}

impl<'a, V: Vector> Copy for VectorTupleReader1<'a, V> {}

impl<'a, V: Vector> Clone for VectorTupleReader1<'a, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, V: Vector> VectorTupleReader1<'a, V> {
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
    }
    pub fn elements(self) -> &'a [V::Element] {
        self.elements
    }
    pub fn index(self) -> u32 {
        self.header.index
    }
}

#[derive(Debug)]
pub enum VectorTupleWriter<'a, V: Vector> {
    _0(VectorTupleWriter0<'a, V>),
    #[expect(dead_code)]
    _1(VectorTupleWriter1<'a, V>),
}

#[derive(Debug)]
pub struct VectorTupleWriter0<'a, V: Vector> {
    header: &'a mut VectorTupleHeader0,
    elements: &'a mut [V::Element],
    metadata: &'a mut V::Metadata,
    neighbours: &'a mut [OptionNeighbour],
}

impl<V: Vector> VectorTupleWriter0<'_, V> {
    #[expect(dead_code)]
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
    pub fn version(&mut self) -> &mut Wrapping<u32> {
        &mut self.header.version
    }
}

#[derive(Debug)]
pub struct VectorTupleWriter1<'a, V: Vector> {
    header: &'a mut VectorTupleHeader1,
    elements: &'a mut [V::Element],
}

impl<V: Vector> VectorTupleWriter1<'_, V> {
    #[expect(dead_code)]
    pub fn payload(&mut self) -> &mut Option<NonZero<u64>> {
        &mut self.header.payload
    }
    #[expect(dead_code)]
    pub fn elements(&mut self) -> &mut [V::Element] {
        self.elements
    }
    #[expect(dead_code)]
    pub fn index(&mut self) -> &mut u32 {
        &mut self.header.index
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
