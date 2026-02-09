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
use index::tuples::{Bool, MutChecker, Padding, RefChecker};
use std::num::NonZero;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: Tag = Tag::from_ne_bytes(*b"vchordrq");
const VERSION: u64 = 1001;

#[inline(always)]
fn tag(source: &[u8]) -> Tag {
    assert!(source.len() >= size_of::<Tag>());
    #[allow(unsafe_code)]
    unsafe {
        source.as_ptr().cast::<Tag>().read_unaligned()
    }
}

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
    dim: u32,
    height_of_root: u32,
    is_residual: Bool,
    rerank_in_heap: Bool,
    cells_s: u16,
    cells_e: u16,
    _padding_0: [Padding; 2],
    centroids_first: u32,
    vectors_first_s: u16,
    vectors_first_e: u16,
    freepages_first: u32,
    _padding_1: [Padding; 6],
    // tree
    centroid_prefetch_s: u16,
    centroid_prefetch_e: u16,
    centroid_head: u16,
    centroid_norm: f32,
    first: u32,
}

pub struct MetaTuple {
    pub dim: u32,
    pub height_of_root: u32,
    pub is_residual: bool,
    pub rerank_in_heap: bool,
    pub cells: Vec<u32>,
    pub centroids_first: u32,
    pub vectors_first: Vec<u32>,
    pub freepages_first: u32,
    pub centroid_prefetch: Vec<u32>,
    pub centroid_head: u16,
    pub centroid_norm: f32,
    pub first: u32,
}

impl Tuple for MetaTuple {
    #[allow(clippy::match_single_binding)]
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            MetaTuple {
                dim,
                height_of_root,
                is_residual,
                rerank_in_heap,
                cells,
                centroids_first,
                vectors_first,
                freepages_first,
                centroid_prefetch,
                centroid_head,
                centroid_norm,
                first,
            } => {
                buffer.extend((MAGIC as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<MetaTupleHeader>()));
                // cells
                let cells_s = buffer.len() as u16;
                buffer.extend(cells.as_bytes());
                let cells_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // vectors_first
                let vectors_first_s = buffer.len() as u16;
                buffer.extend(vectors_first.as_bytes());
                let vectors_first_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // centroid_prefetch
                let centroid_prefetch_s = buffer.len() as u16;
                buffer.extend(centroid_prefetch.as_bytes());
                let centroid_prefetch_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<MetaTupleHeader>()].copy_from_slice(
                    MetaTupleHeader {
                        version: VERSION,
                        dim: *dim,
                        height_of_root: *height_of_root,
                        is_residual: (*is_residual).into(),
                        rerank_in_heap: (*rerank_in_heap).into(),
                        cells_s,
                        cells_e,
                        centroids_first: *centroids_first,
                        vectors_first_s,
                        vectors_first_e,
                        freepages_first: *freepages_first,
                        centroid_prefetch_s,
                        centroid_prefetch_e,
                        centroid_head: *centroid_head,
                        centroid_norm: *centroid_norm,
                        first: *first,
                        _padding_0: Default::default(),
                        _padding_1: Default::default(),
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
        let tag = tag(source);
        match tag {
            MAGIC => {
                let checker = RefChecker::new(source);
                if VERSION != *checker.prefix::<u64>(size_of::<Tag>()) {
                    panic!(
                        "deserialization: bad version number; {}",
                        "after upgrading VectorChord, please use REINDEX to rebuild the index."
                    );
                }
                let header: &MetaTupleHeader = checker.prefix(size_of::<Tag>());
                let cells = checker.bytes(header.cells_s, header.cells_e);
                let vectors_first = checker.bytes(header.vectors_first_s, header.vectors_first_e);
                let centroid_prefetch =
                    checker.bytes(header.centroid_prefetch_s, header.centroid_prefetch_e);
                MetaTupleReader {
                    header,
                    cells,
                    vectors_first,
                    centroid_prefetch,
                }
            }
            _ => panic!("deserialization: bad magic number"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaTupleReader<'a> {
    header: &'a MetaTupleHeader,
    cells: &'a [u32],
    vectors_first: &'a [u32],
    centroid_prefetch: &'a [u32],
}

impl<'a> MetaTupleReader<'a> {
    pub fn dim(self) -> u32 {
        self.header.dim
    }
    pub fn height_of_root(self) -> u32 {
        self.header.height_of_root
    }
    pub fn is_residual(self) -> bool {
        self.header.is_residual.into()
    }
    pub fn rerank_in_heap(self) -> bool {
        self.header.rerank_in_heap.into()
    }
    pub fn cells(self) -> &'a [u32] {
        self.cells
    }
    pub fn centroids_first(self) -> u32 {
        self.header.centroids_first
    }
    pub fn vectors_first(self) -> &'a [u32] {
        self.vectors_first
    }
    pub fn freepages_first(self) -> u32 {
        self.header.freepages_first
    }
    pub fn centroid_prefetch(self) -> &'a [u32] {
        self.centroid_prefetch
    }
    pub fn centroid_head(self) -> u16 {
        self.header.centroid_head
    }
    pub fn centroid_norm(self) -> f32 {
        self.header.centroid_norm
    }
    pub fn first(self) -> u32 {
        self.header.first
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FreepagesTupleHeader {
    first: u32,
    _padding_0: [Padding; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct FreepagesTuple {}

impl Tuple for FreepagesTuple {
    fn serialize(&self) -> Vec<u8> {
        FreepagesTupleHeader {
            first: u32::MAX,
            _padding_0: Default::default(),
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithWriter for FreepagesTuple {
    type Writer<'a> = FreepagesTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> FreepagesTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header = checker.prefix(0_u16);
        FreepagesTupleWriter { header }
    }
}

pub struct FreepagesTupleWriter<'a> {
    header: &'a mut FreepagesTupleHeader,
}

impl FreepagesTupleWriter<'_> {
    pub fn first(&mut self) -> &mut u32 {
        &mut self.header.first
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct CentroidTupleHeader0 {
    metadata_s: u16,
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 2],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct CentroidTupleHeader1 {
    head: u16,
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub enum CentroidTuple<V: Vector> {
    _0 {
        metadata: V::Metadata,
        elements: Vec<V::Element>,
    },
    _1 {
        head: u16,
        elements: Vec<V::Element>,
    },
}

impl<V: Vector> Tuple for CentroidTuple<V> {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            CentroidTuple::_0 { metadata, elements } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<CentroidTupleHeader0>()));
                // metadata
                let metadata_s = buffer.len() as u16;
                buffer.extend(metadata.as_bytes());
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<CentroidTupleHeader0>()].copy_from_slice(
                    CentroidTupleHeader0 {
                        metadata_s,
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            CentroidTuple::_1 { head, elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<CentroidTupleHeader1>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<CentroidTupleHeader1>()].copy_from_slice(
                    CentroidTupleHeader1 {
                        head: *head,
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl<V: Vector> WithReader for CentroidTuple<V> {
    type Reader<'a> = CentroidTupleReader<'a, V>;

    fn deserialize_ref(source: &[u8]) -> CentroidTupleReader<'_, V> {
        let tag = tag(source);
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &CentroidTupleHeader0 = checker.prefix(size_of::<Tag>());
                let metadata = checker.prefix(header.metadata_s);
                let elements = checker.bytes(header.elements_s, header.elements_e);
                CentroidTupleReader::_0(CentroidTupleReader0 {
                    header,
                    elements,
                    metadata,
                })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &CentroidTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                CentroidTupleReader::_1(CentroidTupleReader1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

#[derive(Clone)]
pub struct CentroidTupleReader0<'a, V: Vector> {
    #[allow(dead_code)]
    header: &'a CentroidTupleHeader0,
    metadata: &'a V::Metadata,
    elements: &'a [V::Element],
}

impl<V: Vector> Copy for CentroidTupleReader0<'_, V> {}

#[derive(Clone)]
pub struct CentroidTupleReader1<'a, V: Vector> {
    header: &'a CentroidTupleHeader1,
    elements: &'a [V::Element],
}

impl<V: Vector> Copy for CentroidTupleReader1<'_, V> {}

#[derive(Clone)]
pub enum CentroidTupleReader<'a, V: Vector> {
    _0(CentroidTupleReader0<'a, V>),
    _1(CentroidTupleReader1<'a, V>),
}

impl<V: Vector> Copy for CentroidTupleReader<'_, V> {}

impl<'a, V: Vector> CentroidTupleReader<'a, V> {
    pub fn elements(self) -> &'a [<V as Vector>::Element] {
        match self {
            CentroidTupleReader::_0(this) => this.elements,
            CentroidTupleReader::_1(this) => this.elements,
        }
    }
    pub fn metadata_or_head(self) -> Result<V::Metadata, u16> {
        match self {
            CentroidTupleReader::_0(this) => Ok(*this.metadata),
            CentroidTupleReader::_1(this) => Err(this.header.head),
        }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader0 {
    payload: Option<NonZero<u64>>,
    metadata_s: u16,
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 2],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader1 {
    payload: Option<NonZero<u64>>,
    head: u16,
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorTuple<V: Vector> {
    _0 {
        payload: Option<NonZero<u64>>,
        metadata: V::Metadata,
        elements: Vec<V::Element>,
    },
    _1 {
        payload: Option<NonZero<u64>>,
        head: u16,
        elements: Vec<V::Element>,
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
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader0>()));
                // metadata
                let metadata_s = buffer.len() as u16;
                buffer.extend(metadata.as_bytes());
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<VectorTupleHeader0>()].copy_from_slice(
                    VectorTupleHeader0 {
                        payload: *payload,
                        metadata_s,
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            VectorTuple::_1 {
                payload,
                head,
                elements,
            } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader1>()));
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
                        payload: *payload,
                        head: *head,
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
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
        let tag = tag(source);
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &VectorTupleHeader0 = checker.prefix(size_of::<Tag>());
                let metadata = checker.prefix(header.metadata_s);
                let elements = checker.bytes(header.elements_s, header.elements_e);
                VectorTupleReader::_0(VectorTupleReader0 {
                    header,
                    elements,
                    metadata,
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

#[derive(Clone)]
pub struct VectorTupleReader0<'a, V: Vector> {
    header: &'a VectorTupleHeader0,
    metadata: &'a V::Metadata,
    elements: &'a [V::Element],
}

impl<V: Vector> Copy for VectorTupleReader0<'_, V> {}

#[derive(Clone)]
pub struct VectorTupleReader1<'a, V: Vector> {
    header: &'a VectorTupleHeader1,
    elements: &'a [V::Element],
}

impl<V: Vector> Copy for VectorTupleReader1<'_, V> {}

#[derive(Clone)]
pub enum VectorTupleReader<'a, V: Vector> {
    _0(VectorTupleReader0<'a, V>),
    _1(VectorTupleReader1<'a, V>),
}

impl<V: Vector> Copy for VectorTupleReader<'_, V> {}

impl<'a, V: Vector> VectorTupleReader<'a, V> {
    pub fn payload(self) -> Option<NonZero<u64>> {
        match self {
            VectorTupleReader::_0(this) => this.header.payload,
            VectorTupleReader::_1(this) => this.header.payload,
        }
    }
    pub fn elements(self) -> &'a [<V as Vector>::Element] {
        match self {
            VectorTupleReader::_0(this) => this.elements,
            VectorTupleReader::_1(this) => this.elements,
        }
    }
    pub fn metadata_or_head(self) -> Result<V::Metadata, u16> {
        match self {
            VectorTupleReader::_0(this) => Ok(*this.metadata),
            VectorTupleReader::_1(this) => Err(this.header.head),
        }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct DirectoryTupleHeader0 {
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 4],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct DirectoryTupleHeader1 {
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub enum DirectoryTuple {
    _0 { elements: Vec<u32> },
    _1 { elements: Vec<u32> },
}

impl DirectoryTuple {
    pub fn estimate_size_0(elements: usize) -> usize {
        let mut size = 0_usize;
        size += size_of::<Tag>();
        size += size_of::<DirectoryTupleHeader0>();
        size += (elements * size_of::<u32>()).next_multiple_of(ALIGN);
        size
    }
    pub fn fit_1(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<DirectoryTupleHeader1>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<u32>())
        } else {
            None
        }
    }
}

impl Tuple for DirectoryTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            Self::_0 { elements } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<DirectoryTupleHeader0>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<DirectoryTupleHeader0>()].copy_from_slice(
                    DirectoryTupleHeader0 {
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            Self::_1 { elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<DirectoryTupleHeader1>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<DirectoryTupleHeader1>()].copy_from_slice(
                    DirectoryTupleHeader1 {
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl WithReader for DirectoryTuple {
    type Reader<'a> = DirectoryTupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> DirectoryTupleReader<'_> {
        let tag = tag(source);
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &DirectoryTupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                DirectoryTupleReader::_0(DirectoryTupleReader0 { header, elements })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &DirectoryTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                DirectoryTupleReader::_1(DirectoryTupleReader1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectoryTupleReader<'a> {
    _0(DirectoryTupleReader0<'a>),
    _1(DirectoryTupleReader1<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectoryTupleReader0<'a> {
    header: &'a DirectoryTupleHeader0,
    elements: &'a [u32],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectoryTupleReader1<'a> {
    header: &'a DirectoryTupleHeader1,
    elements: &'a [u32],
}

impl<'a> DirectoryTupleReader0<'a> {
    pub fn elements(&self) -> &'a [u32] {
        self.elements
    }
}

impl<'a> DirectoryTupleReader1<'a> {
    pub fn elements(&self) -> &'a [u32] {
        self.elements
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H1TupleHeader0 {
    metadata: [[f32; 32]; 4],
    delta: [f32; 32],
    prefetch_s: u16,
    prefetch_e: u16,
    head: [u16; 32],
    norm: [f32; 32],
    first: [u32; 32],
    len: u32,
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 4],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H1TupleHeader1 {
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 4],
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum H1Tuple {
    _0 {
        metadata: [[f32; 32]; 4],
        delta: [f32; 32],
        prefetch: Vec<[u32; 32]>,
        head: [u16; 32],
        norm: [f32; 32],
        first: [u32; 32],
        len: u32,
        elements: Vec<[u8; 16]>,
    },
    _1 {
        elements: Vec<[u8; 16]>,
    },
}

impl H1Tuple {
    pub fn estimate_size_0(prefetch: usize, elements: usize) -> usize {
        let mut size = 0_usize;
        size += size_of::<Tag>();
        size += size_of::<H1TupleHeader0>();
        size += (prefetch * size_of::<[u32; 32]>()).next_multiple_of(ALIGN);
        size += (elements * size_of::<[u8; 16]>()).next_multiple_of(ALIGN);
        size
    }
    pub fn fit_1(prefetch: usize, freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<H1TupleHeader1>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= (prefetch * size_of::<[u32; 32]>()).next_multiple_of(ALIGN) as isize;
        freespace &= !(ALIGN - 1) as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<[u8; 16]>())
        } else {
            None
        }
    }
}

impl Tuple for H1Tuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            Self::_0 {
                head,
                norm,
                metadata,
                delta,
                first,
                prefetch,
                len,
                elements,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H1TupleHeader0>()));
                // prefetch
                let prefetch_s = buffer.len() as u16;
                buffer.extend(prefetch.as_bytes());
                let prefetch_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<H1TupleHeader0>()].copy_from_slice(
                    H1TupleHeader0 {
                        head: *head,
                        norm: *norm,
                        metadata: *metadata,
                        delta: *delta,
                        first: *first,
                        len: *len,
                        prefetch_s,
                        prefetch_e,
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            Self::_1 { elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H1TupleHeader1>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<H1TupleHeader1>()].copy_from_slice(
                    H1TupleHeader1 {
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl WithReader for H1Tuple {
    type Reader<'a> = H1TupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> H1TupleReader<'_> {
        let tag = tag(source);
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &H1TupleHeader0 = checker.prefix(size_of::<Tag>());
                let prefetch = checker.bytes(header.prefetch_s, header.prefetch_e);
                let elements = checker.bytes(header.elements_s, header.elements_e);
                H1TupleReader::_0(H1TupleReader0 {
                    header,
                    prefetch,
                    elements,
                })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &H1TupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                H1TupleReader::_1(H1TupleReader1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H1TupleReader<'a> {
    _0(H1TupleReader0<'a>),
    _1(H1TupleReader1<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H1TupleReader0<'a> {
    header: &'a H1TupleHeader0,
    prefetch: &'a [[u32; 32]],
    elements: &'a [[u8; 16]],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H1TupleReader1<'a> {
    header: &'a H1TupleHeader1,
    elements: &'a [[u8; 16]],
}

impl<'a> H1TupleReader0<'a> {
    pub fn metadata(self) -> &'a [[f32; 32]; 4] {
        &self.header.metadata
    }
    pub fn delta(self) -> &'a [f32; 32] {
        &self.header.delta
    }
    pub fn prefetch(self) -> &'a [[u32; 32]] {
        self.prefetch
    }
    pub fn head(self) -> &'a [u16; 32] {
        &self.header.head
    }
    pub fn norm(self) -> &'a [f32; 32] {
        &self.header.norm
    }
    pub fn first(self) -> &'a [u32; 32] {
        &self.header.first
    }
    pub fn len(self) -> u32 {
        self.header.len
    }
    pub fn elements(&self) -> &'a [[u8; 16]] {
        self.elements
    }
}

impl<'a> H1TupleReader1<'a> {
    pub fn elements(&self) -> &'a [[u8; 16]] {
        self.elements
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct JumpTupleHeader {
    centroid_prefetch_s: u16,
    centroid_prefetch_e: u16,
    centroid_head: u16,
    _padding_0: [Padding; 6],
    directory_first: u32,
    frozen_first: u32,
    appendable_first: u32,
    tuples: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JumpTuple {
    pub centroid_prefetch: Vec<u32>,
    pub centroid_head: u16,
    pub directory_first: u32,
    pub frozen_first: u32,
    pub appendable_first: u32,
    pub tuples: u64,
}

impl Tuple for JumpTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<JumpTupleHeader>()));
        // centroid_prefetch
        let centroid_prefetch_s = buffer.len() as u16;
        buffer.extend(self.centroid_prefetch.as_bytes());
        let centroid_prefetch_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // header
        buffer[..size_of::<JumpTupleHeader>()].copy_from_slice(
            JumpTupleHeader {
                centroid_prefetch_s,
                centroid_prefetch_e,
                centroid_head: self.centroid_head,
                directory_first: self.directory_first,
                frozen_first: self.frozen_first,
                appendable_first: self.appendable_first,
                tuples: self.tuples,
                _padding_0: Default::default(),
            }
            .as_bytes(),
        );
        buffer
    }
}

impl WithReader for JumpTuple {
    type Reader<'a> = JumpTupleReader<'a>;
    fn deserialize_ref(source: &[u8]) -> JumpTupleReader<'_> {
        let checker = RefChecker::new(source);
        let header: &JumpTupleHeader = checker.prefix(0_u16);
        let centroid_prefetch =
            checker.bytes(header.centroid_prefetch_s, header.centroid_prefetch_e);
        JumpTupleReader {
            header,
            centroid_prefetch,
        }
    }
}

impl WithWriter for JumpTuple {
    type Writer<'a> = JumpTupleWriter<'a>;
    fn deserialize_mut(source: &mut [u8]) -> JumpTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut JumpTupleHeader = checker.prefix(0_u16);
        JumpTupleWriter { header }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct JumpTupleReader<'a> {
    header: &'a JumpTupleHeader,
    centroid_prefetch: &'a [u32],
}

impl<'a> JumpTupleReader<'a> {
    pub fn centroid_prefetch(self) -> &'a [u32] {
        self.centroid_prefetch
    }
    pub fn centroid_head(self) -> u16 {
        self.header.centroid_head
    }
    pub fn directory_first(self) -> u32 {
        self.header.directory_first
    }
    pub fn frozen_first(self) -> u32 {
        self.header.frozen_first
    }
    pub fn appendable_first(self) -> u32 {
        self.header.appendable_first
    }
    pub fn tuples(self) -> u64 {
        self.header.tuples
    }
}

#[derive(Debug)]
pub struct JumpTupleWriter<'a> {
    header: &'a mut JumpTupleHeader,
}

impl JumpTupleWriter<'_> {
    pub fn directory_first(&mut self) -> &mut u32 {
        &mut self.header.directory_first
    }
    pub fn frozen_first(&mut self) -> &mut u32 {
        &mut self.header.frozen_first
    }
    pub fn appendable_first(&mut self) -> &mut u32 {
        &mut self.header.appendable_first
    }
    pub fn tuples(&mut self) -> &mut u64 {
        &mut self.header.tuples
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FrozenTupleHeader0 {
    metadata: [[f32; 32]; 4],
    delta: [f32; 32],
    // it's not last field for reducing padding bytes
    payload: [Option<NonZero<u64>>; 32],
    prefetch_s: u16,
    prefetch_e: u16,
    head: [u16; 32],
    elements_s: u16,
    elements_e: u16,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FrozenTupleHeader1 {
    elements_s: u16,
    elements_e: u16,
    _padding_0: [Padding; 4],
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum FrozenTuple {
    _0 {
        metadata: [[f32; 32]; 4],
        delta: [f32; 32],
        payload: [Option<NonZero<u64>>; 32],
        prefetch: Vec<[u32; 32]>,
        head: [u16; 32],
        elements: Vec<[u8; 16]>,
    },
    _1 {
        elements: Vec<[u8; 16]>,
    },
}

impl FrozenTuple {
    pub fn estimate_size_0(prefetch: usize, elements: usize) -> usize {
        let mut size = 0_usize;
        size += size_of::<Tag>();
        size += size_of::<FrozenTupleHeader0>();
        size += (prefetch * size_of::<[u32; 32]>()).next_multiple_of(ALIGN);
        size += (elements * size_of::<[u8; 16]>()).next_multiple_of(ALIGN);
        size
    }
    pub fn fit_1(prefetch: usize, freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= size_of::<FrozenTupleHeader1>() as isize;
        freespace &= !(ALIGN - 1) as isize;
        freespace -= (prefetch * size_of::<[u32; 32]>()).next_multiple_of(ALIGN) as isize;
        freespace &= !(ALIGN - 1) as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<[u8; 16]>())
        } else {
            None
        }
    }
}

impl Tuple for FrozenTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            FrozenTuple::_0 {
                metadata,
                delta,
                payload,
                prefetch,
                head,
                elements,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<FrozenTupleHeader0>()));
                // prefetch
                let prefetch_s = buffer.len() as u16;
                buffer.extend(prefetch.as_bytes());
                let prefetch_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<FrozenTupleHeader0>()].copy_from_slice(
                    FrozenTupleHeader0 {
                        head: *head,
                        metadata: *metadata,
                        delta: *delta,
                        payload: *payload,
                        elements_s,
                        elements_e,
                        prefetch_s,
                        prefetch_e,
                    }
                    .as_bytes(),
                );
            }
            Self::_1 { elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<FrozenTupleHeader1>()));
                // elements
                let elements_s = buffer.len() as u16;
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len() as u16;
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                // header
                buffer[size_of::<Tag>()..][..size_of::<FrozenTupleHeader1>()].copy_from_slice(
                    FrozenTupleHeader1 {
                        elements_s,
                        elements_e,
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
        }
        buffer
    }
}

impl WithReader for FrozenTuple {
    type Reader<'a> = FrozenTupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> FrozenTupleReader<'_> {
        let tag = tag(source);
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &FrozenTupleHeader0 = checker.prefix(size_of::<Tag>());
                let prefetch = checker.bytes(header.prefetch_s, header.prefetch_e);
                let elements = checker.bytes(header.elements_s, header.elements_e);
                FrozenTupleReader::_0(FrozenTupleReader0 {
                    header,
                    prefetch,
                    elements,
                })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &FrozenTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                FrozenTupleReader::_1(FrozenTupleReader1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

impl WithWriter for FrozenTuple {
    type Writer<'a> = FrozenTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> FrozenTupleWriter<'_> {
        let tag = tag(source);
        match tag {
            0 => {
                let mut checker = MutChecker::new(source);
                let header: &mut FrozenTupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                FrozenTupleWriter::_0(FrozenTupleWriter0 { header, elements })
            }
            1 => {
                let mut checker = MutChecker::new(source);
                let header: &mut FrozenTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                FrozenTupleWriter::_1(FrozenTupleWriter1 { header, elements })
            }
            _ => panic!("deserialization: bad bytes"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrozenTupleReader<'a> {
    _0(FrozenTupleReader0<'a>),
    _1(FrozenTupleReader1<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenTupleReader0<'a> {
    header: &'a FrozenTupleHeader0,
    prefetch: &'a [[u32; 32]],
    elements: &'a [[u8; 16]],
}

impl<'a> FrozenTupleReader0<'a> {
    pub fn metadata(self) -> &'a [[f32; 32]; 4] {
        &self.header.metadata
    }
    pub fn delta(self) -> &'a [f32; 32] {
        &self.header.delta
    }
    pub fn payload(self) -> &'a [Option<NonZero<u64>>; 32] {
        &self.header.payload
    }
    pub fn prefetch(self) -> &'a [[u32; 32]] {
        self.prefetch
    }
    pub fn head(self) -> &'a [u16; 32] {
        &self.header.head
    }
    pub fn elements(self) -> &'a [[u8; 16]] {
        self.elements
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenTupleReader1<'a> {
    header: &'a FrozenTupleHeader1,
    elements: &'a [[u8; 16]],
}

impl<'a> FrozenTupleReader1<'a> {
    pub fn elements(self) -> &'a [[u8; 16]] {
        self.elements
    }
}

#[derive(Debug)]
pub enum FrozenTupleWriter<'a> {
    _0(FrozenTupleWriter0<'a>),
    #[allow(dead_code)]
    _1(FrozenTupleWriter1<'a>),
}

#[derive(Debug)]
pub struct FrozenTupleWriter0<'a> {
    header: &'a mut FrozenTupleHeader0,
    #[allow(dead_code)]
    elements: &'a mut [[u8; 16]],
}

#[derive(Debug)]
pub struct FrozenTupleWriter1<'a> {
    #[allow(dead_code)]
    header: &'a mut FrozenTupleHeader1,
    #[allow(dead_code)]
    elements: &'a mut [[u8; 16]],
}

impl FrozenTupleWriter0<'_> {
    pub fn payload(&mut self) -> &mut [Option<NonZero<u64>>; 32] {
        &mut self.header.payload
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct AppendableTupleHeader {
    metadata: [f32; 4],
    delta: f32,
    prefetch_s: u16,
    prefetch_e: u16,
    head: u16,
    _padding_0: [Padding; 2],
    elements_s: u16,
    elements_e: u16,
    // it's the last field for reducing padding bytes
    payload: Option<NonZero<u64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppendableTuple {
    pub metadata: [f32; 4],
    pub delta: f32,
    pub prefetch: Vec<u32>,
    pub head: u16,
    pub elements: Vec<u64>,
    pub payload: Option<NonZero<u64>>,
}

impl Tuple for AppendableTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<AppendableTupleHeader>()));
        // prefetch
        let prefetch_s = buffer.len() as u16;
        buffer.extend(self.prefetch.as_bytes());
        let prefetch_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // elements
        let elements_s = buffer.len() as u16;
        buffer.extend(self.elements.as_bytes());
        let elements_e = buffer.len() as u16;
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
        // header
        buffer[..size_of::<AppendableTupleHeader>()].copy_from_slice(
            AppendableTupleHeader {
                metadata: self.metadata,
                delta: self.delta,
                prefetch_s,
                prefetch_e,
                head: self.head,
                elements_s,
                elements_e,
                payload: self.payload,
                _padding_0: Default::default(),
            }
            .as_bytes(),
        );
        buffer
    }
}

impl WithReader for AppendableTuple {
    type Reader<'a> = AppendableTupleReader<'a>;

    fn deserialize_ref(source: &[u8]) -> AppendableTupleReader<'_> {
        let checker = RefChecker::new(source);
        let header: &AppendableTupleHeader = checker.prefix(0_u16);
        let prefetch = checker.bytes(header.prefetch_s, header.prefetch_e);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        AppendableTupleReader {
            header,
            prefetch,
            elements,
        }
    }
}

impl WithWriter for AppendableTuple {
    type Writer<'a> = AppendableTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> AppendableTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut AppendableTupleHeader = checker.prefix(0_u16);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        AppendableTupleWriter { header, elements }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppendableTupleReader<'a> {
    header: &'a AppendableTupleHeader,
    prefetch: &'a [u32],
    elements: &'a [u64],
}

impl<'a> AppendableTupleReader<'a> {
    pub fn metadata(self) -> [f32; 4] {
        self.header.metadata
    }
    pub fn delta(self) -> f32 {
        self.header.delta
    }
    pub fn prefetch(self) -> &'a [u32] {
        self.prefetch
    }
    pub fn head(self) -> u16 {
        self.header.head
    }
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
    }
    pub fn elements(self) -> &'a [u64] {
        self.elements
    }
}

#[derive(Debug)]
pub struct AppendableTupleWriter<'a> {
    header: &'a mut AppendableTupleHeader,
    #[allow(dead_code)]
    elements: &'a mut [u64],
}

impl AppendableTupleWriter<'_> {
    pub fn payload(&mut self) -> &mut Option<NonZero<u64>> {
        &mut self.header.payload
    }
}
