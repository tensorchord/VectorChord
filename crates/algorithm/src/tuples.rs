use crate::IndexPointer;
use crate::operator::Vector;
use rabitq::binary::BinaryCode;
use std::marker::PhantomData;
use std::num::NonZero;
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: u64 = u64::from_ne_bytes(*b"vchordrq");
const VERSION: u64 = 4;

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
    magic: u64,
    version: u64,
    dims: u32,
    height_of_root: u32,
    is_residual: Bool,
    rerank_in_heap: Bool,
    _padding_0: [ZeroU8; 2],
    vectors_first: u32,
    // raw vector
    root_mean: IndexPointer,
    // for meta tuple, it's pointers to next level
    root_first: u32,
    freepage_first: u32,
}

pub struct MetaTuple {
    pub dims: u32,
    pub height_of_root: u32,
    pub is_residual: bool,
    pub rerank_in_heap: bool,
    pub vectors_first: u32,
    pub root_mean: IndexPointer,
    pub root_first: u32,
    pub freepage_first: u32,
}

impl Tuple for MetaTuple {
    fn serialize(&self) -> Vec<u8> {
        MetaTupleHeader {
            magic: MAGIC,
            version: VERSION,
            dims: self.dims,
            height_of_root: self.height_of_root,
            is_residual: self.is_residual.into(),
            rerank_in_heap: self.rerank_in_heap.into(),
            _padding_0: Default::default(),
            vectors_first: self.vectors_first,
            root_mean: self.root_mean,
            root_first: self.root_first,
            freepage_first: self.freepage_first,
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithReader for MetaTuple {
    type Reader<'a> = MetaTupleReader<'a>;
    fn deserialize_ref(source: &[u8]) -> MetaTupleReader<'_> {
        if source.len() < 16 {
            panic!("deserialization: bad bytes")
        }
        let magic = u64::from_ne_bytes(std::array::from_fn(|i| source[i + 0]));
        if magic != MAGIC {
            panic!("deserialization: bad magic number");
        }
        let version = u64::from_ne_bytes(std::array::from_fn(|i| source[i + 8]));
        if version != VERSION {
            panic!("deserialization: bad version number");
        }
        let checker = RefChecker::new(source);
        let header = checker.prefix(0);
        MetaTupleReader { header }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaTupleReader<'a> {
    header: &'a MetaTupleHeader,
}

impl MetaTupleReader<'_> {
    pub fn dims(self) -> u32 {
        self.header.dims
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
    pub fn vectors_first(self) -> u32 {
        self.header.vectors_first
    }
    pub fn root_mean(self) -> IndexPointer {
        self.header.root_mean
    }
    pub fn root_first(self) -> u32 {
        self.header.root_first
    }
    pub fn freepage_first(self) -> u32 {
        self.header.freepage_first
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FreepageTupleHeader {
    level_0: [u32; 1 << 10],
    level_1: [u32; 1 << 5],
    level_2: [u32; 1 << 0],
    _padding_0: [ZeroU8; 4],
}

const _: () = assert!(size_of::<FreepageTupleHeader>() == 4232);

#[derive(Debug, Clone, PartialEq)]
pub struct FreepageTuple {}

impl Tuple for FreepageTuple {
    fn serialize(&self) -> Vec<u8> {
        FreepageTupleHeader {
            level_0: FromZeros::new_zeroed(),
            level_1: FromZeros::new_zeroed(),
            level_2: FromZeros::new_zeroed(),
            _padding_0: Default::default(),
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithWriter for FreepageTuple {
    type Writer<'a> = FreepageTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> FreepageTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header = checker.prefix(0);
        FreepageTupleWriter { header }
    }
}

pub struct FreepageTupleWriter<'a> {
    header: &'a mut FreepageTupleHeader,
}

impl FreepageTupleWriter<'_> {
    pub fn mark(&mut self, i: usize) {
        assert!(i < 32768, "out of bound: {i}");
        set(&mut self.header.level_0[i >> 5], (i >> 0) % 32, true);
        set(&mut self.header.level_1[i >> 10], (i >> 5) % 32, true);
        set(&mut self.header.level_2[i >> 15], (i >> 10) % 32, true);
    }
    fn find(&self) -> Option<usize> {
        let i_3 = 0_usize;
        let i_2 = self.header.level_2[i_3 << 0].trailing_zeros() as usize;
        if i_2 == 32 {
            return None;
        }
        let i_1 = self.header.level_1[i_3 << 5 | i_2 << 0].trailing_zeros() as usize;
        if i_1 == 32 {
            panic!("deserialization: bad bytes");
        }
        let i_0 = self.header.level_0[i_3 << 10 | i_2 << 5 | i_1 << 0].trailing_zeros() as usize;
        if i_0 == 32 {
            panic!("deserialization: bad bytes");
        }
        Some(i_3 << 15 | i_2 << 10 | i_1 << 5 | i_0 << 0)
    }
    pub fn fetch(&mut self) -> Option<usize> {
        let i = self.find()?;
        let x = false;
        set(&mut self.header.level_0[i >> 5], (i >> 0) % 32, x);
        let x = self.header.level_0[i >> 5] != 0;
        set(&mut self.header.level_1[i >> 10], (i >> 5) % 32, x);
        let x = self.header.level_1[i >> 10] != 0;
        set(&mut self.header.level_2[i >> 15], (i >> 10) % 32, x);
        Some(i)
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader0 {
    payload: Option<NonZero<u64>>,
    metadata_s: usize,
    elements_s: usize,
    elements_e: usize,
    #[cfg(target_pointer_width = "32")]
    _padding_0: [ZeroU8; 4],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader1 {
    payload: Option<NonZero<u64>>,
    pointer: IndexPointer,
    elements_s: usize,
    elements_e: usize,
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
        pointer: IndexPointer,
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
                let metadata_s = buffer.len();
                buffer.extend(metadata.as_bytes());
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                buffer[size_of::<Tag>()..][..size_of::<VectorTupleHeader0>()].copy_from_slice(
                    VectorTupleHeader0 {
                        payload: *payload,
                        metadata_s,
                        elements_s,
                        elements_e,
                        #[cfg(target_pointer_width = "32")]
                        _padding_0: Default::default(),
                    }
                    .as_bytes(),
                );
            }
            VectorTuple::_1 {
                payload,
                pointer,
                elements,
            } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<VectorTupleHeader1>()));
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                buffer[size_of::<Tag>()..][..size_of::<VectorTupleHeader1>()].copy_from_slice(
                    VectorTupleHeader1 {
                        payload: *payload,
                        pointer: *pointer,
                        elements_s,
                        elements_e,
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
    pub fn metadata_or_pointer(self) -> Result<V::Metadata, IndexPointer> {
        match self {
            VectorTupleReader::_0(this) => Ok(*this.metadata),
            VectorTupleReader::_1(this) => Err(this.header.pointer),
        }
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H1TupleHeader0 {
    mean: [IndexPointer; 32],
    dis_u_2: [f32; 32],
    factor_ppc: [f32; 32],
    factor_ip: [f32; 32],
    factor_err: [f32; 32],
    first: [u32; 32],
    len: u32,
    _padding_0: [ZeroU8; 4],
    elements_s: usize,
    elements_e: usize,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H1TupleHeader1 {
    elements_s: usize,
    elements_e: usize,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum H1Tuple {
    _0 {
        mean: [IndexPointer; 32],
        dis_u_2: [f32; 32],
        factor_ppc: [f32; 32],
        factor_ip: [f32; 32],
        factor_err: [f32; 32],
        first: [u32; 32],
        len: u32,
        elements: Vec<[u8; 16]>,
    },
    _1 {
        elements: Vec<[u8; 16]>,
    },
}

impl H1Tuple {
    pub fn estimate_size_0(elements: usize) -> usize {
        let mut size = 0_usize;
        size += size_of::<Tag>();
        size += size_of::<H1TupleHeader0>();
        size += elements * size_of::<[u8; 16]>();
        size
    }
    pub fn fit_1(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace -= size_of::<H1TupleHeader1>() as isize;
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
                mean,
                dis_u_2,
                factor_ppc,
                factor_ip,
                factor_err,
                first,
                len,
                elements,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H1TupleHeader0>()));
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<H1TupleHeader0>()].copy_from_slice(
                    H1TupleHeader0 {
                        mean: *mean,
                        dis_u_2: *dis_u_2,
                        factor_ppc: *factor_ppc,
                        factor_ip: *factor_ip,
                        factor_err: *factor_err,
                        first: *first,
                        len: *len,
                        _padding_0: Default::default(),
                        elements_s,
                        elements_e,
                    }
                    .as_bytes(),
                );
            }
            Self::_1 { elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H1TupleHeader1>()));
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<H1TupleHeader1>()].copy_from_slice(
                    H1TupleHeader1 {
                        elements_s,
                        elements_e,
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
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &H1TupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                H1TupleReader::_0(H1TupleReader0 { header, elements })
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
    elements: &'a [[u8; 16]],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H1TupleReader1<'a> {
    header: &'a H1TupleHeader1,
    elements: &'a [[u8; 16]],
}

impl<'a> H1TupleReader0<'a> {
    pub fn len(self) -> u32 {
        self.header.len
    }
    pub fn mean(self) -> &'a [IndexPointer] {
        &self.header.mean[..self.header.len as usize]
    }
    pub fn metadata(self) -> (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]) {
        (
            &self.header.dis_u_2,
            &self.header.factor_ppc,
            &self.header.factor_ip,
            &self.header.factor_err,
        )
    }
    pub fn first(self) -> &'a [u32] {
        &self.header.first[..self.header.len as usize]
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
    frozen_first: u32,
    appendable_first: u32,
    tuples: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JumpTuple {
    pub frozen_first: u32,
    pub appendable_first: u32,
    pub tuples: u64,
}

impl Tuple for JumpTuple {
    fn serialize(&self) -> Vec<u8> {
        JumpTupleHeader {
            frozen_first: self.frozen_first,
            appendable_first: self.appendable_first,
            tuples: self.tuples,
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithReader for JumpTuple {
    type Reader<'a> = JumpTupleReader<'a>;
    fn deserialize_ref(source: &[u8]) -> JumpTupleReader<'_> {
        let checker = RefChecker::new(source);
        let header: &JumpTupleHeader = checker.prefix(0);
        JumpTupleReader { header }
    }
}

impl WithWriter for JumpTuple {
    type Writer<'a> = JumpTupleWriter<'a>;
    fn deserialize_mut(source: &mut [u8]) -> JumpTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut JumpTupleHeader = checker.prefix(0);
        JumpTupleWriter { header }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct JumpTupleReader<'a> {
    header: &'a JumpTupleHeader,
}

impl JumpTupleReader<'_> {
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
    mean: [IndexPointer; 32],
    dis_u_2: [f32; 32],
    factor_ppc: [f32; 32],
    factor_ip: [f32; 32],
    factor_err: [f32; 32],
    payload: [Option<NonZero<u64>>; 32],
    elements_s: usize,
    elements_e: usize,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FrozenTupleHeader1 {
    elements_s: usize,
    elements_e: usize,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum FrozenTuple {
    _0 {
        mean: [IndexPointer; 32],
        dis_u_2: [f32; 32],
        factor_ppc: [f32; 32],
        factor_ip: [f32; 32],
        factor_err: [f32; 32],
        payload: [Option<NonZero<u64>>; 32],
        elements: Vec<[u8; 16]>,
    },
    _1 {
        elements: Vec<[u8; 16]>,
    },
}

impl FrozenTuple {
    pub fn estimate_size_0(elements: usize) -> usize {
        let mut size = 0_usize;
        size += size_of::<Tag>();
        size += size_of::<FrozenTupleHeader0>();
        size += elements * size_of::<[u8; 16]>();
        size
    }
    pub fn fit_1(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace -= size_of::<FrozenTupleHeader1>() as isize;
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
                mean,
                dis_u_2,
                factor_ppc,
                factor_ip,
                factor_err,
                payload,
                elements,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<FrozenTupleHeader0>()));
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<FrozenTupleHeader0>()].copy_from_slice(
                    FrozenTupleHeader0 {
                        mean: *mean,
                        dis_u_2: *dis_u_2,
                        factor_ppc: *factor_ppc,
                        factor_ip: *factor_ip,
                        factor_err: *factor_err,
                        payload: *payload,
                        elements_s,
                        elements_e,
                    }
                    .as_bytes(),
                );
            }
            Self::_1 { elements } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<FrozenTupleHeader1>()));
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<FrozenTupleHeader1>()].copy_from_slice(
                    FrozenTupleHeader1 {
                        elements_s,
                        elements_e,
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
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &FrozenTupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                FrozenTupleReader::_0(FrozenTupleReader0 { header, elements })
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
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
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
    elements: &'a [[u8; 16]],
}

impl<'a> FrozenTupleReader0<'a> {
    pub fn mean(self) -> &'a [IndexPointer; 32] {
        &self.header.mean
    }
    pub fn metadata(self) -> (&'a [f32; 32], &'a [f32; 32], &'a [f32; 32], &'a [f32; 32]) {
        (
            &self.header.dis_u_2,
            &self.header.factor_ppc,
            &self.header.factor_ip,
            &self.header.factor_err,
        )
    }
    pub fn elements(self) -> &'a [[u8; 16]] {
        self.elements
    }
    pub fn payload(self) -> &'a [Option<NonZero<u64>>; 32] {
        &self.header.payload
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
    mean: IndexPointer,
    dis_u_2: f32,
    factor_ppc: f32,
    factor_ip: f32,
    factor_err: f32,
    payload: Option<NonZero<u64>>,
    elements_s: usize,
    elements_e: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppendableTuple {
    pub mean: IndexPointer,
    pub dis_u_2: f32,
    pub factor_ppc: f32,
    pub factor_ip: f32,
    pub factor_err: f32,
    pub payload: Option<NonZero<u64>>,
    pub elements: Vec<u64>,
}

impl Tuple for AppendableTuple {
    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<AppendableTupleHeader>()));
        let elements_s = buffer.len();
        buffer.extend(self.elements.as_bytes());
        let elements_e = buffer.len();
        buffer[..size_of::<AppendableTupleHeader>()].copy_from_slice(
            AppendableTupleHeader {
                mean: self.mean,
                dis_u_2: self.dis_u_2,
                factor_ppc: self.factor_ppc,
                factor_ip: self.factor_ip,
                factor_err: self.factor_err,
                payload: self.payload,
                elements_s,
                elements_e,
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
        let header: &AppendableTupleHeader = checker.prefix(0);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        AppendableTupleReader { header, elements }
    }
}

impl WithWriter for AppendableTuple {
    type Writer<'a> = AppendableTupleWriter<'a>;

    fn deserialize_mut(source: &mut [u8]) -> AppendableTupleWriter<'_> {
        let mut checker = MutChecker::new(source);
        let header: &mut AppendableTupleHeader = checker.prefix(0);
        let elements = checker.bytes(header.elements_s, header.elements_e);
        AppendableTupleWriter { header, elements }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppendableTupleReader<'a> {
    header: &'a AppendableTupleHeader,
    elements: &'a [u64],
}

impl<'a> AppendableTupleReader<'a> {
    pub fn mean(self) -> IndexPointer {
        self.header.mean
    }
    pub fn code(self) -> BinaryCode<'a> {
        (
            self.header.dis_u_2,
            self.header.factor_ppc,
            self.header.factor_ip,
            self.header.factor_err,
            self.elements,
        )
    }
    pub fn payload(self) -> Option<NonZero<u64>> {
        self.header.payload
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

pub const fn pointer_to_pair(pointer: IndexPointer) -> (u32, u16) {
    let value = pointer.0;
    (((value >> 16) & 0xffffffff) as u32, (value & 0xffff) as u16)
}

pub const fn pair_to_pointer(pair: (u32, u16)) -> IndexPointer {
    let mut value = 0;
    value |= (pair.0 as u64) << 16;
    value |= pair.1 as u64;
    IndexPointer(value)
}

#[test]
const fn soundness_check() {
    let a = (111, 222);
    let b = pair_to_pointer(a);
    let c = pointer_to_pair(b);
    assert!(a.0 == c.0);
    assert!(a.1 == c.1);
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
pub struct ZeroU8(Option<NonZero<u8>>);

#[repr(transparent)]
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
        if value { Self::TRUE } else { Self::FALSE }
    }
}

pub struct RefChecker<'a> {
    bytes: &'a [u8],
}

impl<'a> RefChecker<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    pub fn prefix<T: FromBytes + IntoBytes + KnownLayout + Immutable + Sized>(
        &self,
        s: usize,
    ) -> &'a T {
        let start = s;
        let end = s + size_of::<T>();
        let bytes = &self.bytes[start..end];
        FromBytes::ref_from_bytes(bytes).expect("deserialization: bad bytes")
    }
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + Immutable + ?Sized>(
        &self,
        s: usize,
        e: usize,
    ) -> &'a T {
        let start = s;
        let end = e;
        let bytes = &self.bytes[start..end];
        FromBytes::ref_from_bytes(bytes).expect("deserialization: bad bytes")
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
    pub fn prefix<T: FromBytes + IntoBytes + KnownLayout + Sized>(
        &mut self,
        s: usize,
    ) -> &'a mut T {
        let start = s;
        let end = s + size_of::<T>();
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
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + ?Sized>(
        &mut self,
        s: usize,
        e: usize,
    ) -> &'a mut T {
        let start = s;
        let end = e;
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

#[inline(always)]
fn set(a: &mut u32, i: usize, x: bool) {
    assert!(i < 32, "out of bound: {i}");
    if x {
        *a |= 1 << i;
    } else {
        *a &= !(1 << i);
    }
}

#[test]
fn aliasing_test() {
    #[repr(C, align(8))]
    #[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    struct ExampleHeader {
        elements_s: usize,
        elements_e: usize,
    }
    let serialized = {
        let elements = (0u32..1111).collect::<Vec<u32>>();
        let mut buffer = Vec::<u8>::new();
        buffer.extend(std::iter::repeat_n(0, size_of::<ExampleHeader>()));
        let elements_s = buffer.len();
        buffer.extend(elements.as_bytes());
        let elements_e = buffer.len();
        buffer[..size_of::<ExampleHeader>()].copy_from_slice(
            ExampleHeader {
                elements_s,
                elements_e,
            }
            .as_bytes(),
        );
        buffer
    };
    let mut source = vec![0u64; serialized.len().next_multiple_of(8)];
    source.as_mut_bytes()[..serialized.len()].copy_from_slice(&serialized);
    let deserialized = {
        let mut checker = MutChecker::new(source.as_mut_bytes());
        let header: &mut ExampleHeader = checker.prefix(0);
        let elements: &mut [u32] = checker.bytes(header.elements_s, header.elements_e);
        (header, elements)
    };
    assert_eq!(
        deserialized.1,
        (0u32..1111).collect::<Vec<u32>>().as_slice()
    );
}
