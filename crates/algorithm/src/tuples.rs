use crate::operator::Vector;
use std::marker::PhantomData;
use std::num::{NonZeroU8, NonZeroU64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};
use zerocopy_derive::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const ALIGN: usize = 8;
pub type Tag = u64;
const MAGIC: u64 = u64::from_ne_bytes(*b"vchordrq");
const VERSION: u64 = 2;

pub trait Tuple: 'static {
    type Reader<'a>: TupleReader<'a, Tuple = Self>;
    fn serialize(&self) -> Vec<u8>;
}

pub trait WithWriter: Tuple {
    type Writer<'a>: TupleWriter<'a, Tuple = Self>;
}

pub trait TupleReader<'a>: Copy {
    type Tuple: Tuple;
    fn deserialize_ref(source: &'a [u8]) -> Self;
}

pub trait TupleWriter<'a> {
    type Tuple: Tuple;
    fn deserialize_mut(source: &'a mut [u8]) -> Self;
}

pub fn serialize<T: Tuple>(tuple: &T) -> Vec<u8> {
    Tuple::serialize(tuple)
}

pub fn read_tuple<T: Tuple>(source: &[u8]) -> T::Reader<'_> {
    TupleReader::deserialize_ref(source)
}

pub fn write_tuple<T: Tuple + WithWriter>(source: &mut [u8]) -> T::Writer<'_> {
    TupleWriter::deserialize_mut(source)
}

// meta tuple

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
    type Reader<'a> = MetaTupleReader<'a>;

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

#[derive(Debug, Clone, Copy)]
pub struct MetaTupleReader<'a> {
    header: &'a MetaTupleHeader,
}

impl<'a> TupleReader<'a> for MetaTupleReader<'a> {
    type Tuple = MetaTuple;
    fn deserialize_ref(source: &'a [u8]) -> Self {
        if source.len() < 16 {
            panic!("bad bytes")
        }
        let magic = u64::from_ne_bytes(std::array::from_fn(|i| source[i + 0]));
        if magic != MAGIC {
            panic!("bad magic number");
        }
        let version = u64::from_ne_bytes(std::array::from_fn(|i| source[i + 8]));
        if version != VERSION {
            panic!("bad version number");
        }
        let checker = RefChecker::new(source);
        let header = checker.prefix(0);
        Self { header }
    }
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

// freepage tuple

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct FreepageTupleHeader {
    a: [u32; 1],
    b: [u32; 32],
    c: [u32; 32 * 32],
    _padding_0: [ZeroU8; 4],
}

const _: () = assert!(size_of::<FreepageTupleHeader>() == 4232);

#[derive(Debug, Clone, PartialEq)]
pub struct FreepageTuple {}

impl Tuple for FreepageTuple {
    type Reader<'a> = FreepageTupleReader<'a>;

    fn serialize(&self) -> Vec<u8> {
        FreepageTupleHeader {
            a: std::array::from_fn(|_| 0),
            b: std::array::from_fn(|_| 0),
            c: std::array::from_fn(|_| 0),
            _padding_0: Default::default(),
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithWriter for FreepageTuple {
    type Writer<'a> = FreepageTupleWriter<'a>;
}

#[derive(Debug, Clone, Copy)]
pub struct FreepageTupleReader<'a> {
    #[allow(dead_code)]
    header: &'a FreepageTupleHeader,
}

impl<'a> TupleReader<'a> for FreepageTupleReader<'a> {
    type Tuple = FreepageTuple;

    fn deserialize_ref(source: &'a [u8]) -> Self {
        let checker = RefChecker::new(source);
        let header = checker.prefix(0);
        Self { header }
    }
}

pub struct FreepageTupleWriter<'a> {
    header: &'a mut FreepageTupleHeader,
}

impl<'a> TupleWriter<'a> for FreepageTupleWriter<'a> {
    type Tuple = FreepageTuple;

    fn deserialize_mut(source: &'a mut [u8]) -> Self {
        let mut checker = MutChecker::new(source);
        let header = checker.prefix(0);
        Self { header }
    }
}

impl FreepageTupleWriter<'_> {
    pub fn mark(&mut self, i: usize) {
        let c_i = i;
        self.header.c[c_i / 32] |= 1 << (c_i % 32);
        let b_i = i / 32;
        self.header.b[b_i / 32] |= 1 << (b_i % 32);
        let a_i = i / 32 / 32;
        self.header.a[a_i / 32] |= 1 << (a_i % 32);
    }
    pub fn fetch(&mut self) -> Option<usize> {
        if self.header.a[0].trailing_ones() == 32 {
            return None;
        }
        let a_i = self.header.a[0].trailing_zeros() as usize;
        let b_i = self.header.b[a_i].trailing_zeros() as usize + a_i * 32;
        let c_i = self.header.c[b_i].trailing_zeros() as usize + b_i * 32;
        self.header.c[c_i / 32] &= !(1 << (c_i % 32));
        if self.header.c[b_i] == 0 {
            self.header.b[b_i / 32] &= !(1 << (b_i % 32));
            if self.header.b[a_i] == 0 {
                self.header.a[a_i / 32] &= !(1 << (a_i % 32));
            }
        }
        Some(c_i)
    }
}

// vector tuple

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader0 {
    payload: Option<NonZeroU64>,
    metadata_s: usize,
    elements_s: usize,
    elements_e: usize,
    #[cfg(target_pointer_width = "32")]
    _padding_0: [ZeroU8; 4],
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct VectorTupleHeader1 {
    payload: Option<NonZeroU64>,
    pointer: IndexPointer,
    elements_s: usize,
    elements_e: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorTuple<V: Vector> {
    _0 {
        payload: Option<NonZeroU64>,
        metadata: V::Metadata,
        elements: Vec<V::Element>,
    },
    _1 {
        payload: Option<NonZeroU64>,
        pointer: IndexPointer,
        elements: Vec<V::Element>,
    },
}

impl<V: Vector> Tuple for VectorTuple<V> {
    type Reader<'a> = VectorTupleReader<'a, V>;

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
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let metadata_s = buffer.len();
                buffer.extend(metadata.as_bytes());
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
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
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
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

impl<'a, V: Vector> TupleReader<'a> for VectorTupleReader<'a, V> {
    type Tuple = VectorTuple<V>;

    fn deserialize_ref(source: &'a [u8]) -> Self {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &VectorTupleHeader0 = checker.prefix(size_of::<Tag>());
                let metadata = checker.prefix(header.metadata_s);
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_0(VectorTupleReader0 {
                    header,
                    elements,
                    metadata,
                })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &VectorTupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_1(VectorTupleReader1 { header, elements })
            }
            _ => panic!("bad bytes"),
        }
    }
}

impl<'a, V: Vector> VectorTupleReader<'a, V> {
    pub fn payload(self) -> Option<NonZeroU64> {
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

// height1tuple

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
    pub fn fit_0(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace -= size_of::<H1TupleHeader0>() as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<[u8; 16]>())
        } else {
            None
        }
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
    type Reader<'a> = H1TupleReader<'a>;

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
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
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
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
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

impl<'a> TupleReader<'a> for H1TupleReader<'a> {
    type Tuple = H1Tuple;

    fn deserialize_ref(source: &'a [u8]) -> Self {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &H1TupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_0(H1TupleReader0 { header, elements })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &H1TupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_1(H1TupleReader1 { header, elements })
            }
            _ => panic!("bad bytes"),
        }
    }
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
    first: u32,
    _padding_0: [ZeroU8; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub struct JumpTuple {
    pub first: u32,
}

impl Tuple for JumpTuple {
    type Reader<'a> = JumpTupleReader<'a>;

    fn serialize(&self) -> Vec<u8> {
        JumpTupleHeader {
            first: self.first,
            _padding_0: Default::default(),
        }
        .as_bytes()
        .to_vec()
    }
}

impl WithWriter for JumpTuple {
    type Writer<'a> = JumpTupleWriter<'a>;
}

#[derive(Debug, Clone, Copy)]
pub struct JumpTupleReader<'a> {
    header: &'a JumpTupleHeader,
}

impl<'a> TupleReader<'a> for JumpTupleReader<'a> {
    type Tuple = JumpTuple;

    fn deserialize_ref(source: &'a [u8]) -> Self {
        let checker = RefChecker::new(source);
        let header: &JumpTupleHeader = checker.prefix(0);
        Self { header }
    }
}

impl JumpTupleReader<'_> {
    pub fn first(self) -> u32 {
        self.header.first
    }
}

#[derive(Debug)]
pub struct JumpTupleWriter<'a> {
    header: &'a mut JumpTupleHeader,
}

impl<'a> TupleWriter<'a> for JumpTupleWriter<'a> {
    type Tuple = JumpTuple;

    fn deserialize_mut(source: &'a mut [u8]) -> Self {
        let mut checker = MutChecker::new(source);
        let header: &mut JumpTupleHeader = checker.prefix(0);
        Self { header }
    }
}

impl JumpTupleWriter<'_> {
    pub fn first(&mut self) -> &mut u32 {
        &mut self.header.first
    }
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H0TupleHeader0 {
    mean: IndexPointer,
    dis_u_2: f32,
    factor_ppc: f32,
    factor_ip: f32,
    factor_err: f32,
    payload: Option<NonZeroU64>,
    elements_s: usize,
    elements_e: usize,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H0TupleHeader1 {
    mean: [IndexPointer; 32],
    dis_u_2: [f32; 32],
    factor_ppc: [f32; 32],
    factor_ip: [f32; 32],
    factor_err: [f32; 32],
    payload: [Option<NonZeroU64>; 32],
    elements_s: usize,
    elements_e: usize,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
struct H0TupleHeader2 {
    elements_s: usize,
    elements_e: usize,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum H0Tuple {
    _0 {
        mean: IndexPointer,
        dis_u_2: f32,
        factor_ppc: f32,
        factor_ip: f32,
        factor_err: f32,
        payload: Option<NonZeroU64>,
        elements: Vec<u64>,
    },
    _1 {
        mean: [IndexPointer; 32],
        dis_u_2: [f32; 32],
        factor_ppc: [f32; 32],
        factor_ip: [f32; 32],
        factor_err: [f32; 32],
        payload: [Option<NonZeroU64>; 32],
        elements: Vec<[u8; 16]>,
    },
    _2 {
        elements: Vec<[u8; 16]>,
    },
}

impl H0Tuple {
    pub fn fit_1(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace -= size_of::<H0TupleHeader1>() as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<[u8; 16]>())
        } else {
            None
        }
    }
    pub fn fit_2(freespace: u16) -> Option<usize> {
        let mut freespace = freespace as isize;
        freespace -= size_of::<Tag>() as isize;
        freespace -= size_of::<H0TupleHeader2>() as isize;
        if freespace >= 0 {
            Some(freespace as usize / size_of::<[u8; 16]>())
        } else {
            None
        }
    }
}

impl Tuple for H0Tuple {
    type Reader<'a> = H0TupleReader<'a>;

    fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::<u8>::new();
        match self {
            H0Tuple::_0 {
                mean,
                dis_u_2,
                factor_ppc,
                factor_ip,
                factor_err,
                payload,
                elements,
            } => {
                buffer.extend((0 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H0TupleHeader0>()));
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<H0TupleHeader0>()].copy_from_slice(
                    H0TupleHeader0 {
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
            H0Tuple::_1 {
                mean,
                dis_u_2,
                factor_ppc,
                factor_ip,
                factor_err,
                payload,
                elements,
            } => {
                buffer.extend((1 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H0TupleHeader1>()));
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<H0TupleHeader1>()].copy_from_slice(
                    H0TupleHeader1 {
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
            Self::_2 { elements } => {
                buffer.extend((2 as Tag).to_ne_bytes());
                buffer.extend(std::iter::repeat_n(0, size_of::<H0TupleHeader2>()));
                while buffer.len() % ALIGN != 0 {
                    buffer.push(0);
                }
                let elements_s = buffer.len();
                buffer.extend(elements.as_bytes());
                let elements_e = buffer.len();
                buffer[size_of::<Tag>()..][..size_of::<H0TupleHeader2>()].copy_from_slice(
                    H0TupleHeader2 {
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

impl WithWriter for H0Tuple {
    type Writer<'a> = H0TupleWriter<'a>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum H0TupleReader<'a> {
    _0(H0TupleReader0<'a>),
    _1(H0TupleReader1<'a>),
    _2(H0TupleReader2<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H0TupleReader0<'a> {
    header: &'a H0TupleHeader0,
    elements: &'a [u64],
}

impl<'a> H0TupleReader0<'a> {
    pub fn mean(self) -> IndexPointer {
        self.header.mean
    }
    pub fn code(self) -> (f32, f32, f32, f32, &'a [u64]) {
        (
            self.header.dis_u_2,
            self.header.factor_ppc,
            self.header.factor_ip,
            self.header.factor_err,
            self.elements,
        )
    }
    pub fn payload(self) -> Option<NonZeroU64> {
        self.header.payload
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H0TupleReader1<'a> {
    header: &'a H0TupleHeader1,
    elements: &'a [[u8; 16]],
}

impl<'a> H0TupleReader1<'a> {
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
    pub fn payload(self) -> &'a [Option<NonZeroU64>; 32] {
        &self.header.payload
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H0TupleReader2<'a> {
    header: &'a H0TupleHeader2,
    elements: &'a [[u8; 16]],
}

impl<'a> H0TupleReader2<'a> {
    pub fn elements(self) -> &'a [[u8; 16]] {
        self.elements
    }
}

impl<'a> TupleReader<'a> for H0TupleReader<'a> {
    type Tuple = H0Tuple;

    fn deserialize_ref(source: &'a [u8]) -> Self {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let checker = RefChecker::new(source);
                let header: &H0TupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_0(H0TupleReader0 { header, elements })
            }
            1 => {
                let checker = RefChecker::new(source);
                let header: &H0TupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_1(H0TupleReader1 { header, elements })
            }
            2 => {
                let checker = RefChecker::new(source);
                let header: &H0TupleHeader2 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_2(H0TupleReader2 { header, elements })
            }
            _ => panic!("bad bytes"),
        }
    }
}

#[derive(Debug)]
pub enum H0TupleWriter<'a> {
    _0(H0TupleWriter0<'a>),
    _1(H0TupleWriter1<'a>),
    #[allow(dead_code)]
    _2(H0TupleWriter2<'a>),
}

#[derive(Debug)]
pub struct H0TupleWriter0<'a> {
    header: &'a mut H0TupleHeader0,
    #[allow(dead_code)]
    elements: &'a mut [u64],
}

#[derive(Debug)]
pub struct H0TupleWriter1<'a> {
    header: &'a mut H0TupleHeader1,
    #[allow(dead_code)]
    elements: &'a mut [[u8; 16]],
}

#[derive(Debug)]
pub struct H0TupleWriter2<'a> {
    #[allow(dead_code)]
    header: &'a mut H0TupleHeader2,
    #[allow(dead_code)]
    elements: &'a mut [[u8; 16]],
}

impl<'a> TupleWriter<'a> for H0TupleWriter<'a> {
    type Tuple = H0Tuple;

    fn deserialize_mut(source: &'a mut [u8]) -> Self {
        let tag = Tag::from_ne_bytes(std::array::from_fn(|i| source[i]));
        match tag {
            0 => {
                let mut checker = MutChecker::new(source);
                let header: &mut H0TupleHeader0 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_0(H0TupleWriter0 { header, elements })
            }
            1 => {
                let mut checker = MutChecker::new(source);
                let header: &mut H0TupleHeader1 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_1(H0TupleWriter1 { header, elements })
            }
            2 => {
                let mut checker = MutChecker::new(source);
                let header: &mut H0TupleHeader2 = checker.prefix(size_of::<Tag>());
                let elements = checker.bytes(header.elements_s, header.elements_e);
                Self::_2(H0TupleWriter2 { header, elements })
            }
            _ => panic!("bad bytes"),
        }
    }
}

impl H0TupleWriter0<'_> {
    pub fn payload(&mut self) -> &mut Option<NonZeroU64> {
        &mut self.header.payload
    }
}

impl H0TupleWriter1<'_> {
    pub fn payload(&mut self) -> &mut [Option<NonZeroU64>; 32] {
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
pub struct ZeroU8(Option<NonZeroU8>);

#[repr(transparent)]
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable, KnownLayout,
)]
pub struct IndexPointer(pub u64);

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
        FromBytes::ref_from_bytes(bytes).expect("bad bytes")
    }
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + Immutable + ?Sized>(
        &self,
        s: usize,
        e: usize,
    ) -> &'a T {
        let start = s;
        let end = e;
        let bytes = &self.bytes[start..end];
        FromBytes::ref_from_bytes(bytes).expect("bad bytes")
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
            panic!("bad bytes");
        }
        if !(self.flag <= start) {
            panic!("bad bytes");
        } else {
            self.flag = end;
        }
        #[allow(unsafe_code)]
        let bytes = unsafe {
            std::slice::from_raw_parts_mut((self.bytes as *mut u8).add(start), end - start)
        };
        FromBytes::mut_from_bytes(bytes).expect("bad bytes")
    }
    pub fn bytes<T: FromBytes + IntoBytes + KnownLayout + ?Sized>(
        &mut self,
        s: usize,
        e: usize,
    ) -> &'a mut T {
        let start = s;
        let end = e;
        if !(start <= end && end <= self.bytes.len()) {
            panic!("bad bytes");
        }
        if !(self.flag <= start) {
            panic!("bad bytes");
        } else {
            self.flag = end;
        }
        #[allow(unsafe_code)]
        let bytes = unsafe {
            std::slice::from_raw_parts_mut((self.bytes as *mut u8).add(start), end - start)
        };
        FromBytes::mut_from_bytes(bytes).expect("bad bytes")
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
        while buffer.len() % ALIGN != 0 {
            buffer.push(0);
        }
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
