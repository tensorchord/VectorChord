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

use super::opclass::Opfamily;
use crate::index::am::am_build::InternalBuild;
use algorithm::operator::{Dot, L2, Op};
use algorithm::types::*;
use algorithm::*;
use half::f16;
use std::cell::UnsafeCell;
use std::collections::BinaryHeap;
use std::mem::MaybeUninit;
use std::num::NonZero;
use vector::VectorOwned;
use vector::vect::{VectBorrowed, VectOwned};

#[repr(C, align(4096))]
struct Chunk([u8; 2 * 1024 * 1024]);

struct Allocator {
    used: Vec<*mut MaybeUninit<Chunk>>,
    free: Vec<*mut MaybeUninit<Chunk>>,
    this: *mut MaybeUninit<u8>,
    size: usize,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            used: Vec::new(),
            free: Vec::new(),
            this: Box::into_raw(Box::<Chunk>::new_uninit()).cast(),
            size: size_of::<Chunk>(),
        }
    }
    pub fn malloc<T>(&mut self) -> *mut T {
        const {
            assert!(align_of::<T>() <= align_of::<Chunk>());
            assert!(size_of::<T>() <= size_of::<Chunk>());
        }
        if size_of::<T>() <= self.size {
            self.size = (self.size - size_of::<T>()) / align_of::<T>() * align_of::<T>();
            unsafe { self.this.add(self.size).cast::<T>() }
        } else {
            #[cold]
            fn cold<T>(sel: &mut Allocator) -> *mut T {
                abort_unwind(|| {
                    let raw = std::mem::replace(&mut sel.this, std::ptr::null_mut());
                    sel.used.push(raw.cast());
                    sel.this = sel
                        .free
                        .pop()
                        .unwrap_or_else(|| Box::into_raw(Box::<Chunk>::new_uninit()))
                        .cast();
                    sel.size = size_of::<Chunk>();
                });
                sel.size = (sel.size - size_of::<T>()) / align_of::<T>() * align_of::<T>();
                unsafe { sel.this.add(sel.size).cast::<T>() }
            }
            cold(self)
        }
    }
    pub fn malloc_n<T>(&mut self, n: usize) -> *mut T {
        const {
            assert!(align_of::<T>() <= align_of::<Chunk>());
        }
        let limit = const {
            if size_of::<T>() > 0 {
                size_of::<Chunk>() / size_of::<T>()
            } else {
                usize::MAX
            }
        };
        if n <= limit && n * size_of::<T>() <= self.size {
            self.size = (self.size - n * size_of::<T>()) / align_of::<T>() * align_of::<T>();
            unsafe { self.this.add(self.size).cast::<T>() }
        } else {
            #[cold]
            fn cold<T>(sel: &mut Allocator, n: usize) -> *mut T {
                abort_unwind(|| {
                    let raw = std::mem::replace(&mut sel.this, std::ptr::null_mut());
                    sel.used.push(raw.cast());
                    sel.this = sel
                        .free
                        .pop()
                        .unwrap_or_else(|| Box::into_raw(Box::<Chunk>::new_uninit()))
                        .cast();
                    sel.size = size_of::<Chunk>();
                });
                sel.size = (sel.size - n * size_of::<T>()) / align_of::<T>() * align_of::<T>();
                unsafe { sel.this.add(sel.size).cast::<T>() }
            }
            if n > limit {
                panic!("failed to allocate memory");
            }
            cold(self, n)
        }
    }
    pub fn reset(&mut self) {
        abort_unwind(|| {
            self.free.extend(std::mem::take(&mut self.used));
            self.size = size_of::<Chunk>();
        });
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        for raw in self.used.iter().copied() {
            unsafe {
                let _ = Box::<MaybeUninit<Chunk>>::from_raw(raw);
            }
        }
        for raw in self.free.iter().copied() {
            unsafe {
                let _ = Box::<MaybeUninit<Chunk>>::from_raw(raw);
            }
        }
        unsafe {
            let _ = Box::<MaybeUninit<Chunk>>::from_raw(self.this.cast());
        }
    }
}

#[test]
fn test_allocator() {
    let mut allocator = Allocator::new();
    for _ in 0..1024 * 8 {
        allocator.malloc::<()>();
        allocator.malloc::<u8>();
        allocator.malloc::<u32>();
        allocator.malloc::<u8>();
        allocator.malloc::<[u8; 32]>();
        allocator.malloc_n::<u32>(2);
        allocator.malloc::<[u8; 32]>();
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
    }
    let number_of_chunks_0 = 1 + allocator.used.len() + allocator.free.len();
    allocator.reset();
    for _ in 0..1024 * 8 {
        allocator.malloc::<()>();
        allocator.malloc::<u8>();
        allocator.malloc::<u32>();
        allocator.malloc::<u8>();
        allocator.malloc::<[u8; 32]>();
        allocator.malloc_n::<u32>(2);
        allocator.malloc::<[u8; 32]>();
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
        allocator.malloc_n::<u32>(2);
    }
    let number_of_chunks_1 = 1 + allocator.used.len() + allocator.free.len();
    assert_eq!(number_of_chunks_0, number_of_chunks_1);
}

pub struct BumpAlloc {
    inner: UnsafeCell<Allocator>,
}

impl BumpAlloc {
    pub fn new() -> Self {
        Self {
            inner: UnsafeCell::new(Allocator::new()),
        }
    }
    pub fn reset(&mut self) {
        self.inner.get_mut().reset();
    }
}

impl Bump for BumpAlloc {
    fn alloc<T>(&self, value: T) -> &mut T {
        unsafe {
            let ptr = (*self.inner.get()).malloc::<T>();
            ptr.write(value);
            &mut *ptr
        }
    }

    fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        unsafe {
            let ptr = (*self.inner.get()).malloc_n::<T>(slice.len());
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            std::slice::from_raw_parts_mut(ptr, slice.len())
        }
    }
}

pub fn prewarm(opfamily: Opfamily, index: &impl RelationRead, height: i32) -> String {
    let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::prewarm::<_, Op<VectOwned<f32>, L2>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::prewarm::<_, Op<VectOwned<f32>, Dot>>(
                index,
                height,
                make_h0_plain_prefetcher,
            )
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::prewarm::<_, Op<VectOwned<f16>, L2>>(index, height, make_h0_plain_prefetcher)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::prewarm::<_, Op<VectOwned<f16>, Dot>>(
                index,
                height,
                make_h0_plain_prefetcher,
            )
        }
    }
}

pub fn bulkdelete(
    opfamily: Opfamily,
    index: &(impl RelationRead + RelationWrite),
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::bulkdelete::<_, Op<VectOwned<f32>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::bulkdelete::<_, Op<VectOwned<f32>, Dot>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::bulkdelete::<_, Op<VectOwned<f16>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::bulkdelete::<_, Op<VectOwned<f16>, Dot>>(index, check, callback)
        }
    }
}

pub fn maintain(opfamily: Opfamily, index: &(impl RelationRead + RelationWrite), check: impl Fn()) {
    let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::maintain::<_, Op<VectOwned<f32>, L2>>(index, make_h0_plain_prefetcher, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::maintain::<_, Op<VectOwned<f32>, Dot>>(
                index,
                make_h0_plain_prefetcher,
                check,
            )
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::maintain::<_, Op<VectOwned<f16>, L2>>(index, make_h0_plain_prefetcher, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::maintain::<_, Op<VectOwned<f16>, Dot>>(
                index,
                make_h0_plain_prefetcher,
                check,
            )
        }
    }
}

pub fn build(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: &impl RelationWrite,
    structures: Vec<Structure<Vec<f32>>>,
) {
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2) => algorithm::build::<_, Op<VectOwned<f32>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => algorithm::build::<_, Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::L2) => algorithm::build::<_, Op<VectOwned<f16>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => algorithm::build::<_, Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
    }
}

pub fn insert(
    opfamily: Opfamily,
    index: &(impl RelationRead + RelationWrite),
    payload: NonZero<u64>,
    vector: OwnedVector,
) {
    let bump = BumpAlloc::new();
    let make_h1_plain_prefetcher = MakeH1PlainPrefetcherForInsertion { index };
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let (list, head) = insert_vector::<_, Op<VectOwned<f32>, L2>>(index, payload, &vector);
            insert_index::<_, Op<VectOwned<f32>, L2>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                make_h1_plain_prefetcher,
                list,
                head,
            )
        }
        (OwnedVector::Vecf32(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let (list, head) = insert_vector::<_, Op<VectOwned<f32>, Dot>>(index, payload, &vector);
            insert_index::<_, Op<VectOwned<f32>, Dot>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                make_h1_plain_prefetcher,
                list,
                head,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let (list, head) = insert_vector::<_, Op<VectOwned<f16>, L2>>(index, payload, &vector);
            insert_index::<_, Op<VectOwned<f16>, L2>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                make_h1_plain_prefetcher,
                list,
                head,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let (list, head) = insert_vector::<_, Op<VectOwned<f16>, Dot>>(index, payload, &vector);
            insert_index::<_, Op<VectOwned<f16>, Dot>>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                make_h1_plain_prefetcher,
                list,
                head,
            )
        }
    }
}

fn map_structures<T, U>(x: Vec<Structure<T>>, f: impl Fn(T) -> U + Copy) -> Vec<Structure<U>> {
    x.into_iter()
        .map(|Structure { means, children }| Structure {
            means: means.into_iter().map(f).collect(),
            children,
        })
        .collect()
}

pub trait RandomProject {
    type Output;
    fn project(self) -> Self::Output;
}

impl RandomProject for VectBorrowed<'_, f32> {
    type Output = VectOwned<f32>;
    fn project(self) -> VectOwned<f32> {
        use crate::index::projection::project;
        let input = self.slice();
        VectOwned::new(project(input))
    }
}

impl RandomProject for VectBorrowed<'_, f16> {
    type Output = VectOwned<f16>;
    fn project(self) -> VectOwned<f16> {
        use crate::index::projection::project;
        use simd::Floating;
        let input = f16::vector_to_f32(self.slice());
        VectOwned::new(f16::vector_from_f32(&project(&input)))
    }
}

// Emulate unstable library feature `abort_unwind`.
// See https://github.com/rust-lang/rust/issues/130338.

#[inline(never)]
extern "C" fn abort_unwind<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

#[derive(Debug)]
pub struct MakeH1PlainPrefetcherForInsertion<'r, R> {
    pub index: &'r R,
}

impl<'r, R> Clone for MakeH1PlainPrefetcherForInsertion<'r, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'r, R: RelationRead> PrefetcherHeapFamily<'r, R> for MakeH1PlainPrefetcherForInsertion<'r, R> {
    type P<T>
        = PlainPrefetcher<'r, R, FastHeap<T>>
    where
        T: Ord + Fetch + 'r;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch + 'r,
    {
        PlainPrefetcher::new(self.index, FastHeap::from(seq))
    }
}

#[derive(Debug)]
pub struct MakeH1PlainPrefetcher<'r, R> {
    pub index: &'r R,
}

impl<'r, R> Clone for MakeH1PlainPrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'r, R: RelationRead> PrefetcherHeapFamily<'r, R> for MakeH1PlainPrefetcher<'r, R> {
    type P<T>
        = PlainPrefetcher<'r, R, BinaryHeap<T>>
    where
        T: Ord + Fetch + 'r;

    fn prefetch<T>(&mut self, seq: Vec<T>) -> Self::P<T>
    where
        T: Ord + Fetch + 'r,
    {
        PlainPrefetcher::new(self.index, BinaryHeap::from(seq))
    }
}

#[derive(Debug)]
pub struct MakeH0PlainPrefetcher<'r, R> {
    pub index: &'r R,
}

impl<'r, R> Clone for MakeH0PlainPrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'r, R: RelationRead> PrefetcherSequenceFamily<'r, R> for MakeH0PlainPrefetcher<'r, R> {
    type P<S: Sequence>
        = PlainPrefetcher<'r, R, S>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch,
    {
        PlainPrefetcher::new(self.index, seq)
    }
}

#[derive(Debug)]
pub struct MakeH0SimplePrefetcher<'r, R> {
    pub index: &'r R,
}

impl<'r, R> Clone for MakeH0SimplePrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self { index: self.index }
    }
}

impl<'r, R: RelationRead + RelationPrefetch> PrefetcherSequenceFamily<'r, R>
    for MakeH0SimplePrefetcher<'r, R>
{
    type P<S: Sequence>
        = SimplePrefetcher<'r, R, S>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch,
    {
        SimplePrefetcher::new(self.index, seq)
    }
}

#[derive(Debug)]
pub struct MakeH0StreamPrefetcher<'r, R> {
    pub index: &'r R,
    pub hints: Hints,
}

impl<'r, R> Clone for MakeH0StreamPrefetcher<'r, R> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            hints: self.hints.clone(),
        }
    }
}

impl<'r, R: RelationReadStream> PrefetcherSequenceFamily<'r, R> for MakeH0StreamPrefetcher<'r, R> {
    type P<S: Sequence>
        = StreamPrefetcher<'r, R, S>
    where
        S::Item: Fetch;

    fn prefetch<S: Sequence>(&mut self, seq: S) -> Self::P<S>
    where
        S::Item: Fetch,
    {
        StreamPrefetcher::new(self.index, seq, self.hints.clone())
    }
}
