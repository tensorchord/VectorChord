use super::opclass::Opfamily;
use crate::index::am::am_build::InternalBuild;
use algorithm::operator::{Dot, L2, Op};
use algorithm::types::*;
use algorithm::{Bump, RelationRead, RelationWrite};
use half::f16;
use std::cell::UnsafeCell;
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

pub fn prewarm(
    opfamily: Opfamily,
    index: impl RelationRead,
    height: i32,
    check: impl Fn(),
) -> String {
    let message = match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::prewarm::<_, Op<VectOwned<f32>, L2>>(index, height, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::prewarm::<_, Op<VectOwned<f32>, Dot>>(index, height, check)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::prewarm::<_, Op<VectOwned<f16>, L2>>(index, height, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::prewarm::<_, Op<VectOwned<f16>, Dot>>(index, height, check)
        }
    };
    match message {
        Ok(message) => message,
        Err(e) => panic!("{e}"),
    }
}

pub fn bulkdelete(
    opfamily: Opfamily,
    index: impl RelationRead + RelationWrite,
    check: impl Fn(),
    callback: impl Fn(NonZero<u64>) -> bool,
) {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f32>, Dot>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, L2>>(index, check, callback)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::bulkdelete::<Op<VectOwned<f16>, Dot>>(index, check, callback)
        }
    }
}

pub fn maintain(opfamily: Opfamily, index: impl RelationRead + RelationWrite, check: impl Fn()) {
    match (opfamily.vector_kind(), opfamily.distance_kind()) {
        (VectorKind::Vecf32, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f32>, L2>, _>(index, check)
        }
        (VectorKind::Vecf32, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f32>, Dot>, _>(index, check)
        }
        (VectorKind::Vecf16, DistanceKind::L2) => {
            algorithm::maintain::<Op<VectOwned<f16>, L2>, _>(index, check)
        }
        (VectorKind::Vecf16, DistanceKind::Dot) => {
            algorithm::maintain::<Op<VectOwned<f16>, Dot>, _>(index, check)
        }
    }
}

pub fn build(
    vector_options: VectorOptions,
    vchordrq_options: VchordrqIndexOptions,
    index: impl RelationWrite,
    structures: Vec<Structure<Vec<f32>>>,
) {
    match (vector_options.v, vector_options.d) {
        (VectorKind::Vecf32, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f32>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf32, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f32>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::L2) => algorithm::build::<Op<VectOwned<f16>, L2>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
        (VectorKind::Vecf16, DistanceKind::Dot) => algorithm::build::<Op<VectOwned<f16>, Dot>>(
            vector_options,
            vchordrq_options,
            index,
            map_structures(structures, |x| InternalBuild::build_from_vecf32(&x)),
        ),
    }
}

pub fn insert(
    opfamily: Opfamily,
    index: impl RelationRead + RelationWrite,
    payload: NonZero<u64>,
    vector: OwnedVector,
) {
    use algorithm::{FastHeap, PlainPrefetcher};
    let bump = BumpAlloc::new();
    match (vector, opfamily.distance_kind()) {
        (OwnedVector::Vecf32(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let prefetch = {
                let index = index.clone();
                move |results| {
                    PlainPrefetcher::<_, FastHeap<_>, _, Op<VectOwned<f32>, L2>>::new(
                        index.clone(),
                        results,
                        |_, _| (true, None),
                        false,
                    )
                }
            };
            let (list, head) =
                algorithm::insert_vector::<_, Op<VectOwned<f32>, L2>>(&index, payload, &vector);
            algorithm::insert_index::<_, Op<VectOwned<f32>, L2>, _>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                prefetch,
                list,
                head,
            )
        }
        (OwnedVector::Vecf32(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf32);
            let prefetch = {
                let index = index.clone();
                move |results| {
                    PlainPrefetcher::<_, FastHeap<_>, _, Op<VectOwned<f32>, Dot>>::new(
                        index.clone(),
                        results,
                        |_, _| (true, None),
                        false,
                    )
                }
            };
            let (list, head) =
                algorithm::insert_vector::<_, Op<VectOwned<f32>, Dot>>(&index, payload, &vector);
            algorithm::insert_index::<_, Op<VectOwned<f32>, Dot>, _>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                prefetch,
                list,
                head,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::L2) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let prefetch = {
                let index = index.clone();
                move |results| {
                    PlainPrefetcher::<_, FastHeap<_>, _, Op<VectOwned<f16>, L2>>::new(
                        index.clone(),
                        results,
                        |_, _| (true, None),
                        false,
                    )
                }
            };
            let (list, head) =
                algorithm::insert_vector::<_, Op<VectOwned<f16>, L2>>(&index, payload, &vector);
            algorithm::insert_index::<_, Op<VectOwned<f16>, L2>, _>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                prefetch,
                list,
                head,
            )
        }
        (OwnedVector::Vecf16(vector), DistanceKind::Dot) => {
            assert!(opfamily.vector_kind() == VectorKind::Vecf16);
            let prefetch = {
                let index = index.clone();
                move |results| {
                    PlainPrefetcher::<_, FastHeap<_>, _, Op<VectOwned<f16>, Dot>>::new(
                        index.clone(),
                        results,
                        |_, _| (true, None),
                        false,
                    )
                }
            };
            let (list, head) =
                algorithm::insert_vector::<_, Op<VectOwned<f16>, Dot>>(&index, payload, &vector);
            algorithm::insert_index::<_, Op<VectOwned<f16>, Dot>, _>(
                index,
                payload,
                RandomProject::project(vector.as_borrowed()),
                &bump,
                prefetch,
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

extern "C" fn abort_unwind<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}
