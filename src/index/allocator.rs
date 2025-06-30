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

use algo::Bump;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

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

// Emulate unstable library feature `abort_unwind`.
// See https://github.com/rust-lang/rust/issues/130338.

#[inline(never)]
extern "C" fn abort_unwind<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}
