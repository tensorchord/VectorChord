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

use algo::{Opaque, *};
use std::collections::VecDeque;
use std::iter::{Chain, Flatten};
use std::marker::PhantomData;
use std::mem::{MaybeUninit, offset_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[repr(C, align(8))]
#[derive(Debug)]
pub struct PostgresPage<O> {
    header: pgrx::pg_sys::PageHeaderData,
    content: [u8; pgrx::pg_sys::BLCKSZ as usize - size_of::<pgrx::pg_sys::PageHeaderData>()],
    _opaque: PhantomData<fn(O) -> O>,
}

// It is a non-guaranteed detection.
// If `PageHeaderData` contains padding bytes, const-eval probably fails.
const _: () = {
    use pgrx::pg_sys::PageHeaderData as T;
    use std::mem::{transmute, zeroed};
    const _ZERO: &[u8; size_of::<T>()] = unsafe { transmute(&zeroed::<T>()) };
};

// Layout checks of header.
const _: () = {
    use pgrx::pg_sys::{MAXIMUM_ALIGNOF, PageHeaderData as T};
    assert!(size_of::<T>() == offset_of!(T, pd_linp));
    assert!(size_of::<T>() % MAXIMUM_ALIGNOF as usize == 0);
};

const _: () = assert!(align_of::<PostgresPage<()>>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);
const _: () = assert!(size_of::<PostgresPage<()>>() == pgrx::pg_sys::BLCKSZ as usize);

impl<O: Opaque> PostgresPage<O> {
    pub fn clone_into_boxed(&self) -> Box<Self> {
        let mut result = Box::new_uninit();
        unsafe {
            std::ptr::copy(self as *const Self, result.as_mut_ptr(), 1);
            result.assume_init()
        }
    }
}

impl<O: Opaque> Page for PostgresPage<O> {
    type Opaque = O;
    fn get_opaque(&self) -> &O {
        assert!(self.header.pd_special as usize + size_of::<O>() == size_of::<Self>());
        unsafe { &*((self as *const _ as *const O).byte_add(self.header.pd_special as _)) }
    }
    fn get_opaque_mut(&mut self) -> &mut O {
        assert!(self.header.pd_special as usize + size_of::<O>() == size_of::<Self>());
        unsafe { &mut *((self as *mut _ as *mut O).byte_add(self.header.pd_special as _)) }
    }
    fn len(&self) -> u16 {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        assert!(self.header.pd_upper as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let upper = self.header.pd_upper as usize;
        assert!(lower <= upper);
        ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16
    }
    fn get(&self, i: u16) -> Option<&[u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        match lp_flags(iid) {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *const Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts(ptr, lp_len as _))
        }
    }
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]> {
        use pgrx::pg_sys::{ItemIdData, PageHeaderData};
        if i == 0 {
            return None;
        }
        assert!(self.header.pd_lower as usize <= size_of::<Self>());
        let lower = self.header.pd_lower as usize;
        let n = ((lower - offset_of!(PageHeaderData, pd_linp)) / size_of::<ItemIdData>()) as u16;
        if i > n {
            return None;
        }
        let iid = unsafe { self.header.pd_linp.as_ptr().add((i - 1) as _).read() };
        let lp_off = iid.lp_off() as usize;
        let lp_len = iid.lp_len() as usize;
        match lp_flags(iid) {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off);
        assert!(lp_off <= size_of::<Self>());
        assert!(lp_len <= size_of::<Self>());
        assert!(lp_off + lp_len <= size_of::<Self>());
        unsafe {
            let ptr = (self as *mut Self).cast::<u8>().add(lp_off as _);
            Some(std::slice::from_raw_parts_mut(ptr, lp_len as _))
        }
    }
    fn alloc(&mut self, data: &[u8]) -> Option<u16> {
        unsafe {
            let i = pgrx::pg_sys::PageAddItemExtended(
                (self as *const Self).cast_mut().cast(),
                data.as_ptr().cast_mut().cast(),
                data.len(),
                0,
                0,
            );
            if i == 0 { None } else { Some(i) }
        }
    }
    fn free(&mut self, i: u16) {
        unsafe {
            pgrx::pg_sys::PageIndexTupleDeleteNoCompact((self as *mut Self).cast(), i);
        }
    }
    fn freespace(&self) -> u16 {
        unsafe { pgrx::pg_sys::PageGetFreeSpace((self as *const Self).cast_mut().cast()) as u16 }
    }
    fn clear(&mut self, opaque: O) {
        unsafe {
            page_init(self, opaque);
        }
    }
}

unsafe fn page_init<O: Opaque>(this: *mut PostgresPage<O>, opaque: O) {
    unsafe {
        use pgrx::pg_sys::{BLCKSZ, PageHeaderData, PageInit};
        PageInit(this.cast(), BLCKSZ as usize, size_of::<O>());
        assert_eq!(
            (*this.cast::<PageHeaderData>()).pd_special as usize + size_of::<O>(),
            size_of::<PostgresPage<O>>()
        );
        this.cast::<O>()
            .byte_add(size_of::<PostgresPage<O>>() - size_of::<O>())
            .write(opaque);
    }
}

pub struct PostgresBufferReadGuard<Opaque> {
    buf: i32,
    page: NonNull<PostgresPage<Opaque>>,
    id: u32,
}

impl<Opaque> PageGuard for PostgresBufferReadGuard<Opaque> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<Opaque> Deref for PostgresBufferReadGuard<Opaque> {
    type Target = PostgresPage<Opaque>;

    fn deref(&self) -> &PostgresPage<Opaque> {
        unsafe { self.page.as_ref() }
    }
}

impl<Opaque> Drop for PostgresBufferReadGuard<Opaque> {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

pub struct PostgresBufferWriteGuard<O: Opaque> {
    raw: pgrx::pg_sys::Relation,
    buf: i32,
    page: NonNull<PostgresPage<O>>,
    state: *mut pgrx::pg_sys::GenericXLogState,
    id: u32,
    tracking_freespace: bool,
}

impl<O: Opaque> PageGuard for PostgresBufferWriteGuard<O> {
    fn id(&self) -> u32 {
        self.id
    }
}

impl<O: Opaque> Deref for PostgresBufferWriteGuard<O> {
    type Target = PostgresPage<O>;

    fn deref(&self) -> &PostgresPage<O> {
        unsafe { self.page.as_ref() }
    }
}

impl<O: Opaque> DerefMut for PostgresBufferWriteGuard<O> {
    fn deref_mut(&mut self) -> &mut PostgresPage<O> {
        unsafe { self.page.as_mut() }
    }
}

impl<O: Opaque> Drop for PostgresBufferWriteGuard<O> {
    fn drop(&mut self) {
        unsafe {
            if std::thread::panicking() {
                pgrx::pg_sys::GenericXLogAbort(self.state);
            } else {
                if self.tracking_freespace {
                    pgrx::pg_sys::RecordPageWithFreeSpace(self.raw, self.id, self.freespace() as _);
                    pgrx::pg_sys::FreeSpaceMapVacuumRange(self.raw, self.id, self.id + 1);
                }
                pgrx::pg_sys::GenericXLogFinish(self.state);
            }
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PostgresRelation<Opaque> {
    raw: pgrx::pg_sys::Relation,
    _phantom: PhantomData<fn(Opaque) -> Opaque>,
}

impl<Opaque> PostgresRelation<Opaque> {
    pub unsafe fn new(raw: pgrx::pg_sys::Relation) -> Self {
        Self {
            raw,
            _phantom: PhantomData,
        }
    }
}

impl<O: Opaque> Relation for PostgresRelation<O> {
    type Page = PostgresPage<O>;
}

impl<O: Opaque> RelationReadTypes for PostgresRelation<O> {
    type ReadGuard<'a> = PostgresBufferReadGuard<O>;
}

impl<O: Opaque> RelationRead for PostgresRelation<O> {
    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_SHARE, BufferGetPage, LockBuffer, ReadBufferExtended, ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                0,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            PostgresBufferReadGuard { buf, page, id }
        }
    }
}

impl<O: Opaque> RelationWriteTypes for PostgresRelation<O> {
    type WriteGuard<'a> = PostgresBufferWriteGuard<O>;
}

impl<O: Opaque> RelationWrite for PostgresRelation<O> {
    fn write(&self, id: u32, tracking_freespace: bool) -> PostgresBufferWriteGuard<O> {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ForkNumber, GenericXLogRegisterBuffer, GenericXLogStart,
                LockBuffer, ReadBufferExtended, ReadBufferMode,
            };
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                id,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.raw);
            let page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, 0).cast::<MaybeUninit<PostgresPage<O>>>(),
            )
            .expect("failed to get page");
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id,
                tracking_freespace,
            }
        }
    }
    fn extend(
        &self,
        opaque: <Self::Page as Page>::Opaque,
        tracking_freespace: bool,
    ) -> PostgresBufferWriteGuard<O> {
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, GENERIC_XLOG_FULL_IMAGE, GenericXLogRegisterBuffer,
                GenericXLogStart, LockBuffer,
            };
            let buf;
            #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15"))]
            {
                use pgrx::pg_sys::{
                    ExclusiveLock, ForkNumber, LockRelationForExtension, ReadBufferExtended,
                    ReadBufferMode, UnlockRelationForExtension,
                };
                LockRelationForExtension(self.raw, ExclusiveLock as _);
                buf = ReadBufferExtended(
                    self.raw,
                    ForkNumber::MAIN_FORKNUM,
                    u32::MAX,
                    ReadBufferMode::RBM_NORMAL,
                    std::ptr::null_mut(),
                );
                UnlockRelationForExtension(self.raw, ExclusiveLock as _);
            }
            #[cfg(any(feature = "pg16", feature = "pg17", feature = "pg18"))]
            {
                use pgrx::pg_sys::{BufferManagerRelation, ExtendBufferedRel, ForkNumber};
                let bmr = BufferManagerRelation {
                    rel: self.raw,
                    smgr: std::ptr::null_mut(),
                    relpersistence: 0,
                };
                buf = ExtendBufferedRel(bmr, ForkNumber::MAIN_FORKNUM, std::ptr::null_mut(), 0);
            }
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.raw);
            let mut page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage<O>>>(),
            )
            .expect("failed to get page");
            page_init(page.as_mut().as_mut_ptr(), opaque);
            PostgresBufferWriteGuard {
                raw: self.raw,
                buf,
                page: page.cast(),
                state,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
                tracking_freespace,
            }
        }
    }
    fn search(&self, freespace: usize) -> Option<PostgresBufferWriteGuard<O>> {
        unsafe {
            loop {
                let id = pgrx::pg_sys::GetPageWithFreeSpace(self.raw, freespace);
                if id == u32::MAX {
                    return None;
                }
                let write = self.write(id, true);
                if write.freespace() < freespace as _ {
                    // the free space is recorded incorrectly
                    pgrx::pg_sys::RecordPageWithFreeSpace(self.raw, id, write.freespace() as _);
                    pgrx::pg_sys::FreeSpaceMapVacuumRange(self.raw, id, id + 1);
                    continue;
                }
                return Some(write);
            }
        }
    }
}

impl<O: Opaque> RelationPrefetch for PostgresRelation<O> {
    fn prefetch(&self, id: u32) {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::PrefetchBuffer;
            PrefetchBuffer(self.raw, 0, id);
        }
    }
}

pub struct Cache<'b, I: Iterator> {
    window: VecDeque<I::Item>,
    tail: VecDeque<u32>,
    iter: Option<I>,
    _phantom: PhantomData<&'b mut ()>,
}

impl<'b, I: Iterator> Default for Cache<'b, I> {
    fn default() -> Self {
        Self {
            window: Default::default(),
            tail: Default::default(),
            iter: Default::default(),
            _phantom: PhantomData,
        }
    }
}

impl<'b, I: Iterator> Cache<'b, I>
where
    I::Item: Fetch<'b>,
{
    #[allow(dead_code)]
    pub fn pop_id(&mut self) -> Option<u32> {
        while self.tail.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.tail.pop_front()
    }
    #[allow(dead_code)]
    pub fn pop_item_if(&mut self, predicate: impl FnOnce(&I::Item) -> bool) -> Option<I::Item> {
        while self.window.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        vec_deque_pop_front_if(&mut self.window, predicate)
    }
    #[allow(dead_code)]
    pub fn pop_item(&mut self) -> Option<I::Item> {
        while self.window.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.window.pop_front()
    }
}

pub struct PostgresReadStream<'b, O: Opaque, I: Iterator> {
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    raw: *mut pgrx::pg_sys::ReadStream,
    // Because of `Box`'s special alias rules, `Box` cannot be used here.
    cache: NonNull<Cache<'b, I>>,
    _phantom: PhantomData<fn(O) -> O>,
}

pub struct PostgresReadStreamGuards<O, I, L> {
    #[cfg(any(feature = "pg17", feature = "pg18"))]
    raw: *mut pgrx::pg_sys::ReadStream,
    list: L,
    _phantom: PhantomData<fn(O, I) -> (O, I)>,
}

impl<O: Opaque, I: Iterator, L> Iterator for PostgresReadStreamGuards<O, I, L>
where
    L: Iterator<Item = u32>,
{
    type Item = PostgresBufferReadGuard<O>;

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }

    #[cfg(any(feature = "pg17", feature = "pg18"))]
    fn next(&mut self) -> Option<Self::Item> {
        let _id = self.list.next()?;
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_SHARE, BufferGetPage, LockBuffer, read_stream_next_buffer,
            };
            let buf = read_stream_next_buffer(self.raw, core::ptr::null_mut());
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            Some(PostgresBufferReadGuard {
                buf,
                page,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.list.size_hint()
    }
}

impl<O: Opaque, I: Iterator, L> ExactSizeIterator for PostgresReadStreamGuards<O, I, L> where
    L: ExactSizeIterator<Item = u32>
{
}

#[cfg(any(feature = "pg17", feature = "pg18"))]
impl<'b, O: Opaque, I: Iterator> PostgresReadStream<'b, O, I>
where
    I::Item: Fetch<'b>,
{
    fn read<L: Iterator<Item = u32>>(&mut self, fetch: L) -> PostgresReadStreamGuards<O, I, L> {
        PostgresReadStreamGuards {
            raw: self.raw,
            list: fetch,
            _phantom: PhantomData,
        }
    }
}

impl<'b, O: Opaque, I: Iterator> ReadStream<'b> for PostgresReadStream<'b, O, I>
where
    I::Item: Fetch<'b>,
{
    type Relation = PostgresRelation<O>;

    type Guards = PostgresReadStreamGuards<O, I, <I::Item as Fetch<'b>>::Iter>;

    type Item = I::Item;

    type Inner =
        Chain<std::collections::vec_deque::IntoIter<I::Item>, Flatten<std::option::IntoIter<I>>>;

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next(&mut self) -> Option<(I::Item, Self::Guards)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next_if<P: FnOnce(&I::Item) -> bool>(
        &mut self,
        _predicate: P,
    ) -> Option<(I::Item, Self::Guards)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg17", feature = "pg18"))]
    fn next(&mut self) -> Option<(I::Item, Self::Guards)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item() } {
            let list = self.read(e.fetch());
            Some((e, list))
        } else {
            None
        }
    }

    #[cfg(any(feature = "pg17", feature = "pg18"))]
    fn next_if<P: FnOnce(&I::Item) -> bool>(
        &mut self,
        predicate: P,
    ) -> Option<(I::Item, Self::Guards)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item_if(predicate) } {
            let list = self.read(e.fetch());
            Some((e, list))
        } else {
            None
        }
    }

    fn into_inner(mut self) -> Self::Inner {
        let cache = unsafe { std::mem::take(self.cache.as_mut()) };
        cache
            .window
            .into_iter()
            .chain(cache.iter.into_iter().flatten())
    }
}

impl<'b, O: Opaque, I: Iterator> Drop for PostgresReadStream<'b, O, I> {
    fn drop(&mut self) {
        unsafe {
            let _ = std::mem::take(self.cache.as_mut());
            #[cfg(any(feature = "pg17", feature = "pg18"))]
            if !std::thread::panicking() && pgrx::pg_sys::IsTransactionState() {
                pgrx::pg_sys::read_stream_end(self.raw);
            }
            let _ = box_from_non_null(self.cache);
        }
    }
}

impl<O: Opaque> RelationReadStreamTypes for PostgresRelation<O> {
    type ReadStream<'b, I: Iterator>
        = PostgresReadStream<'b, O, I>
    where
        I::Item: Fetch<'b>;
}

impl<O: Opaque> RelationReadStream for PostgresRelation<O> {
    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn read_stream<'b, I: Iterator>(&'b self, _iter: I, _hints: Hints) -> Self::ReadStream<'b, I>
    where
        I::Item: Fetch<'b>,
    {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg17", feature = "pg18"))]
    fn read_stream<'b, I: Iterator>(&'b self, iter: I, hints: Hints) -> Self::ReadStream<'b, I>
    where
        I::Item: Fetch<'b>,
    {
        #[pgrx::pg_guard]
        unsafe extern "C-unwind" fn callback<'b, I: Iterator>(
            _stream: *mut pgrx::pg_sys::ReadStream,
            callback_private_data: *mut core::ffi::c_void,
            _per_buffer_data: *mut core::ffi::c_void,
        ) -> pgrx::pg_sys::BlockNumber
        where
            I::Item: Fetch<'b>,
        {
            unsafe {
                use pgrx::pg_sys::InvalidBlockNumber;
                let inner = callback_private_data.cast::<Cache<I>>();
                (*inner).pop_id().unwrap_or(InvalidBlockNumber)
            }
        }
        let cache = box_into_non_null(Box::new(Cache {
            window: VecDeque::new(),
            tail: VecDeque::new(),
            iter: Some(iter),
            _phantom: PhantomData,
        }));
        let raw = unsafe {
            use pgrx::pg_sys::{
                ForkNumber, READ_STREAM_DEFAULT, READ_STREAM_FULL, read_stream_begin_relation,
            };
            let mut flags = READ_STREAM_DEFAULT;
            if hints.full {
                flags |= READ_STREAM_FULL;
            }
            read_stream_begin_relation(
                flags as i32,
                core::ptr::null_mut(),
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                Some(callback::<I>),
                cache.as_ptr().cast(),
                0,
            )
        };
        PostgresReadStream {
            raw,
            cache,
            _phantom: PhantomData,
        }
    }
}

#[inline(always)]
fn lp_flags(x: pgrx::pg_sys::ItemIdData) -> u32 {
    let x: u32 = unsafe { std::mem::transmute(x) };
    (x >> 15) & 0b11
}

// Emulate unstable library feature `vec_deque_pop_if`.
// See https://github.com/rust-lang/rust/issues/135889.

fn vec_deque_pop_front_if<T>(
    this: &mut VecDeque<T>,
    predicate: impl FnOnce(&T) -> bool,
) -> Option<T> {
    let first = this.front()?;
    if predicate(first) {
        this.pop_front()
    } else {
        None
    }
}

// Emulate unstable library feature `box_vec_non_null`.
// See https://github.com/rust-lang/rust/issues/130364.

#[allow(dead_code)]
#[must_use]
fn box_into_non_null<T>(b: Box<T>) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(Box::into_raw(b)) }
}

#[must_use]
unsafe fn box_from_non_null<T>(ptr: NonNull<T>) -> Box<T> {
    unsafe { Box::from_raw(ptr.as_ptr()) }
}
