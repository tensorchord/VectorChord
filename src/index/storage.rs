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

use algorithm::*;
use std::collections::VecDeque;
use std::iter::{Chain, Flatten};
use std::mem::{MaybeUninit, offset_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const _: () = assert!(
    offset_of!(pgrx::pg_sys::PageHeaderData, pd_linp) % pgrx::pg_sys::MAXIMUM_ALIGNOF as usize == 0
);

const fn size_of_contents() -> usize {
    use pgrx::pg_sys::{BLCKSZ, PageHeaderData};
    let size_of_page = BLCKSZ as usize;
    let size_of_header = offset_of!(PageHeaderData, pd_linp);
    let size_of_opaque = size_of::<Opaque>();
    size_of_page - size_of_header - size_of_opaque
}

#[repr(C, align(8))]
#[derive(Debug)]
pub struct PostgresPage {
    header: pgrx::pg_sys::PageHeaderData,
    content: [u8; size_of_contents()],
    opaque: Opaque,
}

const _: () = assert!(align_of::<PostgresPage>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);
const _: () = assert!(size_of::<PostgresPage>() == pgrx::pg_sys::BLCKSZ as usize);

impl PostgresPage {
    fn init_mut(this: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            pgrx::pg_sys::PageInit(
                this.as_mut_ptr() as pgrx::pg_sys::Page,
                pgrx::pg_sys::BLCKSZ as usize,
                size_of::<Opaque>(),
            );
            (&raw mut (*this.as_mut_ptr()).opaque).write(Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            });
        }
        let this = unsafe { MaybeUninit::assume_init_mut(this) };
        assert_eq!(offset_of!(Self, opaque), this.header.pd_special as usize);
        this
    }
    pub fn clone_into_boxed(&self) -> Box<Self> {
        let mut result = Box::new_uninit();
        unsafe {
            std::ptr::copy(self as *const Self, result.as_mut_ptr(), 1);
            result.assume_init()
        }
    }
}

impl Page for PostgresPage {
    fn get_opaque(&self) -> &Opaque {
        &self.opaque
    }
    fn get_opaque_mut(&mut self) -> &mut Opaque {
        &mut self.opaque
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
        match iid.lp_flags() {
            pgrx::pg_sys::LP_UNUSED => return None,
            pgrx::pg_sys::LP_NORMAL => (),
            pgrx::pg_sys::LP_REDIRECT => unimplemented!(),
            pgrx::pg_sys::LP_DEAD => unimplemented!(),
            _ => unreachable!(),
        }
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off && lp_off <= size_of::<Self>());
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
        assert!(offset_of!(PageHeaderData, pd_linp) <= lp_off && lp_off <= size_of::<Self>());
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
    fn clear(&mut self) {
        unsafe {
            pgrx::pg_sys::PageInit(
                (self as *mut PostgresPage as pgrx::pg_sys::Page).cast(),
                pgrx::pg_sys::BLCKSZ as usize,
                size_of::<Opaque>(),
            );
            (&raw mut self.opaque).write(Opaque {
                next: u32::MAX,
                skip: u32::MAX,
            });
        }
        assert_eq!(offset_of!(Self, opaque), self.header.pd_special as usize);
    }
}

const _: () = assert!(align_of::<Opaque>() == pgrx::pg_sys::MAXIMUM_ALIGNOF as usize);

pub struct PostgresBufferReadGuard {
    buf: i32,
    page: NonNull<PostgresPage>,
    id: u32,
}

impl PageGuard for PostgresBufferReadGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for PostgresBufferReadGuard {
    type Target = PostgresPage;

    fn deref(&self) -> &PostgresPage {
        unsafe { self.page.as_ref() }
    }
}

impl Drop for PostgresBufferReadGuard {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::UnlockReleaseBuffer(self.buf);
        }
    }
}

pub struct PostgresBufferWriteGuard {
    raw: pgrx::pg_sys::Relation,
    buf: i32,
    page: NonNull<PostgresPage>,
    state: *mut pgrx::pg_sys::GenericXLogState,
    id: u32,
    tracking_freespace: bool,
}

impl PageGuard for PostgresBufferWriteGuard {
    fn id(&self) -> u32 {
        self.id
    }
}

impl Deref for PostgresBufferWriteGuard {
    type Target = PostgresPage;

    fn deref(&self) -> &PostgresPage {
        unsafe { self.page.as_ref() }
    }
}

impl DerefMut for PostgresBufferWriteGuard {
    fn deref_mut(&mut self) -> &mut PostgresPage {
        unsafe { self.page.as_mut() }
    }
}

impl Drop for PostgresBufferWriteGuard {
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
pub struct PostgresRelation {
    raw: pgrx::pg_sys::Relation,
}

impl PostgresRelation {
    pub unsafe fn new(raw: pgrx::pg_sys::Relation) -> Self {
        Self { raw }
    }
}

impl Relation for PostgresRelation {
    type Page = PostgresPage;
}

impl RelationRead for PostgresRelation {
    type ReadGuard<'a> = PostgresBufferReadGuard;

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

impl RelationWrite for PostgresRelation {
    type WriteGuard<'a> = PostgresBufferWriteGuard;

    fn write(&self, id: u32, tracking_freespace: bool) -> PostgresBufferWriteGuard {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ForkNumber, GENERIC_XLOG_FULL_IMAGE,
                GenericXLogRegisterBuffer, GenericXLogStart, LockBuffer, ReadBufferExtended,
                ReadBufferMode,
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
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage>>(),
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
    fn extend(&self, tracking_freespace: bool) -> PostgresBufferWriteGuard {
        unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_EXCLUSIVE, ExclusiveLock, ForkNumber, GENERIC_XLOG_FULL_IMAGE,
                GenericXLogRegisterBuffer, GenericXLogStart, LockBuffer, LockRelationForExtension,
                ReadBufferExtended, ReadBufferMode, UnlockRelationForExtension,
            };
            LockRelationForExtension(self.raw, ExclusiveLock as _);
            let buf = ReadBufferExtended(
                self.raw,
                ForkNumber::MAIN_FORKNUM,
                u32::MAX,
                ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            UnlockRelationForExtension(self.raw, ExclusiveLock as _);
            LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE as _);
            let state = GenericXLogStart(self.raw);
            let mut page = NonNull::new(
                GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE as _)
                    .cast::<MaybeUninit<PostgresPage>>(),
            )
            .expect("failed to get page");
            PostgresPage::init_mut(page.as_mut());
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
    fn search(&self, freespace: usize) -> Option<PostgresBufferWriteGuard> {
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

impl RelationPrefetch for PostgresRelation {
    fn prefetch(&self, id: u32) {
        assert!(id != u32::MAX, "no such page");
        unsafe {
            use pgrx::pg_sys::PrefetchBuffer;
            PrefetchBuffer(self.raw, 0, id);
        }
    }
}

pub struct Cache<I: Iterator> {
    window: VecDeque<I::Item>,
    tail: VecDeque<u32>,
    iter: Option<I>,
}

impl<I: Iterator> Default for Cache<I> {
    fn default() -> Self {
        Self {
            window: Default::default(),
            tail: Default::default(),
            iter: Default::default(),
        }
    }
}

impl<I: Iterator> Cache<I>
where
    I::Item: Fetch,
{
    #[allow(dead_code)]
    pub fn pop_id(&mut self) -> Option<u32> {
        while self.tail.is_empty()
            && let Some(iter) = self.iter.as_mut()
            && let Some(e) = iter.next()
        {
            for id in e.fetch().iter().copied() {
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
            for id in e.fetch().iter().copied() {
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
            for id in e.fetch().iter().copied() {
                self.tail.push_back(id);
            }
            self.window.push_back(e);
        }
        self.window.pop_front()
    }
}

pub struct PostgresReadStream<I: Iterator> {
    #[cfg(feature = "pg17")]
    raw: *mut pgrx::pg_sys::ReadStream,
    // Because of `Box`'s special alias rules, `Box` cannot be used here.
    cache: NonNull<Cache<I>>,
}

#[cfg(feature = "pg17")]
impl<I: Iterator> PostgresReadStream<I>
where
    I::Item: Fetch,
{
    fn read(&mut self, fetch: &[u32]) -> impl Iterator<Item = PostgresBufferReadGuard> {
        fetch.iter().map(|_| unsafe {
            use pgrx::pg_sys::{
                BUFFER_LOCK_SHARE, BufferGetPage, LockBuffer, read_stream_next_buffer,
            };
            let buf = read_stream_next_buffer(self.raw, core::ptr::null_mut());
            LockBuffer(buf, BUFFER_LOCK_SHARE as _);
            let page = NonNull::new(BufferGetPage(buf).cast()).expect("failed to get page");
            PostgresBufferReadGuard {
                buf,
                page,
                id: pgrx::pg_sys::BufferGetBlockNumber(buf),
            }
        })
    }
}

impl<'r, I: Iterator> ReadStream<'r> for PostgresReadStream<I>
where
    I::Item: Fetch,
{
    type Relation = PostgresRelation;

    type Item = I::Item;

    type Inner =
        Chain<std::collections::vec_deque::IntoIter<I::Item>, Flatten<std::option::IntoIter<I>>>;

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next(&mut self) -> Option<(I::Item, Vec<PostgresBufferReadGuard>)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn next_if(
        &mut self,
        _predicate: impl FnOnce(&I::Item) -> bool,
    ) -> Option<(I::Item, Vec<PostgresBufferReadGuard>)> {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(feature = "pg17")]
    fn next(&mut self) -> Option<(I::Item, Vec<PostgresBufferReadGuard>)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item() } {
            let list = self.read(e.fetch()).collect();
            Some((e, list))
        } else {
            None
        }
    }

    #[cfg(feature = "pg17")]
    fn next_if(
        &mut self,
        predicate: impl FnOnce(&I::Item) -> bool,
    ) -> Option<(I::Item, Vec<PostgresBufferReadGuard>)> {
        if let Some(e) = unsafe { self.cache.as_mut().pop_item_if(predicate) } {
            let list = self.read(e.fetch()).collect();
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

impl<I: Iterator> Drop for PostgresReadStream<I> {
    fn drop(&mut self) {
        unsafe {
            let _ = std::mem::take(self.cache.as_mut());
            #[cfg(feature = "pg17")]
            if !std::thread::panicking() && pgrx::pg_sys::IsTransactionState() {
                pgrx::pg_sys::read_stream_end(self.raw);
            }
            let _ = box_from_non_null(self.cache);
        }
    }
}

impl RelationReadStream for PostgresRelation {
    type ReadStream<'s, I: Iterator>
        = PostgresReadStream<I>
    where
        I::Item: Fetch;

    #[cfg(any(feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16"))]
    fn read_stream<I: Iterator>(&self, _iter: I, _hints: Hints) -> Self::ReadStream<'_, I>
    where
        I::Item: Fetch,
    {
        panic!("read_stream is not supported on PostgreSQL versions earlier than 17.");
    }

    #[cfg(feature = "pg17")]
    fn read_stream<I: Iterator>(&self, iter: I, hints: Hints) -> Self::ReadStream<'_, I>
    where
        I::Item: Fetch,
    {
        #[pgrx::pg_guard]
        unsafe extern "C-unwind" fn callback<I: Iterator>(
            _stream: *mut pgrx::pg_sys::ReadStream,
            callback_private_data: *mut core::ffi::c_void,
            _per_buffer_data: *mut core::ffi::c_void,
        ) -> pgrx::pg_sys::BlockNumber
        where
            I::Item: Fetch,
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
        PostgresReadStream { raw, cache }
    }
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
