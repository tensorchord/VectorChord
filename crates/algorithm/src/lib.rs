use std::ops::{Deref, DerefMut};

#[repr(C, align(8))]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

#[allow(clippy::len_without_is_empty)]
pub trait Page: Sized {
    fn get_opaque(&self) -> &Opaque;
    fn get_opaque_mut(&mut self) -> &mut Opaque;
    fn len(&self) -> u16;
    fn get(&self, i: u16) -> Option<&[u8]>;
    fn get_mut(&mut self, i: u16) -> Option<&mut [u8]>;
    fn alloc(&mut self, data: &[u8]) -> Option<u16>;
    fn free(&mut self, i: u16);
    fn reconstruct(&mut self, removes: &[u16]);
    fn freespace(&self) -> u16;
}

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait RelationRead {
    type Page: Page;
    type ReadGuard<'a>: PageGuard + Deref<Target = Self::Page>
    where
        Self: 'a;
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWrite: RelationRead {
    type WriteGuard<'a>: PageGuard + DerefMut<Target = Self::Page>
    where
        Self: 'a;
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}
