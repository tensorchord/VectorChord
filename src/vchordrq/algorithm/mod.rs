pub mod build;
pub mod insert;
pub mod prewarm;
pub mod rabitq;
pub mod scan;
pub mod tuples;
pub mod vacuum;
pub mod vectors;

use crate::postgres::Page;
use std::ops::{Deref, DerefMut};

pub trait PageGuard {
    fn id(&self) -> u32;
}

pub trait RelationRead {
    type ReadGuard<'a>: PageGuard + Deref<Target = Page>
    where
        Self: 'a;
    fn read(&self, id: u32) -> Self::ReadGuard<'_>;
}

pub trait RelationWrite: RelationRead {
    type WriteGuard<'a>: PageGuard + DerefMut<Target = Page>
    where
        Self: 'a;
    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_>;
    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>>;
}

impl PageGuard for crate::postgres::BufferReadGuard {
    fn id(&self) -> u32 {
        self.id()
    }
}

impl PageGuard for crate::postgres::BufferWriteGuard {
    fn id(&self) -> u32 {
        self.id()
    }
}

impl RelationRead for crate::postgres::Relation {
    type ReadGuard<'a> = crate::postgres::BufferReadGuard;

    fn read(&self, id: u32) -> Self::ReadGuard<'_> {
        self.read(id)
    }
}

impl RelationWrite for crate::postgres::Relation {
    type WriteGuard<'a> = crate::postgres::BufferWriteGuard;

    fn write(&self, id: u32, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.write(id, tracking_freespace)
    }

    fn extend(&self, tracking_freespace: bool) -> Self::WriteGuard<'_> {
        self.extend(tracking_freespace)
    }

    fn search(&self, freespace: usize) -> Option<Self::WriteGuard<'_>> {
        self.search(freespace)
    }
}
