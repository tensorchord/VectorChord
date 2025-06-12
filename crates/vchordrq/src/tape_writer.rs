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

use crate::tape::TapeWriter;
use crate::tuples::*;
use crate::{Branch, Opaque};
use algo::{Page, RelationWrite};
use rabitq::packing::{any_pack, padding_pack};
use std::num::NonZero;

pub struct DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, DirectoryTuple>,
}

impl<'a, R> DirectoryTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
        }
    }
    pub fn push(&mut self, branch: &[u32]) {
        let mut remain = branch.to_vec();
        loop {
            let freespace = self.tape.freespace();
            if DirectoryTuple::estimate_size_0(remain.len()) <= freespace as usize {
                self.tape.tape_put(DirectoryTuple::_0 { elements: remain });
                break;
            }
            if let Some(w) = DirectoryTuple::fit_1(freespace) {
                let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                self.tape.tape_put(DirectoryTuple::_1 {
                    elements: left.to_vec(),
                });
                remain = right.to_vec();
            } else {
                self.tape.tape_move();
            }
        }
    }
    pub fn into_inner(self) -> TapeWriter<'a, R, DirectoryTuple> {
        self.tape
    }
}

pub struct H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, H1Tuple>,
    branches: Vec<Branch<u32>>,
    prefetch: usize,
}

impl<'a, R> H1TapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, prefetch: usize, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            prefetch,
        }
    }
    pub fn push(&mut self, branch: Branch<u32>) {
        self.branches.push(branch);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            let mut remain =
                padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
            loop {
                let freespace = self.tape.freespace();
                if H1Tuple::estimate_size_0(self.prefetch, remain.len()) <= freespace as usize {
                    self.tape.tape_put(H1Tuple::_0 {
                        metadata: [
                            chunk.each_ref().map(|x| x.code.0.dis_u_2),
                            chunk.each_ref().map(|x| x.code.0.factor_cnt),
                            chunk.each_ref().map(|x| x.code.0.factor_ip),
                            chunk.each_ref().map(|x| x.code.0.factor_err),
                        ],
                        delta: chunk.each_ref().map(|x| x.delta),
                        prefetch: fix(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                        norm: chunk.each_ref().map(|x| x.norm),
                        head: chunk.each_ref().map(|x| x.head),
                        first: chunk.each_ref().map(|x| x.extra),
                        len: chunk.len() as _,
                        elements: remain,
                    });
                    break;
                }
                if let Some(w) = H1Tuple::fit_1(self.prefetch, freespace) {
                    let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                    self.tape.tape_put(H1Tuple::_1 {
                        elements: left.to_vec(),
                    });
                    remain = right.to_vec();
                } else {
                    self.tape.tape_move();
                }
            }
            self.branches.clear();
        }
    }
    pub fn into_inner(self) -> (TapeWriter<'a, R, H1Tuple>, Vec<Branch<u32>>) {
        (self.tape, self.branches)
    }
    pub fn flush(tape: &mut TapeWriter<'_, R, H1Tuple>, prefetch: usize, chunk: Vec<Branch<u32>>) {
        if chunk.is_empty() {
            return;
        }
        let mut remain = padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
        loop {
            let freespace = tape.freespace();
            if H1Tuple::estimate_size_0(prefetch, remain.len()) <= freespace as usize {
                tape.tape_put(H1Tuple::_0 {
                    metadata: [
                        any_pack(chunk.iter().map(|x| x.code.0.dis_u_2)),
                        any_pack(chunk.iter().map(|x| x.code.0.factor_cnt)),
                        any_pack(chunk.iter().map(|x| x.code.0.factor_ip)),
                        any_pack(chunk.iter().map(|x| x.code.0.factor_err)),
                    ],
                    delta: any_pack(chunk.iter().map(|x| x.delta)),
                    prefetch: fix(chunk.iter().map(|x| x.prefetch.as_slice())),
                    head: any_pack(chunk.iter().map(|x| x.head)),
                    norm: any_pack(chunk.iter().map(|x| x.norm)),
                    first: any_pack(chunk.iter().map(|x| x.extra)),
                    len: chunk.len() as _,
                    elements: remain,
                });
                break;
            }
            if let Some(w) = H1Tuple::fit_1(prefetch, freespace) {
                let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                tape.tape_put(H1Tuple::_1 {
                    elements: left.to_vec(),
                });
                remain = right.to_vec();
            } else {
                tape.tape_move();
            }
        }
    }
}

pub struct FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
{
    tape: TapeWriter<'a, R, FrozenTuple>,
    branches: Vec<Branch<NonZero<u64>>>,
    prefetch: usize,
}

impl<'a, R> FrozenTapeWriter<'a, R>
where
    R: RelationWrite + 'a,
    R::Page: Page<Opaque = Opaque>,
{
    pub fn create(index: &'a R, prefetch: usize, tracking_freespace: bool) -> Self {
        Self {
            tape: TapeWriter::create(index, tracking_freespace),
            branches: Vec::new(),
            prefetch,
        }
    }
    pub fn push(&mut self, branch: Branch<NonZero<u64>>) {
        self.branches.push(branch);
        if let Ok(chunk) = <&[_; 32]>::try_from(self.branches.as_slice()) {
            let mut remain =
                padding_pack(chunk.iter().map(|x| rabitq::packing::pack_to_u4(&x.code.1)));
            loop {
                let freespace = self.tape.freespace();
                if FrozenTuple::estimate_size_0(self.prefetch, remain.len()) <= freespace as usize {
                    self.tape.tape_put(FrozenTuple::_0 {
                        metadata: [
                            chunk.each_ref().map(|x| x.code.0.dis_u_2),
                            chunk.each_ref().map(|x| x.code.0.factor_cnt),
                            chunk.each_ref().map(|x| x.code.0.factor_ip),
                            chunk.each_ref().map(|x| x.code.0.factor_err),
                        ],
                        delta: chunk.each_ref().map(|x| x.delta),
                        prefetch: fix(chunk.each_ref().map(|x| x.prefetch.as_slice())),
                        head: chunk.each_ref().map(|x| x.head),
                        payload: chunk.each_ref().map(|x| Some(x.extra)),
                        elements: remain,
                    });
                    break;
                }
                if let Some(w) = FrozenTuple::fit_1(self.prefetch, freespace) {
                    let (left, right) = remain.split_at(std::cmp::min(w, remain.len()));
                    self.tape.tape_put(FrozenTuple::_1 {
                        elements: left.to_vec(),
                    });
                    remain = right.to_vec();
                } else {
                    self.tape.tape_move();
                }
            }
            self.branches.clear();
        }
    }
    pub fn into_inner(self) -> (TapeWriter<'a, R, FrozenTuple>, Vec<Branch<NonZero<u64>>>) {
        (self.tape, self.branches)
    }
}

fn fix<'a>(into_iter: impl IntoIterator<Item = &'a [u32]>) -> Vec<[u32; 32]> {
    use std::array::from_fn;
    let mut iter = into_iter.into_iter();
    let mut array: [_; 32] = from_fn(|_| iter.next().map(<[u32]>::to_vec).unwrap_or_default());
    if iter.next().is_some() {
        panic!("too many slices");
    }
    let step = array.iter().map(Vec::len).max().unwrap_or_default();
    array.iter_mut().for_each(|x| x.resize(step, u32::MAX));
    let flat = array.into_iter().flatten().collect::<Vec<_>>();
    (0..step).map(|i| from_fn(|j| flat[i * 32 + j])).collect()
}
