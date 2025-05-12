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

// Use stable language features as an alternative to `closure_lifetime_binder`.
// See https://github.com/rust-lang/rust/issues/97362.

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&'a mut A, &'a B) -> R,
{
    f
}

#[inline(always)]
pub fn id_1<F, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(A, (&'a B, &'a B, &'a B, &'a B)) -> R,
{
    f
}

#[inline(always)]
pub fn id_2<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(A, B, C, &'a D) -> R,
{
    f
}

#[inline(always)]
pub fn id_3<F, T, A: ?Sized>(f: F) -> F
where
    T: crate::RelationWrite,
    F: for<'a> Fn(&'a T, A) -> T::WriteGuard<'a>,
{
    f
}

#[inline(always)]
pub fn id_4<F, T, A: ?Sized, B: ?Sized, R: ?Sized>(f: F) -> F
where
    T: crate::RelationRead,
    F: FnMut(A, Vec<T::ReadGuard<'_>>, B) -> R,
{
    f
}
