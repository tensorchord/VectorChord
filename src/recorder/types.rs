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

use crate::recorder::worker::push;
use rand::Rng;
use std::cell::RefMut;

pub trait Recorder {
    fn is_enabled(&self) -> bool;
    fn send(&self, sample: &str);
}

#[derive(Debug)]
pub struct DefaultRecorder {
    pub enable: bool,
    pub rate: Option<f64>,
    pub max_records: u32,
    pub index: u32,
}

pub struct PgRefCell<T>(std::cell::RefCell<T>);

unsafe impl<T: Send> Send for PgRefCell<T> {}
unsafe impl<T: Send> Sync for PgRefCell<T> {}

impl<T> PgRefCell<T> {
    pub const fn new(x: T) -> Self {
        Self(std::cell::RefCell::new(x))
    }
    pub fn borrow_mut(&self) -> RefMut<'_, T> {
        assert!(
            crate::is_main(),
            "cannot borrow the value outside main thread"
        );
        self.0.borrow_mut()
    }
}

impl Recorder for DefaultRecorder {
    fn is_enabled(&self) -> bool {
        self.enable
    }
    fn send(&self, sample: &str) {
        if let Some(rate) = self.rate {
            let mut rng = rand::rng();
            if rng.random_bool(rate) {
                push(self.index, sample, self.max_records);
            }
        }
    }
}
