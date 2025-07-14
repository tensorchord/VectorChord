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

use crate::collector::{Query, QueryCollectorMaster};
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::SearchOptions;
use rand::Rng;
use vchordrq::types::OwnedVector;

pub trait CollectorSender {
    fn exec(&self, options: &SearchOptions, vector: &OwnedVector);
}

pub struct DefaultSender {
    pub database_oid: u32,
    pub table_oid: u32,
    pub index_oid: u32,
    pub opfamily: Opfamily,
}

impl CollectorSender for DefaultSender {
    fn exec(&self, options: &SearchOptions, vector: &OwnedVector) {
        if options.collect_enable {
            let mut rng = rand::rng();
            if rng.random_bool(options.collect_rate) {
                let query = Query::new(
                    self.database_oid,
                    self.table_oid,
                    self.index_oid,
                    self.opfamily,
                    vector.clone(),
                );
                if let Some(q) = query {
                    unsafe {
                        QueryCollectorMaster::push(q);
                    }
                }
            }
        }
    }
}
