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

use crate::collector::{Operator, Query, QueryCollectorMaster};
use crate::index::vchordg::opclass::Opfamily as OpfamilyG;
use crate::index::vchordrq::opclass::Opfamily as OpfamilyRQ;
use rand::Rng;
use vchordg::types::OwnedVector as OwnedVectorG;
use vchordrq::types::OwnedVector as OwnedVectorRQ;

pub trait CollectorSender {
    fn send(&self, vector: UniVec);
}

#[derive(Debug)]
pub enum UniOp {
    RQ(OpfamilyRQ),
    G(OpfamilyG),
}

impl UniOp {
    fn operator(&self) -> Option<Operator> {
        match self {
            UniOp::RQ(OpfamilyRQ::HalfvecCosine)
            | UniOp::RQ(OpfamilyRQ::VectorCosine)
            | UniOp::G(OpfamilyG::HalfvecCosine)
            | UniOp::G(OpfamilyG::VectorCosine) => Some(Operator::Cosine),
            UniOp::RQ(OpfamilyRQ::HalfvecIp)
            | UniOp::RQ(OpfamilyRQ::VectorIp)
            | UniOp::G(OpfamilyG::HalfvecIp)
            | UniOp::G(OpfamilyG::VectorIp) => Some(Operator::Dot),
            UniOp::RQ(OpfamilyRQ::HalfvecL2)
            | UniOp::RQ(OpfamilyRQ::VectorL2)
            | UniOp::G(OpfamilyG::HalfvecL2)
            | UniOp::G(OpfamilyG::VectorL2) => Some(Operator::L2),
            UniOp::RQ(OpfamilyRQ::VectorMaxsim) | UniOp::RQ(OpfamilyRQ::HalfvecMaxsim) => None,
        }
    }
}

#[derive(Debug)]
pub enum UniVec {
    RQ(OwnedVectorRQ),
    G(OwnedVectorG),
}

impl From<UniVec> for Vec<f32> {
    fn from(vec: UniVec) -> Self {
        match vec {
            UniVec::RQ(OwnedVectorRQ::Vecf32(v)) | UniVec::G(OwnedVectorG::Vecf32(v)) => {
                v.into_vec()
            }
            UniVec::RQ(OwnedVectorRQ::Vecf16(v)) | UniVec::G(OwnedVectorG::Vecf16(v)) => {
                v.into_vec().into_iter().map(|f| f.to_f32()).collect()
            }
        }
    }
}

#[derive(Debug)]
pub struct DefaultSender {
    pub send_prob: Option<f64>,
    pub database_oid: u32,
    pub table_oid: u32,
    pub index_oid: u32,
    pub opfamily: UniOp,
}

impl CollectorSender for DefaultSender {
    fn send(&self, vector: UniVec) {
        if let Some(rate) = self.send_prob {
            let mut rng = rand::rng();
            if rng.random_bool(rate) {
                let query = Query::new(
                    self.database_oid,
                    self.table_oid,
                    self.index_oid,
                    self.opfamily.operator(),
                    vector.into(),
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
