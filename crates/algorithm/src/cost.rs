use crate::tuples::{MetaTuple, WithReader};
use crate::{Page, RelationRead};
use std::num::NonZero;

pub struct Cost {
    pub dims: u32,
    pub is_residual: bool,
    pub cells: Vec<NonZero<u32>>,
}

#[must_use]
pub fn cost(index: impl RelationRead) -> Cost {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dims = meta_tuple.dims();
    let is_residual = meta_tuple.is_residual();
    let cells = meta_tuple.cells();
    drop(meta_guard);

    Cost {
        dims,
        is_residual,
        cells: cells.into_iter().map_while(NonZero::new).collect(),
    }
}
