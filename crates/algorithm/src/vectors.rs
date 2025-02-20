use crate::operator::*;
use crate::pipe::Pipe;
use crate::tuples::*;
use crate::{Page, PageGuard, RelationRead, RelationWrite, tape};
use std::num::NonZeroU64;
use vector::VectorOwned;

pub fn access_1<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    index: impl RelationRead,
    mean: IndexPointer,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let vector_guard = index.read(mean.0);
        let vector_tuple = vector_guard
            .get(mean.1)
            .expect("data corruption")
            .pipe(read_tuple::<VectorTuple<O::Vector>>);
        if vector_tuple.payload().is_some() {
            panic!("data corruption");
        }
        result.push(vector_tuple.elements());
        cursor = vector_tuple.metadata_or_pointer();
    }
    result.finish(cursor.expect("data corruption"))
}

pub fn access_0<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    index: impl RelationRead,
    mean: IndexPointer,
    payload: NonZeroU64,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let vector_guard = index.read(mean.0);
        let vector_tuple = vector_guard
            .get(mean.1)?
            .pipe(read_tuple::<VectorTuple<O::Vector>>);
        if vector_tuple.payload().is_none() {
            panic!("data corruption");
        }
        if vector_tuple.payload() != Some(payload) {
            return None;
        }
        result.push(vector_tuple.elements());
        cursor = vector_tuple.metadata_or_pointer();
    }
    Some(result.finish(cursor.ok()?))
}

pub fn append<O: Operator>(
    index: impl RelationWrite,
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZeroU64,
) -> IndexPointer {
    fn append(index: impl RelationWrite, first: u32, bytes: &[u8]) -> IndexPointer {
        if let Some(mut write) = index.search(bytes.len()) {
            let i = write.alloc(bytes).unwrap();
            return pair_to_pointer((write.id(), i));
        }
        tape::append(index, first, bytes, true)
    }
    let (metadata, slices) = O::Vector::vector_split(vector);
    let mut chain = Ok(metadata);
    for i in (0..slices.len()).rev() {
        let bytes = serialize::<VectorTuple<O::Vector>>(&match chain {
            Ok(metadata) => VectorTuple::_0 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                metadata,
            },
            Err(pointer) => VectorTuple::_1 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                pointer,
            },
        });
        chain = Err(append(index.clone(), vectors_first, &bytes));
    }
    chain.err().unwrap()
}
