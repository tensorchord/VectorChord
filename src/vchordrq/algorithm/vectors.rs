use super::tuples::Vector;
use super::RelationRead;
use crate::vchordrq::algorithm::tuples::VectorTuple;
use base::distance::Distance;
use base::distance::DistanceKind;

pub fn vector_dist<V: Vector>(
    relation: impl RelationRead,
    vector: V::Borrowed<'_>,
    mean: (u32, u16),
    payload: Option<u64>,
    for_distance: Option<DistanceKind>,
    for_original: bool,
) -> Option<(Option<Distance>, Option<V>)> {
    if for_distance.is_none() && !for_original && payload.is_none() {
        return Some((None, None));
    }
    let (left_metadata, slices) = V::vector_split(vector);
    let mut cursor = Ok(mean);
    let mut result = for_distance.map(|x| V::distance_begin(x));
    let mut original = Vec::new();
    for i in 0..slices.len() {
        let Ok(mean) = cursor else {
            // fails consistency check
            return None;
        };
        let vector_guard = relation.read(mean.0);
        let Some(vector_tuple) = vector_guard.get(mean.1) else {
            // fails consistency check
            return None;
        };
        let vector_tuple = unsafe { rkyv::archived_root::<VectorTuple<V>>(vector_tuple) };
        if vector_tuple.payload != payload {
            // fails consistency check
            return None;
        }
        if let Some(result) = result.as_mut() {
            V::distance_next(result, slices[i], &vector_tuple.slice);
        }
        if for_original {
            original.extend_from_slice(&vector_tuple.slice);
        }
        cursor = match &vector_tuple.chain {
            rkyv::result::ArchivedResult::Ok(x) => Ok(*x),
            rkyv::result::ArchivedResult::Err(x) => Err(V::metadata_from_archived(x)),
        };
    }
    let Err(right_metadata) = cursor else {
        panic!("data corruption")
    };
    Some((
        result.map(|r| Distance::from_f32(V::distance_end(r, left_metadata, right_metadata))),
        for_original.then(|| V::vector_merge(right_metadata, &original)),
    ))
}

pub fn vector_warm<V: Vector>(relation: impl RelationRead, mean: (u32, u16)) {
    let mut cursor = Ok(mean);
    while let Ok(mean) = cursor {
        let vector_guard = relation.read(mean.0);
        let Some(vector_tuple) = vector_guard.get(mean.1) else {
            // fails consistency check
            return;
        };
        let vector_tuple = unsafe { rkyv::archived_root::<VectorTuple<V>>(vector_tuple) };
        if vector_tuple.payload.is_some() {
            // fails consistency check
            return;
        }
        cursor = match &vector_tuple.chain {
            rkyv::result::ArchivedResult::Ok(x) => Ok(*x),
            rkyv::result::ArchivedResult::Err(x) => Err(V::metadata_from_archived(x)),
        };
    }
}
