use crate::postgres::Relation;
use crate::vchordrq::algorithm::tuples::VectorTuple;
use crate::vchordrq::index::utils::distance;
use base::distance::Distance;
use base::distance::DistanceKind;

pub fn vector_split(vector: &[f32]) -> Vec<&[f32]> {
    match vector.len() {
        0..=960 => vec![vector],
        961..=1280 => vec![&vector[..640], &vector[640..]],
        1281.. => vector.chunks(1920).collect(),
    }
}

pub fn vector_dist(
    relation: Relation,
    vector: &[f32],
    mean: (u32, u16),
    payload: Option<u64>,
    for_distance: Option<DistanceKind>,
    for_original: bool,
) -> Option<(Option<Distance>, Option<Vec<f32>>)> {
    if for_distance.is_none() && !for_original && payload.is_none() {
        return Some((None, None));
    }
    let slices = vector_split(vector);
    let mut cursor = Some(mean);
    let mut result = 0.0f32;
    let mut original = Vec::new();
    for i in 0..slices.len() {
        let Some(mean) = cursor else {
            // fails consistency check
            return None;
        };
        let vector_guard = relation.read(mean.0);
        let Some(vector_tuple) = vector_guard.get().get(mean.1) else {
            // fails consistency check
            return None;
        };
        let vector_tuple =
            rkyv::check_archived_root::<VectorTuple>(vector_tuple).expect("data corruption");
        if vector_tuple.payload != payload {
            // fails consistency check
            return None;
        }
        if let Some(distance_kind) = for_distance {
            result += distance(distance_kind, slices[i], &vector_tuple.slice).to_f32();
        }
        if for_original {
            original.extend_from_slice(&vector_tuple.slice);
        }
        cursor = vector_tuple.chain.as_ref().cloned();
    }
    Some((
        for_distance.map(|_| Distance::from_f32(result)),
        for_original.then_some(original),
    ))
}

pub fn vector_warm(relation: Relation, mean: (u32, u16)) {
    let mut cursor = Some(mean);
    while let Some(mean) = cursor {
        let vector_guard = relation.read(mean.0);
        let Some(vector_tuple) = vector_guard.get().get(mean.1) else {
            // fails consistency check
            return;
        };
        let vector_tuple =
            rkyv::check_archived_root::<VectorTuple>(vector_tuple).expect("data corruption");
        if vector_tuple.payload.is_some() {
            // fails consistency check
            return;
        }
        cursor = vector_tuple.chain.as_ref().cloned();
    }
}
