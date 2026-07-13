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
// Copyright (c) 2025-2026 TensorChord Inc.

use super::MaxsimBuilder;
use super::candidate::{HeapKey, PageCandidate};
use super::external::{
    CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorSourceBinding,
    ExternalTensorStorage, resolve_external_tensor_source, validate_descriptor,
};
use super::gpu::{GpuExternalTileMaxsimBackend, UnixSocketTransport};
use super::rerank::RerankError;
use crate::index::fetcher::{
    Fetcher, FilterableTuple, HeapFetcher, Tuple, TupleAttribute, ctid_to_key,
};
use crate::index::gucs::{self, PostgresMaxsimBackend};
use crate::index::scanners::SearchBuilder;
use crate::index::storage::PostgresRelation;
use crate::index::vchordrq::opclass::{Opfamily, opfamily};
use crate::index::vchordrq::scanners::SearchOptions;
use crate::recorder::DefaultRecorder;
use distance::Distance;
use pgrx::datum::{DatumWithOid, FromDatum};
use pgrx::iter::TableIterator;
use pgrx::{AnyArray, IntoDatum, name};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

const MAX_EXPLICIT_CANDIDATES: i32 = 65_536;

/// Restricted Phase 3B search surface. Candidate generation reads only the
/// named index. Descriptor projection happens later through SPI, under the
/// caller's normal SELECT privileges and active MVCC snapshot.
#[pgrx::pg_extern(sql = "")]
fn _vchordrq_maxsim_search_external(
    index_oid: pgrx::pg_sys::Oid,
    query: AnyArray,
    candidate_limit: i32,
    top_k: i32,
) -> TableIterator<'static, (name!(public_id, i64), name!(similarity, f32))> {
    let rows = execute_external_search(index_oid, query, candidate_limit, top_k)
        .unwrap_or_else(|error| pgrx::error!("{error}"));
    TableIterator::new(rows)
}

fn execute_external_search(
    index_oid: pgrx::pg_sys::Oid,
    query: AnyArray,
    candidate_limit: i32,
    top_k: i32,
) -> Result<std::vec::IntoIter<(i64, f32)>, RerankError> {
    validate_search_limits(candidate_limit, top_k)?;
    if !matches!(gucs::vchordrq_maxsim_backend(), PostgresMaxsimBackend::Gpu) {
        return Err(RerankError::Configuration(
            "external MaxSim search currently requires vchordrq.maxsim_backend = 'gpu'",
        ));
    }

    let binding = resolve_external_tensor_source(index_oid)?;
    let index_lock = RelationLock::open(index_oid, pgrx::pg_sys::AccessShareLock as _)?;
    let heap_lock = RelationLock::open(binding.heap_oid, pgrx::pg_sys::AccessShareLock as _)?;
    let descriptor_lock = binding
        .descriptor_oid
        .map(|oid| RelationLock::open(oid, pgrx::pg_sys::AccessShareLock as _))
        .transpose()?;
    if binding.index_oid != index_lock.oid() || binding.heap_oid != heap_lock.oid() {
        return Err(RerankError::Registry(
            "registered MaxSim tensor source changed during execution".into(),
        ));
    }
    if descriptor_lock.as_ref().map(RelationLock::oid) != binding.descriptor_oid {
        return Err(RerankError::Registry(
            "registered descriptor relation changed during execution".into(),
        ));
    }

    let opfamily = unsafe { opfamily(index_lock.raw()) };
    if !matches!(opfamily, Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim) {
        return Err(RerankError::UnsupportedTensorKind);
    }
    let indexed_type = unsafe { pgrx::pg_sys::get_atttype(index_oid, 1) };
    if indexed_type != query.oid() {
        return Err(RerankError::TensorMismatch);
    }
    let query_vectors =
        unsafe { opfamily.input_vectors(query.datum()) }.ok_or(RerankError::TensorMismatch)?;

    // The registry resolution happens before the index read, and this
    // privilege-only SELECT is planned/executed before any candidate CTID is
    // generated. It fails early when the caller cannot project the registered
    // descriptor columns. The same query shape is used for the actual fetch.
    preflight_descriptor_access(&binding)?;

    let candidates = generate_candidates(
        index_lock.raw(),
        opfamily,
        query.datum(),
        candidate_limit as u32,
    )?;
    let resolved =
        resolve_visible_candidates(index_lock.raw(), heap_lock.raw(), candidates.into_iter())?;
    let (mut source, public_ids) = load_visible_descriptors(&binding, &resolved)?;
    let visible_candidates = resolved
        .into_iter()
        .filter_map(|resolved| {
            public_ids
                .contains_key(&resolved.candidate.heap_key)
                .then_some(resolved.candidate)
        })
        .collect::<Vec<_>>();
    let endpoint = gucs::vchordrq_maxsim_gpu_endpoint()
        .map(|endpoint| endpoint.to_string_lossy().into_owned())
        .unwrap_or_default();
    let transport = UnixSocketTransport::new(endpoint);
    let mut backend = GpuExternalTileMaxsimBackend::new(
        transport,
        binding.model_contract_id,
        Duration::from_millis(gucs::vchordrq_maxsim_gpu_timeout_ms() as u64),
        gucs::vchordrq_maxsim_gpu_max_batch_tokens() as usize,
        gucs::vchordrq_maxsim_gpu_max_batch_bytes() as usize,
    );
    let mut candidate_iter = visible_candidates.into_iter();
    let exact = backend.rerank(&query_vectors, &mut candidate_iter, &mut source)?;
    let mut rows = exact
        .map(|result| {
            let public_id = public_ids.get(&result.heap_key).copied().ok_or_else(|| {
                RerankError::Protocol("sidecar result has no visible public ID".into())
            })?;
            Ok((result.distance, public_id))
        })
        .collect::<Result<Vec<_>, RerankError>>()?;
    rows.sort_unstable_by(|(left_distance, left_id), (right_distance, right_id)| {
        left_distance
            .cmp(right_distance)
            .then_with(|| left_id.cmp(right_id))
    });
    rows.truncate(top_k as usize);
    Ok(rows
        .into_iter()
        .map(|(distance, public_id)| (public_id, -distance.to_f32()))
        .collect::<Vec<_>>()
        .into_iter())
}

#[derive(Clone, Copy)]
struct ResolvedCandidate {
    candidate: PageCandidate,
    current_ctid: pgrx::pg_sys::ItemPointerData,
}

fn resolve_visible_candidates(
    index_relation: pgrx::pg_sys::Relation,
    heap_relation: pgrx::pg_sys::Relation,
    candidates: impl Iterator<Item = PageCandidate>,
) -> Result<Vec<ResolvedCandidate>, RerankError> {
    let snapshot = unsafe { pgrx::pg_sys::GetActiveSnapshot() };
    if snapshot.is_null() {
        return Err(RerankError::Configuration(
            "external MaxSim search requires an active MVCC snapshot",
        ));
    }
    if unsafe { (*snapshot).snapshot_type } != pgrx::pg_sys::SnapshotType::SNAPSHOT_MVCC {
        return Err(RerankError::Configuration(
            "external MaxSim search requires an MVCC snapshot",
        ));
    }

    let mut fetcher =
        unsafe { HeapFetcher::new_standalone(index_relation, heap_relation, snapshot) };
    let mut resolved = Vec::new();
    for candidate in candidates {
        pgrx::check_for_interrupts!();
        if let Some(tuple) = fetcher.fetch(candidate.heap_key) {
            resolved.push(ResolvedCandidate {
                candidate,
                current_ctid: tuple.ctid(),
            });
        }
    }
    Ok(resolved)
}

fn validate_search_limits(candidate_limit: i32, top_k: i32) -> Result<(), RerankError> {
    if !(1..=MAX_EXPLICIT_CANDIDATES).contains(&candidate_limit) {
        return Err(RerankError::Configuration(
            "candidate_limit must be between 1 and 65536",
        ));
    }
    if top_k <= 0 || top_k > candidate_limit {
        return Err(RerankError::Configuration(
            "top_k must be positive and no greater than candidate_limit",
        ));
    }
    Ok(())
}

fn generate_candidates(
    index_relation: pgrx::pg_sys::Relation,
    opfamily: Opfamily,
    query: pgrx::pg_sys::Datum,
    candidate_limit: u32,
) -> Result<Vec<PageCandidate>, RerankError> {
    let index = unsafe { PostgresRelation::<vchordrq::Opaque>::new(index_relation) };
    let mut builder = MaxsimBuilder::new(opfamily);
    unsafe { builder.add(3, Some(query)) };
    let options = SearchOptions {
        epsilon: unsafe { gucs::vchordrq_epsilon(index_relation) },
        probes: unsafe { gucs::vchordrq_probes(index_relation) },
        max_scan_tuples: None,
        maxsim_refine: gucs::vchordrq_maxsim_refine(index_relation),
        maxsim_threshold: gucs::vchordrq_maxsim_threshold(index_relation),
        maxsim_candidate_limit: Some(candidate_limit),
        maxsim_backend: PostgresMaxsimBackend::CoarseOnly,
        maxsim_gpu_endpoint: None,
        maxsim_gpu_timeout_ms: 1,
        maxsim_gpu_max_batch_tokens: 1,
        maxsim_gpu_max_batch_bytes: 1,
        io_search: gucs::vchordrq_io_search(),
        io_rerank: gucs::vchordrq_io_rerank(),
        // General same-relation quals would require the optional Phase 3C
        // CustomScan. The restricted function applies PostgreSQL row
        // visibility during the following SPI descriptor fetch.
        prefilter: false,
    };
    let bump = bumpalo::Bump::new();
    let recorder = DefaultRecorder {
        enable: false,
        rate: None,
        max_records: 0,
        index: unsafe { (*index_relation).rd_id.to_u32() },
    };
    let candidates = builder
        .build(&index, options, NoHeapFetcher, &bump, recorder)
        .map(|(distance, heap_key, _)| PageCandidate {
            approximate_distance: Distance::from_f32(distance),
            heap_key,
        })
        .collect();
    Ok(candidates)
}

fn preflight_descriptor_access(binding: &ExternalTensorSourceBinding) -> Result<(), RerankError> {
    let query = descriptor_query(binding, true)?;
    pgrx::spi::Spi::connect(|client| {
        let privilege = client
            .prepare(
                "SELECT pg_catalog.has_table_privilege($1, 'SELECT') AS allowed",
                &pgrx::oids_of![pgrx::pg_sys::Oid],
            )
            .map_err(registry_error)?;
        for relation_oid in std::iter::once(binding.heap_oid).chain(binding.descriptor_oid) {
            let allowed = client
                .select(&privilege, Some(1), &[relation_oid.into()])
                .map_err(registry_error)?
                .first()
                .get_by_name::<bool, _>("allowed")
                .map_err(registry_error)?
                .unwrap_or(false);
            if !allowed {
                return Err(RerankError::Registry(
                    "table-level SELECT privilege is required for every external MaxSim relation"
                        .into(),
                ));
            }
        }
        client
            .select(query.as_str(), Some(1), &[])
            .map(|_| ())
            .map_err(registry_error)
    })
}

fn load_visible_descriptors(
    binding: &ExternalTensorSourceBinding,
    candidates: &[ResolvedCandidate],
) -> Result<(MaterializedDescriptorSource, BTreeMap<HeapKey, i64>), RerankError> {
    if candidates.is_empty() {
        return Ok((MaterializedDescriptorSource::default(), BTreeMap::new()));
    }
    let mut candidates_by_key = BTreeMap::new();
    for resolved in candidates {
        if candidates_by_key
            .insert(ctid_to_key(resolved.current_ctid), resolved.candidate)
            .is_some()
        {
            return Err(RerankError::Registry(
                "multiple index candidates resolved to the same visible CTID".into(),
            ));
        }
    }
    let ctids = candidates
        .iter()
        .map(|resolved| resolved.current_ctid)
        .collect::<Vec<_>>();
    let query = descriptor_query(binding, false)?;
    let (descriptors, public_ids) = pgrx::spi::Spi::connect(|client| {
        let tid_array_oid = unsafe { pgrx::pg_sys::get_array_type(pgrx::pg_sys::TIDOID) };
        let prepared = client
            .prepare(
                query.as_str(),
                &[
                    pgrx::pg_sys::PgOid::from(tid_array_oid),
                    pgrx::pg_sys::PgOid::from(pgrx::pg_sys::TEXTOID),
                ],
            )
            .map_err(registry_error)?;
        let args: [DatumWithOid<'_>; 2] = [ctids.into(), binding.model_contract_id.clone().into()];
        let rows = client
            .select(&prepared, Some(candidates.len() as _), &args)
            .map_err(registry_error)?;
        let mut descriptors = BTreeMap::new();
        let mut public_ids = BTreeMap::new();
        let mut unique_public_ids = BTreeSet::new();
        for row in rows {
            let ctid = required_heap_column::<pgrx::pg_sys::ItemPointerData>(&row, "heap_tid")?;
            let heap_key = ctid_to_key(ctid);
            let candidate = candidates_by_key.get(&heap_key).copied().ok_or_else(|| {
                RerankError::Registry("descriptor query returned an unknown CTID".into())
            })?;
            let public_id = required_heap_column::<i64>(&row, "public_id")?;
            if !unique_public_ids.insert(public_id) {
                return Err(RerankError::InvalidDescriptor(
                    "public IDs are not unique in the visible candidate batch",
                ));
            }
            let descriptor = validate_descriptor(
                candidate,
                public_id,
                required_heap_column::<String>(&row, "tensor_ref")?,
                required_heap_column::<i32>(&row, "tensor_rows")?,
                required_heap_column::<i32>(&row, "tensor_dimension")?,
                required_heap_column::<String>(&row, "tensor_dtype")?,
                required_heap_column::<String>(&row, "tensor_checksum")?,
            )?;
            if descriptors.insert(candidate.heap_key, descriptor).is_some() {
                return Err(RerankError::Registry(
                    "descriptor query returned a duplicate CTID".into(),
                ));
            }
            public_ids.insert(candidate.heap_key, public_id);
        }
        Ok((descriptors, public_ids))
    })?;
    Ok((MaterializedDescriptorSource(descriptors), public_ids))
}

fn descriptor_query(
    binding: &ExternalTensorSourceBinding,
    preflight: bool,
) -> Result<String, RerankError> {
    let heap_relation = relation_name(binding.heap_oid)?;
    let names = &binding.column_names;
    let model_contract = pgrx::spi::quote_identifier(&names.model_contract);
    let public_id = pgrx::spi::quote_identifier(&names.public_id);
    let tensor_ref = pgrx::spi::quote_identifier(&names.tensor_ref);
    let tensor_rows = pgrx::spi::quote_identifier(&names.tensor_rows);
    let tensor_dimension = pgrx::spi::quote_identifier(&names.tensor_dimension);
    let tensor_dtype = pgrx::spi::quote_identifier(&names.tensor_dtype);
    let tensor_checksum = pgrx::spi::quote_identifier(&names.tensor_checksum);
    let predicate = if preflight {
        "false".to_string()
    } else {
        format!("h.ctid = ANY($1) AND h.{model_contract} = $2")
    };
    match binding.storage {
        ExternalTensorStorage::SameHeap => Ok(format!(
            "SELECT h.ctid AS heap_tid,
                    h.{model_contract} AS model_contract,
                    h.{public_id} AS public_id,
                    h.{tensor_ref} AS tensor_ref,
                    h.{tensor_rows} AS tensor_rows,
                    h.{tensor_dimension} AS tensor_dimension,
                    h.{tensor_dtype} AS tensor_dtype,
                    h.{tensor_checksum} AS tensor_checksum
               FROM ONLY {heap_relation} AS h
              WHERE {predicate}"
        )),
        ExternalTensorStorage::DescriptorRelation => {
            let descriptor_oid = binding.descriptor_oid.ok_or_else(|| {
                RerankError::Registry("registered descriptor relation is missing".into())
            })?;
            let descriptor_relation = relation_name(descriptor_oid)?;
            let descriptor_public_id = pgrx::spi::quote_identifier(
                names.descriptor_public_id.as_deref().ok_or_else(|| {
                    RerankError::Registry(
                        "registered descriptor public ID column is missing".into(),
                    )
                })?,
            );
            Ok(format!(
                "SELECT h.ctid AS heap_tid,
                        h.{model_contract} AS model_contract,
                        h.{public_id} AS public_id,
                        d.{tensor_ref} AS tensor_ref,
                        d.{tensor_rows} AS tensor_rows,
                        d.{tensor_dimension} AS tensor_dimension,
                        d.{tensor_dtype} AS tensor_dtype,
                        d.{tensor_checksum} AS tensor_checksum
                   FROM ONLY {heap_relation} AS h
                   LEFT JOIN ONLY {descriptor_relation} AS d
                     ON d.{descriptor_public_id} = h.{public_id}
                  WHERE {predicate}"
            ))
        }
    }
}

fn relation_name(relation_oid: pgrx::pg_sys::Oid) -> Result<String, RerankError> {
    pgrx::spi::Spi::connect(|client| {
        let prepared = client
            .prepare(
                "SELECT n.nspname::text AS schema_name, c.relname::text AS relation_name
                   FROM pg_catalog.pg_class AS c
                   JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                  WHERE c.oid = $1",
                &pgrx::oids_of![pgrx::pg_sys::Oid],
            )
            .map_err(registry_error)?;
        let rows = client
            .select(&prepared, Some(1), &[relation_oid.into()])
            .map_err(registry_error)?;
        if rows.is_empty() {
            return Err(RerankError::Registry(
                "registered relation disappeared".into(),
            ));
        }
        let row = rows.first();
        Ok(pgrx::spi::quote_qualified_identifier(
            required_column::<String>(&row, "schema_name")?,
            required_column::<String>(&row, "relation_name")?,
        ))
    })
}

fn required_column<T>(row: &pgrx::spi::SpiTupleTable<'_>, name: &str) -> Result<T, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name)
        .map_err(registry_error)?
        .ok_or(RerankError::InvalidDescriptor(
            "descriptor query returned NULL",
        ))
}

fn required_heap_column<T>(
    row: &pgrx::spi::SpiHeapTupleData<'_>,
    name: &str,
) -> Result<T, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name)
        .map_err(registry_error)?
        .ok_or(RerankError::InvalidDescriptor(
            "descriptor query returned NULL",
        ))
}

fn registry_error(error: impl std::fmt::Display) -> RerankError {
    RerankError::Registry(error.to_string())
}

#[derive(Default)]
struct MaterializedDescriptorSource(BTreeMap<HeapKey, ExternalTensorDescriptor>);

impl CandidateTensorDescriptorSource for MaterializedDescriptorSource {
    fn fetch(
        &mut self,
        candidate: PageCandidate,
    ) -> Result<Option<ExternalTensorDescriptor>, RerankError> {
        Ok(self.0.remove(&candidate.heap_key))
    }
}

struct RelationLock {
    raw: pgrx::pg_sys::Relation,
    lockmode: pgrx::pg_sys::LOCKMODE,
}

impl RelationLock {
    fn open(oid: pgrx::pg_sys::Oid, lockmode: pgrx::pg_sys::LOCKMODE) -> Result<Self, RerankError> {
        let raw = unsafe { pgrx::pg_sys::relation_open(oid, lockmode) };
        if raw.is_null() {
            return Err(RerankError::Registry("relation open returned NULL".into()));
        }
        Ok(Self { raw, lockmode })
    }

    fn raw(&self) -> pgrx::pg_sys::Relation {
        self.raw
    }

    fn oid(&self) -> pgrx::pg_sys::Oid {
        unsafe { (*self.raw).rd_id }
    }
}

impl Drop for RelationLock {
    fn drop(&mut self) {
        unsafe { pgrx::pg_sys::relation_close(self.raw, self.lockmode) };
    }
}

struct NoHeapFetcher;
struct NoHeapTuple;

impl Tuple for NoHeapTuple {
    fn build(&mut self) -> (&[pgrx::pg_sys::Datum; 32], &[bool; 32]) {
        unreachable!("restricted external candidate generation must not read the heap")
    }

    fn attribute(&mut self, _attnum: i16) -> Option<TupleAttribute> {
        unreachable!("restricted external candidate generation must not read the heap")
    }
}

impl FilterableTuple for NoHeapTuple {
    fn filter(&mut self) -> bool {
        unreachable!("restricted external candidate generation must not read the heap")
    }
}

impl Fetcher for NoHeapFetcher {
    type Tuple<'a> = NoHeapTuple;

    fn fetch(&mut self, _key: HeapKey) -> Option<Self::Tuple<'_>> {
        unreachable!("restricted external candidate generation must not read the heap")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_search_limits_are_bounded() {
        assert!(validate_search_limits(256, 10).is_ok());
        for (candidates, top_k) in [(0, 1), (65_537, 1), (1, 0), (10, 11)] {
            assert!(matches!(
                validate_search_limits(candidates, top_k),
                Err(RerankError::Configuration(_))
            ));
        }
    }

    #[test]
    fn materialized_source_only_returns_visible_descriptors_once() {
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 1],
        };
        let descriptor = validate_descriptor(
            candidate,
            42,
            "s3://immutable/tensor".into(),
            1,
            2,
            "float16".into(),
            format!("sha256:{}", "a".repeat(64)),
        )
        .unwrap();
        let mut source =
            MaterializedDescriptorSource(BTreeMap::from([(candidate.heap_key, descriptor)]));
        assert_eq!(source.fetch(candidate).unwrap().unwrap().public_id, 42);
        assert!(source.fetch(candidate).unwrap().is_none());
    }
}
