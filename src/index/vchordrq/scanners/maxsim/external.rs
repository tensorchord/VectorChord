// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): This software is also available under the ELv2,
// which has specific restrictions.
//
// Copyright (c) 2025-2026 TensorChord Inc.

use super::candidate::PageCandidate;
use super::rerank::RerankError;
use crate::index::fetcher::{Fetcher, FilterableTuple, Tuple};
use pgrx::datum::{FromDatum, IntoDatum};
use std::ffi::CString;

const MAX_MODEL_CONTRACT_BYTES: usize = 512;
const MAX_TENSOR_REF_BYTES: usize = 4096;
const MAX_CHECKSUM_BYTES: usize = 512;
const MAX_TENSOR_ROWS: u32 = 65_536;
const MAX_TENSOR_DIMENSION: u32 = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExternalTensorDtype {
    F32,
    F16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExternalTensorStorage {
    SameHeap,
    DescriptorRelation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExternalTensorColumns {
    pub model_contract: i16,
    pub public_id: i16,
    pub tensor_ref: i16,
    pub tensor_rows: i16,
    pub tensor_dimension: i16,
    pub tensor_dtype: i16,
    pub tensor_checksum: i16,
}

impl ExternalTensorColumns {
    pub(super) fn validate(self) -> Result<Self, RerankError> {
        let columns = [
            self.model_contract,
            self.public_id,
            self.tensor_ref,
            self.tensor_rows,
            self.tensor_dimension,
            self.tensor_dtype,
            self.tensor_checksum,
        ];
        if columns.iter().any(|column| *column <= 0) {
            return Err(RerankError::InvalidDescriptor(
                "registered attribute number is invalid",
            ));
        }
        for (index, column) in columns.iter().enumerate() {
            if columns[..index].contains(column) {
                return Err(RerankError::InvalidDescriptor(
                    "registered descriptor columns are not distinct",
                ));
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub(super) struct ExternalTensorDescriptor {
    pub candidate: PageCandidate,
    /// Stable application identifier. It stays inside PostgreSQL and is never
    /// encoded into the sidecar request.
    #[allow(
        dead_code,
        reason = "returned by the score API through a separate visible-row map"
    )]
    pub public_id: i64,
    pub tensor_ref: String,
    pub rows: u32,
    pub dimension: u32,
    pub dtype: ExternalTensorDtype,
    pub checksum: String,
}

pub(super) trait CandidateTensorDescriptorSource {
    fn fetch(
        &mut self,
        candidate: PageCandidate,
    ) -> Result<Option<ExternalTensorDescriptor>, RerankError>;
}

#[allow(dead_code, reason = "reserved for the optional Phase 3C heap source")]
pub(super) struct HeapTensorRefSource<'a, F> {
    fetcher: &'a mut F,
    model_contract_id: String,
    columns: ExternalTensorColumns,
}

#[derive(Clone, Debug)]
pub(super) struct ExternalTensorSourceBinding {
    pub index_oid: pgrx::pg_sys::Oid,
    pub heap_oid: pgrx::pg_sys::Oid,
    pub descriptor_oid: Option<pgrx::pg_sys::Oid>,
    pub storage: ExternalTensorStorage,
    pub model_contract_id: String,
    #[allow(
        dead_code,
        reason = "physical attnums are consumed by the optional Phase 3C heap source"
    )]
    pub columns: Option<ExternalTensorColumns>,
    pub column_names: ExternalTensorColumnNames,
}

#[derive(Clone, Debug)]
pub(super) struct ExternalTensorColumnNames {
    pub model_contract: String,
    pub public_id: String,
    pub descriptor_public_id: Option<String>,
    pub tensor_ref: String,
    pub tensor_rows: String,
    pub tensor_dimension: String,
    pub tensor_dtype: String,
    pub tensor_checksum: String,
}

/// Resolve the privilege-aware SQL registry boundary. Same-heap bindings also
/// resolve the physical attribute numbers consumed by [`HeapTensorRefSource`];
/// independent descriptor relations are projected later through SPI.
///
/// The SECURITY DEFINER SQL function performs ownership/SELECT checks and
/// revalidates the live index, heap relation, opclass, column types, and NOT
/// NULL constraints. This Rust layer deliberately calls that function rather
/// than reading the private registry table with extension privileges.
pub(super) fn resolve_external_tensor_source(
    index_oid: pgrx::pg_sys::Oid,
) -> Result<ExternalTensorSourceBinding, RerankError> {
    pgrx::spi::Spi::connect(|client| {
        let schema_rows = client
            .select(
                "SELECT n.nspname::text AS schema_name
                   FROM pg_catalog.pg_extension AS e
                   JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
                  WHERE e.extname = 'vchord'",
                Some(1),
                &[],
            )
            .map_err(registry_error)?;
        if schema_rows.is_empty() {
            return Err(RerankError::Registry(
                "vchord extension schema is unavailable".into(),
            ));
        }
        let schema_name = schema_rows
            .first()
            .get_by_name::<String, _>("schema_name")
            .map_err(registry_error)?
            .ok_or_else(|| RerankError::Registry("vchord extension schema is NULL".into()))?;
        let resolver = pgrx::spi::quote_qualified_identifier(
            schema_name,
            "vchordrq_maxsim_source_info".to_string(),
        );
        let query = format!(
            "SELECT registered_index::oid AS index_oid,
                    heap_relation::oid AS heap_oid,
                    descriptor_relation::oid AS descriptor_oid,
                    model_contract_id,
                    source_storage,
                    model_contract_column::text AS model_contract_column,
                    public_id_column::text AS public_id_column,
                    descriptor_public_id_column::text AS descriptor_public_id_column,
                    tensor_ref_column::text AS tensor_ref_column,
                    tensor_rows_column::text AS tensor_rows_column,
                    tensor_dim_column::text AS tensor_dim_column,
                    tensor_dtype_column::text AS tensor_dtype_column,
                    tensor_checksum_column::text AS tensor_checksum_column
               FROM {resolver}($1::regclass)"
        );
        let prepared = client
            .prepare(query.as_str(), &pgrx::oids_of![pgrx::pg_sys::Oid])
            .map_err(registry_error)?;
        let rows = client
            .select(&prepared, Some(1), &[index_oid.into()])
            .map_err(registry_error)?;
        if rows.is_empty() {
            return Err(RerankError::Registry(
                "registered MaxSim tensor source resolution returned no row".into(),
            ));
        }
        let row = rows.first();
        let resolved_index_oid = required_column::<pgrx::pg_sys::Oid>(&row, "index_oid")?;
        let heap_oid = required_column::<pgrx::pg_sys::Oid>(&row, "heap_oid")?;
        let descriptor_oid = optional_column::<pgrx::pg_sys::Oid>(&row, "descriptor_oid")?;
        let model_contract_id = required_column::<String>(&row, "model_contract_id")?;
        let storage = required_column::<String>(&row, "source_storage")?;
        if resolved_index_oid != index_oid {
            return Err(RerankError::Registry(
                "registered MaxSim tensor source resolved a different index".into(),
            ));
        }
        let storage = match storage.as_str() {
            "external_ref" if descriptor_oid.is_none() => ExternalTensorStorage::SameHeap,
            "external_relation" if descriptor_oid.is_some() => {
                ExternalTensorStorage::DescriptorRelation
            }
            "heap_array" => {
                return Err(RerankError::Registry(
                    "registered MaxSim tensor source is not external".into(),
                ));
            }
            _ => {
                return Err(RerankError::Registry(
                    "registered external MaxSim tensor source is inconsistent".into(),
                ));
            }
        };

        let column_names = ExternalTensorColumnNames {
            model_contract: required_column::<String>(&row, "model_contract_column")?,
            public_id: required_column::<String>(&row, "public_id_column")?,
            descriptor_public_id: optional_column::<String>(&row, "descriptor_public_id_column")?,
            tensor_ref: required_column::<String>(&row, "tensor_ref_column")?,
            tensor_rows: required_column::<String>(&row, "tensor_rows_column")?,
            tensor_dimension: required_column::<String>(&row, "tensor_dim_column")?,
            tensor_dtype: required_column::<String>(&row, "tensor_dtype_column")?,
            tensor_checksum: required_column::<String>(&row, "tensor_checksum_column")?,
        };
        let columns = match storage {
            ExternalTensorStorage::SameHeap => Some(
                ExternalTensorColumns {
                    model_contract: resolve_attnum(heap_oid, &column_names.model_contract)?,
                    public_id: resolve_attnum(heap_oid, &column_names.public_id)?,
                    tensor_ref: resolve_attnum(heap_oid, &column_names.tensor_ref)?,
                    tensor_rows: resolve_attnum(heap_oid, &column_names.tensor_rows)?,
                    tensor_dimension: resolve_attnum(heap_oid, &column_names.tensor_dimension)?,
                    tensor_dtype: resolve_attnum(heap_oid, &column_names.tensor_dtype)?,
                    tensor_checksum: resolve_attnum(heap_oid, &column_names.tensor_checksum)?,
                }
                .validate()?,
            ),
            ExternalTensorStorage::DescriptorRelation => {
                if column_names.descriptor_public_id.is_none() {
                    return Err(RerankError::Registry(
                        "registered descriptor relation has no public ID column".into(),
                    ));
                }
                None
            }
        };
        validate_model_contract(&model_contract_id)?;
        Ok(ExternalTensorSourceBinding {
            index_oid,
            heap_oid,
            descriptor_oid,
            storage,
            model_contract_id,
            columns,
            column_names,
        })
    })
}

fn registry_error(error: impl std::fmt::Display) -> RerankError {
    RerankError::Registry(error.to_string())
}

fn required_column<T>(row: &pgrx::spi::SpiTupleTable<'_>, name: &str) -> Result<T, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name)
        .map_err(registry_error)?
        .ok_or_else(|| RerankError::Registry(format!("registry resolver returned NULL {name}")))
}

fn optional_column<T>(
    row: &pgrx::spi::SpiTupleTable<'_>,
    name: &str,
) -> Result<Option<T>, RerankError>
where
    T: FromDatum + IntoDatum,
{
    row.get_by_name::<T, _>(name).map_err(registry_error)
}

fn resolve_attnum(heap_oid: pgrx::pg_sys::Oid, column_name: &str) -> Result<i16, RerankError> {
    let column_name = CString::new(column_name)
        .map_err(|_| RerankError::Registry("registry column contains a NUL byte".into()))?;
    let attnum = unsafe { pgrx::pg_sys::get_attnum(heap_oid, column_name.as_ptr()) };
    if attnum <= 0 {
        return Err(RerankError::Registry(
            "registered descriptor column disappeared during resolution".into(),
        ));
    }
    Ok(attnum)
}

#[allow(dead_code, reason = "reserved for the optional Phase 3C heap source")]
impl<'a, F> HeapTensorRefSource<'a, F> {
    pub(super) fn new(
        fetcher: &'a mut F,
        model_contract_id: String,
        columns: ExternalTensorColumns,
    ) -> Result<Self, RerankError> {
        validate_model_contract(&model_contract_id)?;
        Ok(Self {
            fetcher,
            model_contract_id,
            columns: columns.validate()?,
        })
    }
}

impl<F: Fetcher> CandidateTensorDescriptorSource for HeapTensorRefSource<'_, F> {
    fn fetch(
        &mut self,
        candidate: PageCandidate,
    ) -> Result<Option<ExternalTensorDescriptor>, RerankError> {
        let Some(mut tuple) = self.fetcher.fetch(candidate.heap_key) else {
            return Ok(None);
        };

        // This is the data-exfiltration boundary: no descriptor value may be
        // read before PostgreSQL has accepted the active base-relation qual.
        if !tuple.filter() {
            return Ok(None);
        }

        let model_contract = read_attribute::<String>(&mut tuple, self.columns.model_contract)?;
        if model_contract != self.model_contract_id {
            return Err(RerankError::ModelContractMismatch);
        }
        let public_id = read_attribute::<i64>(&mut tuple, self.columns.public_id)?;
        let tensor_ref = read_attribute::<String>(&mut tuple, self.columns.tensor_ref)?;
        let rows = read_attribute::<i32>(&mut tuple, self.columns.tensor_rows)?;
        let dimension = read_attribute::<i32>(&mut tuple, self.columns.tensor_dimension)?;
        let dtype = read_attribute::<String>(&mut tuple, self.columns.tensor_dtype)?;
        let checksum = read_attribute::<String>(&mut tuple, self.columns.tensor_checksum)?;

        validate_descriptor(
            candidate, public_id, tensor_ref, rows, dimension, dtype, checksum,
        )
        .map(Some)
    }
}

#[allow(dead_code, reason = "reserved for the optional Phase 3C heap source")]
fn read_attribute<T: FromDatum>(tuple: &mut impl Tuple, attnum: i16) -> Result<T, RerankError> {
    let attribute = tuple
        .attribute(attnum)
        .ok_or(RerankError::InvalidDescriptor(
            "registered attribute is unavailable",
        ))?;
    unsafe { T::from_datum(attribute.datum, attribute.is_null) }.ok_or(
        RerankError::InvalidDescriptor("registered descriptor value is NULL or malformed"),
    )
}

fn validate_model_contract(value: &str) -> Result<(), RerankError> {
    if value.is_empty() || value.len() > MAX_MODEL_CONTRACT_BYTES || contains_control(value) {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_descriptor(
    candidate: PageCandidate,
    public_id: i64,
    tensor_ref: String,
    rows: i32,
    dimension: i32,
    dtype: String,
    checksum: String,
) -> Result<ExternalTensorDescriptor, RerankError> {
    if tensor_ref.is_empty()
        || tensor_ref.len() > MAX_TENSOR_REF_BYTES
        || contains_control(&tensor_ref)
    {
        return Err(RerankError::InvalidDescriptor(
            "tensor reference is empty, oversized, or contains control characters",
        ));
    }
    let rows = u32::try_from(rows)
        .ok()
        .filter(|rows| (1..=MAX_TENSOR_ROWS).contains(rows))
        .ok_or(RerankError::InvalidDescriptor(
            "tensor row count is invalid",
        ))?;
    let dimension = u32::try_from(dimension)
        .ok()
        .filter(|dimension| (1..=MAX_TENSOR_DIMENSION).contains(dimension))
        .ok_or(RerankError::InvalidDescriptor(
            "tensor dimension is invalid",
        ))?;
    let dtype = match dtype.as_str() {
        "float32" => ExternalTensorDtype::F32,
        "float16" => ExternalTensorDtype::F16,
        _ => {
            return Err(RerankError::InvalidDescriptor(
                "tensor dtype must be float16 or float32",
            ));
        }
    };
    if checksum.len() > MAX_CHECKSUM_BYTES || !is_sha256_checksum(&checksum) {
        return Err(RerankError::InvalidDescriptor(
            "tensor checksum must be a lowercase sha256 digest",
        ));
    }
    Ok(ExternalTensorDescriptor {
        candidate,
        public_id,
        tensor_ref,
        rows,
        dimension,
        dtype,
        checksum,
    })
}

fn contains_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}

fn is_sha256_checksum(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::fetcher::TupleAttribute;
    use distance::Distance;

    fn candidate() -> PageCandidate {
        PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: [0, 0, 1],
        }
    }

    fn columns() -> ExternalTensorColumns {
        ExternalTensorColumns {
            model_contract: 1,
            public_id: 2,
            tensor_ref: 3,
            tensor_rows: 4,
            tensor_dimension: 5,
            tensor_dtype: 6,
            tensor_checksum: 7,
        }
    }

    struct RejectingFetcher;
    struct RejectingTuple;

    impl Tuple for RejectingTuple {
        fn build(&mut self) -> (&[pgrx::pg_sys::Datum; 32], &[bool; 32]) {
            panic!("external descriptor sources do not build index expressions")
        }

        fn attribute(&mut self, _attnum: i16) -> Option<TupleAttribute> {
            panic!("a rejected tuple must not expose descriptor attributes")
        }
    }

    impl FilterableTuple for RejectingTuple {
        fn filter(&mut self) -> bool {
            false
        }
    }

    impl Fetcher for RejectingFetcher {
        type Tuple<'a> = RejectingTuple;

        fn fetch(&mut self, _key: [u16; 3]) -> Option<Self::Tuple<'_>> {
            Some(RejectingTuple)
        }
    }

    #[test]
    fn source_filters_before_reading_descriptor_attributes() {
        let mut fetcher = RejectingFetcher;
        let mut source =
            HeapTensorRefSource::new(&mut fetcher, "contract@1".into(), columns()).unwrap();
        assert!(source.fetch(candidate()).unwrap().is_none());
    }

    #[test]
    fn descriptor_validation_accepts_bounded_immutable_metadata() {
        let descriptor = validate_descriptor(
            candidate(),
            42,
            "s3://immutable-bucket/page-42.tensor".into(),
            747,
            320,
            "float16".into(),
            format!("sha256:{}", "a".repeat(64)),
        )
        .unwrap();
        assert_eq!(descriptor.public_id, 42);
        assert_eq!(descriptor.rows, 747);
        assert_eq!(descriptor.dimension, 320);
        assert_eq!(descriptor.dtype, ExternalTensorDtype::F16);
    }

    #[test]
    fn descriptor_validation_rejects_ambiguous_or_unbounded_metadata() {
        assert!(matches!(
            columns_with_duplicate().validate(),
            Err(RerankError::InvalidDescriptor(_))
        ));
        for (rows, dimension, dtype, checksum) in [
            (0, 320, "float16", format!("sha256:{}", "a".repeat(64))),
            (1, 0, "float16", format!("sha256:{}", "a".repeat(64))),
            (1, 320, "bf16", format!("sha256:{}", "a".repeat(64))),
            (1, 320, "float16", "sha256:short".into()),
        ] {
            assert!(matches!(
                validate_descriptor(
                    candidate(),
                    1,
                    "s3://immutable/tensor".into(),
                    rows,
                    dimension,
                    dtype.into(),
                    checksum,
                ),
                Err(RerankError::InvalidDescriptor(_))
            ));
        }
    }

    fn columns_with_duplicate() -> ExternalTensorColumns {
        ExternalTensorColumns {
            public_id: 1,
            ..columns()
        }
    }
}
