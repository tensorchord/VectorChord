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

use super::candidate::{HeapKey, PageCandidate};
use super::external::{
    CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorDtype,
};
use super::rerank::{CandidateTensorSource, ExactMaxsimBackend, RerankError, RerankResults};
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::mem::size_of_val;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use vchordrq::types::OwnedVector;

const MAGIC: &[u8; 4] = b"VCTM";
const VERSION: u16 = 1;
const EXTERNAL_VERSION: u16 = 2;
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;
const HEADER_LEN: usize = 24;
const MAX_REMOTE_ERROR_BYTES: usize = 64 * 1024;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static LAST_FALLBACK_WARNING_SECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) fn report_gpu_fallback(error: &RerankError) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let should_warn = LAST_FALLBACK_WARNING_SECONDS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
            (last == 0 || now.saturating_sub(last) >= 60).then_some(now)
        })
        .is_ok();
    if should_warn {
        pgrx::warning!("GPU MaxSim failed; using cpu_exact: {error}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorDtype {
    F32 = 1,
    F16 = 2,
}

pub(super) trait TileMaxsimTransport {
    fn round_trip(
        &mut self,
        request: &[u8],
        timeout: Duration,
        max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError>;
}

pub(super) struct GpuTileMaxsimBackend<T> {
    transport: T,
    timeout: Duration,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
}

impl<T> GpuTileMaxsimBackend<T> {
    pub fn new(
        transport: T,
        timeout: Duration,
        max_batch_tokens: usize,
        max_batch_bytes: usize,
    ) -> Self {
        Self {
            transport,
            timeout,
            max_batch_tokens,
            max_batch_bytes,
        }
    }
}

impl<T: TileMaxsimTransport> ExactMaxsimBackend for GpuTileMaxsimBackend<T> {
    type Results = RerankResults;

    fn rerank<S: CandidateTensorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<Self::Results, RerankError> {
        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let encoded = encode_request(
            request_id,
            query,
            candidates,
            source,
            self.max_batch_tokens,
            self.max_batch_bytes,
        )?;
        if encoded.heap_keys.is_empty() {
            return Ok(RerankResults {
                inner: BinaryHeap::new(),
            });
        }
        let max_response_bytes = HEADER_LEN
            .checked_add(8)
            .and_then(|size| size.checked_add(encoded.heap_keys.len().checked_mul(8)?))
            .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
            .ok_or(RerankError::RequestTooLarge)?;
        let response =
            self.transport
                .round_trip(&encoded.frame, self.timeout, max_response_bytes)?;
        decode_response(&response, request_id, &encoded.heap_keys)
    }
}

/// GPU backend for Phase 3B external tensor descriptors.
///
/// This codec is intentionally separate from [`ExactMaxsimBackend`]: an
/// external full tensor may have different logical values from the indexed
/// sketch, so it cannot be substituted into ordinary `@#` execution.
pub(super) struct GpuExternalTileMaxsimBackend<T> {
    transport: T,
    model_contract_id: String,
    timeout: Duration,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
}

impl<T> GpuExternalTileMaxsimBackend<T> {
    pub(super) fn new(
        transport: T,
        model_contract_id: String,
        timeout: Duration,
        max_batch_tokens: usize,
        max_batch_bytes: usize,
    ) -> Self {
        Self {
            transport,
            model_contract_id,
            timeout,
            max_batch_tokens,
            max_batch_bytes,
        }
    }
}

impl<T: TileMaxsimTransport> GpuExternalTileMaxsimBackend<T> {
    pub(super) fn rerank<S: CandidateTensorDescriptorSource>(
        &mut self,
        query: &[OwnedVector],
        candidates: &mut dyn Iterator<Item = PageCandidate>,
        source: &mut S,
    ) -> Result<RerankResults, RerankError> {
        let request_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let encoded = encode_external_request(
            request_id,
            &self.model_contract_id,
            query,
            candidates,
            source,
            self.max_batch_tokens,
            self.max_batch_bytes,
        )?;
        if encoded.heap_keys.is_empty() {
            return Ok(RerankResults {
                inner: BinaryHeap::new(),
            });
        }
        let max_response_bytes = HEADER_LEN
            .checked_add(8)
            .and_then(|size| size.checked_add(encoded.heap_keys.len().checked_mul(8)?))
            .map(|size| size.max(HEADER_LEN + 8 + MAX_REMOTE_ERROR_BYTES))
            .ok_or(RerankError::RequestTooLarge)?;
        let response =
            self.transport
                .round_trip(&encoded.frame, self.timeout, max_response_bytes)?;
        decode_response_for_version(&response, EXTERNAL_VERSION, request_id, &encoded.heap_keys)
    }
}

struct EncodedRequest {
    frame: Vec<u8>,
    heap_keys: Vec<HeapKey>,
}

fn encode_external_request<S: CandidateTensorDescriptorSource>(
    request_id: u64,
    model_contract_id: &str,
    query: &[OwnedVector],
    candidates: &mut dyn Iterator<Item = PageCandidate>,
    source: &mut S,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
) -> Result<EncodedRequest, RerankError> {
    const MAX_MODEL_CONTRACT_BYTES: usize = 512;
    const MAX_TENSOR_REF_BYTES: usize = 4096;
    const MAX_CHECKSUM_BYTES: usize = 512;

    if model_contract_id.is_empty()
        || model_contract_id.len() > MAX_MODEL_CONTRACT_BYTES
        || model_contract_id.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "model contract is empty, oversized, or contains control characters",
        ));
    }
    let (dtype, dimension) = tensor_metadata(query)?;
    let external_dtype = match dtype {
        TensorDtype::F32 => ExternalTensorDtype::F32,
        TensorDtype::F16 => ExternalTensorDtype::F16,
    };
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    let mut total_tokens = query.len();
    if total_tokens > max_batch_tokens {
        return Err(RerankError::RequestTooLarge);
    }
    let mut declared_tensor_bytes = tensor_bytes(query_rows, dimension, dtype)?;
    if declared_tensor_bytes > max_batch_bytes {
        return Err(RerankError::RequestTooLarge);
    }

    let mut writer = BoundedWriter::new(max_batch_bytes);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    let candidate_count_offset = writer.len();
    writer.u32(0)?;
    writer.u8(dtype as u8)?;
    writer.u8(1)?; // sum_query_max_document_dot
    writer.u16(0)?;
    writer
        .u32(u32::try_from(model_contract_id.len()).map_err(|_| RerankError::RequestTooLarge)?)?;
    writer.bytes(model_contract_id.as_bytes())?;
    encode_tensor_values(&mut writer, query, dtype)?;

    let mut heap_keys = Vec::new();
    for candidate in candidates {
        let Some(descriptor) = source.fetch(candidate)? else {
            continue;
        };
        validate_external_for_request(
            &descriptor,
            dimension,
            external_dtype,
            MAX_TENSOR_REF_BYTES,
            MAX_CHECKSUM_BYTES,
        )?;
        total_tokens = total_tokens
            .checked_add(descriptor.rows as usize)
            .ok_or(RerankError::RequestTooLarge)?;
        if total_tokens > max_batch_tokens {
            return Err(RerankError::RequestTooLarge);
        }
        declared_tensor_bytes = declared_tensor_bytes
            .checked_add(tensor_bytes(descriptor.rows, dimension, dtype)?)
            .ok_or(RerankError::RequestTooLarge)?;
        if declared_tensor_bytes > max_batch_bytes {
            return Err(RerankError::RequestTooLarge);
        }

        let candidate_id =
            u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
        writer.u32(candidate_id)?;
        writer.u32(descriptor.rows)?;
        writer.u32(
            u32::try_from(descriptor.tensor_ref.len()).map_err(|_| RerankError::RequestTooLarge)?,
        )?;
        writer.u32(
            u32::try_from(descriptor.checksum.len()).map_err(|_| RerankError::RequestTooLarge)?,
        )?;
        writer.bytes(descriptor.tensor_ref.as_bytes())?;
        writer.bytes(descriptor.checksum.as_bytes())?;
        heap_keys.push(descriptor.candidate.heap_key);
    }

    let candidate_count =
        u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_u32(candidate_count_offset, candidate_count);
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, EXTERNAL_VERSION);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(
        16,
        u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?,
    );
    Ok(EncodedRequest {
        frame: writer.finish(),
        heap_keys,
    })
}

fn tensor_bytes(rows: u32, dimension: u32, dtype: TensorDtype) -> Result<usize, RerankError> {
    let scalar_bytes = match dtype {
        TensorDtype::F32 => 4usize,
        TensorDtype::F16 => 2usize,
    };
    usize::try_from(rows)
        .ok()
        .and_then(|rows| rows.checked_mul(dimension as usize))
        .and_then(|elements| elements.checked_mul(scalar_bytes))
        .ok_or(RerankError::RequestTooLarge)
}

fn validate_external_for_request(
    descriptor: &ExternalTensorDescriptor,
    dimension: u32,
    dtype: ExternalTensorDtype,
    max_tensor_ref_bytes: usize,
    max_checksum_bytes: usize,
) -> Result<(), RerankError> {
    if descriptor.rows == 0
        || descriptor.rows > 65_536
        || dimension > 60_000
        || descriptor.dimension != dimension
        || descriptor.dtype != dtype
    {
        return Err(RerankError::TensorMismatch);
    }
    if descriptor.tensor_ref.is_empty()
        || descriptor.tensor_ref.len() > max_tensor_ref_bytes
        || descriptor.tensor_ref.chars().any(char::is_control)
    {
        return Err(RerankError::InvalidDescriptor(
            "tensor reference is empty, oversized, or contains control characters",
        ));
    }
    if descriptor.checksum.len() > max_checksum_bytes
        || !descriptor
            .checksum
            .strip_prefix("sha256:")
            .is_some_and(|digest| {
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            })
    {
        return Err(RerankError::InvalidDescriptor(
            "tensor checksum must be a lowercase sha256 digest",
        ));
    }
    Ok(())
}

fn encode_request<S: CandidateTensorSource>(
    request_id: u64,
    query: &[OwnedVector],
    candidates: &mut dyn Iterator<Item = PageCandidate>,
    source: &mut S,
    max_batch_tokens: usize,
    max_batch_bytes: usize,
) -> Result<EncodedRequest, RerankError> {
    let (dtype, dimension) = tensor_metadata(query)?;
    let query_rows = u32::try_from(query.len()).map_err(|_| RerankError::RequestTooLarge)?;
    let mut total_tokens = query.len();
    if total_tokens > max_batch_tokens {
        return Err(RerankError::RequestTooLarge);
    }

    let mut writer = BoundedWriter::new(max_batch_bytes);
    writer.zeros(HEADER_LEN)?;
    writer.u32(dimension)?;
    writer.u32(query_rows)?;
    let candidate_count_offset = writer.len();
    writer.u32(0)?;
    writer.u8(dtype as u8)?;
    writer.u8(1)?; // sum_query_max_document_dot
    writer.u16(0)?;
    encode_tensor_values(&mut writer, query, dtype)?;

    let mut heap_keys = Vec::new();
    for candidate in candidates {
        let Some(tensor) = source.fetch(candidate)? else {
            continue;
        };
        let (candidate_dtype, candidate_dimension) = tensor_metadata(&tensor.vectors)?;
        if candidate_dtype != dtype || candidate_dimension != dimension {
            return Err(RerankError::TensorMismatch);
        }
        total_tokens = total_tokens
            .checked_add(tensor.vectors.len())
            .ok_or(RerankError::RequestTooLarge)?;
        if total_tokens > max_batch_tokens {
            return Err(RerankError::RequestTooLarge);
        }
        let candidate_id =
            u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
        let rows = u32::try_from(tensor.vectors.len()).map_err(|_| RerankError::RequestTooLarge)?;
        writer.u32(candidate_id)?;
        writer.u32(rows)?;
        encode_tensor_values(&mut writer, &tensor.vectors, dtype)?;
        heap_keys.push(tensor.candidate.heap_key);
    }
    let candidate_count =
        u32::try_from(heap_keys.len()).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_u32(candidate_count_offset, candidate_count);
    let body_len = writer
        .len()
        .checked_sub(HEADER_LEN)
        .ok_or_else(|| RerankError::Protocol("invalid request length".into()))?;
    let body_len = u64::try_from(body_len).map_err(|_| RerankError::RequestTooLarge)?;
    writer.patch_bytes(0, MAGIC);
    writer.patch_u16(4, VERSION);
    writer.patch_u16(6, REQUEST_KIND);
    writer.patch_u64(8, request_id);
    writer.patch_u64(16, body_len);
    Ok(EncodedRequest {
        frame: writer.finish(),
        heap_keys,
    })
}

fn tensor_metadata(vectors: &[OwnedVector]) -> Result<(TensorDtype, u32), RerankError> {
    let Some(first) = vectors.first() else {
        return Err(RerankError::TensorMismatch);
    };
    let dtype = match first {
        OwnedVector::Vecf32(_) => TensorDtype::F32,
        OwnedVector::Vecf16(_) => TensorDtype::F16,
        OwnedVector::Rabitq8(_) | OwnedVector::Rabitq4(_) => {
            return Err(RerankError::UnsupportedTensorKind);
        }
    };
    let dimension = first.dim();
    for vector in vectors {
        let this_dtype = match vector {
            OwnedVector::Vecf32(_) => TensorDtype::F32,
            OwnedVector::Vecf16(_) => TensorDtype::F16,
            OwnedVector::Rabitq8(_) | OwnedVector::Rabitq4(_) => {
                return Err(RerankError::UnsupportedTensorKind);
            }
        };
        if this_dtype != dtype || vector.dim() != dimension {
            return Err(RerankError::TensorMismatch);
        }
    }
    Ok((dtype, dimension))
}

fn encode_tensor_values(
    writer: &mut BoundedWriter,
    vectors: &[OwnedVector],
    dtype: TensorDtype,
) -> Result<(), RerankError> {
    for vector in vectors {
        match (dtype, vector) {
            (TensorDtype::F32, OwnedVector::Vecf32(vector)) => {
                for value in vector.slice() {
                    writer.bytes(&value.to_le_bytes())?;
                }
            }
            (TensorDtype::F16, OwnedVector::Vecf16(vector)) => {
                for value in vector.slice() {
                    writer.u16(value.to_bits())?;
                }
            }
            _ => return Err(RerankError::TensorMismatch),
        }
    }
    Ok(())
}

fn decode_response(
    frame: &[u8],
    request_id: u64,
    heap_keys: &[HeapKey],
) -> Result<RerankResults, RerankError> {
    decode_response_for_version(frame, VERSION, request_id, heap_keys)
}

fn decode_response_for_version(
    frame: &[u8],
    expected_version: u16,
    request_id: u64,
    heap_keys: &[HeapKey],
) -> Result<RerankResults, RerankError> {
    let mut cursor = Cursor::new(frame);
    if cursor.bytes(4)? != MAGIC {
        return Err(RerankError::Protocol("invalid magic".into()));
    }
    if cursor.u16()? != expected_version {
        return Err(RerankError::Protocol("unsupported version".into()));
    }
    if cursor.u16()? != RESPONSE_KIND {
        return Err(RerankError::Protocol("unexpected message kind".into()));
    }
    if cursor.u64()? != request_id {
        return Err(RerankError::Protocol("request ID mismatch".into()));
    }
    let body_len = usize::try_from(cursor.u64()?)
        .map_err(|_| RerankError::Protocol("response body is too large".into()))?;
    if body_len != frame.len().saturating_sub(HEADER_LEN) {
        return Err(RerankError::Protocol("response length mismatch".into()));
    }
    let status = cursor.u32()?;
    if status != 0 {
        let length = usize::try_from(cursor.u32()?)
            .map_err(|_| RerankError::Protocol("remote error is too large".into()))?;
        if length > MAX_REMOTE_ERROR_BYTES {
            return Err(RerankError::Protocol("remote error is too large".into()));
        }
        let message = std::str::from_utf8(cursor.bytes(length)?)
            .map_err(|_| RerankError::Protocol("remote error is not UTF-8".into()))?;
        cursor.finish()?;
        return Err(RerankError::Remote(message.into()));
    }
    let result_count = usize::try_from(cursor.u32()?)
        .map_err(|_| RerankError::Protocol("result count is too large".into()))?;
    if result_count != heap_keys.len() {
        return Err(RerankError::Protocol("partial result set".into()));
    }
    let mut seen = vec![false; heap_keys.len()];
    let mut results = BinaryHeap::new();
    for _ in 0..result_count {
        let candidate_id = usize::try_from(cursor.u32()?)
            .map_err(|_| RerankError::Protocol("candidate ID is too large".into()))?;
        let Some(heap_key) = heap_keys.get(candidate_id).copied() else {
            return Err(RerankError::Protocol("unknown candidate ID".into()));
        };
        if std::mem::replace(&mut seen[candidate_id], true) {
            return Err(RerankError::Protocol("duplicate candidate ID".into()));
        }
        let similarity = f32::from_bits(cursor.u32()?);
        if !similarity.is_finite() {
            return Err(RerankError::Protocol("non-finite similarity".into()));
        }
        let distance = Distance::from_f32(-similarity);
        results.push((Reverse(distance), Reverse(heap_key)));
    }
    cursor.finish()?;
    if seen.iter().any(|seen| !seen) {
        return Err(RerankError::Protocol("partial result set".into()));
    }
    Ok(RerankResults { inner: results })
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn ensure(&self, additional: usize) -> Result<(), RerankError> {
        let size = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or(RerankError::RequestTooLarge)?;
        if size > self.limit {
            return Err(RerankError::RequestTooLarge);
        }
        Ok(())
    }

    fn zeros(&mut self, count: usize) -> Result<(), RerankError> {
        self.ensure(count)?;
        self.bytes.resize(self.bytes.len() + count, 0);
        Ok(())
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), RerankError> {
        self.ensure(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), RerankError> {
        self.bytes(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), RerankError> {
        self.bytes(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), RerankError> {
        self.bytes(&value.to_le_bytes())
    }

    fn patch_bytes(&mut self, offset: usize, bytes: &[u8]) {
        self.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    fn patch_u16(&mut self, offset: usize, value: u16) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn patch_u32(&mut self, offset: usize, value: u32) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn patch_u64(&mut self, offset: usize, value: u64) {
        self.patch_bytes(offset, &value.to_le_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], RerankError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| RerankError::Protocol("message offset overflow".into()))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| RerankError::Protocol("truncated message".into()))?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, RerankError> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, RerankError> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, RerankError> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn finish(self) -> Result<(), RerankError> {
        if self.offset != self.bytes.len() {
            return Err(RerankError::Protocol("trailing response bytes".into()));
        }
        Ok(())
    }
}

pub(super) struct UnixSocketTransport {
    endpoint: String,
}

impl UnixSocketTransport {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
}

#[cfg(unix)]
impl TileMaxsimTransport for UnixSocketTransport {
    fn round_trip(
        &mut self,
        request: &[u8],
        timeout: Duration,
        max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError> {
        use std::time::Instant;

        if self.endpoint.is_empty() {
            return Err(RerankError::Transport("endpoint is empty".into()));
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| RerankError::Transport("invalid timeout".into()))?;
        let mut stream = connect_interruptible(&self.endpoint, deadline)?;
        let poll = remaining_until(deadline)?.min(Duration::from_millis(50));
        stream
            .set_read_timeout(Some(poll))
            .map_err(|error| RerankError::Transport(error.to_string()))?;
        stream
            .set_write_timeout(Some(poll))
            .map_err(|error| RerankError::Transport(error.to_string()))?;

        write_interruptible(&mut stream, request, deadline)?;
        let mut header = [0u8; HEADER_LEN];
        read_interruptible(&mut stream, &mut header, deadline)?;
        let body_len = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
            .map_err(|_| RerankError::Protocol("response body is too large".into()))?;
        let response_len = HEADER_LEN
            .checked_add(body_len)
            .ok_or_else(|| RerankError::Protocol("response length overflow".into()))?;
        if response_len > max_response_bytes {
            return Err(RerankError::Protocol(
                "response exceeds configured limit".into(),
            ));
        }
        let mut response = Vec::with_capacity(response_len);
        response.extend_from_slice(&header);
        response.resize(response_len, 0);
        read_interruptible(&mut stream, &mut response[HEADER_LEN..], deadline)?;
        Ok(response)
    }
}

#[cfg(unix)]
fn connect_interruptible(
    endpoint: &str,
    deadline: std::time::Instant,
) -> Result<std::os::unix::net::UnixStream, RerankError> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let (address, address_len) = unix_socket_address(endpoint)?;
    let raw_fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if raw_fd < 0 {
        return Err(last_transport_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFD,
        libc::F_SETFD,
        libc::FD_CLOEXEC,
        true,
    )?;
    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFL,
        libc::F_SETFL,
        libc::O_NONBLOCK,
        true,
    )?;

    let connected = unsafe {
        libc::connect(
            fd.as_raw_fd(),
            (&raw const address).cast::<libc::sockaddr>(),
            address_len,
        )
    } == 0;
    if !connected {
        let error = std::io::Error::last_os_error();
        let raw_error = error.raw_os_error();
        if raw_error != Some(libc::EINPROGRESS)
            && raw_error != Some(libc::EAGAIN)
            && raw_error != Some(libc::EWOULDBLOCK)
        {
            return Err(RerankError::Transport(error.to_string()));
        }
        wait_for_connect(fd.as_raw_fd(), deadline)?;
    }

    update_fd_flag(
        fd.as_raw_fd(),
        libc::F_GETFL,
        libc::F_SETFL,
        libc::O_NONBLOCK,
        false,
    )?;
    Ok(std::os::unix::net::UnixStream::from(fd))
}

#[cfg(unix)]
fn unix_socket_address(
    endpoint: &str,
) -> Result<(libc::sockaddr_un, libc::socklen_t), RerankError> {
    let path = endpoint.as_bytes();
    let mut address = unsafe { std::mem::zeroed::<libc::sockaddr_un>() };
    if path.contains(&0) {
        return Err(RerankError::Transport(
            "endpoint contains a NUL byte".into(),
        ));
    }
    if path.len() >= address.sun_path.len() {
        return Err(RerankError::Transport("endpoint path is too long".into()));
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    unsafe {
        std::ptr::copy_nonoverlapping(
            path.as_ptr().cast::<libc::c_char>(),
            address.sun_path.as_mut_ptr(),
            path.len(),
        );
    }
    let length = std::mem::offset_of!(libc::sockaddr_un, sun_path)
        .checked_add(path.len())
        .and_then(|length| length.checked_add(1))
        .and_then(|length| libc::socklen_t::try_from(length).ok())
        .ok_or_else(|| RerankError::Transport("endpoint path is too long".into()))?;
    #[cfg(any(
        target_os = "aix",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "hurd",
        target_os = "ios",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "tvos",
        target_os = "visionos",
        target_os = "watchos"
    ))]
    {
        address.sun_len = u8::try_from(length)
            .map_err(|_| RerankError::Transport("endpoint path is too long".into()))?;
    }
    Ok((address, length))
}

#[cfg(unix)]
fn update_fd_flag(
    fd: std::os::fd::RawFd,
    get_command: libc::c_int,
    set_command: libc::c_int,
    flag: libc::c_int,
    enabled: bool,
) -> Result<(), RerankError> {
    let current = unsafe { libc::fcntl(fd, get_command) };
    if current < 0 {
        return Err(last_transport_error());
    }
    let updated = if enabled {
        current | flag
    } else {
        current & !flag
    };
    if unsafe { libc::fcntl(fd, set_command, updated) } < 0 {
        return Err(last_transport_error());
    }
    Ok(())
}

#[cfg(unix)]
fn wait_for_connect(
    fd: std::os::fd::RawFd,
    deadline: std::time::Instant,
) -> Result<(), RerankError> {
    loop {
        pgrx::check_for_interrupts!();
        let remaining = remaining_until(deadline)?;
        let timeout_ms = remaining.min(Duration::from_millis(50)).as_millis().max(1) as libc::c_int;
        let mut poll_fd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut poll_fd, 1, timeout_ms) };
        if result == 0 {
            continue;
        }
        if result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(RerankError::Transport(error.to_string()));
        }
        let mut socket_error = 0;
        let mut socket_error_len = size_of_val(&socket_error) as libc::socklen_t;
        if unsafe {
            libc::getsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_ERROR,
                (&raw mut socket_error).cast(),
                &raw mut socket_error_len,
            )
        } < 0
        {
            return Err(last_transport_error());
        }
        if socket_error != 0 {
            return Err(RerankError::Transport(
                std::io::Error::from_raw_os_error(socket_error).to_string(),
            ));
        }
        return Ok(());
    }
}

#[cfg(unix)]
fn remaining_until(deadline: std::time::Instant) -> Result<Duration, RerankError> {
    deadline
        .checked_duration_since(std::time::Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| RerankError::Transport("request timed out".into()))
}

#[cfg(unix)]
fn last_transport_error() -> RerankError {
    RerankError::Transport(std::io::Error::last_os_error().to_string())
}

#[cfg(unix)]
fn write_interruptible(
    stream: &mut std::os::unix::net::UnixStream,
    mut bytes: &[u8],
    deadline: std::time::Instant,
) -> Result<(), RerankError> {
    use std::io::Write;

    while !bytes.is_empty() {
        match stream.write(bytes) {
            Ok(0) => return Err(RerankError::Transport("connection closed".into())),
            Ok(count) => bytes = &bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(RerankError::Transport(error.to_string())),
        }
        pgrx::check_for_interrupts!();
        if std::time::Instant::now() >= deadline {
            return Err(RerankError::Transport("request timed out".into()));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_interruptible(
    stream: &mut std::os::unix::net::UnixStream,
    mut bytes: &mut [u8],
    deadline: std::time::Instant,
) -> Result<(), RerankError> {
    use std::io::Read;

    while !bytes.is_empty() {
        match stream.read(bytes) {
            Ok(0) => return Err(RerankError::Transport("connection closed".into())),
            Ok(count) => bytes = &mut bytes[count..],
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => return Err(RerankError::Transport(error.to_string())),
        }
        pgrx::check_for_interrupts!();
        if std::time::Instant::now() >= deadline {
            return Err(RerankError::Transport("request timed out".into()));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
impl TileMaxsimTransport for UnixSocketTransport {
    fn round_trip(
        &mut self,
        _request: &[u8],
        _timeout: Duration,
        _max_response_bytes: usize,
    ) -> Result<Vec<u8>, RerankError> {
        if self.endpoint.is_empty() {
            return Err(RerankError::Transport("endpoint is empty".into()));
        }
        Err(RerankError::Transport(
            "Unix sockets are not supported on this platform".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::external::{
        CandidateTensorDescriptorSource, ExternalTensorDescriptor, ExternalTensorDtype,
    };
    use super::super::rerank::CandidateTensor;
    use super::*;
    use std::collections::BTreeMap;
    use vector::vect::VectOwned;

    struct MockTensorSource(BTreeMap<HeapKey, Vec<OwnedVector>>);

    impl CandidateTensorSource for MockTensorSource {
        fn fetch(
            &mut self,
            candidate: PageCandidate,
        ) -> Result<Option<CandidateTensor>, RerankError> {
            Ok(Some(CandidateTensor {
                candidate,
                vectors: self
                    .0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            }))
        }
    }

    struct MockDescriptorSource(BTreeMap<HeapKey, ExternalTensorDescriptor>);

    impl CandidateTensorDescriptorSource for MockDescriptorSource {
        fn fetch(
            &mut self,
            candidate: PageCandidate,
        ) -> Result<Option<ExternalTensorDescriptor>, RerankError> {
            Ok(Some(
                self.0
                    .remove(&candidate.heap_key)
                    .ok_or(RerankError::TensorMismatch)?,
            ))
        }
    }

    struct MockTransport {
        similarities: Vec<(u32, f32)>,
    }

    impl TileMaxsimTransport for MockTransport {
        fn round_trip(
            &mut self,
            request: &[u8],
            _timeout: Duration,
            _max_response_bytes: usize,
        ) -> Result<Vec<u8>, RerankError> {
            let request_id = u64::from_le_bytes(request[8..16].try_into().unwrap());
            let version = u16::from_le_bytes(request[4..6].try_into().unwrap());
            Ok(success_response_with_version(
                version,
                request_id,
                &self.similarities,
            ))
        }
    }

    fn vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf32(VectOwned::new(values.to_vec()))
    }

    fn half_vector(values: &[f32]) -> OwnedVector {
        OwnedVector::Vecf16(VectOwned::new(
            values.iter().copied().map(simd::f16::from_f32).collect(),
        ))
    }

    fn success_response(request_id: u64, similarities: &[(u32, f32)]) -> Vec<u8> {
        success_response_with_version(VERSION, request_id, similarities)
    }

    fn success_response_with_version(
        version: u16,
        request_id: u64,
        similarities: &[(u32, f32)],
    ) -> Vec<u8> {
        let body_len = 8 + similarities.len() * 8;
        let mut response = Vec::with_capacity(HEADER_LEN + body_len);
        response.extend_from_slice(MAGIC);
        response.extend_from_slice(&version.to_le_bytes());
        response.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
        response.extend_from_slice(&request_id.to_le_bytes());
        response.extend_from_slice(&(body_len as u64).to_le_bytes());
        response.extend_from_slice(&0u32.to_le_bytes());
        response.extend_from_slice(&(similarities.len() as u32).to_le_bytes());
        for (candidate_id, similarity) in similarities {
            response.extend_from_slice(&candidate_id.to_le_bytes());
            response.extend_from_slice(&similarity.to_bits().to_le_bytes());
        }
        response
    }

    fn external_descriptor(
        candidate: PageCandidate,
        public_id: i64,
        tensor_ref: &str,
        rows: u32,
        dimension: u32,
        dtype: ExternalTensorDtype,
    ) -> ExternalTensorDescriptor {
        ExternalTensorDescriptor {
            candidate,
            public_id,
            tensor_ref: tensor_ref.into(),
            rows,
            dimension,
            dtype,
            checksum: format!("sha256:{}", "a".repeat(64)),
        }
    }

    fn error_response(request_id: u64, message: &str) -> Vec<u8> {
        let body_len = 8 + message.len();
        let mut response = Vec::with_capacity(HEADER_LEN + body_len);
        response.extend_from_slice(MAGIC);
        response.extend_from_slice(&VERSION.to_le_bytes());
        response.extend_from_slice(&RESPONSE_KIND.to_le_bytes());
        response.extend_from_slice(&request_id.to_le_bytes());
        response.extend_from_slice(&(body_len as u64).to_le_bytes());
        response.extend_from_slice(&1u32.to_le_bytes());
        response.extend_from_slice(&(message.len() as u32).to_le_bytes());
        response.extend_from_slice(message.as_bytes());
        response
    }

    #[test]
    fn gpu_backend_maps_positive_similarity_to_ascending_distance() {
        let page_1 = [0, 0, 1];
        let page_2 = [0, 0, 2];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![
            PageCandidate {
                approximate_distance: Distance::ZERO,
                heap_key: page_1,
            },
            PageCandidate {
                approximate_distance: Distance::ZERO,
                heap_key: page_2,
            },
        ]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([
            (page_1, vec![vector(&[1.0, 0.0])]),
            (page_2, vec![vector(&[0.5, 0.0])]),
        ]));
        let transport = MockTransport {
            similarities: vec![(0, 1.0), (1, 2.0)],
        };
        let results = GpuTileMaxsimBackend::new(transport, Duration::from_secs(1), 100, 4096)
            .rerank(&query, &mut candidates, &mut source)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(results[0].heap_key, page_2);
        assert_eq!(results[0].distance.to_f32(), -2.0);
        assert_eq!(results[1].heap_key, page_1);
        assert_eq!(results[1].distance.to_f32(), -1.0);
    }

    #[test]
    fn response_rejects_partial_and_duplicate_results() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let partial = success_response(7, &[(0, 1.0)]);
        assert!(matches!(
            decode_response(&partial, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let duplicate = success_response(7, &[(0, 1.0), (0, 2.0)]);
        assert!(matches!(
            decode_response(&duplicate, 7, &keys),
            Err(RerankError::Protocol(_))
        ));
    }

    #[test]
    fn response_ids_may_arrive_out_of_order() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let response = success_response(7, &[(1, 0.5), (0, 1.0)]);
        let results = decode_response(&response, 7, &keys)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(results[0].heap_key, keys[0]);
        assert_eq!(results[0].distance.to_f32(), -1.0);
        assert_eq!(results[1].heap_key, keys[1]);
        assert_eq!(results[1].distance.to_f32(), -0.5);
    }

    #[test]
    fn response_rejects_unknown_non_finite_and_trailing_results() {
        let keys = [[0, 0, 1], [0, 0, 2]];
        let unknown = success_response(7, &[(0, 1.0), (2, 2.0)]);
        assert!(matches!(
            decode_response(&unknown, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let non_finite = success_response(7, &[(0, f32::NAN), (1, 2.0)]);
        assert!(matches!(
            decode_response(&non_finite, 7, &keys),
            Err(RerankError::Protocol(_))
        ));

        let mut trailing = success_response(7, &[(0, 1.0), (1, 2.0)]);
        trailing.push(0);
        assert!(matches!(
            decode_response(&trailing, 7, &keys),
            Err(RerankError::Protocol(_))
        ));
    }

    #[test]
    fn response_rejects_invalid_header_fields() {
        let keys = [[0, 0, 1]];
        let valid = success_response(7, &[(0, 1.0)]);

        for (offset, replacement) in [(0, 0u8), (4, 2), (6, 1), (8, 8), (16, 0)] {
            let mut invalid = valid.clone();
            invalid[offset] = replacement;
            assert!(matches!(
                decode_response(&invalid, 7, &keys),
                Err(RerankError::Protocol(_))
            ));
        }
    }

    #[test]
    fn response_surfaces_remote_error() {
        let response = error_response(7, "CUDA queue is unavailable");
        assert!(matches!(
            decode_response(&response, 7, &[]),
            Err(RerankError::Remote(message)) if message == "CUDA queue is unavailable"
        ));
    }

    #[test]
    fn request_frame_is_versioned_and_length_prefixed() {
        let page = [0, 0, 1];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([(page, vec![vector(&[0.5, 0.0])])]));
        let encoded = encode_request(9, &query, &mut candidates, &mut source, 100, 4096).unwrap();

        assert_eq!(&encoded.frame[0..4], MAGIC);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[4..6].try_into().unwrap()),
            VERSION
        );
        assert_eq!(
            u16::from_le_bytes(encoded.frame[6..8].try_into().unwrap()),
            REQUEST_KIND
        );
        assert_eq!(
            u64::from_le_bytes(encoded.frame[8..16].try_into().unwrap()),
            9
        );
        assert_eq!(
            u64::from_le_bytes(encoded.frame[16..24].try_into().unwrap()) as usize,
            encoded.frame.len() - HEADER_LEN
        );
        assert_eq!(
            u32::from_le_bytes(encoded.frame[32..36].try_into().unwrap()),
            1
        );
        assert_eq!(encoded.heap_keys, vec![page]);
    }

    #[test]
    fn external_request_encodes_contract_and_opaque_descriptor_ids() {
        let page = [0, 0, 7];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![half_vector(&[1.0, -0.5])];
        let tensor_ref = "s3://immutable/page-9001.tensor";
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(candidate, 9001, tensor_ref, 2, 2, ExternalTensorDtype::F16),
        )]));
        let contract = "colqwen@immutable-revision";
        let encoded = encode_external_request(
            19,
            contract,
            &query,
            &mut candidates,
            &mut source,
            100,
            4096,
        )
        .unwrap();

        assert_eq!(&encoded.frame[0..4], MAGIC);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[4..6].try_into().unwrap()),
            EXTERNAL_VERSION
        );
        assert_eq!(
            u32::from_le_bytes(encoded.frame[32..36].try_into().unwrap()),
            1
        );
        let contract_len = u32::from_le_bytes(encoded.frame[40..44].try_into().unwrap()) as usize;
        assert_eq!(&encoded.frame[44..44 + contract_len], contract.as_bytes());
        let candidate_offset = 44 + contract_len + 4; // one 2-D f16 query row
        assert_eq!(
            u32::from_le_bytes(
                encoded.frame[candidate_offset..candidate_offset + 4]
                    .try_into()
                    .unwrap()
            ),
            0
        );
        assert_eq!(
            u32::from_le_bytes(
                encoded.frame[candidate_offset + 4..candidate_offset + 8]
                    .try_into()
                    .unwrap()
            ),
            2
        );
        let reference_len = u32::from_le_bytes(
            encoded.frame[candidate_offset + 8..candidate_offset + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        let reference_offset = candidate_offset + 16;
        assert_eq!(
            &encoded.frame[reference_offset..reference_offset + reference_len],
            tensor_ref.as_bytes()
        );
        assert_eq!(encoded.heap_keys, vec![page]);
    }

    #[test]
    fn external_backend_maps_scores_without_exposing_public_ids() {
        let page = [0, 0, 8];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![candidate].into_iter();
        let mut source = MockDescriptorSource(BTreeMap::from([(
            page,
            external_descriptor(
                candidate,
                i64::MAX,
                "object://immutable/tensor",
                4,
                2,
                ExternalTensorDtype::F32,
            ),
        )]));
        let transport = MockTransport {
            similarities: vec![(0, 3.5)],
        };
        let results = GpuExternalTileMaxsimBackend::new(
            transport,
            "contract@1".into(),
            Duration::from_secs(1),
            100,
            4096,
        )
        .rerank(&query, &mut candidates, &mut source)
        .unwrap()
        .collect::<Vec<_>>();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].heap_key, page);
        assert_eq!(results[0].distance.to_f32(), -3.5);
    }

    #[test]
    fn external_request_rejects_shape_and_declared_payload_overflow() {
        let page = [0, 0, 9];
        let candidate = PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        };
        let query = vec![half_vector(&[1.0, 0.0])];
        let make_source = |dimension, rows| {
            MockDescriptorSource(BTreeMap::from([(
                page,
                external_descriptor(
                    candidate,
                    9,
                    "object://immutable/tensor",
                    rows,
                    dimension,
                    ExternalTensorDtype::F16,
                ),
            )]))
        };

        let mut candidates = vec![candidate].into_iter();
        assert!(matches!(
            encode_external_request(
                1,
                "contract@1",
                &query,
                &mut candidates,
                &mut make_source(3, 1),
                100,
                4096,
            ),
            Err(RerankError::TensorMismatch)
        ));

        let mut candidates = vec![candidate].into_iter();
        assert!(matches!(
            encode_external_request(
                1,
                "contract@1",
                &query,
                &mut candidates,
                &mut make_source(2, 100),
                1000,
                64,
            ),
            Err(RerankError::RequestTooLarge)
        ));
    }

    #[test]
    fn request_frame_encodes_f16_tensor_bits() {
        let page = [0, 0, 1];
        let query = vec![half_vector(&[1.0, -0.5])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source =
            MockTensorSource(BTreeMap::from([(page, vec![half_vector(&[0.25, 2.0])])]));
        let encoded = encode_request(9, &query, &mut candidates, &mut source, 100, 4096).unwrap();

        assert_eq!(encoded.frame[36], TensorDtype::F16 as u8);
        assert_eq!(
            u16::from_le_bytes(encoded.frame[40..42].try_into().unwrap()),
            simd::f16::from_f32(1.0).to_bits()
        );
        assert_eq!(
            u16::from_le_bytes(encoded.frame[42..44].try_into().unwrap()),
            simd::f16::from_f32(-0.5).to_bits()
        );
    }

    #[test]
    fn request_limits_are_enforced_before_transport() {
        let page = [0, 0, 1];
        let query = vec![vector(&[1.0, 0.0])];
        let mut candidates = vec![PageCandidate {
            approximate_distance: Distance::ZERO,
            heap_key: page,
        }]
        .into_iter();
        let mut source = MockTensorSource(BTreeMap::from([(page, vec![vector(&[1.0, 0.0])])]));
        assert!(matches!(
            encode_request(1, &query, &mut candidates, &mut source, 1, 4096),
            Err(RerankError::RequestTooLarge)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_address_is_length_bounded_and_nul_terminated() {
        let (address, length) = unix_socket_address("/tmp/vectorchord.sock").unwrap();
        let path_offset = std::mem::offset_of!(libc::sockaddr_un, sun_path);

        assert_eq!(address.sun_family, libc::AF_UNIX as libc::sa_family_t);
        assert_eq!(
            length as usize,
            path_offset + "/tmp/vectorchord.sock".len() + 1
        );
        assert_eq!(
            &address.sun_path[.."/tmp/vectorchord.sock".len()],
            "/tmp/vectorchord.sock"
                .as_bytes()
                .iter()
                .map(|byte| *byte as libc::c_char)
                .collect::<Vec<_>>()
        );
        assert_eq!(address.sun_path["/tmp/vectorchord.sock".len()], 0);

        let too_long = "x".repeat(address.sun_path.len());
        assert!(matches!(
            unix_socket_address(&too_long),
            Err(RerankError::Transport(message)) if message == "endpoint path is too long"
        ));
        assert!(matches!(
            unix_socket_address("invalid\0path"),
            Err(RerankError::Transport(message)) if message == "endpoint contains a NUL byte"
        ));
    }
}
