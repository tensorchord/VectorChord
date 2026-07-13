# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the terms of the ELv2, which has specific restrictions.
#
# We welcome any commercial collaboration or support. For inquiries
# regarding the licenses, please contact us at:
# vectorchord-inquiry@tensorchord.ai
#
# Copyright (c) 2025-2026 TensorChord Inc.

"""CPU reference implementation of the VectorChord TileMaxSim IPC sidecar.

This executable is intentionally simple and single-threaded. It is a protocol
oracle and end-to-end development aid, not a production or GPU implementation.
The CLI serves inline v1 requests. External-descriptor v2 requests require an
explicit resolver injected by a test or embedding application and fail closed
when no resolver is configured.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import math
import os
import signal
import socket
import stat
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

MAGIC = b"VCTM"
VERSION = 1
EXTERNAL_VERSION = 2
REQUEST_KIND = 1
RESPONSE_KIND = 2
SCORING_SUM_QUERY_MAX_DOCUMENT_DOT = 1
DTYPE_F32 = 1
DTYPE_F16 = 2

HEADER = struct.Struct("<4sHHQQ")
REQUEST_FIXED = struct.Struct("<IIIBBH")
CANDIDATE_FIXED = struct.Struct("<II")
EXTERNAL_REQUEST_FIXED = struct.Struct("<IIIBBHI")
EXTERNAL_CANDIDATE_FIXED = struct.Struct("<IIII")
RESPONSE_FIXED = struct.Struct("<II")
RESULT = struct.Struct("<If")
ERROR_FIXED = struct.Struct("<II")

STATUS_INVALID_REQUEST = 1
STATUS_RESOURCE_LIMIT = 2
STATUS_COMPUTE_ERROR = 3
MAX_ERROR_BYTES = 64 * 1024


class SidecarError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Limits:
    max_request_bytes: int = 64 * 1024 * 1024
    max_batch_tokens: int = 1_000_000
    max_tensor_bytes: int = 1024 * 1024 * 1024
    max_candidates: int = 65_536


@dataclass(frozen=True)
class ExternalTensorRequest:
    model_contract_id: str
    tensor_ref: str
    rows: int
    dimension: int
    dtype: int
    checksum: str


@dataclass(frozen=True)
class InlineTensorCandidate:
    candidate_id: int
    rows: int
    payload: bytes


@dataclass(frozen=True)
class InlineTensorRequest:
    request_id: int
    dimension: int
    query_rows: int
    dtype: int
    query_payload: bytes
    candidates: tuple[InlineTensorCandidate, ...]


@dataclass(frozen=True)
class ExternalTensorCandidate:
    candidate_id: int
    descriptor: ExternalTensorRequest


@dataclass(frozen=True)
class ParsedExternalTensorRequest:
    request_id: int
    dimension: int
    query_rows: int
    dtype: int
    model_contract_id: str
    query_payload: bytes
    candidates: tuple[ExternalTensorCandidate, ...]


ParsedRequest = InlineTensorRequest | ParsedExternalTensorRequest


# A reference resolver returns the exact row-major scalar bytes whose SHA-256
# digest is stored in the descriptor. Production resolvers may adapt richer
# immutable object formats before returning this canonical tensor payload.
ExternalTensorResolver = Callable[[ExternalTensorRequest], bytes]


class Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, count: int) -> bytes:
        end = self.offset + count
        if count < 0 or end > len(self.payload):
            raise SidecarError(STATUS_INVALID_REQUEST, "truncated request")
        chunk = self.payload[self.offset : end]
        self.offset = end
        return chunk

    def unpack(self, layout: struct.Struct) -> tuple:
        return layout.unpack(self.take(layout.size))

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise SidecarError(STATUS_INVALID_REQUEST, "trailing request bytes")


def checked_elements(rows: int, dimension: int) -> int:
    if rows <= 0 or dimension <= 0:
        raise SidecarError(
            STATUS_INVALID_REQUEST, "tensor rows and dimension must be positive"
        )
    elements = rows * dimension
    if elements > (1 << 63) - 1:
        raise SidecarError(STATUS_RESOURCE_LIMIT, "tensor shape is too large")
    return elements


def dtype_size(dtype: int) -> int:
    if dtype == DTYPE_F32:
        return 4
    if dtype == DTYPE_F16:
        return 2
    raise SidecarError(STATUS_INVALID_REQUEST, "unsupported tensor dtype")


def checked_tensor_bytes(rows: int, dimension: int, dtype: int) -> int:
    return checked_elements(rows, dimension) * dtype_size(dtype)


def validate_finite_tensor_payload(
    payload: bytes, rows: int, dimension: int, dtype: int
) -> None:
    expected = checked_tensor_bytes(rows, dimension, dtype)
    if len(payload) != expected:
        raise SidecarError(
            STATUS_INVALID_REQUEST, "tensor byte length does not match its shape"
        )
    code = "f" if dtype == DTYPE_F32 else "e"
    try:
        values = struct.iter_unpack(f"<{code}", payload)
        if any(not math.isfinite(value[0]) for value in values):
            raise SidecarError(
                STATUS_INVALID_REQUEST, "tensor contains non-finite value"
            )
    except struct.error as error:
        raise SidecarError(STATUS_INVALID_REQUEST, str(error)) from error


def read_text(reader: Reader, length: int, maximum: int, field: str) -> str:
    if length <= 0 or length > maximum:
        raise SidecarError(STATUS_RESOURCE_LIMIT, f"invalid {field} length")
    try:
        value = reader.take(length).decode("utf-8")
    except UnicodeDecodeError as error:
        raise SidecarError(STATUS_INVALID_REQUEST, f"{field} is not UTF-8") from error
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SidecarError(
            STATUS_INVALID_REQUEST, f"{field} contains control characters"
        )
    return value


def read_tensor(
    reader: Reader, rows: int, dimension: int, dtype: int
) -> list[tuple[float, ...]]:
    elements = checked_elements(rows, dimension)
    if dtype == DTYPE_F32:
        code = "f"
        element_size = 4
    elif dtype == DTYPE_F16:
        code = "e"
        element_size = 2
    else:
        raise SidecarError(STATUS_INVALID_REQUEST, "unsupported tensor dtype")
    raw = reader.take(elements * element_size)
    try:
        values = struct.unpack(f"<{elements}{code}", raw)
    except struct.error as error:
        raise SidecarError(STATUS_INVALID_REQUEST, str(error)) from error
    if not all(math.isfinite(value) for value in values):
        raise SidecarError(STATUS_INVALID_REQUEST, "tensor contains non-finite value")
    return [
        tuple(values[offset : offset + dimension])
        for offset in range(0, elements, dimension)
    ]


def decode_resolved_tensor(
    payload: bytes, request: ExternalTensorRequest
) -> list[tuple[float, ...]]:
    expected_bytes = checked_tensor_bytes(
        request.rows, request.dimension, request.dtype
    )
    if len(payload) != expected_bytes:
        raise SidecarError(
            STATUS_INVALID_REQUEST,
            "resolved tensor byte length does not match descriptor",
        )
    expected_checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not hmac.compare_digest(expected_checksum, request.checksum):
        raise SidecarError(STATUS_INVALID_REQUEST, "resolved tensor checksum mismatch")
    reader = Reader(payload)
    tensor = read_tensor(reader, request.rows, request.dimension, request.dtype)
    reader.finish()
    return tensor


def tilemaxsim(
    query: list[tuple[float, ...]], document: list[tuple[float, ...]]
) -> float:
    if not query or not document:
        raise SidecarError(STATUS_INVALID_REQUEST, "tensor must not be empty")
    score = 0.0
    try:
        for query_vector in query:
            best = -math.inf
            for document_vector in document:
                dot = math.fsum(
                    left * right
                    for left, right in zip(query_vector, document_vector, strict=True)
                )
                best = max(best, dot)
            score += best
    except (OverflowError, ValueError) as error:
        raise SidecarError(STATUS_COMPUTE_ERROR, str(error)) from error
    if not math.isfinite(score):
        raise SidecarError(STATUS_COMPUTE_ERROR, "TileMaxSim result is non-finite")
    return score


def success_response(
    request_id: int,
    results: Iterable[tuple[int, float]],
    version: int = VERSION,
) -> bytes:
    results = list(results)
    body = bytearray(RESPONSE_FIXED.pack(0, len(results)))
    for candidate_id, similarity in results:
        body.extend(RESULT.pack(candidate_id, similarity))
    return HEADER.pack(MAGIC, version, RESPONSE_KIND, request_id, len(body)) + body


def error_response(
    request_id: int,
    status_code: int,
    message: str,
    version: int = VERSION,
) -> bytes:
    encoded = message.encode("utf-8", errors="replace")[:MAX_ERROR_BYTES]
    body = ERROR_FIXED.pack(status_code or STATUS_COMPUTE_ERROR, len(encoded)) + encoded
    return HEADER.pack(MAGIC, version, RESPONSE_KIND, request_id, len(body)) + body


def validate_request_fixed(
    dimension: int,
    query_rows: int,
    candidate_count: int,
    dtype: int,
    scoring: int,
    reserved: int,
    limits: Limits,
) -> None:
    if dimension == 0 or dimension > 60_000:
        raise SidecarError(STATUS_INVALID_REQUEST, "invalid tensor dimension")
    if query_rows == 0:
        raise SidecarError(STATUS_INVALID_REQUEST, "query tensor is empty")
    if candidate_count > limits.max_candidates:
        raise SidecarError(STATUS_RESOURCE_LIMIT, "too many candidates")
    dtype_size(dtype)
    if scoring != SCORING_SUM_QUERY_MAX_DOCUMENT_DOT:
        raise SidecarError(STATUS_INVALID_REQUEST, "unsupported scoring function")
    if reserved != 0:
        raise SidecarError(STATUS_INVALID_REQUEST, "reserved field must be zero")
    if query_rows > limits.max_batch_tokens:
        raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds token limit")
    if checked_tensor_bytes(query_rows, dimension, dtype) > limits.max_tensor_bytes:
        raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds tensor byte limit")


def process_inline_request(reader: Reader, limits: Limits) -> list[tuple[int, float]]:
    (
        dimension,
        query_rows,
        candidate_count,
        dtype,
        scoring,
        reserved,
    ) = reader.unpack(REQUEST_FIXED)
    validate_request_fixed(
        dimension, query_rows, candidate_count, dtype, scoring, reserved, limits
    )

    query = read_tensor(reader, query_rows, dimension, dtype)
    total_tokens = query_rows
    total_tensor_bytes = checked_tensor_bytes(query_rows, dimension, dtype)
    candidates: list[tuple[int, list[tuple[float, ...]]]] = []
    candidate_ids: set[int] = set()
    for _ in range(candidate_count):
        candidate_id, rows = reader.unpack(CANDIDATE_FIXED)
        if candidate_id in candidate_ids:
            raise SidecarError(STATUS_INVALID_REQUEST, "duplicate candidate ID")
        candidate_ids.add(candidate_id)
        total_tokens += rows
        total_tensor_bytes += checked_tensor_bytes(rows, dimension, dtype)
        if total_tokens > limits.max_batch_tokens:
            raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds token limit")
        if total_tensor_bytes > limits.max_tensor_bytes:
            raise SidecarError(
                STATUS_RESOURCE_LIMIT, "request exceeds tensor byte limit"
            )
        candidates.append((candidate_id, read_tensor(reader, rows, dimension, dtype)))
    reader.finish()
    return [
        (candidate_id, tilemaxsim(query, document))
        for candidate_id, document in candidates
    ]


def process_external_request(
    reader: Reader,
    limits: Limits,
    resolver: ExternalTensorResolver | None,
) -> list[tuple[int, float]]:
    (
        dimension,
        query_rows,
        candidate_count,
        dtype,
        scoring,
        reserved,
        contract_length,
    ) = reader.unpack(EXTERNAL_REQUEST_FIXED)
    validate_request_fixed(
        dimension, query_rows, candidate_count, dtype, scoring, reserved, limits
    )
    model_contract_id = read_text(reader, contract_length, 512, "model contract")
    query = read_tensor(reader, query_rows, dimension, dtype)
    total_tokens = query_rows
    total_tensor_bytes = checked_tensor_bytes(query_rows, dimension, dtype)
    descriptors: list[tuple[int, ExternalTensorRequest]] = []
    candidate_ids: set[int] = set()
    for _ in range(candidate_count):
        candidate_id, rows, reference_length, checksum_length = reader.unpack(
            EXTERNAL_CANDIDATE_FIXED
        )
        if candidate_id in candidate_ids:
            raise SidecarError(STATUS_INVALID_REQUEST, "duplicate candidate ID")
        candidate_ids.add(candidate_id)
        tensor_ref = read_text(reader, reference_length, 4096, "tensor reference")
        checksum = read_text(reader, checksum_length, 512, "tensor checksum")
        digest = checksum.removeprefix("sha256:")
        if (
            not checksum.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SidecarError(
                STATUS_INVALID_REQUEST,
                "tensor checksum must be a lowercase sha256 digest",
            )
        total_tokens += rows
        total_tensor_bytes += checked_tensor_bytes(rows, dimension, dtype)
        if total_tokens > limits.max_batch_tokens:
            raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds token limit")
        if total_tensor_bytes > limits.max_tensor_bytes:
            raise SidecarError(
                STATUS_RESOURCE_LIMIT, "request exceeds tensor byte limit"
            )
        descriptors.append(
            (
                candidate_id,
                ExternalTensorRequest(
                    model_contract_id=model_contract_id,
                    tensor_ref=tensor_ref,
                    rows=rows,
                    dimension=dimension,
                    dtype=dtype,
                    checksum=checksum,
                ),
            )
        )
    reader.finish()

    # Validate the complete control frame before performing any external I/O.
    if resolver is None:
        raise SidecarError(
            STATUS_COMPUTE_ERROR, "external tensor resolver is not configured"
        )
    results = []
    for candidate_id, descriptor in descriptors:
        payload = resolver(descriptor)
        if not isinstance(payload, bytes):
            raise SidecarError(
                STATUS_COMPUTE_ERROR,
                "external tensor resolver returned a non-bytes value",
            )
        document = decode_resolved_tensor(payload, descriptor)
        results.append((candidate_id, tilemaxsim(query, document)))
    return results


def parse_request_frame(
    frame: bytes,
    limits: Limits = Limits(),
    *,
    validate_finite: bool = True,
) -> ParsedRequest:
    """Decode and validate a request without resolving or computing tensors.

    The production CUDA sidecar uses this function so the protocol oracle and
    deployable executor share one strict wire decoder. All control data and the
    inline query are validated before a v2 resolver performs external I/O.
    """

    if len(frame) < HEADER.size:
        raise SidecarError(STATUS_INVALID_REQUEST, "truncated frame header")
    magic, version, kind, request_id, body_len = HEADER.unpack_from(frame)
    if magic != MAGIC:
        raise SidecarError(STATUS_INVALID_REQUEST, "invalid frame magic")
    if version not in (VERSION, EXTERNAL_VERSION):
        raise SidecarError(STATUS_INVALID_REQUEST, "unsupported protocol version")
    if kind != REQUEST_KIND:
        raise SidecarError(STATUS_INVALID_REQUEST, "unexpected message kind")
    if body_len != len(frame) - HEADER.size:
        raise SidecarError(STATUS_INVALID_REQUEST, "request length mismatch")
    if len(frame) > limits.max_request_bytes:
        raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds byte limit")

    reader = Reader(frame[HEADER.size :])
    if version == VERSION:
        (
            dimension,
            query_rows,
            candidate_count,
            dtype,
            scoring,
            reserved,
        ) = reader.unpack(REQUEST_FIXED)
        validate_request_fixed(
            dimension,
            query_rows,
            candidate_count,
            dtype,
            scoring,
            reserved,
            limits,
        )
        query_payload = reader.take(checked_tensor_bytes(query_rows, dimension, dtype))
        if validate_finite:
            validate_finite_tensor_payload(query_payload, query_rows, dimension, dtype)
        total_tokens = query_rows
        total_tensor_bytes = len(query_payload)
        candidate_ids: set[int] = set()
        candidates = []
        for _ in range(candidate_count):
            candidate_id, rows = reader.unpack(CANDIDATE_FIXED)
            if candidate_id in candidate_ids:
                raise SidecarError(STATUS_INVALID_REQUEST, "duplicate candidate ID")
            candidate_ids.add(candidate_id)
            payload_bytes = checked_tensor_bytes(rows, dimension, dtype)
            total_tokens += rows
            total_tensor_bytes += payload_bytes
            if total_tokens > limits.max_batch_tokens:
                raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds token limit")
            if total_tensor_bytes > limits.max_tensor_bytes:
                raise SidecarError(
                    STATUS_RESOURCE_LIMIT, "request exceeds tensor byte limit"
                )
            payload = reader.take(payload_bytes)
            if validate_finite:
                validate_finite_tensor_payload(payload, rows, dimension, dtype)
            candidates.append(InlineTensorCandidate(candidate_id, rows, payload))
        reader.finish()
        return InlineTensorRequest(
            request_id,
            dimension,
            query_rows,
            dtype,
            query_payload,
            tuple(candidates),
        )

    (
        dimension,
        query_rows,
        candidate_count,
        dtype,
        scoring,
        reserved,
        contract_length,
    ) = reader.unpack(EXTERNAL_REQUEST_FIXED)
    validate_request_fixed(
        dimension,
        query_rows,
        candidate_count,
        dtype,
        scoring,
        reserved,
        limits,
    )
    model_contract_id = read_text(reader, contract_length, 512, "model contract")
    query_payload = reader.take(checked_tensor_bytes(query_rows, dimension, dtype))
    if validate_finite:
        validate_finite_tensor_payload(query_payload, query_rows, dimension, dtype)
    total_tokens = query_rows
    total_tensor_bytes = len(query_payload)
    candidate_ids = set()
    candidates = []
    for _ in range(candidate_count):
        candidate_id, rows, reference_length, checksum_length = reader.unpack(
            EXTERNAL_CANDIDATE_FIXED
        )
        if candidate_id in candidate_ids:
            raise SidecarError(STATUS_INVALID_REQUEST, "duplicate candidate ID")
        candidate_ids.add(candidate_id)
        tensor_ref = read_text(reader, reference_length, 4096, "tensor reference")
        checksum = read_text(reader, checksum_length, 512, "tensor checksum")
        digest = checksum.removeprefix("sha256:")
        if (
            not checksum.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SidecarError(
                STATUS_INVALID_REQUEST,
                "tensor checksum must be a lowercase sha256 digest",
            )
        total_tokens += rows
        total_tensor_bytes += checked_tensor_bytes(rows, dimension, dtype)
        if total_tokens > limits.max_batch_tokens:
            raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds token limit")
        if total_tensor_bytes > limits.max_tensor_bytes:
            raise SidecarError(
                STATUS_RESOURCE_LIMIT, "request exceeds tensor byte limit"
            )
        candidates.append(
            ExternalTensorCandidate(
                candidate_id,
                ExternalTensorRequest(
                    model_contract_id=model_contract_id,
                    tensor_ref=tensor_ref,
                    rows=rows,
                    dimension=dimension,
                    dtype=dtype,
                    checksum=checksum,
                ),
            )
        )
    reader.finish()
    return ParsedExternalTensorRequest(
        request_id,
        dimension,
        query_rows,
        dtype,
        model_contract_id,
        query_payload,
        tuple(candidates),
    )


def process_frame(
    frame: bytes,
    limits: Limits = Limits(),
    resolver: ExternalTensorResolver | None = None,
) -> bytes:
    request_id = 0
    response_version = VERSION
    try:
        if len(frame) < HEADER.size:
            raise SidecarError(STATUS_INVALID_REQUEST, "truncated frame header")
        magic, version, kind, request_id, body_len = HEADER.unpack_from(frame)
        if version in (VERSION, EXTERNAL_VERSION):
            response_version = version
        if magic != MAGIC:
            raise SidecarError(STATUS_INVALID_REQUEST, "invalid frame magic")
        if version not in (VERSION, EXTERNAL_VERSION):
            raise SidecarError(STATUS_INVALID_REQUEST, "unsupported protocol version")
        if kind != REQUEST_KIND:
            raise SidecarError(STATUS_INVALID_REQUEST, "unexpected message kind")
        if body_len != len(frame) - HEADER.size:
            raise SidecarError(STATUS_INVALID_REQUEST, "request length mismatch")
        if len(frame) > limits.max_request_bytes:
            raise SidecarError(STATUS_RESOURCE_LIMIT, "request exceeds byte limit")

        reader = Reader(frame[HEADER.size :])
        if version == VERSION:
            results = process_inline_request(reader, limits)
        else:
            results = process_external_request(reader, limits, resolver)
        return success_response(request_id, results, response_version)
    except SidecarError as error:
        return error_response(request_id, error.status, str(error), response_version)
    except Exception as error:  # Keep protocol failures inside the response boundary.
        return error_response(
            request_id, STATUS_COMPUTE_ERROR, str(error), response_version
        )


def receive_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise SidecarError(
                STATUS_INVALID_REQUEST, "connection closed during request"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def handle_connection(
    connection: socket.socket,
    limits: Limits,
    resolver: ExternalTensorResolver | None = None,
) -> None:
    request_id = 0
    response_version = VERSION
    try:
        header = receive_exact(connection, HEADER.size)
        _, version, _, request_id, body_len = HEADER.unpack(header)
        if version in (VERSION, EXTERNAL_VERSION):
            response_version = version
        if body_len > limits.max_request_bytes - HEADER.size:
            response = error_response(
                request_id,
                STATUS_RESOURCE_LIMIT,
                "request exceeds byte limit",
                response_version,
            )
        else:
            body = receive_exact(connection, body_len)
            response = process_frame(header + body, limits, resolver)
    except SidecarError as error:
        response = error_response(
            request_id, error.status, str(error), response_version
        )
    except Exception as error:
        response = error_response(
            request_id, STATUS_COMPUTE_ERROR, str(error), response_version
        )
    connection.sendall(response)


def remove_stale_socket(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket path: {path}")
    path.unlink()


def serve(
    socket_path: Path,
    limits: Limits,
    socket_mode: int = 0o600,
    once: bool = False,
    stop: threading.Event | None = None,
    resolver: ExternalTensorResolver | None = None,
) -> None:
    stop = stop or threading.Event()
    remove_stale_socket(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(os.fspath(socket_path))
        os.chmod(socket_path, socket_mode)
        listener.listen(16)
        listener.settimeout(0.25)
        bound_identity = socket_path.lstat().st_dev, socket_path.lstat().st_ino
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                handle_connection(connection, limits, resolver)
            if once:
                break
    finally:
        listener.close()
        try:
            current = socket_path.lstat()
            if (current.st_dev, current.st_ino) == bound_identity:
                socket_path.unlink()
        except (FileNotFoundError, UnboundLocalError):
            pass


def parse_mode(value: str) -> int:
    mode = int(value, 8)
    if mode < 0 or mode > 0o777:
        raise argparse.ArgumentTypeError("socket mode must be between 000 and 777")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--socket-mode", type=parse_mode, default=0o600)
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-batch-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-tensor-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--max-candidates", type=int, default=65_536)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    limits = Limits(
        max_request_bytes=args.max_request_bytes,
        max_batch_tokens=args.max_batch_tokens,
        max_tensor_bytes=args.max_tensor_bytes,
        max_candidates=args.max_candidates,
    )
    if (
        min(
            limits.max_request_bytes,
            limits.max_batch_tokens,
            limits.max_tensor_bytes,
            limits.max_candidates,
        )
        <= 0
    ):
        parser.error("all limits must be positive")

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    serve(args.socket, limits, args.socket_mode, args.once, stop)


if __name__ == "__main__":
    main()
