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

from __future__ import annotations

import hashlib
import os
import socket
import stat
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    from . import tilemaxsim_reference_sidecar as sidecar
except ImportError:
    import tilemaxsim_reference_sidecar as sidecar


def request_frame(
    request_id: int,
    dtype: int,
    query: list[list[float]],
    candidates: list[tuple[int, list[list[float]]]],
) -> bytes:
    dimension = len(query[0])
    code = "f" if dtype == sidecar.DTYPE_F32 else "e"
    body = bytearray(
        sidecar.REQUEST_FIXED.pack(
            dimension,
            len(query),
            len(candidates),
            dtype,
            sidecar.SCORING_SUM_QUERY_MAX_DOCUMENT_DOT,
            0,
        )
    )
    body.extend(struct.pack(f"<{len(query) * dimension}{code}", *sum(query, [])))
    for candidate_id, tensor in candidates:
        body.extend(sidecar.CANDIDATE_FIXED.pack(candidate_id, len(tensor)))
        body.extend(struct.pack(f"<{len(tensor) * dimension}{code}", *sum(tensor, [])))
    return (
        sidecar.HEADER.pack(
            sidecar.MAGIC,
            sidecar.VERSION,
            sidecar.REQUEST_KIND,
            request_id,
            len(body),
        )
        + body
    )


def external_request_frame(
    request_id: int,
    dtype: int,
    query: list[list[float]],
    model_contract_id: str,
    candidates: list[tuple[int, str, list[list[float]]]],
) -> tuple[bytes, dict[str, bytes]]:
    dimension = len(query[0])
    code = "f" if dtype == sidecar.DTYPE_F32 else "e"
    contract = model_contract_id.encode()
    body = bytearray(
        sidecar.EXTERNAL_REQUEST_FIXED.pack(
            dimension,
            len(query),
            len(candidates),
            dtype,
            sidecar.SCORING_SUM_QUERY_MAX_DOCUMENT_DOT,
            0,
            len(contract),
        )
    )
    body.extend(contract)
    body.extend(struct.pack(f"<{len(query) * dimension}{code}", *sum(query, [])))
    objects = {}
    for candidate_id, tensor_ref, tensor in candidates:
        payload = struct.pack(f"<{len(tensor) * dimension}{code}", *sum(tensor, []))
        objects[tensor_ref] = payload
        reference = tensor_ref.encode()
        checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}".encode()
        body.extend(
            sidecar.EXTERNAL_CANDIDATE_FIXED.pack(
                candidate_id, len(tensor), len(reference), len(checksum)
            )
        )
        body.extend(reference)
        body.extend(checksum)
    return (
        sidecar.HEADER.pack(
            sidecar.MAGIC,
            sidecar.EXTERNAL_VERSION,
            sidecar.REQUEST_KIND,
            request_id,
            len(body),
        )
        + body,
        objects,
    )


def decode_response(frame: bytes) -> tuple[int, int, list[tuple[int, float]] | str]:
    magic, version, kind, request_id, body_len = sidecar.HEADER.unpack_from(frame)
    assert magic == sidecar.MAGIC
    assert version in (sidecar.VERSION, sidecar.EXTERNAL_VERSION)
    assert kind == sidecar.RESPONSE_KIND
    assert len(frame) == sidecar.HEADER.size + body_len
    status, count_or_length = sidecar.RESPONSE_FIXED.unpack_from(
        frame, sidecar.HEADER.size
    )
    offset = sidecar.HEADER.size + sidecar.RESPONSE_FIXED.size
    if status:
        return request_id, status, frame[offset : offset + count_or_length].decode()
    results = []
    for _ in range(count_or_length):
        results.append(sidecar.RESULT.unpack_from(frame, offset))
        offset += sidecar.RESULT.size
    assert offset == len(frame)
    return request_id, status, results


class ReferenceSidecarTest(unittest.TestCase):
    def test_f32_exact_scores_and_opaque_ids(self) -> None:
        frame = request_frame(
            41,
            sidecar.DTYPE_F32,
            [[1.0, 0.0], [0.0, 1.0]],
            [
                (17, [[1.0, 0.0], [0.0, 1.0]]),
                (3, [[0.5, 0.5]]),
            ],
        )
        request_id, status, results = decode_response(sidecar.process_frame(frame))

        self.assertEqual(request_id, 41)
        self.assertEqual(status, 0)
        self.assertEqual(results, [(17, 2.0), (3, 1.0)])

    def test_f16_exact_scores(self) -> None:
        frame = request_frame(
            42,
            sidecar.DTYPE_F16,
            [[1.0, 0.0], [0.0, 1.0]],
            [(0, [[0.75, 0.0], [0.0, 0.5]])],
        )
        _, status, results = decode_response(sidecar.process_frame(frame))

        self.assertEqual(status, 0)
        self.assertEqual(results, [(0, 1.25)])

    def test_shared_raw_decoder_preserves_inline_payloads(self) -> None:
        frame = request_frame(
            420,
            sidecar.DTYPE_F16,
            [[1.0, 0.0]],
            [(11, [[0.5, 0.25]])],
        )
        request = sidecar.parse_request_frame(frame)
        self.assertIsInstance(request, sidecar.InlineTensorRequest)
        assert isinstance(request, sidecar.InlineTensorRequest)
        self.assertEqual(request.request_id, 420)
        self.assertEqual(request.query_rows, 1)
        self.assertEqual(request.dimension, 2)
        self.assertEqual([item.candidate_id for item in request.candidates], [11])
        self.assertEqual(len(request.query_payload), 4)
        self.assertEqual(len(request.candidates[0].payload), 4)

    def test_duplicate_id_and_non_finite_input_fail_closed(self) -> None:
        duplicate = request_frame(
            43,
            sidecar.DTYPE_F32,
            [[1.0]],
            [(7, [[1.0]]), (7, [[2.0]])],
        )
        _, status, message = decode_response(sidecar.process_frame(duplicate))
        self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)
        self.assertIn("duplicate candidate ID", message)

        non_finite = request_frame(
            44,
            sidecar.DTYPE_F32,
            [[float("nan")]],
            [(0, [[1.0]])],
        )
        _, status, message = decode_response(sidecar.process_frame(non_finite))
        self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)
        self.assertIn("non-finite", message)

    def test_truncated_and_oversized_frames_fail_closed(self) -> None:
        valid = request_frame(45, sidecar.DTYPE_F32, [[1.0]], [(0, [[1.0]])])
        _, status, message = decode_response(sidecar.process_frame(valid[:-1]))
        self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)
        self.assertIn("length mismatch", message)

        _, status, message = decode_response(
            sidecar.process_frame(valid, sidecar.Limits(max_request_bytes=32))
        )
        self.assertEqual(status, sidecar.STATUS_RESOURCE_LIMIT)
        self.assertIn("byte limit", message)

    def test_header_reserved_trailing_and_token_limit_fail_closed(self) -> None:
        valid = request_frame(47, sidecar.DTYPE_F32, [[1.0]], [(9, [[1.0]])])
        invalid_frames = []
        for offset, value in ((0, 0), (4, 3), (6, 2)):
            invalid = bytearray(valid)
            invalid[offset] = value
            invalid_frames.append(bytes(invalid))

        reserved = bytearray(valid)
        reserved[sidecar.HEADER.size + 14] = 1
        invalid_frames.append(bytes(reserved))

        trailing = bytearray(valid)
        trailing.extend(b"x")
        struct.pack_into("<Q", trailing, 16, len(trailing) - sidecar.HEADER.size)
        invalid_frames.append(bytes(trailing))

        for invalid in invalid_frames:
            _, status, _ = decode_response(sidecar.process_frame(invalid))
            self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)

        _, status, message = decode_response(
            sidecar.process_frame(valid, sidecar.Limits(max_batch_tokens=1))
        )
        self.assertEqual(status, sidecar.STATUS_RESOURCE_LIMIT)
        self.assertIn("token limit", message)

    def test_external_v2_resolves_checksums_and_scores_opaque_ids(self) -> None:
        frame, objects = external_request_frame(
            48,
            sidecar.DTYPE_F16,
            [[1.0, 0.0], [0.0, 1.0]],
            "colqwen@immutable-revision",
            [
                (77, "object://fixture/page-1", [[1.0, 0.0], [0.0, 1.0]]),
                (4, "object://fixture/page-2", [[0.5, 0.5]]),
            ],
        )
        seen = []

        def resolver(request: sidecar.ExternalTensorRequest) -> bytes:
            seen.append(request)
            return objects[request.tensor_ref]

        request_id, status, results = decode_response(
            sidecar.process_frame(frame, resolver=resolver)
        )

        self.assertEqual(request_id, 48)
        self.assertEqual(status, 0)
        self.assertEqual(results, [(77, 2.0), (4, 1.0)])
        self.assertEqual(
            {request.model_contract_id for request in seen},
            {"colqwen@immutable-revision"},
        )
        self.assertEqual({request.tensor_ref for request in seen}, set(objects))

        parsed = sidecar.parse_request_frame(frame)
        self.assertIsInstance(parsed, sidecar.ParsedExternalTensorRequest)
        assert isinstance(parsed, sidecar.ParsedExternalTensorRequest)
        self.assertEqual(parsed.model_contract_id, "colqwen@immutable-revision")
        self.assertEqual(
            [candidate.candidate_id for candidate in parsed.candidates], [77, 4]
        )

    def test_external_v2_fails_closed_without_resolver_or_on_checksum_mismatch(
        self,
    ) -> None:
        frame, objects = external_request_frame(
            49,
            sidecar.DTYPE_F32,
            [[1.0]],
            "contract@1",
            [(0, "object://immutable/page", [[2.0]])],
        )
        _, status, message = decode_response(sidecar.process_frame(frame))
        self.assertEqual(status, sidecar.STATUS_COMPUTE_ERROR)
        self.assertIn("resolver is not configured", message)

        def corrupt_resolver(request: sidecar.ExternalTensorRequest) -> bytes:
            return objects[request.tensor_ref] + b"x"

        _, status, message = decode_response(
            sidecar.process_frame(frame, resolver=corrupt_resolver)
        )
        self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)
        self.assertIn("byte length", message)

    def test_external_v2_validates_complete_control_frame_before_resolution(
        self,
    ) -> None:
        frame, objects = external_request_frame(
            50,
            sidecar.DTYPE_F32,
            [[1.0, 0.0]],
            "contract@1",
            [(0, "object://immutable/page", [[1.0, 0.0]])],
        )
        invalid = bytearray(frame)
        contract_length = len("contract@1")
        candidate_offset = (
            sidecar.HEADER.size
            + sidecar.EXTERNAL_REQUEST_FIXED.size
            + contract_length
            + 8
        )
        reference_offset = candidate_offset + sidecar.EXTERNAL_CANDIDATE_FIXED.size
        invalid[reference_offset] = 0
        called = False

        def resolver(request: sidecar.ExternalTensorRequest) -> bytes:
            nonlocal called
            called = True
            return objects[request.tensor_ref]

        _, status, message = decode_response(
            sidecar.process_frame(bytes(invalid), resolver=resolver)
        )
        self.assertEqual(status, sidecar.STATUS_INVALID_REQUEST)
        self.assertIn("control characters", message)
        self.assertFalse(called)

    def test_unix_socket_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tilemaxsim.sock"
            thread = threading.Thread(
                target=sidecar.serve,
                args=(path, sidecar.Limits()),
                kwargs={"once": True},
                daemon=True,
            )
            thread.start()
            for _ in range(100):
                if path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o600:
                    break
                time.sleep(0.01)
            else:
                self.fail("sidecar socket was not created with mode 0600")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            frame = request_frame(
                46,
                sidecar.DTYPE_F32,
                [[1.0, 0.0], [0.0, 1.0]],
                [(5, [[1.0, 0.0], [0.0, 1.0]])],
            )
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(os.fspath(path))
                connection.sendall(frame)
                header = connection.recv(sidecar.HEADER.size)
                while len(header) < sidecar.HEADER.size:
                    header += connection.recv(sidecar.HEADER.size - len(header))
                body_len = sidecar.HEADER.unpack(header)[4]
                body = b""
                while len(body) < body_len:
                    body += connection.recv(body_len - len(body))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

            request_id, status, results = decode_response(header + body)
            self.assertEqual(request_id, 46)
            self.assertEqual(status, 0)
            self.assertEqual(results, [(5, 2.0)])
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
