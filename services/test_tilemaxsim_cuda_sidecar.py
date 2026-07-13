# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
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

import numpy as np
import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from devtools.test_tilemaxsim_reference_sidecar import (
    decode_response,
    external_request_frame,
    request_frame,
)
from services import tilemaxsim_cuda_sidecar as cuda_sidecar
from services.build_tilemaxsim_tensor_cache import process_record


class CapturingMetrics(cuda_sidecar.JsonMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def emit(self, fields: dict[str, object]) -> None:
        with self.lock:
            self.events.append(fields.copy())


def write_content_addressed(root: Path, payload: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    directory = root / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{digest}.bin").write_bytes(payload)
    return f"sha256://{digest}", f"sha256:{digest}"


class CudaSidecarTest(unittest.TestCase):
    def test_cache_builder_publishes_resolver_compatible_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            cache_root = root / "cache"
            source_root.mkdir()
            tensor = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f2")
            np.save(source_root / "page.npy", tensor, allow_pickle=False)
            descriptor = process_record(
                {
                    "page_key": "page-1",
                    "embedding_file": "page.npy",
                    "n_tokens": 2,
                    "dim": 2,
                },
                source_root,
                cache_root,
                False,
                False,
            )
            self.assertEqual(descriptor["tensor_dtype"], "float16")
            resolver = cuda_sidecar.ContentAddressedResolver({"model@1": cache_root}, 0)
            try:
                resolved = resolver.resolve(
                    protocol.ExternalTensorRequest(
                        "model@1",
                        str(descriptor["tensor_ref"]),
                        2,
                        2,
                        protocol.DTYPE_F16,
                        str(descriptor["tensor_checksum"]),
                    )
                )
                self.assertEqual(resolved.payload, tensor.tobytes())
            finally:
                resolver.close()

    def test_vectorized_finite_validation_rejects_non_finite_values(self) -> None:
        cuda_sidecar.validate_finite_payload(
            struct.pack("<2e", 1.0, 0.0), 1, 2, protocol.DTYPE_F16
        )
        with self.assertRaisesRegex(protocol.SidecarError, "non-finite"):
            cuda_sidecar.validate_finite_payload(
                struct.pack("<2f", 1.0, float("nan")),
                1,
                2,
                protocol.DTYPE_F32,
            )

    def test_content_addressed_resolver_validates_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = struct.pack("<4e", 1.0, 0.0, 0.0, 1.0)
            tensor_ref, checksum = write_content_addressed(root, payload)
            resolver = cuda_sidecar.ContentAddressedResolver({"model@1": root}, 1024)
            try:
                request = protocol.ExternalTensorRequest(
                    "model@1",
                    tensor_ref,
                    2,
                    2,
                    protocol.DTYPE_F16,
                    checksum,
                )
                first = resolver.resolve(request)
                second = resolver.resolve(request)
                self.assertEqual(first.payload, payload)
                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)

                bad = protocol.ExternalTensorRequest(
                    "model@1",
                    tensor_ref,
                    2,
                    2,
                    protocol.DTYPE_F16,
                    "sha256:" + "0" * 64,
                )
                with self.assertRaisesRegex(protocol.SidecarError, "disagree"):
                    resolver.resolve(bad)
            finally:
                resolver.close()

    def test_content_addressed_resolver_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            payload = struct.pack("<f", 1.0)
            tensor_ref, checksum = write_content_addressed(root, payload)
            digest = checksum.removeprefix("sha256:")
            tensor_path = root / digest[:2] / f"{digest}.bin"
            target = Path(directory) / "target.bin"
            target.write_bytes(payload)
            tensor_path.unlink()
            tensor_path.symlink_to(target)
            resolver = cuda_sidecar.ContentAddressedResolver({"model@1": root}, 0)
            try:
                request = protocol.ExternalTensorRequest(
                    "model@1",
                    tensor_ref,
                    1,
                    1,
                    protocol.DTYPE_F32,
                    checksum,
                )
                with self.assertRaises(protocol.SidecarError):
                    resolver.resolve(request)
            finally:
                resolver.close()

    def test_cpu_engine_matches_oracle_across_device_chunks(self) -> None:
        query = [[1.0, 0.0], [0.0, 1.0]]
        candidates = [
            (17, [[1.0, 0.0], [0.0, 1.0]]),
            (3, [[0.5, 0.5], [0.25, 0.25]]),
        ]
        frame = request_frame(41, protocol.DTYPE_F32, query, candidates)
        parsed = protocol.parse_request_frame(frame)
        self.assertIsInstance(parsed, protocol.InlineTensorRequest)
        assert isinstance(parsed, protocol.InlineTensorRequest)
        documents = [
            (candidate.candidate_id, candidate.rows, candidate.payload)
            for candidate in parsed.candidates
        ]
        # 64 bytes fits one candidate but not both, exercising internal
        # all-or-nothing device chunking.
        engine = cuda_sidecar.TorchTileMaxsimEngine("cpu", 64, False, 1)
        results, _, _ = engine.score(
            parsed.query_payload,
            parsed.query_rows,
            parsed.dimension,
            parsed.dtype,
            documents,
            time.monotonic() + 2,
            lambda: False,
        )
        _, status, oracle = decode_response(protocol.process_frame(frame))
        self.assertEqual(status, 0)
        self.assertEqual(results, oracle)

    def test_compute_capacity_wait_uses_overall_deadline(self) -> None:
        engine = cuda_sidecar.TorchTileMaxsimEngine("cpu", 1024, False, 1)
        self.assertTrue(engine.compute_slots.acquire(blocking=False))
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(protocol.SidecarError, "CUDA capacity"):
                engine.score(
                    struct.pack("<f", 1.0),
                    1,
                    1,
                    protocol.DTYPE_F32,
                    [(0, 1, struct.pack("<f", 1.0))],
                    time.monotonic() + 0.05,
                    lambda: False,
                )
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            engine.compute_slots.release()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_f16_matches_cpu_protocol_oracle(self) -> None:
        query = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]]
        candidates = [
            (7, [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]]),
            (2, [[0.5, 0.5, 0.0], [-0.5, 0.25, 1.0]]),
        ]
        frame = request_frame(52, protocol.DTYPE_F16, query, candidates)
        parsed = protocol.parse_request_frame(frame)
        assert isinstance(parsed, protocol.InlineTensorRequest)
        engine = cuda_sidecar.TorchTileMaxsimEngine("cuda:0", 1024 * 1024, False, 1)
        results, _, _ = engine.score(
            parsed.query_payload,
            parsed.query_rows,
            parsed.dimension,
            parsed.dtype,
            [
                (candidate.candidate_id, candidate.rows, candidate.payload)
                for candidate in parsed.candidates
            ],
            time.monotonic() + 5,
            lambda: False,
        )
        _, status, oracle = decode_response(protocol.process_frame(frame))
        self.assertEqual(status, 0)
        assert isinstance(oracle, list)
        self.assertEqual([item[0] for item in results], [item[0] for item in oracle])
        for (_, actual), (_, expected) in zip(results, oracle, strict=True):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_v2_unix_socket_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            root = directory_path / "tensors"
            root.mkdir()
            tensor = [[1.0, 0.0], [0.0, 1.0]]
            payload = struct.pack("<4e", *sum(tensor, []))
            tensor_ref, _ = write_content_addressed(root, payload)
            frame, _ = external_request_frame(
                61,
                protocol.DTYPE_F16,
                [[1.0, 0.0], [0.0, 1.0]],
                "model@1",
                [(9, tensor_ref, tensor)],
            )
            resolver = cuda_sidecar.ContentAddressedResolver({"model@1": root}, 1024)
            metrics = CapturingMetrics()
            service = cuda_sidecar.TileMaxsimService(
                protocol.Limits(),
                resolver,
                cuda_sidecar.TorchTileMaxsimEngine("cpu", 1024 * 1024, False, 1),
                2000,
                metrics,
            )
            socket_path = directory_path / "tilemaxsim.sock"
            stop = threading.Event()
            thread = threading.Thread(
                target=cuda_sidecar.serve,
                args=(socket_path, 0o600, 4, 2, service, stop),
                kwargs={"once": True},
                daemon=True,
            )
            thread.start()
            for _ in range(100):
                if socket_path.exists():
                    break
                time.sleep(0.01)
            else:
                self.fail("CUDA sidecar socket was not created")
            self.assertEqual(stat.S_IMODE(socket_path.stat().st_mode), 0o600)

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(os.fspath(socket_path))
                connection.sendall(frame)
                header = protocol.receive_exact(connection, protocol.HEADER.size)
                body_len = protocol.HEADER.unpack(header)[4]
                response = header + protocol.receive_exact(connection, body_len)
            thread.join(timeout=3)
            resolver.close()
            self.assertFalse(thread.is_alive())
            _, status, results = decode_response(response)
            self.assertEqual(status, 0)
            self.assertEqual(results, [(9, 2.0)])
            request_events = [
                event
                for event in metrics.events
                if event.get("event") == "tilemaxsim_request"
            ]
            self.assertEqual(len(request_events), 1)
            self.assertEqual(request_events[0]["source"], "content_addressed")
            self.assertEqual(request_events[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
