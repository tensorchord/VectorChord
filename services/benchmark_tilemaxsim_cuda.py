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

"""Reproducible synthetic load probe for the CUDA TileMaxSim executor."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from devtools import tilemaxsim_reference_sidecar as protocol
from services.tilemaxsim_cuda_sidecar import TorchTileMaxsimEngine, positive_int


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def canonical_payload(tensor: torch.Tensor, dtype: int) -> bytes:
    scalar_dtype = torch.float32 if dtype == protocol.DTYPE_F32 else torch.float16
    return tensor.to(dtype=scalar_dtype).contiguous().numpy().tobytes()


def normalized_tensor(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    tensor = torch.randn(shape, dtype=torch.float32, generator=generator)
    return torch.nn.functional.normalize(tensor, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--dimension", type=positive_int, default=320)
    parser.add_argument("--query-rows", type=positive_int, default=32)
    parser.add_argument("--document-rows", type=positive_int, default=747)
    parser.add_argument("--candidates", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--max-device-bytes", type=positive_int, default=8 * 1024 * 1024 * 1024
    )
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()

    dtype = protocol.DTYPE_F32 if args.dtype == "f32" else protocol.DTYPE_F16
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    query = normalized_tensor((args.query_rows, args.dimension), generator)
    document_tensors = normalized_tensor(
        (args.candidates, args.document_rows, args.dimension), generator
    )
    query_payload = canonical_payload(query, dtype)
    documents = [
        (
            candidate_id,
            args.document_rows,
            canonical_payload(document_tensors[candidate_id], dtype),
        )
        for candidate_id in range(args.candidates)
    ]
    del document_tensors

    engine = TorchTileMaxsimEngine(
        args.device, args.max_device_bytes, args.allow_tf32, 1
    )
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)

    total_samples: list[float] = []
    queue_samples: list[float] = []
    compute_samples: list[float] = []
    score_checksum = 0.0
    for iteration in range(args.warmup + args.iterations):
        started = time.perf_counter()
        results, queue_ms, compute_ms = engine.score(
            query_payload,
            args.query_rows,
            args.dimension,
            dtype,
            documents,
            time.monotonic() + 300,
            lambda: False,
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        if iteration >= args.warmup:
            total_samples.append(total_ms)
            queue_samples.append(queue_ms)
            compute_samples.append(compute_ms)
            score_checksum = math.fsum(score for _, score in results)

    output = {
        "benchmark": "tilemaxsim_cuda_synthetic_v1",
        "device": str(engine.device),
        "device_name": (
            torch.cuda.get_device_name(engine.device)
            if engine.device.type == "cuda"
            else "cpu"
        ),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "dimension": args.dimension,
        "query_rows": args.query_rows,
        "document_rows": args.document_rows,
        "candidates": args.candidates,
        "candidate_tokens": args.candidates * args.document_rows,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "allow_tf32": args.allow_tf32,
        "max_device_bytes": args.max_device_bytes,
        "latency_ms": {
            "mean": round(statistics.fmean(total_samples), 3),
            "p50": round(percentile(total_samples, 0.50), 3),
            "p95": round(percentile(total_samples, 0.95), 3),
            "p99": round(percentile(total_samples, 0.99), 3),
            "queue_mean": round(statistics.fmean(queue_samples), 3),
            "compute_mean": round(statistics.fmean(compute_samples), 3),
        },
        "score_checksum": score_checksum,
    }
    if engine.device.type == "cuda":
        output["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            engine.device
        )
        output["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            engine.device
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
