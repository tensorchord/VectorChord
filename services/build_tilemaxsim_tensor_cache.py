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

"""Publish NPY page tensors into the sidecar's immutable SHA-256 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from services.tilemaxsim_cuda_sidecar import positive_int


def canonical_tensor(path: Path, expected_rows: int, expected_dim: int) -> np.ndarray:
    tensor = np.load(path, mmap_mode="r", allow_pickle=False)
    if tensor.ndim != 2 or tensor.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"{path}: expected shape {(expected_rows, expected_dim)}, got {tensor.shape}"
        )
    if tensor.dtype == np.dtype("float16"):
        little_dtype = np.dtype("<f2")
    elif tensor.dtype == np.dtype("float32"):
        little_dtype = np.dtype("<f4")
    else:
        raise ValueError(f"{path}: expected float16 or float32, got {tensor.dtype}")
    if not tensor.flags.c_contiguous:
        raise ValueError(f"{path}: tensor must be C-contiguous")
    if tensor.dtype != little_dtype:
        raise ValueError(f"{path}: tensor must use little-endian canonical scalars")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{path}: tensor contains non-finite values")
    return tensor


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_payload(path: Path, payload: memoryview, digest: str, fsync: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != len(payload):
            raise ValueError(f"existing cache payload has wrong size: {path}")
        if file_digest(path) != digest:
            raise ValueError(f"existing cache payload has wrong checksum: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.stat().st_size != len(payload):
                raise ValueError(f"concurrent cache payload has wrong size: {path}")
            if file_digest(path) != digest:
                raise ValueError(f"concurrent cache payload has wrong checksum: {path}")
        if fsync:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def process_record(
    record: dict[str, object],
    source_root: Path,
    cache_root: Path,
    fsync: bool,
    dry_run: bool,
) -> dict[str, object]:
    page_key = record.get("page_key")
    relative = record.get("embedding_file")
    rows = record.get("n_tokens")
    dimension = record.get("dim")
    if not isinstance(page_key, str) or not page_key:
        raise ValueError("manifest record has no page_key")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"manifest page {page_key} has no embedding_file")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"manifest page {page_key} has invalid n_tokens")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(f"manifest page {page_key} has invalid dim")
    source = (source_root / relative).resolve(strict=True)
    try:
        source.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"embedding path escapes the source root: {relative}"
        ) from error
    tensor = canonical_tensor(source, rows, dimension)
    payload = memoryview(tensor).cast("B")
    digest = hashlib.sha256(payload).hexdigest()
    destination = cache_root / digest[:2] / f"{digest}.bin"
    if not dry_run:
        write_payload(destination, payload, digest, fsync)
    dtype_name = "float16" if tensor.dtype == np.dtype("float16") else "float32"
    return {
        "page_key": page_key,
        "tensor_ref": f"sha256://{digest}",
        "tensor_rows": rows,
        "tensor_dim": dimension,
        "tensor_dtype": dtype_name,
        "tensor_checksum": f"sha256:{digest}",
        "canonical_bytes": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--descriptor-manifest", required=True, type=Path)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--no-fsync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.manifest.resolve(strict=True).parent
    cache_root = args.cache_root.resolve()
    if not cache_root.is_absolute():
        parser.error("--cache-root must be absolute")
    records = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at manifest line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError("manifest is empty")

    cache_root.mkdir(parents=True, exist_ok=True)
    descriptors = []
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        results = workers.map(
            lambda record: process_record(
                record,
                source_root,
                cache_root,
                not args.no_fsync,
                args.dry_run,
            ),
            records,
        )
        for completed, item in enumerate(results, 1):
            descriptors.append(item)
            if completed % 1000 == 0 or completed == len(records):
                print(
                    json.dumps(
                        {"event": "tensor_cache_progress", "completed": completed},
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    output_parent = args.descriptor_manifest.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.descriptor_manifest.name}.",
        suffix=".tmp",
        dir=output_parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            for item in descriptors:
                stream.write(json.dumps(item, separators=(",", ":"), sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, args.descriptor_manifest)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass

    total_bytes = sum(int(item["canonical_bytes"]) for item in descriptors)
    print(
        json.dumps(
            {
                "pages": len(descriptors),
                "canonical_bytes": total_bytes,
                "cache_root": os.fspath(cache_root),
                "descriptor_manifest": os.fspath(args.descriptor_manifest),
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
