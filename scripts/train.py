import itertools
from multiprocessing import Pool, cpu_count
from time import perf_counter
import argparse
from pathlib import Path
from sys import version_info
from tqdm import tqdm
from numpy import linalg as LA

if version_info >= (3, 12):
    raise RuntimeError("h5py doesn't support 3.12")

import h5py
from faiss import Kmeans
import numpy as np

DEFAULT_LISTS = 4096
N_ITER = 25
CHUNKS = 10
SEED = 42
MAX_POINTS_PER_CLUSTER = 256


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Train K-means centroids")
    parser.add_argument("-i", "--input", help="input filepath", required=True)
    parser.add_argument("-o", "--output", help="output filepath", required=True)
    parser.add_argument(
        "--lists",
        "--lists-1",
        help="Number of centroids",
        type=int,
        required=False,
        default=DEFAULT_LISTS,
    )
    parser.add_argument("--lists-2", type=int, help="lower layer lists (if enabled)")
    parser.add_argument(
        "--niter", help="number of iterations", type=int, default=N_ITER
    )
    parser.add_argument("-m", "--metric", choices=["l2", "cos", "dot"], default="l2")
    parser.add_argument(
        "-g", "--gpu", help="enable GPU for KMeans", action="store_true"
    )
    parser.add_argument(
        "--mmap",
        help="not load by iter, instead use numpy chunk mmap, faster for large dataset",
        action="store_true",
    )
    parser.add_argument(
        "--chunks",
        help="chunks for in-memory mode. If OOM, increase it",
        type=int,
        default=CHUNKS,
    )
    return parser


def reservoir_sampling(iterator, k: int):
    """Reservoir sampling from an iterator."""
    res = []
    for _ in tqdm(range(k), desc="Collect train subset"):
        try:
            res.append(next(iterator))
        except StopIteration:
            return np.vstack(res)
    for i, vec in tqdm(enumerate(iterator, k + 1), total=k, desc="Random Pick"):
        j = np.random.randint(0, i)
        if j < k:
            res[j] = vec
    return np.vstack(res)


def _slice_chunk(args: tuple[int, str, np.ndarray]):
    k, file_path, chunk, start_idx = args
    dataset = h5py.File(Path(file_path), "r")
    data = dataset["train"]
    start, end = min(chunk), max(chunk)
    indexes = [c - start for c in chunk]
    source = data[start : end + 1]
    select = source[indexes]
    delta, dim = select.shape

    output = np.memmap("index.mmap", dtype=np.float32, mode="r+", shape=(k, dim))
    output[start_idx : start_idx + delta, :] = select
    output.flush()


def reservoir_sampling_np(data, file_path, k: int, chunks: int):
    """Reservoir sampling in memory by numpy."""
    index = np.random.permutation(len(data))[:k]
    indices = np.sort(index)
    num_processes = cpu_count()
    # Split indices into chunks for parallel processing
    index_chunks = np.array_split(indices, chunks)
    _, dim = data.shape
    np.memmap("index.mmap", dtype=np.float32, mode="w+", shape=(k, dim))
    # Create arguments for each chunk
    start_idx_acu = [0]
    start_idx_acu.extend(
        list(itertools.accumulate([len(c) for c in index_chunks[:-1]]))
    )
    chunk_args = [
        (k, file_path, chunk, start_idx_acu[i]) for i, chunk in enumerate(index_chunks)
    ]
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        list(pool.map(_slice_chunk, chunk_args))


def filter_by_label(iter, labels, target):
    for i, vec in enumerate(iter):
        if labels[i] == target:
            yield vec


def kmeans_cluster(
    data,
    file_path,
    k,
    child_k,
    niter,
    metric,
    gpu=False,
    mmap=False,
    chunks=CHUNKS,
):
    n, dim = data.shape
    if n > MAX_POINTS_PER_CLUSTER * k and not mmap:
        train = reservoir_sampling(iter(data), MAX_POINTS_PER_CLUSTER * args.lists)
    elif n > MAX_POINTS_PER_CLUSTER * k and mmap:
        reservoir_sampling_np(
            data, file_path, MAX_POINTS_PER_CLUSTER * args.lists, chunks
        )
        train = np.array(
            np.memmap(
                "index.mmap",
                dtype=np.float32,
                mode="r",
                shape=(MAX_POINTS_PER_CLUSTER * k, dim),
            )
        )
    else:
        train = data[:]
    if metric == "cos":
        train = train / LA.norm(train, axis=1, keepdims=True)
    kmeans = Kmeans(
        dim, k, gpu=gpu, verbose=True, niter=niter, seed=SEED, spherical=metric != "l2"
    )
    kmeans.train(train)
    if not child_k:
        return kmeans.centroids

    # train the lower layer k-means
    labels = np.zeros(n, dtype=np.uint32)
    for i, vec in tqdm(enumerate(data), desc="Assigning labels"):
        _, label = kmeans.assign(vec.reshape((1, -1)))
        labels[i] = label[0]

    centroids = []
    total_k = k * child_k
    for i in tqdm(range(k), desc="training k-means for child layers"):
        samples = np.sum(labels == i) / n * total_k * MAX_POINTS_PER_CLUSTER
        child_train = reservoir_sampling(
            filter_by_label(iter(data), labels, i), samples
        )
        child_kmeans = Kmeans(
            dim,
            child_k,
            gpu=gpu,
            verbose=True,
            niter=niter,
            seed=SEED,
            spherical=metric != "l2",
        )
        child_kmeans.train(child_train)
        centroids.append(child_kmeans.centroids)
    return np.vstack(centroids)


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    dataset = h5py.File(Path(args.input), "r")
    n, dim = dataset["train"].shape

    start_time = perf_counter()
    centroids = kmeans_cluster(
        dataset["train"],
        args.input,
        args.lists,
        args.lists_2,
        args.niter,
        args.metric,
        args.gpu,
        args.mmap,
        args.chunks,
    )
    print(
        f"K-means (k=({args.lists}, {args.lists_2})): {perf_counter() - start_time:.2f}s"
    )

    np.save(Path(args.output), centroids, allow_pickle=False)
