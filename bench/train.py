from time import perf_counter
import argparse
from pathlib import Path
from sys import version_info

if version_info >= (3, 12):
    raise RuntimeError("h5py doesn't support 3.12")

import h5py
from faiss import Kmeans
import numpy as np

DEFAULT_K = 4096
N_ITER = 25
SEED = 42
MAX_POINTS_PER_CLUSTER = 256


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Train K-means centroids")
    parser.add_argument("-i", "--input", help="input filepath", required=True)
    parser.add_argument("-o", "--output", help="output filepath", required=True)
    parser.add_argument(
        "-k",
        help="K-means centroids or nlist",
        type=int,
        default=DEFAULT_K,
    )
    return parser


def reservoir_sampling(iterator, k: int):
    """Reservoir sampling from an iterator."""
    res = []
    while len(res) < k:
        res.append(next(iterator))
    for i, vec in enumerate(iterator, k + 1):
        j = np.random.randint(0, i)
        if j < k:
            res[j] = vec
    return res


def kmeans_cluster(data, k, niter):
    kmeans = Kmeans(data.shape[1], k, verbose=True, niter=niter, seed=SEED)
    start_time = perf_counter()
    kmeans.train(data)
    print(f"K-means (k={k}): {perf_counter() - start_time:.2f}s")
    return kmeans.centroids


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    dataset = h5py.File(Path(args.input), "r")
    n, dim = dataset["train"].shape

    if n > MAX_POINTS_PER_CLUSTER * args.k:
        train = np.array(
            reservoir_sampling(iter(dataset["train"]), MAX_POINTS_PER_CLUSTER * args.k),
        )
    else:
        train = dataset["train"][:]

    centroids = kmeans_cluster(train, args.k, N_ITER)
    np.save(Path(args.output), centroids, allow_pickle=False)
