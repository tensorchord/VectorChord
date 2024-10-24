from time import perf_counter
import argparse
from pathlib import Path
from sys import version_info

if version_info >= (3, 12):
    raise RuntimeError("h5py doesn't support 3.12")

import h5py
from faiss import Kmeans
import numpy as np
from tqdm import tqdm

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
    parser.add_argument("--child-k", help="lower layer nlist (if enabled)")
    parser.add_argument(
        "--niter", help="number of iterations", type=int, default=N_ITER
    )
    parser.add_argument("-m", "--metric", choices=["l2", "cos"], default="l2")
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
    return np.vstack(res)


def filter_by_label(iter, labels, target):
    for i, vec in enumerate(iter):
        if labels[i] == target:
            yield vec


def kmeans_cluster(data, k, child_k, niter, metric):
    n, dim = data.shape
    if n > MAX_POINTS_PER_CLUSTER * k:
        train = reservoir_sampling(iter(data), MAX_POINTS_PER_CLUSTER * args.k)
    else:
        train = data[:]
    kmeans = Kmeans(
        dim, k, verbose=True, niter=niter, seed=SEED, spherical=metric == "cos"
    )
    kmeans.train(train)
    if not child_k:
        return kmeans.centroids

    # train the lower layer k-means
    labels = np.zeros(n, dtype=np.uint32)
    for i, vec in tqdm(enumerate(data), desc="Assigning labels"):
        _, label = kmeans.assign(vec.reshape((1, -1)))
        labels[i] = label

    centroids = []
    for i in tqdm(range(k), desc="training k-means for child layers"):
        child_train = reservoir_sampling(
            filter_by_label(iter(data), labels, i), child_k
        )
        child_kmeans = Kmeans(
            dim,
            child_k,
            verbose=True,
            niter=niter,
            seed=SEED,
            spherical=metric == "cos",
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
        dataset["train"], args.k, args.child_k, args.niter, args.metric
    )
    print(f"K-means (k=({args.k}, {args.child_k})): {perf_counter() - start_time:.2f}s")

    np.save(Path(args.output), centroids, allow_pickle=False)
