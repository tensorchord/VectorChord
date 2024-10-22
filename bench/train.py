from os.path import join
import time
import argparse
from pathlib import Path
import pickle

import h5py
import faiss
import numpy as np

K = 4096
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
parser.add_argument("-k", help="K-means centroids or nlist, default: 4096", required=False, default=4096)
args = parser.parse_args()

K = args.k

HOME = Path.home()
DATA_PATH = join(HOME, f"{args.name}/{args.name}.hdf5")

dataset = h5py.File(DATA_PATH, "r")

if len(dataset["train"]) > 256 * K:
    rs = np.random.RandomState(SEED)
    idx = rs.choice(len(dataset["train"]), size=256 * K, replace=False)
    train = dataset["train"][np.sort(idx)]
else:
    train = dataset["train"][:]

test = dataset["test"][:]

if np.shape(train)[0] > 256 * K:
    rs = np.random.RandomState(SEED)
    idx = rs.choice(np.shape(train)[0], size=256 * K, replace=False)
    train = train[idx]

answer = dataset["neighbors"][:]
n, dims = np.shape(train)
m = np.shape(test)[0]

start = time.perf_counter()

index = faiss.IndexFlatL2(dims)
clustering = faiss.Clustering(dims, K)
clustering.verbose = True
clustering.seed = 42
clustering.niter = 10
clustering.train(train, index)
centroids = faiss.vector_float_to_array(clustering.centroids)

end = time.perf_counter()
delta = end - start
print(f"K-means time: {delta:.2f}s")

centroids = centroids.reshape([K, -1])

with open(f"{args.name}.pickle", "wb") as f:
    pickle.dump(centroids, f)