from os.path import join
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import psycopg
import h5py
from pgvector.psycopg import register_vector
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--metric", help="Metric to pick, in l2 or cos", required=True
)
parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
args = parser.parse_args()

HOME = Path.home()
INDEX_PATH = join(HOME, f"indexes/pg/{args.name}/{args.metric}")
DATA_PATH = join(HOME, f"{args.name}/{args.name}.hdf5")

os.makedirs(join(HOME, f"indexes/pg/{args.name}"), exist_ok=True)
dataset = h5py.File(DATA_PATH, "r")

test = dataset["test"][:]

if args.metric == "l2":
    metric_ops = "<->"
elif args.metric == "cos":
    metric_ops = "<=>"
else:
    raise ValueError

answer = dataset["neighbors"][:]
n, dims = np.shape(test)
m = np.shape(test)[0]

# reconnect for updated GUC variables to take effect
conn = psycopg.connect(
    conninfo="postgres://bench:123@localhost:5432/postgres",
    dbname="postgres",
    autocommit=True,
)
conn.execute("SET search_path TO public, vectors, rabbithole")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.execute("CREATE EXTENSION IF NOT EXISTS rabbithole")
conn.execute("SET rabbithole.nprobe=300")
register_vector(conn)

Ks = [10, 100]
for k in Ks:
    hits = 0
    delta = 0
    pbar = tqdm(enumerate(test), total=m)
    for i, query in pbar:
        start = time.perf_counter()
        result = conn.execute(
            f"SELECT id FROM {args.name} ORDER BY embedding {metric_ops} %s LIMIT {k}",
            (query,),
        ).fetchall()
        end = time.perf_counter()
        hits += len(set([p[0] for p in result[:k]]) & set(answer[i][:k].tolist()))
        delta += end - start
        pbar.set_description(f"recall: {hits / k / (i+1)} QPS: {(i+1) / delta} ")
    recall = hits / k / m
    qps = m / delta
    print(f"Top: {k} recall: {recall:.4f} QPS: {qps:.2f}")
