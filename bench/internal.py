import os
import time
import argparse
from pathlib import Path
import pickle

import psycopg
import h5py
from pgvecto_rs.psycopg import register_vector
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--metric", help="Metric to pick, in l2 or cos", required=True
)
parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
parser.add_argument("-k", help="K-means centroids or nlist, default: 4096", required=False, default=4096)
args = parser.parse_args()

HOME = Path.home()
DATA_PATH = join(HOME, f"{args.name}/{args.name}.hdf5")

os.makedirs(join(HOME, f"indexes/pg/{args.name}"), exist_ok=True)
dataset = h5py.File(DATA_PATH, "r")

train = dataset["train"][:]
test = dataset["test"][:]

K = 4096
if args.metric == "l2":
    metric_ops = "vector_l2_ops"
    ivf_config = f"""
nlist = {K}
residual_quantization = true
spherical_centroids = false
"""
elif args.metric == "cos":
    metric_ops = "vector_cos_ops"
    ivf_config = f"""
nlist = {K}
residual_quantization = false
spherical_centroids = true
"""
else:
    raise ValueError

answer = dataset["neighbors"][:]
n, dims = np.shape(train)
m = np.shape(test)[0]

keepalive_kwargs = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 5,
    "keepalives_count": 5,
}

start = time.perf_counter()

with open(f"{args.name}.pickle", "rb") as f:
    centroids = pickle.load(f)

conn = psycopg.connect(
    conninfo="postgres://bench:123@localhost:5432/postgres",
    dbname="postgres",
    autocommit=True,
    **keepalive_kwargs,
)
conn.execute("SET search_path TO public, vectors, rabbithole")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.execute("CREATE EXTENSION IF NOT EXISTS rabbithole")
register_vector(conn)

conn.execute(f"DROP TABLE IF EXISTS public.centroids")
conn.execute(f"DROP TABLE IF EXISTS {args.name}")
conn.execute(f"CREATE TABLE {args.name} (id integer, embedding vector({dims}))")

with conn.cursor().copy(
    f"COPY {args.name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
) as copy:
    copy.set_types(["integer", "vector"])

    for i in range(n):
        copy.write_row([i, train[i]])
        while conn.pgconn.flush() == 1:
            pass

start = time.perf_counter()
conn.execute(
    f"CREATE INDEX ON {args.name} USING rabbithole (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
)
end = time.perf_counter()

delta = end - start
print(f"Index build time: {delta:.2f}s")