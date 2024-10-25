from sys import version_info
from time import perf_counter
import argparse
from pathlib import Path

if version_info >= (3, 12):
    raise RuntimeError("h5py doesn't support 3.12")

import psycopg
import h5py
from pgvector.psycopg import register_vector
import numpy as np
from tqdm import tqdm


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Build index with K-means centroids")
    parser.add_argument(
        "-m", "--metric", help="Distance metric", default="l2", choices=["l2", "cos"]
    )
    parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
    parser.add_argument(
        "-c", "--centroid", help="K-means centroids file", required=True
    )
    parser.add_argument("-i", "--input", help="Input filepath", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    parser.add_argument("-k", help="Number of centroids", type=int, required=True)
    parser.add_argument("-d", "--dim", help="Dimension", type=int, required=True)
    return parser


def get_ivf_ops_config(metric, k, name=None):
    external_centroids = """
    [external_centroids]
    table = 'public.{name}_centroids'
    h1_means_column = 'coordinate'
    """
    if metric == "l2":
        metric_ops = "vector_l2_ops"
        ivf_config = f"""
        nlist = {k}
        residual_quantization = true
        spherical_centroids = false
        """
    elif metric == "cos":
        metric_ops = "vector_cos_ops"
        ivf_config = f"""
        nlist = {k}
        residual_quantization = false
        spherical_centroids = true
        """
    else:
        raise ValueError

    if name:
        ivf_config += external_centroids.format(name=name)
    return metric_ops, ivf_config


def create_connection(password):
    keepalive_kwargs = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5,
    }
    conn = psycopg.connect(
        conninfo=f"postgresql://postgres:{password}@localhost:5432/postgres",
        dbname="postgres",
        autocommit=True,
        **keepalive_kwargs,
    )
    conn.execute("SET search_path TO public, vectors, rabbithole")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS rabbithole")
    register_vector(conn)
    return conn


def add_centroids(conn, name, centroids):
    dim = centroids.shape[1]
    conn.execute(f"DROP TABLE IF EXISTS public.{name}_centroids")
    conn.execute(f"CREATE TABLE public.{name}_centroids (coordinate vector({dim}))")
    with conn.cursor().copy(
        f"COPY public.{name}_centroids (coordinate) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["vector"])
        for centroid in tqdm(centroids, desc="Adding centroids"):
            copy.write_row((centroid,))
            while conn.pgconn.flush() == 1:
                pass


def build_index(conn, name, metric_ops, ivf_config, dim, train):
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"CREATE TABLE {name} (id integer, embedding vector({dim}))")

    with conn.cursor().copy(
        f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["integer", "vector"])

        for i, vec in tqdm(enumerate(train), desc="Adding embeddings"):
            copy.write_row((i, vec))
            while conn.pgconn.flush() == 1:
                pass

    start_time = perf_counter()
    conn.execute(
        f"CREATE INDEX ON {name} USING rabbithole (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
    )
    print(f"Index build time: {perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    dataset = h5py.File(Path(args.input), "r")
    conn = create_connection(args.password)
    if args.centroids:
        centroids = np.load(args.centroid, allow_pickle=False)
        add_centroids(conn, args.name, centroids)
    metric_ops, ivf_config = get_ivf_ops_config(
        args.metric, args.k, args.name if args.centroids else None
    )
    build_index(
        conn,
        args.name,
        metric_ops,
        ivf_config,
        args.dim,
        dataset["train"],
    )
