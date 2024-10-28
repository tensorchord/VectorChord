import asyncio
from sys import version_info
from time import perf_counter
import argparse
from pathlib import Path
import multiprocessing

if version_info >= (3, 12):
    raise RuntimeError("h5py doesn't support 3.12")

import psycopg
import h5py
from pgvector.psycopg import register_vector_async
import numpy as np
from tqdm import tqdm

KEEPALIVE_KWARGS = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 5,
    "keepalives_count": 5,
}


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Build index with K-means centroids")
    parser.add_argument(
        "-m",
        "--metric",
        help="Distance metric",
        default="l2",
        choices=["l2", "cos", "dot"],
    )
    parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
    parser.add_argument(
        "-c", "--centroids", help="K-means centroids file", required=True
    )
    parser.add_argument("-i", "--input", help="Input filepath", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    parser.add_argument("-k", help="Number of centroids", type=int, required=True)
    parser.add_argument("-d", "--dim", help="Dimension", type=int, required=True)
    parser.add_argument(
        "-w",
        "--workers",
        help="Workers to build index",
        type=int,
        required=False,
        default=max(multiprocessing.cpu_count() - 2, 1),
    )
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
    elif metric == "dot":
        metric_ops = "vector_dot_ops"
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


async def create_connection(password):
    conn = await psycopg.AsyncConnection.connect(
        conninfo=f"postgresql://postgres:{password}@localhost:5432/postgres",
        dbname="postgres",
        autocommit=True,
        **KEEPALIVE_KWARGS,
    )
    await conn.execute("SET search_path TO public, vectors, rabbithole")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS rabbithole")
    await register_vector_async(conn)
    return conn


async def add_centroids(conn, name, centroids):
    dim = centroids.shape[1]
    await conn.execute(f"DROP TABLE IF EXISTS public.{name}_centroids")
    await conn.execute(
        f"CREATE TABLE public.{name}_centroids (coordinate vector({dim}))"
    )
    async with conn.cursor().copy(
        f"COPY public.{name}_centroids (coordinate) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["vector"])
        for centroid in tqdm(centroids, desc="Adding centroids"):
            await copy.write_row((centroid,))
            while conn.pgconn.flush() == 1:
                pass


async def add_embeddings(conn, name, dim, train):
    await conn.execute(f"DROP TABLE IF EXISTS {name}")
    await conn.execute(f"CREATE TABLE {name} (id integer, embedding vector({dim}))")

    async with conn.cursor().copy(
        f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["integer", "vector"])

        for i, vec in tqdm(
            enumerate(train), desc="Adding embeddings", total=len(train)
        ):
            await copy.write_row((i, vec))
            while conn.pgconn.flush() == 1:
                pass


async def build_index(
    conn, name, workers, metric_ops, ivf_config, finish: asyncio.Event
):
    start_time = perf_counter()
    await conn.execute(f"SET max_parallel_maintenance_workers TO {workers}")
    await conn.execute(
        f"CREATE INDEX ON {name} USING rabbithole (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
    )
    print(f"Index build time: {perf_counter() - start_time:.2f}s")
    finish.set()


async def monitor_index_build(password, finish: asyncio.Event):
    conn = await psycopg.AsyncConnection.connect(
        conninfo=f"postgresql://postgres:{password}@localhost:5432/postgres",
        dbname="postgres",
        autocommit=True,
        **KEEPALIVE_KWARGS,
    )
    pbar = tqdm(smoothing=0.0)
    async with conn.cursor() as acur:
        while True:
            if finish.is_set():
                return
            await acur.execute(f"SELECT tuples_done FROM pg_stat_progress_create_index")
            tuples_done = await acur.fetchone()
            update = 0 if tuples_done is None else tuples_done[0]
            pbar.update(update - pbar.n)
            await asyncio.sleep(1)


async def main(dataset):
    dataset = h5py.File(Path(args.input), "r")
    conn = await create_connection(args.password)
    if args.centroids:
        centroids = np.load(args.centroids, allow_pickle=False)
        await add_centroids(conn, args.name, centroids)
    metric_ops, ivf_config = get_ivf_ops_config(
        args.metric, args.k, args.name if args.centroids else None
    )
    await add_embeddings(conn, args.name, args.dim, dataset["train"])

    index_finish = asyncio.Event()
    index_task = build_index(
        conn,
        args.name,
        args.workers,
        metric_ops,
        ivf_config,
        index_finish,
    )
    monitor_task = monitor_index_build(
        args.password,
        index_finish,
    )
    await asyncio.gather(index_task, monitor_task)


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)
    asyncio.run(main(args))
