import asyncio
import math
from time import perf_counter
import argparse
from pathlib import Path
import multiprocessing

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
CHUNKS = 10


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
    parser.add_argument("-i", "--input", help="Input filepath", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    parser.add_argument("-d", "--dim", help="Dimension", type=int, required=True)
    parser.add_argument(
        "-w",
        "--workers",
        help="Workers to build index",
        type=int,
        required=False,
        default=max(multiprocessing.cpu_count() - 1, 1),
    )
    parser.add_argument(
        "--chunks",
        help="chunks for in-memory mode. If OOM, increase it",
        type=int,
        default=CHUNKS,
    )
    # External build
    parser.add_argument(
        "-c", "--centroids", help="K-means centroids file", required=False
    )
    # Internal build
    parser.add_argument("--lists", help="Number of centroids", type=int, required=False)

    return parser


def get_ivf_ops_config(metric, workers, k=None, name=None):
    assert name is not None or k is not None
    external_centroids_cfg = """
    [build.external]
    table = 'public.{name}_centroids'
    """
    if metric == "l2":
        metric_ops = "vector_l2_ops"
        config = "residual_quantization = true"
        internal_centroids_cfg = f"""
        [build.internal]
        lists = [{k}]
        build_threads = {workers}
        spherical_centroids = false
        """
    elif metric == "cos":
        metric_ops = "vector_cosine_ops"
        config = "residual_quantization = false"
        internal_centroids_cfg = f"""
        [build.internal]
        lists = [{k}]
        build_threads = {workers}
        spherical_centroids = true
        """
    elif metric == "dot":
        metric_ops = "vector_ip_ops"
        config = "residual_quantization = false"
        internal_centroids_cfg = f"""
        [build.internal]
        lists = [{k}]
        build_threads = {workers}
        spherical_centroids = true
        """
    else:
        raise ValueError

    build_config = (
        external_centroids_cfg.format(name=name) if name else internal_centroids_cfg
    )
    return metric_ops, "\n".join([config, build_config])


async def create_connection(url):
    conn = await psycopg.AsyncConnection.connect(
        conninfo=url,
        dbname="postgres",
        autocommit=True,
        **KEEPALIVE_KWARGS,
    )
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vchord")
    await register_vector_async(conn)
    return conn


async def add_centroids(conn, name, centroids):
    n, dim = centroids.shape
    root = np.mean(centroids, axis=0)
    await conn.execute(f"DROP TABLE IF EXISTS public.{name}_centroids")
    await conn.execute(
        f"CREATE TABLE public.{name}_centroids (id integer, parent integer, vector vector({dim}))"
    )
    async with conn.cursor().copy(
        f"COPY public.{name}_centroids (id, parent, vector) FROM STDIN WITH (FORMAT BINARY)"
    ) as copy:
        copy.set_types(["integer", "integer", "vector"])
        await copy.write_row((0, None, root))
        for i, centroid in tqdm(enumerate(centroids), desc="Adding centroids", total=n):
            await copy.write_row((i + 1, 0, centroid))
        while conn.pgconn.flush() == 1:
            await asyncio.sleep(0)


async def add_embeddings(conn, name, dim, train, chunks):
    await conn.execute(f"DROP TABLE IF EXISTS {name}")
    await conn.execute(f"CREATE TABLE {name} (id integer, embedding vector({dim}))")

    n, dim = train.shape
    chunk_size = math.ceil(n / chunks)
    pbar = tqdm(desc="Adding embeddings", total=n)
    for i in range(chunks):
        chunk_start = i * chunk_size
        chunk_len = min(chunk_size, n - i * chunk_size)
        data = train[chunk_start : chunk_start + chunk_len]

        async with conn.cursor().copy(
            f"COPY {name} (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["integer", "vector"])

            for i, vec in enumerate(data):
                await copy.write_row((chunk_start + i, vec))
            while conn.pgconn.flush() == 1:
                await asyncio.sleep(0)
        pbar.update(chunk_len)
    pbar.close()


async def build_index(
    conn, name, workers, metric_ops, ivf_config, finish: asyncio.Event
):
    start_time = perf_counter()
    await conn.execute(f"SET max_parallel_maintenance_workers TO {workers}")
    await conn.execute(f"SET max_parallel_workers TO {workers}")
    await conn.execute(
        f"CREATE INDEX {name}_embedding_idx ON {name} USING vchordrq (embedding {metric_ops}) WITH (options = $${ivf_config}$$)"
    )
    print(f"Index build time: {perf_counter() - start_time:.2f}s")
    finish.set()


async def monitor_index_build(conn, finish: asyncio.Event):
    async with conn.cursor() as acur:
        blocks_total = None
        while blocks_total is None:
            await asyncio.sleep(1)
            await acur.execute("SELECT blocks_total FROM pg_stat_progress_create_index")
            blocks_total = await acur.fetchone()
        total = 0 if blocks_total is None else blocks_total[0]
        pbar = tqdm(smoothing=0.0, total=total, desc="Building index")
        while True:
            if finish.is_set():
                pbar.update(pbar.total - pbar.n)
                return
            await acur.execute("SELECT blocks_done FROM pg_stat_progress_create_index")
            blocks_done = await acur.fetchone()
            done = 0 if blocks_done is None else blocks_done[0]
            pbar.update(done - pbar.n)
            await asyncio.sleep(1)
        pbar.close()


async def main(dataset):
    dataset = h5py.File(Path(args.input), "r")
    url = f"postgresql://postgres:{args.password}@localhost:5432/postgres"
    conn = await create_connection(url)
    if args.centroids:
        centroids = np.load(args.centroids, allow_pickle=False)
        await add_centroids(conn, args.name, centroids)
    metric_ops, ivf_config = get_ivf_ops_config(
        args.metric, args.workers, args.lists, args.name if args.centroids else None
    )
    await add_embeddings(conn, args.name, args.dim, dataset["train"], args.chunks)

    index_finish = asyncio.Event()
    # Need a separate connection for monitor process
    monitor_conn = await create_connection(url)
    monitor_task = monitor_index_build(
        monitor_conn,
        index_finish,
    )
    index_task = build_index(
        conn,
        args.name,
        args.workers,
        metric_ops,
        ivf_config,
        index_finish,
    )
    await asyncio.gather(index_task, monitor_task)


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)
    asyncio.run(main(args))
