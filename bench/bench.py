import time
import argparse
from pathlib import Path
from tqdm import tqdm

import psycopg
import h5py
from pgvector.psycopg import register_vector


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--metric",
        help="Metric to pick, in l2 or cos",
        choices=["l2", "cos"],
        default="l2",
    )
    parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
    parser.add_argument("-i", "--input", help="input filepath", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    return parser


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


def bench(name, test, answer, metric_ops, conn):
    m = test.shape[0]
    for k in [10, 100]:
        hits = 0
        delta = 0
        pbar = tqdm(enumerate(test), total=m)
        for i, query in pbar:
            start = time.perf_counter()
            result = conn.execute(
                f"SELECT id FROM {name} ORDER BY embedding {metric_ops} %s LIMIT {k}",
                (query,),
            ).fetchall()
            end = time.perf_counter()
            hits += len(set([p[0] for p in result[:k]]) & set(answer[i][:k].tolist()))
            delta += end - start
            pbar.set_description(f"recall: {hits / k / (i+1)} QPS: {(i+1) / delta} ")
        recall = hits / k / m
        qps = m / delta
        print(f"Top: {k} recall: {recall:.4f} QPS: {qps:.2f}")


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    dataset = h5py.File(Path(args.input), "r")
    test = dataset["test"][:]
    answer = dataset["neighbors"][:]
    metric_ops = "<->" if args.metric == "l2" else "<=>"
    conn = create_connection(args.password)
    conn.execute("SET rabbithole.nprobe=300")

    bench(args.name, test, answer, metric_ops, conn)
