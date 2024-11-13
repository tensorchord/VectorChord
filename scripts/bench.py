import time
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

import psycopg
import h5py
from pgvector.psycopg import register_vector

TOP = [10]


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--metric",
        help="Metric to pick, in l2 or cos",
        choices=["l2", "cos", "dot"],
        default="l2",
    )
    parser.add_argument("-n", "--name", help="Dataset name, like: sift", required=True)
    parser.add_argument("-i", "--input", help="input filepath", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    parser.add_argument(
        "--nprob", help="argument probes for query", default=100, type=int
    )
    parser.add_argument(
        "--epsilon", help="argument epsilon for query", type=float, default=1.0
    )
    parser.add_argument(
        "--processes", help="Number of parallel processes to use", type=int, default=1
    )
    return parser


def create_connection(password, nprob, epsilon):
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
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vchord")
    # Tuning
    conn.execute(f"SET jit=false")
    conn.execute(f"SET effective_io_concurrency=200")

    conn.execute(f"SET vchordrq.probes={nprob}")
    conn.execute(f"SET vchordrq.epsilon={epsilon}")
    conn.execute(f"SELECT vchordrq_prewarm('{args.name}_embedding_idx'::regclass)")
    register_vector(conn)
    return conn


def process_batch(args):
    """Process a batch of queries in a single process"""
    batch_queries, batch_answers, k, metric_ops, password, name, nprob, epsilon = args

    # Create a new connection for this process
    conn = create_connection(password, nprob, epsilon)

    hits = 0
    latencies = []
    results = []

    for query, ground_truth in zip(batch_queries, batch_answers):
        start = time.perf_counter()
        result = conn.execute(
            f"SELECT id FROM {name} ORDER BY embedding {metric_ops} %s LIMIT {k}",
            (query,),
        ).fetchall()
        end = time.perf_counter()

        query_time = end - start
        latencies.append(query_time)

        result_ids = set([p[0] for p in result[:k]])
        ground_truth_ids = set(ground_truth[:k].tolist())
        hit = len(result_ids & ground_truth_ids)
        hits += hit

        results.append((hit, query_time))

    conn.close()
    return results


def calculate_metrics(all_results, k, m):
    """Calculate recall, QPS, and latency percentiles from results"""
    hits, latencies = zip(*all_results)

    total_hits = sum(hits)
    total_time = sum(latencies)

    recall = total_hits / (k * m)
    qps = m / total_time

    # Calculate latency percentiles (in milliseconds)
    latencies_ms = np.array(latencies) * 1000
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)

    return recall, qps, p50, p99


def parallel_bench(
    name, test, answer, metric_ops, num_processes, password, nprob, epsilon
):
    """Run benchmark in parallel using multiple processes"""
    m = test.shape[0]

    for k in TOP:
        # Split data into batches for each process
        batch_size = m // num_processes
        batches = []

        for i in range(num_processes):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_processes - 1 else m

            batch = (
                test[start_idx:end_idx],
                answer[start_idx:end_idx],
                k,
                metric_ops,
                password,
                name,
                nprob,
                epsilon,
            )
            batches.append(batch)

        # Create process pool and execute batches
        with mp.Pool(processes=num_processes) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(process_batch, batches),
                    total=len(batches),
                    desc=f"Processing k={k}",
                )
            )

        # Flatten results from all batches
        all_results = [result for batch in batch_results for result in batch]

        # Calculate metrics
        recall, qps, p50, p99 = calculate_metrics(all_results, k, m)

        print(f"Top: {k}")
        print(f"  Recall: {recall:.4f}")
        print(f"  QPS: {qps*num_processes:.2f}")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P99 latency: {p99:.2f}ms")


def sequential_bench(name, test, answer, metric_ops, conn):
    """Original sequential benchmark implementation with latency tracking"""
    m = test.shape[0]
    for k in TOP:
        results = []
        pbar = tqdm(enumerate(test), total=m)
        for i, query in pbar:
            start = time.perf_counter()
            result = conn.execute(
                f"SELECT id FROM {name} ORDER BY embedding {metric_ops} %s LIMIT {k}",
                (query,),
            ).fetchall()
            end = time.perf_counter()

            query_time = end - start
            hit = len(set([p[0] for p in result[:k]]) & set(answer[i][:k].tolist()))
            results.append((hit, query_time))

            # Update progress bar with running metrics
            curr_results = results[: i + 1]
            curr_recall, curr_qps, curr_p50, _ = calculate_metrics(
                curr_results, k, i + 1
            )
            pbar.set_description(
                f"recall: {curr_recall:.4f} QPS: {curr_qps:.2f} P50: {curr_p50:.2f}ms"
            )

        # Calculate final metrics
        recall, qps, p50, p99 = calculate_metrics(results, k, m)

        print(f"Top: {k}")
        print(f"  Recall: {recall:.4f}")
        print(f"  QPS: {qps:.2f}")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P99 latency: {p99:.2f}ms")


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    dataset = h5py.File(Path(args.input), "r")
    test = dataset["test"][:]
    answer = dataset["neighbors"][:]

    if args.metric == "l2":
        metric_ops = "<->"
    elif args.metric == "cos":
        metric_ops = "<=>"
    elif args.metric == "dot":
        metric_ops = "<#>"
    else:
        raise ValueError

    if args.processes > 1:
        parallel_bench(
            args.name,
            test,
            answer,
            metric_ops,
            args.processes,
            args.password,
            args.nprob,
            args.epsilon,
        )
    else:
        conn = create_connection(args.password, args.nprob, args.epsilon)
        sequential_bench(args.name, test, answer, metric_ops, conn)
