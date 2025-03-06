import time
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

import psycopg
import h5py
from pgvector.psycopg import register_vector


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
        "--url",
        help="url, like `postgresql://postgres:123@localhost:5432/postgres`",
        required=True,
    )
    parser.add_argument(
        "-t", "--top", help="Dimension", type=int, choices=[10, 100], default=10
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


def create_connection(url, nprob, epsilon):
    keepalive_kwargs = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5,
    }
    conn = psycopg.connect(
        conninfo=url,
        dbname="postgres",
        autocommit=True,
        **keepalive_kwargs,
    )
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS vchord")
    except Exception:
        pass
    # Tuning
    conn.execute("SET jit=false")
    conn.execute("SET effective_io_concurrency=200")

    conn.execute(f"SET vchordrq.probes={nprob}")
    conn.execute(f"SET vchordrq.epsilon={epsilon}")
    try:
        conn.execute(f"SELECT vchordrq_prewarm('{args.name}_embedding_idx'::regclass)")
    except Exception:
        pass
    register_vector(conn)
    return conn


def calculate_coverage(time_intervals):
    if not time_intervals:
        return 0
    sorted_intervals = sorted(time_intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = sorted_intervals[0]
    for interval in sorted_intervals[1:]:
        next_start, next_end = interval
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    total_length = 0
    for start, end in merged:
        total_length += end - start
    return total_length


def process_batch(args):
    """Process a batch of queries in a single process"""
    batch_queries, batch_answers, k, metric_ops, url, name, nprob, epsilon = args

    # Create a new connection for this process
    conn = create_connection(url, nprob, epsilon)

    hits = 0
    results = []

    for query, ground_truth in zip(batch_queries, batch_answers):
        start = time.perf_counter()
        result = conn.execute(
            f"SELECT id FROM {name} ORDER BY embedding {metric_ops} %s LIMIT {k}",
            (query,),
        ).fetchall()
        end = time.perf_counter()

        result_ids = set([p[0] for p in result[:k]])
        ground_truth_ids = set(ground_truth[:k].tolist())
        hit = len(result_ids & ground_truth_ids)
        hits += hit

        results.append((hit, (start, end)))

    conn.close()
    return results


def calculate_metrics(all_results, k, m, num_processes=1):
    """Calculate recall, QPS, and latency percentiles from results"""
    hits, latencies = zip(*all_results)

    if isinstance(latencies[0], list | tuple):
        # parallel_bench
        total_time = calculate_coverage(latencies)
        latencies = [(end - start) for start, end in latencies]
    else:
        # sequential_bench
        total_time = sum(latencies)

    total_hits = sum(hits)

    recall = total_hits / (k * m * num_processes)
    qps = (m * num_processes) / total_time

    # Calculate latency percentiles (in milliseconds)
    latencies_ms = np.array(latencies) * 1000
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)

    return recall, qps, p50, p99


def parallel_bench(
    name, test, answer, metric_ops, num_processes, url, top, nprob, epsilon
):
    """Run benchmark in parallel using multiple processes"""
    m = test.shape[0]
    batches = []

    for _ in range(num_processes):
        batch = (
            test,
            answer,
            top,
            metric_ops,
            url,
            name,
            nprob,
            epsilon,
        )
        batches.append(batch)

    # Create process pool and execute batches
    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(pool.map(process_batch, batches))

    # Flatten results from all batches
    all_results = [result for batch in batch_results for result in batch]

    # Calculate metrics
    recall, qps, p50, p99 = calculate_metrics(all_results, top, m, num_processes)

    print(f"Top: {top}")
    print(f"  Recall: {recall:.4f}")
    print(f"  QPS: {qps:.2f}")
    print(f"  P50 latency: {p50:.2f}ms")
    print(f"  P99 latency: {p99:.2f}ms")


def sequential_bench(name, test, answer, metric_ops, conn, top):
    """Original sequential benchmark implementation with latency tracking"""
    m = test.shape[0]
    results = []
    pbar = tqdm(enumerate(test), total=m)
    for i, query in pbar:
        start = time.perf_counter()
        result = conn.execute(
            f"SELECT id FROM {name} ORDER BY embedding {metric_ops} %s LIMIT {top}",
            (query,),
        ).fetchall()
        end = time.perf_counter()

        query_time = end - start
        hit = len(set([p[0] for p in result[:top]]) & set(answer[i][:top].tolist()))
        results.append((hit, query_time))

        # Update progress bar with running metrics
        curr_results = results[: i + 1]
        curr_recall, curr_qps, curr_p50, _ = calculate_metrics(curr_results, top, i + 1)
        pbar.set_description(
            f"recall: {curr_recall:.4f} QPS: {curr_qps:.2f} P50: {curr_p50:.2f}ms"
        )

    # Calculate final metrics
    recall, qps, p50, p99 = calculate_metrics(results, top, m)

    print(f"Top: {top}")
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
            args.url,
            args.top,
            args.nprob,
            args.epsilon,
        )
    else:
        conn = create_connection(args.url, args.nprob, args.epsilon)
        sequential_bench(args.name, test, answer, metric_ops, conn, args.top)