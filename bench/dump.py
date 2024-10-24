import argparse

import h5py
import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Dump embeddings to a local file")
    parser.add_argument("-n", "--name", help="table name", required=True)
    parser.add_argument(
        "-p", "--password", help="Database password", default="password"
    )
    parser.add_argument("-o", "--output", help="Output filepath", required=True)
    parser.add_argument("-c", "--column", help="Column name", default="embedding")
    parser.add_argument("-d", "--dim", help="Dimension", required=True)
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


def extract_vectors(conn, name, column):
    with conn.execute(f"SELECT {column} FROM {name}") as cursor:
        for row in cursor:
            yield row[0]


def write_to_h5(filepath, vecs, dim):
    with h5py.File(filepath, "w") as file:
        dataset = file.create_dataset("train", (0, dim), chunks=True, dtype=np.float32)
        current_size = 0
        for vec in vecs:
            dataset[current_size * dim : (current_size + 1) * dim] = vec
            current_size += 1


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    conn = create_connection(args.password)
    write_to_h5(args.output, extract_vectors(conn, args.name, args.column), args.dim)
