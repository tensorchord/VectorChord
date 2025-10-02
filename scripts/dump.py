# This software is licensed under a dual license model:
#
# GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
# distribute this software under the terms of the AGPLv3.
#
# Elastic License v2 (ELv2): You may also use, modify, and distribute this
# software under the Elastic License v2, which has specific restrictions.
#
# We welcome any commercial collaboration or support. For inquiries
# regarding the licenses, please contact us at:
# vectorchord-inquiry@tensorchord.ai
#
# Copyright (c) 2025 TensorChord Inc.

import argparse

import h5py
import numpy as np
import psycopg
from tqdm import tqdm
from pgvector.psycopg import register_vector


def build_arg_parse():
    parser = argparse.ArgumentParser(description="Dump embeddings to a local file")
    parser.add_argument(
        "--url",
        help="url, like `postgresql://USERNAME:PASSWORD@localhost:5432/DBNAME`",
        default="postgresql://postgres:postgres@localhost:5432/postgres"
    )
    parser.add_argument("-n", "--name", help="Table name", required=True)
    parser.add_argument("-c", "--column", help="Column name", default="embedding")
    parser.add_argument("-d", "--dim", help="Dimension", type=int, required=True)
    parser.add_argument("-o", "--output", help="Output filepath", required=True)
    return parser


def create_connection(url):
    keepalive_kwargs = {
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5,
    }
    conn = psycopg.connect(
        conninfo=url,
        autocommit=True,
        **keepalive_kwargs,
    )
    register_vector(conn)
    return conn


def extract_vectors(conn, name, column):
    with conn.execute(f"SELECT {column} FROM {name}") as cursor:
        for row in cursor:
            yield row[0]


def write_to_h5(filepath, vecs, dim):
    with h5py.File(filepath, "w") as file:
        dataset = file.create_dataset(
            "train", (0, dim), maxshape=(None, dim), chunks=True, dtype=np.float32
        )
        current_size = 0
        for vec in tqdm(vecs):
            if dataset.shape[0] == current_size:
                dataset.resize(current_size + 64, axis=0)
            dataset[current_size] = vec
            current_size += 1
        dataset.resize(current_size, axis=0)
        dataset.flush()


if __name__ == "__main__":
    parser = build_arg_parse()
    args = parser.parse_args()
    print(args)

    conn = create_connection(args.url)
    write_to_h5(args.output, extract_vectors(conn, args.name, args.column), args.dim)
