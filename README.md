<div align="center">
<h1 align=center>VectorChord</h1>
<h4 align=center>Effortlessly host 100 million 768-dimensional vectors (250GB+) on an AWS i4i.xlarge instance ($250/month), featuring 4 vCPUs and 32GB of RAM with VectorChord.</h4>
</div>

<div align=center>


[Official Site][official-site-link] · [Blog][blog-link] · [Feedback][github-issues-link] · [Contact Us][email-link]

<!-- TODO: Add GHCR when 0.3.0 is ready -->

[![][github-release-shield]][github-release-link]
[![][docker-release-shield]][docker-release-link]
[![][docker-pulls-shield]][docker-pulls-link]
[![][discord-shield]][discord-link]
[![][X-shield]][X-link]
[![][cloud-shield]][cloud-link]
[![][license-1-shield]][license-1-link]
[![][license-2-shield]][license-2-link]
</div>


> [!NOTE]
> VectorChord serves as the successor to [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) [![][previous-docker-pulls-shield]][previous-docker-pulls-link] with better stability and performance. If you are interested in this new solution, you may find the [migration guide](https://docs.vectorchord.ai/vectorchord/admin/migration.html) helpful.

VectorChord (vchord) is a PostgreSQL extension designed for scalable, high-performance, and disk-efficient vector similarity search.

With VectorChord, you can store 400,000 vectors for just $1, enabling significant savings: 6x more vectors compared to Pinecone's optimized storage and 26x more than pgvector/pgvecto.rs for the same price[^1].

![][image-compare]

## Features

VectorChord introduces remarkable enhancements over pgvecto.rs and pgvector:

**⚡ Enhanced Performance**: Delivering optimized operations with up to 5x faster queries, 16x higher insert throughput, and 16x quicker[^1] index building compared to pgvector's HNSW implementation.

[^1]: Based on [MyScale Benchmark](https://myscale.github.io/benchmark/#/) with 768-dimensional vectors and 95% recall. Please checkout our [blog post](https://blog.vectorchord.ai/vectorchord-store-400k-vectors-for-1-in-postgresql) for more details.

**💰 Affordable Vector Search**: Query 100M 768-dimensional vectors using just 32GB of memory, achieving 35ms P50 latency with top10 recall@95%, helping you keep infrastructure costs down while maintaining high search quality.

**🔌 Seamless Integration**: Fully compatible with pgvector data types and syntax while providing optimal defaults out of the box - no manual parameter tuning needed. Just drop in VectorChord for enhanced performance.

**🔧 Accelerated Index Build**: Leverage IVF to build indexes externally (e.g., on GPU) for faster KMeans clustering, combined with RaBitQ[^3] compression to efficiently store vectors while maintaining search quality through autonomous reranking.

[^3]: Gao, Jianyang, and Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 2.3 (2024): 1-27.

**📏 Long Vector Support**: Store and search vectors up to 60,000[^4] dimensions, enabling the use of the best high-dimensional models like text-embedding-3-large with ease.

[^4]: There is a [limitation](https://github.com/pgvector/pgvector#vector-type) at pgvector of 16,000 dimensions now. If you really have a large dimension(`16,000<dim<60,000`), consider to change [VECTOR_MAX_DIM](https://github.com/pgvector/pgvector/blob/fef635c9e5512597621e5669dce845c744170822/src/vector.h#L4) and compile pgvector yourself.

**🌐 Scale As You Want**: Based on horizontal expansion, the query of 5M / 100M 768-dimensional vectors can be easily scaled to 10000+ QPS with top10 recall@90% at a competitive cost[^5]

[^5]: Please check our [blog post](https://blog.vectorchord.ai/vector-search-at-10000-qps-in-postgresql-with-vectorchord)  for more details, the PostgreSQL scalability is powered by [CloudNative-PG](https://github.com/cloudnative-pg/cloudnative-pg).

## Requirements

> [!TIP]
> If you are using the official [Docker image](https://hub.docker.com/r/tensorchord/vchord-postgres), you can skip this step.

VectorChord depends on [pgvector](https://github.com/pgvector/pgvector), ensure the pgvector extension is available:

```SQL
SELECT * FROM pg_available_extensions WHERE name = 'vector';
```
If pgvector is not available, install it using the [pgvector installation instructions](https://github.com/pgvector/pgvector#installation).

And make sure to add `vchord.so` to the `shared_preload_libraries` in `postgresql.conf`.

```SQL
-- Add vchord and pgvector to shared_preload_libraries --
-- Note: A restart is required for this setting to take effect.
ALTER SYSTEM SET shared_preload_libraries = 'vchord.so';
```

## Quick Start
For new users, we recommend using the Docker image to get started quickly.
```bash
docker run \
  --name vectorchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/vchord-postgres:pg17-v0.2.1
```

Then you can connect to the database using the `psql` command line tool. The default username is `postgres`, and the default password is `mysecretpassword`.

```bash
psql -h localhost -p 5432 -U postgres
```

Now you can play with VectorChord!

## Documentation

- [Installation](#installation)
  - [Docker](#installation)
  - [APT](#apt)
  - [More Methods](#more-methods)
- [Usage](#usage)
  - [Storing](#storing)
  - [Indexing](#indexing)
  - [Query](#query)
- [Performance Tuning](#performance-tuning)
  - [Index Build Time](#index-build-time)
  - [Query Performance](#query-performance)
- [Advanced Features](#advanced-features)
  - [Indexing Prewarm](#indexing-prewarm)
  - [Indexing Progress](#indexing-progress)
  - [External Index Precomputation](#external-index-precomputation)
  <!--TODO: Here we have a memory leak in rerank_in_table, show it until the feature is ready
  - [Capacity-optimized Index](#capacity-optimized-index) -->
  - [Range Query](#range-query)

## Installation

### [Docker](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html#docker)

You can easily get the Docker image from:

```bash
docker pull tensorchord/vchord-postgres:pg17-v0.2.1
```

### [APT](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html#from-debian-package)

Debian and Ubuntu packages can be found on [release page](https://github.com/tensorchord/VectorChord/releases).

To install it:
```bash
wget https://github.com/tensorchord/VectorChord/releases/download/0.2.1/postgresql-17-vchord_0.2.1-1_amd64.deb
sudo apt install postgresql-17-vchord_*.deb
```

### More Methods

VectorChord also supports other installation methods, including:
- [From ZIP package](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html#from-zip-package)

## Usage

VectorChord depends on pgvector, including the vector representation. 
This way, we can keep the maximum compatibility of `pgvector` for both:
- [vector storing](https://github.com/pgvector/pgvector#storing)
- [vector query](https://github.com/pgvector/pgvector#querying).

Since you can use them directly, your application can be easily migrated without pain!

Before all, you need to run the following SQL to ensure the extension is enabled.

```SQL
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```
It will install both `pgvector` and `VectorChord`, see [requirements](#requirements) for more detail.

### Storing

Similar to pgvector, you can create a table with vector column in VectorChord and insert some rows to it.

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) SELECT ARRAY[random(), random(), random()]::real[] FROM generate_series(1, 1000);
```

### Indexing

Similar to [ivfflat](https://github.com/pgvector/pgvector#ivfflat), the index type of VectorChord, RaBitQ(vchordrq) also divides vectors into lists, and then searches a subset of those lists that are closest to the query vector. It inherits the advantages of `ivfflat`, such as fast build times and less memory usage, but has [much better performance](https://blog.vectorchord.ai/vectorchord-store-400k-vectors-for-1-in-postgresql#heading-ivf-vs-hnsw) than hnsw and ivfflat.

The RaBitQ(vchordrq) index is supported on some pgvector types and metrics:

|                         | vector | halfvec | bit(n) | sparsevec |
| ----------------------- | ------ | ------- | ------ | --------- |
| L2 distance / `<->`     | ✅      | ✅       | 🆖      | ❌         |
| inner product / `<#>`   | ✅      | ✅       | 🆖      | ❌         |
| cosine distance / `<=>` | ✅      | ✅       | 🆖      | ❌         |
| L1 distance / `<+>`     | ❌      | ❌       | 🆖      | ❌         |
| Hamming distance/ `<~>` | 🆖      | 🆖       | ❌      | 🆖         |
| Jaccard distance/ `<%>` | 🆖      | 🆖       | ❌      | 🆖         |

Where:
- ✅ means supported by pgvector and VectorChord
- ❌ means supported by pgvector but not by VectorChord
- 🆖 means not planned by pgvector and VectorChord
- 🔜 means supported by pgvector now and will be supported by VectorChord soon

To create the VectorChord RaBitQ(vchordrq) index, you can use the following SQL.

L2 distance
```sql
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops) WITH (options = $$
residual_quantization = true
[build.internal]
lists = [1000]
spherical_centroids = false
$$);
```

> [!NOTE]
> - Set `residual_quantization` to true and `spherical_centroids` to false for L2 distance
> - Use `halfvec_l2_ops` for `halfvec`
> - The recommend `lists` could be rows / 1000 for up to 1M rows and 4 * sqrt(rows) for over 1M rows

Inner product
```sql
CREATE INDEX ON items USING vchordrq (embedding vector_ip_ops) WITH (options = $$
residual_quantization = false
[build.internal]
lists = [1000]
spherical_centroids = true
$$);
```

Cosine distance
```sql
CREATE INDEX ON items USING vchordrq (embedding vector_cosine_ops) WITH (options = $$
residual_quantization = false
[build.internal]
lists = [1000]
spherical_centroids = true
$$);
```

> [!NOTE]
> - Set `residual_quantization` to false and `spherical_centroids` to true for inner product/cosine distance
> - Use `vector_cosine_ops`/`vector_ip_ops` for `halfvec`

### Query

The query statement is exactly the same as pgvector. VectorChord supports any filter operation and WHERE/JOIN clauses like pgvecto.rs with VBASE.
```SQL
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```
Supported distance functions are:
- <-> - L2 distance
- <#> - (negative) inner product
- <=> - cosine distance

## Performance Tuning

### Index Build Time

Index building can parallelized, and with external centroid precomputation, the total time is primarily limited by disk speed. Optimize parallelism using the following settings:

```SQL
-- Set this to the number of CPU cores available for parallel operations.
SET max_parallel_maintenance_workers = 8;
SET max_parallel_workers = 8;

-- Adjust the total number of worker processes. 
-- Note: A restart is required for this setting to take effect.
ALTER SYSTEM SET max_worker_processes = 8;
```

## Query Performance
You can fine-tune the search performance by adjusting the `probes` and `epsilon` parameters:

```sql
-- Set probes to control the number of lists scanned. 
-- Recommended range: 3%–10% of the total `lists` value.
SET vchordrq.probes = 100;

-- Set epsilon to control the reranking precision.
-- Larger value means more rerank for higher recall rate and latency.
-- If you need a less precise query, set it to 1.0 might be appropriate.
-- Recommended range: 1.0–1.9. Default value is 1.9.
SET vchordrq.epsilon = 1.9;

-- vchordrq relies on a projection matrix to optimize performance.
-- Add your vector dimensions to the `prewarm_dim` list to reduce latency.
-- If this is not configured, the first query will have higher latency as the matrix is generated on demand.
-- Default value: '64,128,256,384,512,768,1024,1536'
-- Note: This setting requires a database restart to take effect.
ALTER SYSTEM SET vchordrq.prewarm_dim = '64,128,256,384,512,768,1024,1536';
```

And for postgres's setting
```SQL
-- If using SSDs, set `effective_io_concurrency` to 200 for faster disk I/O.
SET effective_io_concurrency = 200;

-- Disable JIT (Just-In-Time Compilation) as it offers minimal benefit (1–2%) 
-- and adds overhead for single-query workloads.
SET jit = off;

-- Allocate at least 25% of total memory to `shared_buffers`. 
-- For disk-heavy workloads, you can increase this to up to 90% of total memory. You may also want to disable swap with network storage to avoid io hang.
-- Note: A restart is required for this setting to take effect.
ALTER SYSTEM SET shared_buffers = '8GB';
```

## Advanced Features

### Indexing prewarm

For disk-first indexing, RaBitQ(vchordrq) is loaded from disk for the first query, 
and then cached in memory if `shared_buffer` is sufficient, resulting in a significant cold-start slowdown.

To improve performance for the first query, you can try the following SQL that preloads the index into memory.

```SQL
-- vchordrq_prewarm(index_name::regclass) to prewarm the index into the shared buffer
SELECT vchordrq_prewarm('gist_train_embedding_idx'::regclass)"
```

### Indexing Progress
You can check the indexing progress by querying the `pg_stat_progress_create_index` view.
```SQL
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;
```

### External Index Precomputation

Unlike pure SQL, an external index precomputation will first do clustering outside and insert centroids to a PostgreSQL table. Although it might be more complicated, external build is definitely much faster on larger dataset (>5M).

To get started, you need to do a clustering of vectors using `faiss`, `scikit-learn` or any other clustering library.

The centroids should be preset in a table of any name with 3 columns:
- id(integer): id of each centroid, should be unique
- parent(integer, nullable): parent id of each centroid, should be NULL for normal clustering
- vector(vector): representation of each centroid, `pgvector` vector type

And example could be like this:

```sql
-- Create table of centroids
CREATE TABLE public.centroids (id integer NOT NULL UNIQUE, parent integer, vector vector(768));
-- Insert centroids into it
INSERT INTO public.centroids (id, parent, vector) VALUES (1, NULL, '{0.1, 0.2, 0.3, ..., 0.768}');
INSERT INTO public.centroids (id, parent, vector) VALUES (2, NULL, '{0.4, 0.5, 0.6, ..., 0.768}');
INSERT INTO public.centroids (id, parent, vector) VALUES (3, NULL, '{0.7, 0.8, 0.9, ..., 0.768}');
-- ...

-- Create index using the centroid table
CREATE INDEX ON gist_train USING vchordrq (embedding vector_l2_ops) WITH (options = $$
[build.external]
table = 'public.centroids'
$$);
```

To simplify the workflow, we provide end-to-end scripts for external index pre-computation, see [scripts](./scripts/README.md#run-external-index-precomputation-toolkit).

<!-- TODO: Here we have a memory leak in rerank_in_table, show it until the feature is ready

### Capacity-optimized Index

The default behavior of Vectorchord is `performance-optimized`, 
which uses more disk space but has a better latency:
- About `80G` for `5M` 768 dim vectors
- About `800G` for `100M` 768 dim vectors

Although it is acceptable for such large data, it could be switched to `capacity-optimized` index and save about **50%** of your disk space. 

For `capacity-optimized` index, just enable the `rerank_in_table` option when creating the index:
```sql
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops) WITH (options = $$
residual_quantization = true
rerank_in_table = true
[build.internal]
...
$$);
```

> [!CAUTION]
> Compared to the `performance-optimized` index, the `capacity-optimized` index will have a **30-50%** increase in latency and QPS loss at query.

-->

### Range Query

To query vectors within a certain distance range, you can use the following syntax.
```SQL
-- Query vectors within a certain distance range
SELECT vec FROM t WHERE vec <<->> sphere('[0.24, 0.24, 0.24]'::vector, 0.012) 
ORDER BY embedding <-> '[0.24, 0.24, 0.24]' LIMIT 5;
```

In this expression, `vec <<->> sphere('[0.24, 0.24, 0.24]'::vector, 0.012)` is equals to `vec <-> '[0.24, 0.24, 0.24]' < 0.012`. However, the latter one will trigger a **exact nearest neighbor search** as the grammar could not be pushed down.

Supported range functions are:
- `<<->>` - L2 distance
- `<<#>>` - (negative) inner product
- `<<=>>` - cosine distance

## Development

### Build the Postgres Docker Image with VectorChord extension

Follow the steps in [Dev Guidance](./scripts/README.md#build-docker).

### Installing From Source

Install pgrx according to [pgrx's instruction](https://github.com/pgcentralfoundation/pgrx?tab=readme-ov-file#getting-started).
```bash
cargo install --locked cargo-pgrx
cargo pgrx init --pg17 $(which pg_config) # To init with system postgres, with pg_config in PATH
cargo pgrx install --release --sudo # To install the extension into the system postgres with sudo
```

## Limitations
- KMeans Clustering: The built-in KMeans clustering depends on multi-thread in-memory build and may require substantial memory. We strongly recommend using external centroid precomputation for efficient index construction.


## License

This software is licensed under a dual license model:

1. **GNU Affero General Public License v3 (AGPLv3)**: You may use, modify, and distribute this software under the terms of the AGPLv3.

2. **Elastic License v2 (ELv2)**: You may also use, modify, and distribute this software under the Elastic License v2, which has specific restrictions.

You may choose either license based on your needs. We welcome any commercial collaboration or support, so please email us <vectorchord-inquiry@tensorchord.ai> with any questions or requests regarding the licenses.


[image-compare]: https://github.com/user-attachments/assets/2d985f1e-7093-4c3a-8bf3-9f0b92c0e7e7
[license-1-link]: https://github.com/tensorchord/VectorChord#license
[license-1-shield]: https://img.shields.io/badge/License-AGPLv3-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=
[license-2-link]: https://github.com/tensorchord/VectorChord#license
[license-2-shield]: https://img.shields.io/badge/License-ELv2-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=

[docker-release-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-release-shield]: https://img.shields.io/docker/v/tensorchord/vchord-postgres?color=369eff&label=docker&labelColor=black&logo=docker&logoColor=white&style=flat&sort=semver
[github-release-link]: https://github.com/tensorchord/VectorChord/releases
[github-release-shield]: https://img.shields.io/github/v/release/tensorchord/VectorChord?color=369eff&labelColor=black&logo=github&style=flat
[docker-pulls-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/vchord-postgres?color=45cc11&labelColor=black&style=flat&sort=semver
[previous-docker-pulls-link]: https://hub.docker.com/r/tensorchord/pgvecto-rs
[previous-docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/pgvecto-rs?color=45cc11&labelColor=black&style=flat&sort=semver
[discord-link]: https://discord.gg/KqswhpVgdU
[discord-shield]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[X-link]: https://twitter.com/TensorChord
[X-shield]: https://img.shields.io/twitter/follow/tensorchord?style=flat&logo=X&cacheSeconds=60
[cloud-link]: https://cloud.vectorchord.ai/
[cloud-shield]: https://img.shields.io/badge/VectorChord_Cloud-Try_For_Free-F2B263.svg?labelColor=DAFDBA&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMzMiIGhlaWdodD0iMTMyIiBmaWxsPSJub25lIj48cGF0aCBmaWxsPSIjRTZEQjNEIiBkPSJNNDguNCAzNy41YzAtMSAwLTEuNS0uMi0xLjhhMSAxIDAgMCAwLS44LS40Yy0uMyAwLS43LjMtMS42IDFMMjcuNiA1MC4xYy0xLjIuOC0xLjcgMS4zLTIuMiAxLjhhNSA1IDAgMCAwLS44IDEuNmMtLjIuNy0uMiAxLjQtLjIgMi45djM3LjNjMCAxLjIgMCAxLjguMyAyIC4yLjMuNS41LjguNC40IDAgLjgtLjQgMS42LTEuM2wxOS0xOC42IDEuNS0xLjhjLjMtLjQuNS0xIC42LTEuNC4yLS42LjItMS4yLjItMi40VjM3LjVaTTM1LjIgMTA1LjNjLS44LjgtMS4yIDEuMy0xLjIgMS42IDAgLjQgMCAuNy4zLjkuMy4yLjkuMiAyIC4yaDM3YzEuMyAwIDIgMCAyLjUtLjJhNSA1IDAgMCAwIDEuNS0uNmMuNi0uNCAxLS45IDEuOS0xLjhMOTYuNiA4NmMuNy0uOSAxLjEtMS4zIDEuMS0xLjZhMSAxIDAgMCAwLS4zLS44Yy0uMy0uMy0uOS0uMy0yLS4zaC0zNWMtMS4yIDAtMS44IDAtMi40LjJhNSA1IDAgMCAwLTEuNC42Yy0uNS4zLTEgLjctMS44IDEuNmwtMTkuNiAxOS42Wk05Ni4zIDcwLjFjMSAwIDEuNCAwIDEuNy0uMi40LS4xLjYtLjQuOC0uN2wuMS0xLjdWMzUuM2wtLjEtMS44Yy0uMi0uMy0uNC0uNS0uOC0uNy0uMy0uMi0uOC0uMi0xLjctLjJINjQuMWMtMSAwLTEuNCAwLTEuNy4yLS40LjItLjYuNC0uOC43bC0uMSAxLjh2MzIuMmwuMSAxLjdjLjIuMy40LjYuOC43LjMuMi44LjIgMS43LjJoMzIuMloiLz48cGF0aCBmaWxsPSIjMTAxNTA5IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00My4yIDIxLjVjLTEuMyAwLTIgMC0yLjMtLjMtLjMtLjMtLjUtLjYtLjUtMSAwLS41LjQtMSAxLjEtMi4xTDUzLjEgMi4zYy42LS44LjktMS4yIDEuMi0xLjNoMWMuNC4xLjcuNSAxLjIgMS4zbDExLjYgMTUuOGMuOCAxIDEuMiAxLjYgMS4yIDIgMCAuNS0uMi44LS41IDEtLjQuNC0xIC40LTIuNC40SDU5Yy0uMy4yLS41LjQtLjYuNy0uMi4zLS4yLjYtLjIgMS40VjY4YzAgMS45IDAgMi44LjQgMy41LjMuNi44IDEuMSAxLjQgMS41LjcuMyAxLjcuMyAzLjUuM2g0NGwxLjMtLjFjLjMtLjEuNS0uMy42LS42bC4xLTEuNFY2NWMwLTEuMyAwLTIgLjMtMi4zLjItLjMuNi0uNSAxLS41czEgLjMgMiAxTDEzMCA3NWMuOC42IDEuMy45IDEuNCAxLjJ2MWMtLjEuNC0uNi43LTEuNCAxLjNsLTE3LjMgMTEuN2MtMSAuOC0xLjYgMS4xLTIgMS4xLS40IDAtLjgtLjItMS0uNS0uMy0uNC0uMy0xLS4zLTIuM1Y4MmwtLjEtMS40Yy0uMS0uMi0uMy0uNC0uNi0uNS0uMy0uMi0uNi0uMi0xLjQtLjJINjAuNWMtMS42IDAtMi40IDAtMy4xLjItLjcuMi0xLjMuNC0yIC44bC0yLjMgMi0yOC42IDI4LjNjLS41LjUtLjguOC0uOSAxLjF2LjhjLjEuMy40LjYuOSAxLjFsMy43IDMuN2MuOCAxIDEuMyAxLjMgMS4zIDEuOCAwIC40IDAgLjctLjMgMS0uMy4zLS45LjUtMiAuOGwtMjAgNC43Yy0xLjEuMi0xLjcuMy0yIC4yLS40LS4xLS43LS40LS44LS44LS4xLS4zIDAtMSAuMy0ybDUtMTkuOWMuMy0xLjEuNC0xLjcuOC0yIC4zLS4zLjctLjQgMS0uMy41IDAgLjkuNSAxLjggMS40bDMuNiAzLjcgMSAuOWguOWMuMy0uMS42LS40IDEtLjlsMjguNi0yOC4yYzEuMi0xLjEgMS43LTEuNyAyLjEtMi4zLjQtLjYuNy0xLjMuOC0yIC4yLS43LjItMS41LjItMy4yVjIzLjZsLS4xLTEuNGMtLjEtLjMtLjMtLjUtLjYtLjZsLTEuNC0uMWgtNi4yWiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+
[blog-link]: https://blog.vectorchord.ai/
[official-site-link]: https://vectorchord.ai/
[github-issues-link]: https://github.com/tensorchord/VectorChord/issues
[email-link]: mailto:support@tensorchord.ai