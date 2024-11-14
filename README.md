<div align="center">
<h1 align=center>VectorChord</h1>
<h4 align=center>Host 100M 768-dim vector (250GB+) on an AWS i4i.xlarge machine ($250/month) with 4 vCPUs and 32GB of RAM using VectorChord.</h4>
</div>

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
<a href="https://hub.docker.com/r/tensorchord/vchord-postgres"><img src="https://img.shields.io/docker/pulls/tensorchord/vchord-postgres" /></a>
<a href="https://github.com/tensorchord/VectorChord#contributors-"><img alt="all-contributors" src="https://img.shields.io/github/all-contributors/tensorchord/VectorChord/main"></a>
</p>

VectorChord (vchord) is a PostgreSQL extension designed for scalable, high-performance, and disk-efficient vector similarity search. It serves as the successor to the [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) project. 

VectorChord incorporates the insights and lessons learned from pgvecto.rs, providing faster query speeds, more flexible build options, significantly enhanced index build performance, and greater stability. It is entirely based on Postgres storage, allowing for physical replication and WAL incremental backups for index. This also enables effective ssd usage to reduce memory requirements and can easily handle vectors ranging from hundreds of millions to multi billions of entries.

## Features
- **Blazing-Fast Queries**: Achieve up to 3x faster queries compared to pgvector's HNSW, maintaining the same recall level.
- **External Index Precomputation**: Built on IVF, VectorChord enables KMeans clustering to be performed externally (e.g., on a GPU) and seamlessly imported into the database.
- **Lightning-Fast Index Building**: Build index up to 20x faster than pgvector hnsw with precomputed centroids. (1.5 min for 1M 960-dim vectors)
- **Advanced Quantization**: Uses cutting-edge RaBitQ to compress float vectors into compact bit representations with autonomous reranking.
- **Optimized SIMD Kernels**: Features a highly tuned computation kernel optimized for fast scans using SIMD and efficient register management.
- **Disk-Friendly Performance**: Query laion-100M 768-dim vectors using just 32GB of memory, achieving 35ms P50 latency with top10 recall@95%.
- **Seamless Compatibility**: Compatible with pgvector data types while delivering faster indexing and querying.
- **Simple Configuration**: No need to tweak quantization or rerank parameters — best defaults are provided out of the box.
<!-- - **High-throughput Update**: TODO -->

## Quick Start
<!-- For new users, we recommend using the Docker image to get started quickly.
```bash
docker run \
  --name vectorchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/vectorchord:pg17-v0.1.0
```
Then you can connect to the database using the `psql` command line tool. The default username is `postgres`, and the default password is `mysecretpassword`.
```bash
psql -h localhost -p 5432 -U postgres
``` -->
Run the following SQL to ensure the extension is enabled.

```SQL
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```

And make sure to add vchord.so to the shared_preload_libraries in postgresql.conf.

```SQL
-- Add vchord and pgvector to shared_preload_libraries --
ALTER SYSTEM SET shared_preload_libraries = 'vchord.so';
```

To create the VectorChord RaBitQ(vchordrq) index, you can use the following SQL.

```SQL
CREATE INDEX ON gist_train USING vchordrq (embedding vector_l2_ops) WITH (options = $$
residual_quantization = true
[build.internal]
lists = [4096]
spherical_centroids = false
$$);
```

## Documentation

### Query

The query statement is exactly the same as pgvector. VectorChord supports any filter operation and WHERE/JOIN clauses like pgvecto.rs with VBASE.
```SQL
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```
Supported distance functions are:
- <-> - L2 distance
- <#> - (negative) inner product
- <=> - cosine distance

<!-- ### Range Query

To query vectors within a certain distance range, you can use the following syntax.
```SQL

``` -->

### Query Performance Tuning
You can fine-tune the search performance by adjusting the `probes` and `epsilon` parameters:

```sql
-- Set probes to control the number of lists scanned. 
-- Recommended range: 3%–10% of the total `lists` value.
SET vchordrq.probes = 100;

-- Set epsilon to control the reranking precision.
-- Larger value means more rerank for higher recall rate.
-- Recommended range: 1.0–1.9.
SET vchordrq.epsilon = 1.0;

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

<!-- ### Indexing
TODO prewarm -->


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

### Installing From Source
Install pgrx according to [pgrx's instruction](https://github.com/pgcentralfoundation/pgrx?tab=readme-ov-file#getting-started).
```bash
cargo install --locked cargo-pgrx
cargo pgrx init --pg17 $(which pg_config) # To init with system postgres, with pg_config in PATH
cargo pgrx install --release --sudo # To install the extension into the system postgres with sudo
```

## Limitations
- Data Type Support: Currently, only the `f32` data type is supported for vectors, and the dimensionality is limited to 2000. (Dimension support improvements are planned for future updates.)
- Architecture Compatibility: The fast-scan kernel is optimized for x86_64 architectures. While it runs on aarch64, performance may be significantly lower.
- KMeans Clustering: The built-in KMeans clustering is not yet fully optimized and may require substantial memory. We strongly recommend using external centroid precomputation for efficient index construction.


## License
This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE) and as commercial software. For commercial licensing, please contact us at support@tensorchord.ai.

