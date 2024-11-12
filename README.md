<div align="center">
<h1>VectorChord-Vec</h1>
<h4>Host 100M 768-dim vector (275GB) on a $250 machine (4 vcpu, 32GB) on aws with VectorChord-Vec</h4>
</div>

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
<a href="https://hub.docker.com/r/tensorchord/vcvec"><img src="https://img.shields.io/docker/pulls/tensorchord/vcvec" /></a>
<a href="https://github.com/tensorchord/vcvec#contributors-"><img alt="all-contributors" src="https://img.shields.io/github/all-contributors/tensorchord/vcvec/main"></a>
</p>

VectorChord-Vec (vchordvec) is a PostgreSQL extension designed for scalable, high-performance, and disk-efficient vector similarity search. It serves as the successor to the pgvecto.rs project.

## Features
- **Blazing-Fast Queries**: Achieve up to 3x faster queries compared to pgvector's HNSW, maintaining the same recall level.
- **External Index Precomputation**: Built on IVF, VectorChord-Vec enables KMeans clustering to be performed externally (e.g., on a GPU) and seamlessly imported into the database.
- **Lightning-Fast Index Building**: Build index up to 20x faster than pgvector hnsw with precomputed centroids. (1.5 min for 1M 960-dim vectors)
- **High-throughput Update**: TODO
- **Advanced Quantization**: Uses cutting-edge RaBitQ to compress float vectors into compact bit representations with autonomous reranking.
- **Optimized SIMD Kernels**: Features a highly tuned computation kernel optimized for fast scans using SIMD and efficient register management.
- **Disk-Friendly Performance**: Query 100M 768-dimensional vectors using just 32GB of memory, achieving 35ms latency with top10 recall@95%.
- **Seamless Compatibility**: Compatible with pgvector data types while delivering faster indexing and querying.
- **Simple Configuration**: No need to tweak quantization or rerank parameters — best defaults are provided out of the box.

## Quick Start
For new users, we recommend using the Docker image to get started quickly.
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
```
Run the following SQL to ensure the extension is enabled.

```SQL
CREATE EXTENSION vchordvec CASCADE;
```

To create the index, you can use the following SQL.

```SQL
CREATE INDEX ON gist_train USING vchordrq (embedding vchordrq.vector_l2_ops) WITH (options = $$
lists = 4096
spherical_centroids = true
$$);
```

## Documentation

### Query

The query statement is exactly the same as pgvector.
```SQL
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```
Supported distance functions are:
- <-> - L2 distance
- <#> - (negative) inner product
- <=> - cosine distance

### Query Performance Tuning
You can fine-tune the search performance by adjusting the `probes` and `epsilon` parameters:

```sql
-- Set probes to control the number of lists scanned. 
-- Recommended range: 3%–10% of the total `lists` value.
SET vchordvec.probes = 100;

-- Set epsilon to control the reranking precision. 
-- Recommended range: 1.0–1.9.
SET vchordvec.epsilon = 1.0;
```

And for postgres's setting
```SQL
-- If using SSDs, set `effective_io_concurrency` to 200 for faster disk I/O.
SET effective_io_concurrency = 200;

-- Disable JIT (Just-In-Time Compilation) as it offers minimal benefit (1–2%) 
-- and adds overhead for single-query workloads.
SET jit = off;

-- Allocate at least 25% of total memory to `shared_buffers`. 
-- For disk-heavy workloads, you can increase this to up to 90% of total memory.
-- Note: A restart is required for this setting to take effect.
ALTER SYSTEM SET shared_buffers = '8GB';
```

### Indexing
TODO

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

### Installing From Source
TODO

## Limitations
- Data Type Support: Currently, only the `f32` data type is supported for vectors, and the dimensionality is limited to 2000. (Dimension support improvements are planned for future updates.)
- Architecture Compatibility: The fast-scan kernel is optimized for x86_64 architectures. While it runs on aarch64, performance may be significantly lower.
- KMeans Clustering: The built-in KMeans clustering is not yet fully optimized and may require substantial memory. We strongly recommend using external centroid precomputation for efficient index construction.


## License
This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE) and as commercial software. For commercial licensing, please contact us at support@tensorchord.ai.

