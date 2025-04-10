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

**🔧 Accelerated Index Build**: Leverage IVF to build indexes externally (e.g., on GPU) for faster KMeans clustering, combined with RaBitQ[^2] compression to efficiently store vectors while maintaining search quality through autonomous reranking.

[^2]: Gao, Jianyang, and Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 2.3 (2024): 1-27.

**📏 Long Vector Support**: Store and search vectors up to 60,000[^3] dimensions, enabling the use of the best high-dimensional models like text-embedding-3-large with ease.

[^3]: There is a [limitation](https://github.com/pgvector/pgvector#vector-type) at pgvector of 16,000 dimensions now. If you really have a large dimension(`16,000<dim<60,000`), consider changing [VECTOR_MAX_DIM](https://github.com/pgvector/pgvector/blob/fef635c9e5512597621e5669dce845c744170822/src/vector.h#L4) and compile pgvector yourself.

**🌐 Scale As You Want**: Based on horizontal expansion, the query of 5M / 100M 768-dimensional vectors can be easily scaled to 10000+ QPS with top10 recall@90% at a competitive cost[^4]

[^4]: Please check our [blog post](https://blog.vectorchord.ai/vector-search-at-10000-qps-in-postgresql-with-vectorchord)  for more details, the PostgreSQL scalability is powered by [CloudNative-PG](https://github.com/cloudnative-pg/cloudnative-pg).

## Quick Start

For new users, we recommend using the Docker image to get started quickly. If you do not prefer Docker, please read [installation guide](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html) for other installation methods.

```bash
docker run \
  --name vectorchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/vchord-postgres:pg17-v0.2.2
```
> In addition to the base image with the VectorChord Extension, we provide an all-in-one image, `tensorchord/vchord-suite:pg17-latest`. This comprehensive image includes all official TensorChord plugins. Developers should select an image tag that is compatible with their extension's version, as indicated in [the support matrix](https://github.com/tensorchord/VectorChord-images?tab=readme-ov-file#support-matrix).

Then you can connect to the database using the `psql` command line tool. The default username is `postgres`, and the default password is `mysecretpassword`.

```bash
psql -h localhost -p 5432 -U postgres
```

Now you can play with VectorChord!

VectorChord depends on pgvector, including the vector representation. Since you can use them directly, your application can be easily migrated without pain!

```sql
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
```

Similar to pgvector, you can create a table with vector column and insert some rows to it.

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) SELECT ARRAY[random(), random(), random()]::real[] FROM generate_series(1, 1000);
```

With VectorChord, you can create `vchordrq` indexes.

```SQL
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops) WITH (options = $$
residual_quantization = true
[build.internal]
lists = []
$$);
```

And then perform a vector search using `SELECT ... ORDER BY ... LIMIT ...`.

```SQL
SET vchordrq.probes TO '';
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```

For more usage, please read:

* [Indexing](https://docs.vectorchord.ai/vectorchord/usage/indexing.html)
* [Performance Tuning](https://docs.vectorchord.ai/vectorchord/usage/performance-tuning.html)
* [Advanced Features](https://docs.vectorchord.ai/vectorchord/usage/advanced-features.html)

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
