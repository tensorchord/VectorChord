<div align="center">
<h1 align=center>VectorChord</h1>
<h4 align=center>Ready for the Billion-Scale Era. Host 100M vectors on a single i4i.xlarge ($247/mo) and (scale seamlessly to 1B+)[https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql].</h4>
</div>

<div align=center>


[Official Site][official-site-link] · [Blog][blog-link] · [Docs][docs-link] · [Feedback][github-issues-link] · [Contact Us][email-link]


[![][github-release-shield]][github-release-link]
[![][docker-release-shield]][docker-release-link]
[![][docker-pulls-shield]][docker-pulls-link]
[![][ghcr-release-shield]][ghcr-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][discord-shield]][discord-link]
[![][X-shield]][X-link]
[![][deepwiki-shield]][deepwiki-link]
[![][license-1-shield]][license-1-link]
[![][license-2-shield]][license-2-link]
</div>

VectorChord (vchord) is a PostgreSQL extension engineered for scalable, high-performance, and cost-effective vector search.

To efficiently store vectors while preserving search quality, VectorChord applies RaBitQ[^1] compression together with autonomous reranking. With VectorChord, you can store 400,000 vectors for just $1, enabling significant savings: 6x more vectors compared to Pinecone's optimized storage and 26x more than pgvector/pgvecto.rs for the same price.

[^1]: Gao, Jianyang, and Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 2.3 (2024): 1-27.

![][image-compare]

## Features

VectorChord introduces remarkable enhancements over pgvecto.rs and pgvector:

**💰 Affordable Vector Search**: Host 100M × 768-dimensional vectors → AWS i4i.xlarge ($247/month)[^2], host 1B × 96-dimensional vectors → i7ie.6xlarge ($2246/month)[^3], helping you keep infrastructure costs down while maintaining competitive search quality.

[^2]: Please check out our [blog post](https://blog.vectorchord.ai/vectorchord-store-400k-vectors-for-1-in-postgresql) for more details.
[^3]: Please check out our [blog post](https://blog.vectorchord.ai/scaling-vector-search-to-1-billion-on-postgresql) for more details.

**⚡ Accelerated Index Build**: Index 100 million vectors in just 20 minutes. Powered by hierarchical K-means and highly optimized disk operations, VectorChord eliminates the bottleneck of vector indexing on a single machine with limited hardware resources.

[^4]: Please check out our [blog post](https://blog.vectorchord.ai/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql#heading-hierarchical-k-means) for more technique details and [document](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html#hierarchical-k-means) for usages.

**📈 Smoothly Scale Up**: Scale with confidence as your data grows. Through dimensionality reduction and sampling[^5], VectorChord effectively controls memory growth, enabling 1B-vector indexes to be built on machines with 128GB of memory in practice.

[^5]: Please check out our [blog post](https://blog.vectorchord.ai/how-we-made-100m-vector-indexing-in-20-minutes-possible-on-postgresql#heading-dimensionality-reduction) for more technique details and [document](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html#reduce-sampling-factor) for usages.

**🔌 Seamless Integration**: Fully compatible with pgvector data types and syntax while providing optimal defaults out of the box - no complex parameter tuning needed. Just drop in VectorChord for enhanced experience.

**💾 Efficient Storage with Low-Bit Data type**: Drastically reduce storage costs with our [native 4-bit (RaBitQ4) and 8-bit (RaBitQ8) vector types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html). Achieve massive space savings without compromising search quality—RaBitQ8 maintains high precision with <1% recall loss.

## Quick Start

For new users, we recommend using the Docker image to get started quickly. If you do not prefer Docker, please read [installation guide](https://docs.vectorchord.ai/vectorchord/getting-started/installation.html) for other installation methods.

```bash
docker run \
  --name vectorchord-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d ghcr.io/tensorchord/vchord-postgres:pg18-v1.1.0
```

> [!NOTE]
> In addition to the base image with the VectorChord extension, we provide an all-in-one image, `tensorchord/vchord-suite:pg17-latest`. This comprehensive image includes all official TensorChord extensions, including `VectorChord`, `VectorChord-bm25` and `pg_tokenizer.rs` . Developers should select an image tag that is compatible with their extension's version, as indicated in [the support matrix](https://github.com/tensorchord/VectorChord-images?tab=readme-ov-file#support-matrix).

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
CREATE INDEX ON items USING vchordrq (embedding vector_l2_ops);
```

And then perform a vector search using `SELECT ... ORDER BY ... LIMIT ...`.

```SQL
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```

For more usage, please read:

- [Indexing](https://docs.vectorchord.ai/vectorchord/usage/indexing.html)
- [Multi-Vector Retrieval](https://docs.vectorchord.ai/vectorchord/usage/indexing-with-maxsim-operators.html)
- [Quantization Types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html)
- [Graph Index](https://docs.vectorchord.ai/vectorchord/usage/graph-index.html)
- [Quantization Types](https://docs.vectorchord.ai/vectorchord/usage/quantization-types.html)
- [Similarity Filter](https://docs.vectorchord.ai/vectorchord/usage/range-query.html)
- [PostgreSQL Tuning](https://docs.vectorchord.ai/vectorchord/usage/performance-tuning.html)
- [Monitoring](https://docs.vectorchord.ai/vectorchord/usage/monitoring.html)
- [Fallback Parameters](https://docs.vectorchord.ai/vectorchord/usage/fallback-parameters.html)
- [Measure Recall](https://docs.vectorchord.ai/vectorchord/usage/measure-recall.html)
- [Prewarm](https://docs.vectorchord.ai/vectorchord/usage/prewarm.html)
- [Prefilter](https://docs.vectorchord.ai/vectorchord/usage/prefilter.html)
- [Prefetch](https://docs.vectorchord.ai/vectorchord/usage/prefetch.html)
- [Rerank in Table](https://docs.vectorchord.ai/vectorchord/usage/rerank-in-table.html)
- [Partitioning Tuning](https://docs.vectorchord.ai/vectorchord/usage/partitioning-tuning.html)
- [External Build](https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html)

## License

This software is licensed under a dual license model:

1. **GNU Affero General Public License v3 (AGPLv3)**: You may use, modify, and distribute this software under the terms of the AGPLv3.

2. **Elastic License v2 (ELv2)**: You may also use, modify, and distribute this software under the Elastic License v2, which has specific restrictions.

You may choose either license based on your needs. We welcome any commercial collaboration or support, so please email us <vectorchord-inquiry@tensorchord.ai> with any questions or requests regarding the licenses.

[cost-estimation]: https://github.com/user-attachments/assets/168fe550-6465-4eee-a224-8c848c301e3d
[image-compare]: https://github.com/user-attachments/assets/2d985f1e-7093-4c3a-8bf3-9f0b92c0e7e7
[license-1-link]: https://github.com/tensorchord/VectorChord#license
[license-1-shield]: https://img.shields.io/badge/License-AGPLv3-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=
[license-2-link]: https://github.com/tensorchord/VectorChord#license
[license-2-shield]: https://img.shields.io/badge/License-ELv2-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=

[docker-release-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-release-shield]: https://img.shields.io/docker/v/tensorchord/vchord-postgres?color=369eff&label=docker&labelColor=black&logo=docker&logoColor=white&style=flat
[github-release-link]: https://github.com/tensorchord/VectorChord/releases
[github-release-shield]: https://img.shields.io/github/v/release/tensorchord/VectorChord?color=369eff&labelColor=black&logo=github&style=flat
[ghcr-release-link]: https://github.com/orgs/tensorchord/packages/container/package/vchord-postgres
<!-- GHCR badge is not supported by shields.io yet, use docker badge instead -->
[ghcr-release-shield]: https://img.shields.io/docker/v/tensorchord/vchord-postgres?color=369eff&label=GHCR&labelColor=black&logo=github&logoColor=white&style=flat
[docker-pulls-link]: https://hub.docker.com/r/tensorchord/vchord-postgres
[docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/vchord-postgres?color=45cc11&labelColor=black&style=flat&sort=semver
[previous-docker-pulls-link]: https://hub.docker.com/r/tensorchord/pgvecto-rs
[previous-docker-pulls-shield]: https://img.shields.io/docker/pulls/tensorchord/pgvecto-rs?color=45cc11&labelColor=black&style=flat&sort=semver
[github-downloads-link]: https://github.com/tensorchord/VectorChord/releases
[github-downloads-shield]: https://img.shields.io/github/downloads/tensorchord/VectorChord/total?color=45cc11&labelColor=black&style=flat&sort=semver
[discord-link]: https://discord.gg/KqswhpVgdU
[discord-shield]: https://img.shields.io/discord/974584200327991326?&logoColor=white&color=5865F2&style=flat&logo=discord&cacheSeconds=60
[X-link]: https://x.com/TensorChord
[X-shield]: https://img.shields.io/badge/follow-%40tensorchord-1DA1F2?logo=x&style=flat&logoColor=white&color=1da1f2
[deepwiki-link]: https://deepwiki.com/tensorchord/VectorChord
[deepwiki-shield]: https://deepwiki.com/badge.svg
[blog-link]: https://blog.vectorchord.ai/
[official-site-link]: https://vectorchord.ai/
[github-issues-link]: https://github.com/tensorchord/VectorChord/issues
[email-link]: mailto:support@tensorchord.ai
[docs-link]: https://docs.vectorchord.ai/vectorchord/
