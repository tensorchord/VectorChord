## Build Docker

```shell
cargo build --package rabbithole --lib --features pg16 --target x86_64-unknown-linux-gnu --release
./tools/schema.sh --features pg16 --target x86_64-unknown-linux-gnu --release | expand -t 4 > ./target/schema.sql

export SEMVER="0.0.0"
export VERSION="16"
export ARCH="x86_64"
export PLATFORM="amd64"
./tools/package.sh

docker build -t rabbithole:pg16-latest -f ./docker/Dockerfile .
```

Or you can use `starkind/rabbithole:pg16-latest` to run the bench.

## Run Instance

```shell
docker run --name rabbithole -e POSTGRES_PASSWORD=123 -p 5432:5432 -d starkind/rabbithole:pg16-latest

PGPASSWORD=123 psql -h 127.0.0.1 -U postgres -c "CREATE USER bench WITH PASSWORD '123';"
PGPASSWORD=123 psql -h 127.0.0.1 -U postgres -c "ALTER ROLE bench SUPERUSER;"
```

## Download Data

```shell
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx

aws s3 cp s3://pgvecto.rs-bench/sift_128_1m/sift.hdf5 $(whoami)/sift/sift.hdf5
aws s3 cp s3://pgvecto.rs-bench/gist_960_1m/gist.hdf5 $(whoami)/gist/gist.hdf5
aws s3 cp s3://pgvecto.rs-bench/glove_200_1m/glove.hdf5 $(whoami)/glove/glove.hdf5
aws s3 cp s3://pgvecto.rs-bench/openai_1536_500k/openai.hdf5 $(whoami)/openai/openai.hdf5
aws s3 cp s3://pgvecto.rs-bench/cohere-768-1m-2022/cohere.hdf5 $(whoami)/cohere_1m_22/cohere_1m_22.hdf5
aws s3 cp s3://pgvecto.rs-bench/cohere-1024-1m-2023/cohere-1m-23.hdf5 $(whoami)/cohere_1m_23/cohere_1m_23.hdf5
aws s3 cp s3://pgvecto.rs-bench/cohere-1024-10m-2023/cohere-10m-23.hdf5 $(whoami)/cohere_10m_23/cohere_10m_23.hdf5
```

## Run Bench

Options for `-n`:
- sift
- glove
- gist
- openai
- cohere_1m_22
- cohere_1m_23
- cohere_10m_23

```shell
# pip install pgvector numpy faiss-cpu psycopg h5py

# If you want to use internal k-means(slow)
python internal.py -n sift -m l2

# Or external k-means
## K-means generate centroids
python train.py -n sift

## Load centroids into PG and build index
python load.py -n sift -m l2

# Run bench
python bench.py -n sift -m l2
```