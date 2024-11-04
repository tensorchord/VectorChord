## Build Docker

```shell
sudo apt install -y build-essential libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config clang
cargo install --locked cargo-pgrx
cargo pgrx init
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
docker run --name rabbithole -e POSTGRES_PASSWORD=123 -p 5432:5432 -d rabbithole:pg16-latest

PGPASSWORD=123 psql -h 127.0.0.1 -U postgres -c "CREATE USER bench WITH PASSWORD '123';"
PGPASSWORD=123 psql -h 127.0.0.1 -U postgres -c "ALTER ROLE bench SUPERUSER;"
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
# pip install pgvector numpy faiss-cpu psycopg h5py tqdm

# If using GPU for train.py:
# conda install pytorch::faiss-gpu

# dump table embedding column to a local h5 file["train"]
python dump.py -n sift -o sift.h5 -c embedding -d 128

# external k-means
python train.py -i sift.hdf5 -o sift_centroids_4096 -m l2

# build index (w/wo external centroids)
## with external centroids
python index.py -n sift -c sift_centroids_4096.npy -i sift.hdf5 -m l2 -p 123 -k 4096 -d 768 -w 4
## without external centroids
## python index.py -n sift -i sift.hdf5 -m l2 -p 123 -k 4096 -d 768 -w 4

# bench
python bench.py -n sift -i sift.hdf5 --nprob 100
```
