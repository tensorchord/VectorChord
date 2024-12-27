## Build Docker

Users can choose to build the package with the provided docker image or create the development environment by themselves.

- (option 1) With `vectorchord-pgrx` Image

```shell
# use the required version of `pgrx` and `rust`
export PGRX_VERSION=$(awk -F'version = "=|"' '/^pgrx\s*=.*version/ {print $2}' Cargo.toml)
export RUST_TOOLCHAIN=$(awk -F'"' '/^\s*channel\s*=/ {print $2}' rust-toolchain.toml)
export PGRX_IMAGE=ghcr.io/tensorchord/vectorchord-pgrx:$PGRX_VERSION-$RUST_TOOLCHAIN

docker run --rm -v .:/workspace $PGRX_IMAGE cargo build --lib --features pg16 --profile opt
docker run --rm -v .:/workspace $PGRX_IMAGE ./tools/schema.sh --features pg16 --profile opt
```

- (option 2) With Local Development Environment

```shell
sudo apt install -y build-essential libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config clang
cargo install --locked cargo-pgrx
cargo pgrx init
cargo build --package vchord --lib --features pg16 --profile opt
./tools/schema.sh --features pg16 --profile opt
```

- build the debian package

```shell
export SEMVER="0.0.0"
export VERSION="16"
export ARCH="x86_64"
export PLATFORM="amd64"
export PROFILE="opt"
./tools/package.sh
```

- build the docker image

```shell
docker build -t vchord:pg16-latest --build-arg PG_VERSION=16 -f ./docker/Dockerfile .
```

## Run Instance

```shell
docker run --name vchord -e POSTGRES_PASSWORD=123 -p 5432:5432 -d vchord:pg16-latest
```

## Run External Index Precomputation Toolkit

1. Install requirements

```shell
# PYTHON = 3.11
# When using CPU to train k-means clustering
conda install conda-forge::pgvector-python numpy pytorch::faiss-cpu conda-forge::psycopg h5py tqdm
# or
pip install pgvector-python numpy faiss-cpu psycopg h5py tqdm

# When using GPU to train k-means clustering
conda install conda-forge::pgvector-python numpy pytorch::faiss-gpu conda-forge::psycopg h5py tqdm
```

1. Prepare dataset in `hdf5` format

   - If you already have your vectors stored in `PostgreSQL` using `pgvector`, you can export them to a local file by:
     ```shell
     python script/dump.py -n [table name] -c [column name] -d [dim] -o export.hdf5
     ```

   - If you don't have any data, but would like to give it a try, you can choose one of these datasets:

     ```shell
     wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 # num=1M dim=128 metric=l2
     wget http://ann-benchmarks.com/gist-960-euclidean.hdf5 # num=1M dim=960 metric=l2
     wget https://myscale-datasets.s3.ap-southeast-1.amazonaws.com/laion-5m-test-ip.hdf5 # num=5M dim=768 metric=dot
     wget https://myscale-datasets.s3.ap-southeast-1.amazonaws.com/laion-20m-test-ip.hdf5 # num=20M dim=768 metric=dot
     wget https://myscale-datasets.s3.ap-southeast-1.amazonaws.com/laion-100m-test-ip.hdf5 # num=100M dim=768 metric=dot
     ```

2. Preform clustering of centroids from vectors

   ```shell
   # For small dataset size from 1M to 5M
   python script/train.py -i [dataset file(export.hdf5)] -o [centroid filename(centroid.npy)] -lists [lists] -m [metric(l2/cos/dot)]
   # For large datasets size, 5M to 100M in size, use GPU and mmap chunks
   python script/train.py -i [dataset file(export.hdf5)] -o [centroid filename(centroid.npy)] --lists [lists] -m [metric(l2/cos/dot)] -g --mmap
   ```

   `lists` is the number of centroids for clustering, and a typical value could range from:
   
   $$
   4*\sqrt{len(vectors)} \le lists \le 16*\sqrt{len(vectors)}
   $$

3. To insert vectors and centroids into the database, and then create an index 

   ```shell
   python script/index.py -n [table name] -i [dataset file(export.hdf5)] -c [centroid filename(centroid.npy)] -m [metric(l2/cos/dot)] -d [dim]
   ```

4. Let's start our tour to check the benchmark result of VectorChord

   ```shell
   python script/bench.py -n [table name] -i [dataset file(export.hdf5)] -m [metric(l2/cos/dot)] -p [database password] --nprob 100 --epsilon 1.0
   ```

    Larger `nprobe` and `epsilon` will have a more precise query but at a slower speed.