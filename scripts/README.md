## Build Docker

```shell
SEMVER='0.2.1'
VERSION='17'
ARCH='x86_64'
PLATFORM='amd64'

git clone https://github.com/tensorchord/VectorChord.git
cd VectorChord
git checkout $SEMVER

sudo apt install -y build-essential libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config

sudo apt-get install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt-get install -y postgresql-server-dev-$VERSION
sudo apt-get install -y postgresql-$VERSION
sudo apt-get install -y postgresql-$VERSION-pgvector

curl --proto '=https' --tlsv1.2 -sSf https://apt.llvm.org/llvm.sh | sudo bash -s -- 18
sudo update-alternatives --install /usr/bin/clang clang $(which clang-18) 255

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

cargo install cargo-pgrx@$(sed -n 's/.*pgrx = { version = "\(=.*\)",.*/\1/p' Cargo.toml) --locked
cargo pgrx init --pg$VERSION=$(which pg_config)

cargo build --lib --features pg$VERSION --release

mkdir -p ./build/zip
cp -a ./sql/upgrade/. ./build/zip/
cp ./sql/install/vchord--$SEMVER.sql ./build/zip/vchord--$SEMVER.sql
sed -e "s/@CARGO_VERSION@/$SEMVER/g" < ./vchord.control > ./build/zip/vchord.control
cp ./target/release/libvchord.so ./build/zip/vchord.so
zip ./build/postgresql-${VERSION}-vchord_${SEMVER}_${ARCH}-linux-gnu.zip -j ./build/zip/*

mkdir -p ./build/deb
mkdir -p ./build/deb/DEBIAN
mkdir -p ./build/deb/usr/share/postgresql/$VERSION/extension/
mkdir -p ./build/deb/usr/lib/postgresql/$VERSION/lib/
for file in $(ls ./build/zip/*.sql | xargs -n 1 basename); do
    cp ./build/zip/$file ./build/deb/usr/share/postgresql/$VERSION/extension/$file
done
for file in $(ls ./build/zip/*.control | xargs -n 1 basename); do
    cp ./build/zip/$file ./build/deb/usr/share/postgresql/$VERSION/extension/$file
done
for file in $(ls ./build/zip/*.so | xargs -n 1 basename); do
    cp ./build/zip/$file ./build/deb/usr/lib/postgresql/$VERSION/lib/$file
done
echo "Package: postgresql-${VERSION}-vchord
Version: ${SEMVER}-1
Section: database
Priority: optional
Architecture: ${PLATFORM}
Maintainer: Tensorchord <support@tensorchord.ai>
Description: Vector database plugin for Postgres, written in Rust, specifically designed for LLM
Homepage: https://vectorchord.ai/
License: AGPL-3 or Elastic-2" \
> ./build/deb/DEBIAN/control
(cd ./build/deb && md5sum usr/share/postgresql/$VERSION/extension/* usr/lib/postgresql/$VERSION/lib/*) > ./build/deb/DEBIAN/md5sums
dpkg-deb --root-owner-group -Zxz --build ./build/deb/ ./build/postgresql-${VERSION}-vchord_${SEMVER}-1_${PLATFORM}.deb

ls ./build

docker build -t vchord:pg$VERSION-latest --build-arg PG_VERSION=$VERSION -f ./docker/Dockerfile .
```

## Run Instance

```shell
VERSION='17'

docker run --name vchord -e POSTGRES_PASSWORD=123 -p 5432:5432 -d vchord:pg$VERSION-latest
```

## Run External Index Precomputation Toolkit

1. Install requirements

```shell
# PYTHON = 3.11
# When using CPU to train k-means clustering
conda install conda-forge::pgvector-python numpy pytorch::faiss-cpu conda-forge::psycopg h5py tqdm
# or
pip install pgvector numpy faiss-cpu psycopg h5py tqdm

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

   `lists` is the number of centroids for clustering, and a typical value for large datasets(>5M) could range from:
   
   $$
   4*\sqrt{len(vectors)} \le lists \le 16*\sqrt{len(vectors)}
   $$

3. To insert vectors and centroids into the database, and then create an index 

   ```shell
   python script/index.py -n [table name] -i [dataset file(export.hdf5)] -c [centroid filename(centroid.npy)] -m [metric(l2/cos/dot)] -d [dim] --url postgresql://postgres:123@localhost:5432/postgres
   ```

4. Let's start our tour to check the benchmark result of VectorChord

   ```shell
   python script/bench.py -n [table name] -i [dataset file(export.hdf5)] -m [metric(l2/cos/dot)] --nprob 100 --epsilon 1.0  --url postgresql://postgres:123@localhost:5432/postgres
   ```