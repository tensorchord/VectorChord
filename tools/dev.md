## Release a new version

### Pre-requisite

```shell
SEMVER='0.2.1'
VERSION='17'

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
```

### Step 1 - generate install schema

```shell
SEMVER='0.2.1'
VERSION='17'

cargo build --lib --features pg$VERSION --release
cargo pgrx schema --features pg$VERSION | expand -t 4 > ./sql/install/vchord--$SEMVER.sql
```

### Step 2 - generate upgrade schema

```shell
PREV='0.2.0'
SEMVER='0.2.1'

diff -u ./sql/install/vchord--$PREV.sql ./sql/install/vchord--$SEMVER.sql | awk '
  /^\+/ && !/^+++/ { 
    print substr($0, 2) > "target/upgrade.sql"
    next
  }
  /^-/ && !/^---/ || /^@/ { print }
  { next }
'
cp target/upgrade.sql ./sql/upgrade/vchord--$PREV--$SEMVER.sql
```

### Step 3 - validate

```shell
PREV='0.2.0'
SEMVER='0.2.1'

sudo -u postgres createdb vchord

sudo -u postgres ./tools/dump.sh $PREV $SEMVER > target/upgrade.sql
sudo -u postgres ./tools/dump.sh $SEMVER > target/install.sql
code --diff target/upgrade.sql target/install.sql

sudo -u postgres dropdb vchord
```

### Step 4 - package and download

```shell
SEMVER='0.2.1'
VERSION='17'
ARCH='x86_64'
PLATFORM='amd64'

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

wget https://github.com/tensorchord/VectorChord/releases/download/${PREV}/postgresql-${VERSION}-vchord_${PREV}-1_${PLATFORM}.deb -o ./build/postgresql-${VERSION}-vchord_${PREV}-1_${PLATFORM}.deb
```

### Step 5 - further test

```shell
PREV='0.2.0'
SEMVER='0.2.1'
VERSION='17'
ARCH='x86_64'
PLATFORM='amd64'

cargo install sqllogictest-bin

# upgrade test

sudo apt install -y ./build/postgresql-${VERSION}-vchord_${PREV}-1_${PLATFORM}.deb
sudo -u postgres psql -d vchord -c "ALTER SYSTEM SET SHARED_PRELOAD_LIBRARIES='vchord.so';"
sudo systemctl restart postgresql@$VERSION-main.service
sudo -u postgres psql -d vchord -c 'CREATE EXTENSION vchord CASCADE;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"
sudo -u postgres sqllogictest -d vchord '../tests/**/*.slt'

sudo apt install -y ./build/postgresql-${VERSION}-vchord_${SEMVER}-1_${PLATFORM}.deb
sudo systemctl restart postgresql@$VERSION-main.service
sudo -u postgres psql -d vchord -c 'ALTER EXTENSION vchord UPDATE;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"
sudo -u postgres sqllogictest -d vchord '../tests/**/*.slt'

sudo apt remove -y postgresql-${VERSION}-vchord

# install test

sudo apt install -y ./build/postgresql-${VERSION}-vchord_${SEMVER}-1_${PLATFORM}.deb
sudo -u postgres psql -d vchord -c "ALTER SYSTEM SET SHARED_PRELOAD_LIBRARIES='vchord.so';"
sudo systemctl restart postgresql@$VERSION-main.service
sudo -u postgres psql -d vchord -c 'CREATE EXTENSION vchord CASCADE;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"
sudo -u postgres sqllogictest -d vchord '../tests/**/*.slt'

sudo apt remove -y postgresql-${VERSION}-vchord
```
