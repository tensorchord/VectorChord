## Release a new version

### Pre-requisite

```shell
sudo apt install -y build-essential libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config clang postgresql-server-dev-16
cargo install --locked cargo-pgrx
cargo pgrx init
```

### Step 1 - generate install schema
```shell
# Change to version to release
SEMVER=0.2.0
PREV_VERSION=0.1.0

mkdir temp

cargo build --package vchord --lib --features pg16 --release
./tools/schema.sh --features pg16 --release

cp ./target/release/schema.sql ./sql/install/vchord--$SEMVER.sql
```

### Step 2 - generate upgrade schema
```shell
PREV_VERSION_FILE=sql/install/vchord--$PREV_VERSION.sql

# New lines redirect to install.sql, revised lines & deleted lines redirect to terminal
diff -u $PREV_VERSION_FILE install.sql | awk '
  /^\+/ && !/^+++/ { 
    print substr($0, 2) > "temp/upgrade.sql"
    next
  }
  /^-/ && !/^---/ || /^@/ { print }
  { next }
'
cp temp/upgrade.sql ./sql/upgrade/vchord--$PREV_VERSION--$SEMVER.sql
```

### Step 3 - validate
```shell

# sudo pg_dropcluster --stop 16 main 
sudo pg_createcluster 16 main --start
sudo -u postgres createdb vchord
sudo -u postgres psql -d vchord -c "ALTER USER postgres WITH PASSWORD '123';"

# Dump upgraded schema and compare it
export PGHOST=localhost
export PGPASSWORD=123
export PGUSER=postgres
./tools/dump.sh $PREV_VERSION $SEMVER > temp/dump1.sql
./tools/dump.sh $SEMVER > temp/dump2.sql
code --diff temp/dump1.sql temp/dump2.sql
```

### Step 4 - further test

```shell
sudo apt remove -y vchord-pg16
SEMVER=$SEMVER VERSION="16" ARCH="x86_64" PLATFORM="amd64" ./tools/package.sh
cd temp
wget https://github.com/tensorchord/VectorChord/releases/download/"$PREV_VERSION"/vchord-pg16_"$PREV_VERSION"_amd64.deb
sudo apt install -y ./vchord-pg16_"$PREV_VERSION"_amd64.deb
sudo -u postgres psql -d vchord -c "ALTER SYSTEM SET SHARED_PRELOAD_LIBRARIES='vchord.so';"
sudo systemctl restart postgresql@16-main.service
sudo -u postgres psql -d vchord -c 'CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS vchord;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"

# Run Test -- Upgrade
sudo apt install -f ../build/vchord-pg16_"$SEMVER"_amd64.deb
sudo systemctl restart postgresql@16-main.service
sudo -u postgres psql -d vchord -c 'ALTER EXTENSION vchord UPDATE;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"
sqllogictest -h localhost -u postgres -d vchord -w 123 '../tests/**/*.slt'

# Run Test -- Install
sudo -u postgres psql -d vchord -c 'DROP EXTENSION vchord CASCADE;'
sudo -u postgres psql -d vchord -c 'CREATE EXTENSION vchord;'
sudo -u postgres psql -d vchord -c "SELECT extversion FROM pg_extension WHERE extname = 'vchord';"
sqllogictest -h localhost -u postgres -d vchord -w 123 '../tests/**/*.slt'

sudo -u postgres psql -d vchord -c "DROP SCHEMA IF EXISTS public CASCADE;"
sudo apt remove -y vchord-pg16
cd .. && rm -rf temp
```