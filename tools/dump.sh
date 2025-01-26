#!/usr/bin/env bash
set -e

mkdir -p $(dirname "$0")/../target/tools

if [ $# -eq 0 ] ; then
    echo "Usage: $0 INITIAL_VERSION VERSIONS.."
    echo "Dump the extension members. Install INITIAL_VERSION first and upgrade to every version in VERSIONS."
    echo "The extension members are installed in database \"vchord\", so use \"createdb vchord\" to setup."
    echo "Examples:"
    echo "  ./tools/dump.sh 0.1.11 0.2.0 > ./dump_upgrade.sql"
    echo "  ./tools/dump.sh 0.2.0 > ./dump_install.sql"
    echo "  diff ./dump_upgrade.sql ./dump_install.sql"
    exit 0
fi

f=()
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "" ]; then
        x=$(realpath "$(dirname "$0")/../sql/install/vchord--${arg}.sql")
    else
        x=$(realpath "$(dirname "$0")/../sql/upgrade/vchord--${prev_arg}--${arg}.sql")
    fi
    prev_arg=$arg
    f+=("$x")
done

so=$(realpath $(dirname "$0")/../target/tools/vchord.so)
$(dirname "$0")/dump-codegen.sh | gcc -I $(pg_config --includedir-server) -fPIC -shared -o $so -x c -

sql=$(mktemp)
echo "BEGIN;" >> $sql
echo "CREATE SCHEMA public;" >> $sql
echo "CREATE EXTENSION vector;" >> $sql
cat ${f[@]} \
    | grep -v '^\\' \
    | sed "s|MODULE_PATHNAME|$so|g" \
    >> $sql
echo "END;" >> $sql

psql -d vchord -f $sql 1>&2
pg_dump -d vchord
psql -d vchord -c "DROP SCHEMA IF EXISTS public CASCADE;" 1>&2