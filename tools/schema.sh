#!/usr/bin/env bash
set -e
if [[ " $@ " =~ --target' '([^ ]+) ]]; then
  TARGET="${BASH_REMATCH[1]}"
  if [[ " $@ " =~ " --release " ]]; then
    DIR="./target/$TARGET/release"
  elif [[ " $@ " =~ " --profile opt " ]]; then
    DIR="./target/$TARGET/opt"
  elif [[ " $@ " =~ " --profile release " ]]; then
    DIR="./target/$TARGET/release"
  else
    DIR="./target/$TARGET/debug"
  fi
else
  if [[ " $@ " =~ " --release " ]]; then
    DIR="./target/release"
  elif [[ " $@ " =~ " --profile opt " ]]; then
    DIR="./target/opt"
  elif [[ " $@ " =~ " --profile release " ]]; then
    DIR="./target/release"
  else
    DIR="./target/debug"
  fi
fi

if [ "$TARGET" = "" ]; then
  printf "Target: [not specified]\n" 1>&2
  RUNNER=()
elif [ "$TARGET" = $(rustc -vV | awk '/^host/ { print $2 }') ]; then
  printf "Target: [host]\n" 1>&2
  RUNNER=()
elif [ "$TARGET" = "aarch64-unknown-linux-gnu" ]; then
  printf "Target: $TARGET\n" 1>&2
  QEMU_LD_PREFIX="/usr/aarch64-linux-gnu"
  RUNNER=("qemu-aarch64-static")
elif [ "$TARGET" = "riscv64gc-unknown-linux-gnu" ]; then
  printf "Target: $TARGET\n" 1>&2
  QEMU_LD_PREFIX="/usr/riscv64-linux-gnu"
  RUNNER=("qemu-riscv64-static")
else
  printf "Unknown target: $TARGET\n" 1>&2
  exit 1
fi

code=$(mktemp)
chmod 700 $code
CONTROL_FILEPATH="./vchord.control" SO_FILEPATH="$DIR/libvchord.so" $(dirname "$0")/schema-codegen.sh >> $code

PGRX_EMBED=$code cargo rustc --package vchord --bin pgrx_embed_vchord "$@" -- --cfg pgrx_embed

CARGO_PKG_VERSION=${SEMVER} QEMU_LD_PREFIX=$QEMU_LD_PREFIX "${RUNNER[@]}" "$DIR/pgrx_embed_vchord" | expand -t 4 > $DIR/schema.sql
