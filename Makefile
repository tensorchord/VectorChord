PG_CONFIG ?= $(shell which pg_config)

.PHONY: make package install
.DEFAULT_GOAL: package

all: build-make package

build-make:
	mkdir -p ./build
	cargo build -p make
	cp ./target/debug/make ./build/make

package:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" ./build/make package -o ./build/raw

install:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" ./build/make install -i ./build/raw
