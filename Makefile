PG_CONFIG ?= $(shell which pg_config)

all: package

# Install vchord into the PostgreSQL cluster identified by pg_config.
.PHONY: install
install:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo make install -i ./build/raw

# Build vchord for the PostgreSQL cluster identified by pg_config.
.DEFAULT_GOAL: package
package:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo make package -o ./build/raw

