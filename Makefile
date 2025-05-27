PG_CONFIG ?= $(shell which pg_config)
DISTNAME = vchord
# Extract DISTVERSION from vchord.control
CONTROL_FILE = vchord.control
DISTVERSION := $(shell sed -n "s/^[[:space:]]*default_version[[:space:]]*=[[:space:]]*'\([^']*\)'/\1/p" $(CONTROL_FILE))

all: package

# Install vchord into the PostgreSQL cluster identified by pg_config.
.PHONY: install
install:
    @PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo make install -i ./build/raw

# Build vchord for the PostgreSQL cluster identified by pg_config.
.DEFAULT_GOAL: package
package:
    @PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo make build -o ./build/raw

META.json: META.json.in
	@sed -e "s/@DISTVERSION@/$(DISTVERSION)/g" $< > $@

$(DISTNAME)-$(DISTVERSION).zip: META.json
	mkdir -p dist
	git archive --format zip --prefix $(DISTNAME)-$(DISTVERSION)/ --add-file $< -o dist/$(DISTNAME)-$(DISTVERSION).zip HEAD

# Create a PGXN-compatible zip file.
dist: $(DISTNAME)-$(DISTVERSION).zip
