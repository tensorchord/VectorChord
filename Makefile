PG_CONFIG ?= $(shell which pg_config)
DISTNAME = vchord
DISTVERSION = 0.4.1

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

all: package

# Install vchord into the PostgreSQL cluster identified by pg_config.
.PHONY: install
install:
	@cargo make install --sudo

# Build vchord for the PostgreSQL cluster identified by pg_config.
.DEFAULT_GOAL: package
package:
	mkdir -p dist
	@cargo make package -o dist

META.json: META.json.in
	@sed -e "s/@DISTVERSION@/$(DISTVERSION)/g" $< > $@

$(DISTNAME)-$(DISTVERSION).zip: META.json
	mkdir -p dist
	git archive --format zip --prefix $(DISTNAME)-$(DISTVERSION)/ -o dist/$(DISTNAME)-$(DISTVERSION).zip HEAD

# Create a PGXN-compatible zip file.
dist: $(DISTNAME)-$(DISTVERSION).zip