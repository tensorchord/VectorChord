PG_CONFIG ?= pg_config

.PHONY: all build install uninstall

all: build

build:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo run -p make -- build -o ./build/raw

install:
	cp -r ./build/raw/pkglibdir/. $(shell $(PG_CONFIG) --pkglibdir)
	cp -r ./build/raw/sharedir/. $(shell $(PG_CONFIG) --sharedir)

uninstall:
	rm -f $(shell find $(shell $(PG_CONFIG) --pkglibdir) -type f -name 'vchord.*')
	rm -f $(shell find $(shell $(PG_CONFIG) --sharedir)/extension -type f -name 'vchord.*')
	rm -f $(shell find $(shell $(PG_CONFIG) --sharedir)/extension -type f -name 'vchord--*.sql')
