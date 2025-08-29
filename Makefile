PG_CONFIG ?= pg_config
PKGLIBDIR := $(shell $(PG_CONFIG) --pkglibdir)
SHAREDIR := $(shell $(PG_CONFIG) --sharedir)
MKDIR ?= mkdir
CP ?= cp

.PHONY: all build install uninstall

all: build

build:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo run -p make -- build --output ./build/raw

install:
	$(MKDIR) -p $(DESTDIR)$(PKGLIBDIR) $(DESTDIR)$(SHAREDIR) && \
	$(CP) -r ./build/raw/pkglibdir/. $(DESTDIR)$(PKGLIBDIR) && \
	$(CP) -r ./build/raw/sharedir/. $(DESTDIR)$(SHAREDIR)

uninstall:
	$(RM) $(wildcard $(DESTDIR)$(PKGLIBDIR)/vchord.*) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord.*) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord--*.sql)
