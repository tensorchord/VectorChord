ARG PG_MAJOR
ARG SEMVER
ARG TARGETARCH

FROM tensorchord/vchord-binary:pg${PG_MAJOR}-v${SEMVER}-${TARGETARCH} as binary

FROM modelzai/pg-slim:${PG_MAJOR}
ARG PG_MAJOR
ARG SEMVER
ARG TARGETARCH
ARG LIB_DIR
ARG PGVECTOR
ARG ALTDIR=/var/lib/postgresql/data/tensorchord

USER root

COPY --from=binary /workspace/postgresql-${PG_MAJOR}-vchord_${SEMVER}-1_${TARGETARCH}.deb /tmp/vchord.deb
RUN apt-get install -y /tmp/vchord.deb && rm -f /tmp/vchord.deb

# PGDATA is set in pg-slim and used by dependents on this image.
RUN if [ -z "${PGDATA}" ]; then echo "PGDATA is not set"; exit 1; fi

# Install barman-cloud
COPY requirements.txt .
RUN set -xe; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-psycopg2 \
    python3-setuptools \
    ; \
    pip3 install --upgrade pip; \
    # TODO: Remove --no-deps once https://github.com/pypa/pip/issues/9644 is solved
    pip3 install --no-deps -r requirements.txt; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*;

RUN chown -R postgres:postgres ${ALTDIR}/${PG_MAJOR} && \
    chmod -R 0700 ${ALTDIR}/${PG_MAJOR}
RUN chown postgres /usr/share/postgresql/${PG_MAJOR}/extension

RUN apt-get update && apt-get install -y \
    jq \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install pig
RUN curl -fsSL https://repo.pigsty.io/pig | bash && \
    pig repo add pigsty pgdg -u

# Install pg_stat_statements
RUN pig ext install -y pg_stat_statements -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config 

# Install auto_explain
RUN pig ext install -y auto_explain -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config 

# Install plpython3u
RUN pig ext install -y plpython3u -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# Install pgvector
RUN pig ext install -y pgvector=${PGVECTOR} -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# Install pg_later
RUN pig ext install -y pg_later=0.3.0 -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# Install pgaduit
RUN pig ext install -y pgaudit -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# Install pg_failover_slots
RUN pig ext install -y pg_failover_slots -p /usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# cache all extensions
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Test trunk
COPY extension-install.sh /usr/local/bin/

# Change the uid of postgres to 26
RUN usermod -u 26 postgres
RUN chown -R postgres:postgres ${ALTDIR}
RUN cp /usr/share/postgresql/${PG_MAJOR}/extension/* ${ALTDIR}/extension/
RUN cp /usr/lib/postgresql/${PG_MAJOR}/lib/* ${ALTDIR}/${PG_MAJOR}/lib/

RUN set -eux; \
    mkdir /tmp/pg_pkglibdir; \
    mkdir /tmp/pg_sharedir; \
    cp -r $(pg_config --pkglibdir)/* /tmp/pg_pkglibdir; \
    cp -r $(pg_config --sharedir)/* /tmp/pg_sharedir

RUN chown -R postgres:postgres /tmp
USER 26
ENV PATH $PATH:/usr/lib/postgresql/${PG_MAJOR}/bin
