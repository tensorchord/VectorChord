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
    pig repo add all --ru

# Install pg_stat_statements
RUN trunk install pg_stat_statements

# Install auto_explain
RUN trunk install auto_explain

# Install plpython3u
RUN trunk install plpython3u

# Install pgvector
RUN trunk install pgvector --version ${PGVECTOR}

# Install pgmq
RUN trunk install pgmq --version 1.4.5

# Install pg_later
RUN trunk install pg_later --version 0.3.0

# Clone and build pgaudit
RUN git clone https://github.com/pgaudit/pgaudit.git && \
    cd pgaudit && \
    git checkout REL_${PG_MAJOR}_STABLE && \
    make install USE_PGXS=1 PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config && \
    cd ../ && rm -rf pgaudit

# Clone and build pg_failover_slots
RUN git clone https://github.com/EnterpriseDB/pg_failover_slots.git && \
    cd pg_failover_slots && \
    make install PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config && \
    cd ../ && rm -rf pg_failover_slots 

# cache all extensions
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Test trunk
COPY trunk-install.sh /usr/local/bin/

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
