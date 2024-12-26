# CNPG only support Debian 12 (Bookworm)
FROM ubuntu:22.04

ARG PGRX_VERSION
ARG RUST_TOOLCHAIN
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    RUSTFLAGS="-Dwarnings" \
    RUST_BACKTRACE=1 \
    CARGO_TERM_COLOR=always \
    SCCACHE_VERSION=0.9.0

RUN set -eux; \
    apt update; \
    apt install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        postgresql-common gnupg \
        libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config

RUN set -eux; \
    apt -y install lsb-release wget software-properties-common gnupg; \
    curl --proto '=https' --tlsv1.2 -sSf https://apt.llvm.org/llvm.sh | bash -s -- 18; \
    update-alternatives --install /usr/bin/clang clang $(which clang-18) 255

# set up sccache
RUN set -ex; \
    curl -fsSL -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl.tar.gz; \
    tar -xzf sccache.tar.gz --strip-components=1; \
    rm sccache.tar.gz; \
    mv sccache /usr/local/bin/

RUN /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
# install all the PostgresQL
RUN set -ex; \
    for v in $(seq 14 17); do \
        apt install -y --no-install-recommends postgresql-$v postgresql-server-dev-$v postgresql-$v-pgvector; \
    done; \
    rm -rf /var/lib/apt/lists/*;

# create a non-root user (make it compatible with Ubuntu 24.04)
RUN useradd -u 1000 -U -m ubuntu
RUN chown -R ubuntu:ubuntu /usr/share/postgresql/ /usr/lib/postgresql/
USER ubuntu
ENV PATH="$PATH:/home/ubuntu/.cargo/bin"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /workspace
RUN rustup toolchain install ${RUST_TOOLCHAIN}
RUN rustup target add $(uname -m)-unknown-linux-gnu

RUN cargo install cargo-pgrx --locked --version=${PGRX_VERSION}

RUN set -ex; \
    for v in $(seq 14 17); do \
        cargo pgrx init --pg$v=/usr/lib/postgresql/$v/bin/pg_config; \
    done;

CMD [ "/bin/bash" ]
