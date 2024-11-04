# CNPG only support Debian 12 (Bookworm)
FROM ubuntu:22.04

ARG PGRX_VERSION=0.12.6

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    RUSTFLAGS="-Dwarnings"

RUN apt update && \
    apt install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        crossbuild-essential-arm64 \
        qemu-user-static \
        libreadline-dev zlib1g-dev flex bison libxml2-dev libxslt-dev libssl-dev libxml2-utils xsltproc ccache pkg-config \
        clang && \
    rm -rf /var/lib/apt/lists/*

# create a non-root user (make it compatible with Ubuntu 24.04)
RUN useradd -u 1000 -U -m ubuntu
USER ubuntu
ENV PATH="$PATH:/home/ubuntu/.cargo/bin"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=none -y

WORKDIR /workspace
COPY rust-toolchain.toml /workspace/rust-toolchain.toml
RUN set -ex; \
    echo 'target.aarch64-unknown-linux-gnu.linker = "aarch64-linux-gnu-gcc"' >> ~/.cargo/config.toml; \
    echo 'target.aarch64-unknown-linux-gnu.runner = ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu"]' >> ~/.cargo/config.toml
RUN rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu

RUN cargo install cargo-pgrx --locked --version=${PGRX_VERSION} && \
    cargo pgrx init

ENTRYPOINT [ "cargo" ]
