ARG PG_VERSION=17
ARG PGVECTOR=0.8.0
ARG SEMVER=0.0.0
ARG PLATFORM=amd64

FROM pgvector/pgvector:${PGVECTOR}-pg${PG_VERSION}

ARG PG_VERSION
RUN echo ${PG_VERSION}
COPY ./build/rabbithole-pg${PG_VERSION}_${SEMVER}_${PLATFORM}.deb /tmp/rabbithole.deb
RUN apt-get install -y /tmp/rabbithole.deb && rm -f /tmp/rabbithole.deb

CMD ["postgres", "-c" ,"shared_preload_libraries=rabbithole.so", "-c", "search_path=\"$user\", public, rabbithole"]
