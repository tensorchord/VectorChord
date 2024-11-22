ARG PG_VERSION=17
ARG PGVECTOR=0.8.0

FROM pgvector/pgvector:${PGVECTOR}-pg${PG_VERSION}

ARG PG_VERSION
ARG SEMVER=0.0.0
ARG TARGETARCH

RUN echo ${PG_VERSION}
COPY ./build/vchord-pg${PG_VERSION}_${SEMVER}_${TARGETARCH}.deb /tmp/vchord.deb
RUN apt-get install -y /tmp/vchord.deb && rm -f /tmp/vchord.deb

CMD ["postgres", "-c" ,"shared_preload_libraries=vchord.so"]
