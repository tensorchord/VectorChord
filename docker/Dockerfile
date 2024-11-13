ARG PG_VERSION=17
ARG PGVECTOR=0.8.0

FROM pgvector/pgvector:${PGVECTOR}-pg${PG_VERSION}

ARG PG_VERSION
RUN echo ${PG_VERSION}
COPY ./build/vchord-pg${PG_VERSION}_0.0.0_amd64.deb /tmp/vchord.deb
RUN apt-get install -y /tmp/vchord.deb && rm -f /tmp/vchord.deb

CMD ["postgres", "-c" ,"shared_preload_libraries=vchord.so"]
