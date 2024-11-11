ARG PG_VERSION=17
ARG PGVECTOR=0.8.0

FROM pgvector/pgvector:${PGVECTOR}-pg${PG_VERSION}

ARG PG_VERSION
RUN echo ${PG_VERSION}
COPY ./build/rabbithole-pg${PG_VERSION}_0.0.0_amd64.deb /tmp/rabbithole.deb
RUN apt-get install -y /tmp/rabbithole.deb && rm -f /tmp/rabbithole.deb

CMD ["postgres", "-c" ,"shared_preload_libraries=rabbithole.so"]
