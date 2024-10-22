FROM pgvector/pgvector:0.7.4-pg16

COPY ./build/rabbithole-pg16_0.0.0_amd64.deb /tmp/rabbithole.deb
RUN apt-get install -y /tmp/rabbithole.deb && rm -f /tmp/rabbithole.deb

CMD ["postgres", "-c" ,"shared_preload_libraries=vector,rabbithole.so", "-c", "search_path=\"$user\", public, rabbithole", "-c", "logging_collector=on"]