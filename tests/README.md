# SQL logic test

Requires `sqllogictest` to be installed:

```bash
cargo install sqllogictest-bin
```

To run all the tests:

```bash
PGPASSWORD=postgres psql -h localhost -U postgres -d postgres -c 'CREATE EXTENSION IF NOT EXISTS rabbithole CASCADE;'
sqllogictest './tests/**/*.slt'
```
