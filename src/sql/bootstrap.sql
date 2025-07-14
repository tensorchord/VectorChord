-- List of shell types

CREATE TYPE scalar8;
CREATE TYPE sphere_vector;
CREATE TYPE sphere_halfvec;
CREATE TYPE sphere_scalar8;

CREATE TYPE logged_query AS (
    table_schema TEXT,
    table_name TEXT,
    column_name TEXT,
    operator TEXT,
    vector_text TEXT,
    simplified_query TEXT
);