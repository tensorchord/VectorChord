-- List of data types

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

-- List of operators

CREATE OPERATOR <<->> (
    PROCEDURE = _rabbithole_pgvector_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<->>
);

-- List of functions

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

-- List of access methods

CREATE ACCESS METHOD rabbithole TYPE INDEX HANDLER _rabbithole_amhandler;
COMMENT ON ACCESS METHOD rabbithole IS 'rabbithole index access method';

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING rabbithole;
CREATE OPERATOR FAMILY int4_eq_ops USING rabbithole;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING rabbithole FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH;

CREATE OPERATOR CLASS int4_eq_ops
    FOR TYPE int4 USING rabbithole FAMILY int4_eq_ops AS
    OPERATOR 11 = (int4, int4) FOR SEARCH;
