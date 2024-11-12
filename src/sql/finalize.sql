-- List of data types

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

-- List of operators

CREATE OPERATOR <<->> (
    PROCEDURE = _rabbithole_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _rabbithole_vector_sphere_ip_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _rabbithole_vector_sphere_cosine_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<=>>
);

-- List of functions

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION rabbithole_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_rabbithole_amhandler_wrapper';

CREATE FUNCTION rabbithole_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_rabbithole_prewarm_wrapper';

-- List of access methods

CREATE ACCESS METHOD rabbithole TYPE INDEX HANDLER rabbithole_amhandler;
COMMENT ON ACCESS METHOD rabbithole IS 'rabbithole index access method';

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING rabbithole;
CREATE OPERATOR FAMILY vector_ip_ops USING rabbithole;
CREATE OPERATOR FAMILY vector_cosine_ops USING rabbithole;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING rabbithole FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _rabbithole_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING rabbithole FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _rabbithole_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING rabbithole FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _rabbithole_support_vector_cosine_ops();
