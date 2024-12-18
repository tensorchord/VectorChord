-- List of data types

CREATE TYPE scalar8 (
    INPUT = _vchord_scalar8_in,
    OUTPUT = _vchord_scalar8_out,
    RECEIVE = _vchord_scalar8_recv,
    SEND = _vchord_scalar8_send,
    TYPMOD_IN = _vchord_typmod_in_65535,
    TYPMOD_OUT = _vchord_typmod_out,
    STORAGE = EXTERNAL,
    INTERNALLENGTH = VARIABLE,
    ALIGNMENT = double
);

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

CREATE TYPE sphere_halfvec AS (
    center halfvec,
    radius REAL
);

CREATE TYPE sphere_scalar8 AS (
    center scalar8,
    radius REAL
);

-- List of operators

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_scalar8_operator_l2,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_scalar8_operator_ip,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_scalar8_operator_cosine,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <=>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_halfvec_sphere_l2_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_scalar8_sphere_l2_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_vector_sphere_ip_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_halfvec_sphere_ip_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_scalar8_sphere_ip_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_vector_sphere_cosine_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<=>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_halfvec_sphere_cosine_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<=>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_scalar8_sphere_cosine_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<=>>
);

-- List of functions

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION sphere(scalar8, real) RETURNS sphere_scalar8
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION quantize_to_scalar8(vector) RETURNS scalar8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_scalar8_wrapper';

CREATE FUNCTION quantize_to_scalar8(halfvec) RETURNS scalar8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_scalar8_wrapper';

CREATE FUNCTION vchordrq_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_amhandler_wrapper';

CREATE FUNCTION vchordrq_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_prewarm_wrapper';

CREATE FUNCTION vchordrqfscan_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrqfscan_amhandler_wrapper';

CREATE FUNCTION vchordrqfscan_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrqfscan_prewarm_wrapper';

-- List of access methods

CREATE ACCESS METHOD vchordrq TYPE INDEX HANDLER vchordrq_amhandler;

CREATE ACCESS METHOD Vchordrqfscan TYPE INDEX HANDLER Vchordrqfscan_amhandler;

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY scalar8_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY scalar8_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY scalar8_cosine_ops USING vchordrq;

CREATE OPERATOR FAMILY vector_l2_ops USING Vchordrqfscan;
CREATE OPERATOR FAMILY vector_ip_ops USING Vchordrqfscan;
CREATE OPERATOR FAMILY vector_cosine_ops USING Vchordrqfscan;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordrq FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordrq FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordrq FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_cosine_ops();

CREATE OPERATOR CLASS scalar8_l2_ops
    FOR TYPE scalar8 USING vchordrq FAMILY scalar8_l2_ops AS
    OPERATOR 1 <-> (scalar8, scalar8) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (scalar8, sphere_scalar8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_scalar8_l2_ops();

CREATE OPERATOR CLASS scalar8_ip_ops
    FOR TYPE scalar8 USING vchordrq FAMILY scalar8_ip_ops AS
    OPERATOR 1 <#> (scalar8, scalar8) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (scalar8, sphere_scalar8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_scalar8_ip_ops();

CREATE OPERATOR CLASS scalar8_cosine_ops
    FOR TYPE scalar8 USING vchordrq FAMILY scalar8_cosine_ops AS
    OPERATOR 1 <=> (scalar8, scalar8) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (scalar8, sphere_scalar8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_scalar8_cosine_ops();

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING Vchordrqfscan FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _Vchordrqfscan_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING Vchordrqfscan FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _Vchordrqfscan_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING Vchordrqfscan FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _Vchordrqfscan_support_vector_cosine_ops();
