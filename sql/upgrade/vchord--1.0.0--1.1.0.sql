-- List of shell types

CREATE TYPE rabitq8;
CREATE TYPE rabitq4;
CREATE TYPE sphere_rabitq8;
CREATE TYPE sphere_rabitq4;

-- List of internal changes

DROP OPERATOR <<=>>(sphere_scalar8, scalar8);
DROP OPERATOR <<#>>(sphere_scalar8, scalar8);
DROP OPERATOR <<->>(sphere_scalar8, scalar8);
DROP FUNCTION quantize_to_scalar8(halfvec);
DROP FUNCTION quantize_to_scalar8(vector);
DROP FUNCTION sphere(scalar8, real);
DROP OPERATOR <<=>>(scalar8, sphere_scalar8);
DROP OPERATOR <<#>>(scalar8, sphere_scalar8);
DROP OPERATOR <<->>(scalar8, sphere_scalar8);
DROP OPERATOR <=>(scalar8, scalar8);
DROP OPERATOR <#>(scalar8, scalar8);
DROP OPERATOR <->(scalar8, scalar8);
DROP FUNCTION _vchord_scalar8_operator_cosine;
DROP FUNCTION _vchord_scalar8_operator_ip;
DROP FUNCTION _vchord_scalar8_operator_l2;
DROP FUNCTION _vchord_scalar8_sphere_cosine_in;
DROP FUNCTION _vchord_scalar8_sphere_ip_in;
DROP FUNCTION _vchord_scalar8_sphere_l2_in;
DROP TYPE sphere_scalar8;
ALTER TYPE scalar8 SET (TYPMOD_IN = NONE);
DROP FUNCTION _vchord_typmod_in_65535;
ALTER TYPE scalar8 SET (TYPMOD_OUT = NONE);
DROP FUNCTION _vchord_typmod_out;
ALTER TYPE scalar8 SET (RECEIVE = NONE);
DROP FUNCTION _vchord_scalar8_recv;
ALTER TYPE scalar8 SET (SEND = NONE);
DROP FUNCTION _vchord_scalar8_send;
DROP TYPE scalar8 CASCADE;
-- DROP FUNCTION _vchord_scalar8_in;
-- DROP FUNCTION _vchord_scalar8_out;

CREATE FUNCTION _vchord_rabitq4_in(input cstring, oid oid, typmod INT) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_in_wrapper';

CREATE FUNCTION _vchord_rabitq4_operator_cosine(lhs rabitq4, rhs rabitq4) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_operator_cosine_wrapper';

CREATE FUNCTION _vchord_rabitq4_operator_ip(lhs rabitq4, rhs rabitq4) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_operator_ip_wrapper';

CREATE FUNCTION _vchord_rabitq4_operator_l2(lhs rabitq4, rhs rabitq4) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_operator_l2_wrapper';

CREATE FUNCTION _vchord_rabitq4_out(vector rabitq4) RETURNS cstring
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_out_wrapper';

CREATE FUNCTION _vchord_rabitq4_recv(internal internal, oid oid, typmod INT) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_recv_wrapper';

CREATE FUNCTION _vchord_rabitq4_send(vector rabitq4) RETURNS bytea
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_send_wrapper';

CREATE FUNCTION _vchord_rabitq4_sphere_cosine_in(lhs rabitq4, rhs sphere_rabitq4) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_sphere_cosine_in_wrapper';

CREATE FUNCTION _vchord_rabitq4_sphere_ip_in(lhs rabitq4, rhs sphere_rabitq4) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_sphere_ip_in_wrapper';

CREATE FUNCTION _vchord_rabitq4_sphere_l2_in(lhs rabitq4, rhs sphere_rabitq4) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_sphere_l2_in_wrapper';

CREATE FUNCTION _vchord_rabitq4_typmod_in(list cstring[]) RETURNS INT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_typmod_in_wrapper';

CREATE FUNCTION _vchord_rabitq8_in(input cstring, oid oid, typmod INT) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_in_wrapper';

CREATE FUNCTION _vchord_rabitq8_operator_cosine(lhs rabitq8, rhs rabitq8) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_operator_cosine_wrapper';

CREATE FUNCTION _vchord_rabitq8_operator_ip(lhs rabitq8, rhs rabitq8) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_operator_ip_wrapper';

CREATE FUNCTION _vchord_rabitq8_operator_l2(lhs rabitq8, rhs rabitq8) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_operator_l2_wrapper';

CREATE FUNCTION _vchord_rabitq8_out(vector rabitq8) RETURNS cstring
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_out_wrapper';

CREATE FUNCTION _vchord_rabitq8_recv(internal internal, oid oid, typmod INT) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_recv_wrapper';

CREATE FUNCTION _vchord_rabitq8_send(vector rabitq8) RETURNS bytea
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_send_wrapper';

CREATE FUNCTION _vchord_rabitq8_sphere_cosine_in(lhs rabitq8, rhs sphere_rabitq8) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_sphere_cosine_in_wrapper';

CREATE FUNCTION _vchord_rabitq8_sphere_ip_in(lhs rabitq8, rhs sphere_rabitq8) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_sphere_ip_in_wrapper';

CREATE FUNCTION _vchord_rabitq8_sphere_l2_in(lhs rabitq8, rhs sphere_rabitq8) RETURNS bool
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_sphere_l2_in_wrapper';

CREATE FUNCTION _vchord_rabitq8_typmod_in(list cstring[]) RETURNS INT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_typmod_in_wrapper';

CREATE FUNCTION _vchordg_support_rabitq4_cosine_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq4_cosine_ops_wrapper';

CREATE FUNCTION _vchordg_support_rabitq4_ip_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq4_ip_ops_wrapper';

CREATE FUNCTION _vchordg_support_rabitq4_l2_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq4_l2_ops_wrapper';

CREATE FUNCTION _vchordg_support_rabitq8_cosine_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq8_cosine_ops_wrapper';

CREATE FUNCTION _vchordg_support_rabitq8_ip_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq8_ip_ops_wrapper';

CREATE FUNCTION _vchordg_support_rabitq8_l2_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_support_rabitq8_l2_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq4_cosine_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq4_cosine_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq4_ip_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq4_ip_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq4_l2_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq4_l2_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq4_maxsim_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq4_maxsim_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq8_cosine_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq8_cosine_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq8_ip_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq8_ip_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq8_l2_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq8_l2_ops_wrapper';

CREATE FUNCTION _vchordrq_support_rabitq8_maxsim_ops() RETURNS TEXT
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_support_rabitq8_maxsim_ops_wrapper';

DROP OPERATOR <<=>>(sphere_halfvec, halfvec);
DROP OPERATOR <<#>>(sphere_halfvec, halfvec);
DROP OPERATOR <<->>(sphere_halfvec, halfvec);
DROP OPERATOR <<=>>(sphere_vector, vector);
DROP OPERATOR <<#>>(sphere_vector, vector);
DROP OPERATOR <<->>(sphere_vector, vector);

-- List of types

CREATE TYPE rabitq8 (
    INPUT = _vchord_rabitq8_in,
    OUTPUT = _vchord_rabitq8_out,
    TYPMOD_IN = _vchord_rabitq8_typmod_in,
    RECEIVE = _vchord_rabitq8_recv,
    SEND = _vchord_rabitq8_send,
    STORAGE = external
);

CREATE TYPE rabitq4 (
    INPUT = _vchord_rabitq4_in,
    OUTPUT = _vchord_rabitq4_out,
    TYPMOD_IN = _vchord_rabitq4_typmod_in,
    RECEIVE = _vchord_rabitq4_recv,
    SEND = _vchord_rabitq4_send,
    STORAGE = external
);

CREATE TYPE sphere_rabitq8 AS (
    center rabitq8,
    radius REAL
);

CREATE TYPE sphere_rabitq4 AS (
    center rabitq4,
    radius REAL
);

-- List of internal functions

CREATE FUNCTION _vchord_rabitq8_operator_maxsim(rabitq8[], rabitq8[]) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_operator_maxsim_wrapper';

CREATE FUNCTION _vchord_rabitq4_operator_maxsim(rabitq4[], rabitq4[]) RETURNS real
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_operator_maxsim_wrapper';

-- List of operators

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_rabitq8_operator_l2,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_rabitq8_operator_ip,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_rabitq8_operator_cosine,
    LEFTARG = rabitq8,
    RIGHTARG = rabitq8,
    COMMUTATOR = <=>
);

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_rabitq4_operator_l2,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_rabitq4_operator_ip,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_rabitq4_operator_cosine,
    LEFTARG = rabitq4,
    RIGHTARG = rabitq4,
    COMMUTATOR = <=>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_rabitq8_sphere_l2_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_rabitq4_sphere_l2_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_rabitq8_sphere_ip_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_rabitq4_sphere_ip_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_rabitq8_sphere_cosine_in,
    LEFTARG = rabitq8,
    RIGHTARG = sphere_rabitq8
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_rabitq4_sphere_cosine_in,
    LEFTARG = rabitq4,
    RIGHTARG = sphere_rabitq4
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_rabitq8_operator_maxsim,
    LEFTARG = rabitq8[],
    RIGHTARG = rabitq8[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_rabitq4_operator_maxsim,
    LEFTARG = rabitq4[],
    RIGHTARG = rabitq4[]
);

-- List of functions

CREATE FUNCTION sphere(rabitq8, real) RETURNS sphere_rabitq8
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION sphere(rabitq4, real) RETURNS sphere_rabitq4
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION quantize_to_rabitq8(vector) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq8_wrapper';

CREATE FUNCTION quantize_to_rabitq8(halfvec) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq8_wrapper';

CREATE FUNCTION quantize_to_rabitq4(vector) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq4_wrapper';

CREATE FUNCTION quantize_to_rabitq4(halfvec) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq4_wrapper';

-- List of operator families

CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordg;

-- List of operator classes

CREATE OPERATOR CLASS rabitq8_l2_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_l2_ops AS
    OPERATOR 1 <-> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_l2_ops();

CREATE OPERATOR CLASS rabitq8_ip_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_ip_ops AS
    OPERATOR 1 <#> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_ip_ops();

CREATE OPERATOR CLASS rabitq8_cosine_ops
    FOR TYPE rabitq8 USING vchordrq FAMILY rabitq8_cosine_ops AS
    OPERATOR 1 <=> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq8_cosine_ops();

CREATE OPERATOR CLASS rabitq4_l2_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_l2_ops AS
    OPERATOR 1 <-> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_l2_ops();

CREATE OPERATOR CLASS rabitq4_ip_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_ip_ops AS
    OPERATOR 1 <#> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_ip_ops();

CREATE OPERATOR CLASS rabitq4_cosine_ops
    FOR TYPE rabitq4 USING vchordrq FAMILY rabitq4_cosine_ops AS
    OPERATOR 1 <=> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_rabitq4_cosine_ops();

CREATE OPERATOR CLASS rabitq8_maxsim_ops
    FOR TYPE rabitq8[] USING vchordrq FAMILY rabitq8_maxsim_ops AS
    OPERATOR 3 @# (rabitq8[], rabitq8[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq8_maxsim_ops();

CREATE OPERATOR CLASS rabitq4_maxsim_ops
    FOR TYPE rabitq4[] USING vchordrq FAMILY rabitq4_maxsim_ops AS
    OPERATOR 3 @# (rabitq4[], rabitq4[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq4_maxsim_ops();

CREATE OPERATOR CLASS rabitq8_l2_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_l2_ops AS
    OPERATOR 1 <-> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_l2_ops();

CREATE OPERATOR CLASS rabitq8_ip_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_ip_ops AS
    OPERATOR 1 <#> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_ip_ops();

CREATE OPERATOR CLASS rabitq8_cosine_ops
    FOR TYPE rabitq8 USING vchordg FAMILY rabitq8_cosine_ops AS
    OPERATOR 1 <=> (rabitq8, rabitq8) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq8, sphere_rabitq8) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq8_cosine_ops();

CREATE OPERATOR CLASS rabitq4_l2_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_l2_ops AS
    OPERATOR 1 <-> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_l2_ops();

CREATE OPERATOR CLASS rabitq4_ip_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_ip_ops AS
    OPERATOR 1 <#> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_ip_ops();

CREATE OPERATOR CLASS rabitq4_cosine_ops
    FOR TYPE rabitq4 USING vchordg FAMILY rabitq4_cosine_ops AS
    OPERATOR 1 <=> (rabitq4, rabitq4) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (rabitq4, sphere_rabitq4) FOR SEARCH,
    FUNCTION 1 _vchordg_support_rabitq4_cosine_ops();
