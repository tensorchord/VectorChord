/* <begin connected objects> */
/*
This file is auto generated by pgrx.

The ordering of items is not stable, it is driven by a dependency graph.
*/
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:18
-- bootstrap
-- List of shell types

CREATE TYPE scalar8;
CREATE TYPE sphere_vector;
CREATE TYPE sphere_halfvec;
CREATE TYPE sphere_scalar8;
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/functions_scalar8.rs:18
-- vchord::datatype::functions_scalar8::_vchord_halfvec_quantize_to_scalar8
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_halfvec.rs:54
-- vchord::datatype::operators_pgvector_halfvec::_vchord_halfvec_sphere_cosine_in
CREATE  FUNCTION "_vchord_halfvec_sphere_cosine_in"(
    "lhs" halfvec, /* vchord::datatype::memory_pgvector_halfvec::PgvectorHalfvecInput */
    "rhs" sphere_halfvec /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_halfvec_sphere_cosine_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_halfvec.rs:30
-- vchord::datatype::operators_pgvector_halfvec::_vchord_halfvec_sphere_ip_in
CREATE  FUNCTION "_vchord_halfvec_sphere_ip_in"(
    "lhs" halfvec, /* vchord::datatype::memory_pgvector_halfvec::PgvectorHalfvecInput */
    "rhs" sphere_halfvec /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_halfvec_sphere_ip_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_halfvec.rs:6
-- vchord::datatype::operators_pgvector_halfvec::_vchord_halfvec_sphere_l2_in
CREATE  FUNCTION "_vchord_halfvec_sphere_l2_in"(
    "lhs" halfvec, /* vchord::datatype::memory_pgvector_halfvec::PgvectorHalfvecInput */
    "rhs" sphere_halfvec /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_halfvec_sphere_l2_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/text_scalar8.rs:7
-- vchord::datatype::text_scalar8::_vchord_scalar8_in
CREATE  FUNCTION "_vchord_scalar8_in"(
    "input" cstring, /* &core::ffi::c_str::CStr */
    "oid" oid, /* pgrx_pg_sys::submodules::oids::Oid */
    "typmod" INT /* i32 */
) RETURNS scalar8 /* vchord::datatype::memory_scalar8::Scalar8Output */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:26
-- vchord::datatype::operators_scalar8::_vchord_scalar8_operator_cosine
CREATE  FUNCTION "_vchord_scalar8_operator_cosine"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" scalar8 /* vchord::datatype::memory_scalar8::Scalar8Input */
) RETURNS real /* f32 */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_operator_cosine_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:6
-- vchord::datatype::operators_scalar8::_vchord_scalar8_operator_ip
CREATE  FUNCTION "_vchord_scalar8_operator_ip"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" scalar8 /* vchord::datatype::memory_scalar8::Scalar8Input */
) RETURNS real /* f32 */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_operator_ip_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:16
-- vchord::datatype::operators_scalar8::_vchord_scalar8_operator_l2
CREATE  FUNCTION "_vchord_scalar8_operator_l2"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" scalar8 /* vchord::datatype::memory_scalar8::Scalar8Input */
) RETURNS real /* f32 */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_operator_l2_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/text_scalar8.rs:119
-- vchord::datatype::text_scalar8::_vchord_scalar8_out
CREATE  FUNCTION "_vchord_scalar8_out"(
    "vector" scalar8 /* vchord::datatype::memory_scalar8::Scalar8Input */
) RETURNS cstring /* alloc::ffi::c_str::CString */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_out_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/binary_scalar8.rs:22
-- vchord::datatype::binary_scalar8::_vchord_scalar8_recv
CREATE  FUNCTION "_vchord_scalar8_recv"(
    "internal" internal, /* pgrx::datum::internal::Internal */
    "oid" oid, /* pgrx_pg_sys::submodules::oids::Oid */
    "typmod" INT /* i32 */
) RETURNS scalar8 /* vchord::datatype::memory_scalar8::Scalar8Output */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_recv_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/binary_scalar8.rs:7
-- vchord::datatype::binary_scalar8::_vchord_scalar8_send
CREATE  FUNCTION "_vchord_scalar8_send"(
    "vector" scalar8 /* vchord::datatype::memory_scalar8::Scalar8Input */
) RETURNS bytea /* alloc::vec::Vec<u8> */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_send_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:84
-- vchord::datatype::operators_scalar8::_vchord_scalar8_sphere_cosine_in
CREATE  FUNCTION "_vchord_scalar8_sphere_cosine_in"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" sphere_scalar8 /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_sphere_cosine_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:36
-- vchord::datatype::operators_scalar8::_vchord_scalar8_sphere_ip_in
CREATE  FUNCTION "_vchord_scalar8_sphere_ip_in"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" sphere_scalar8 /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_sphere_ip_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_scalar8.rs:60
-- vchord::datatype::operators_scalar8::_vchord_scalar8_sphere_l2_in
CREATE  FUNCTION "_vchord_scalar8_sphere_l2_in"(
    "lhs" scalar8, /* vchord::datatype::memory_scalar8::Scalar8Input */
    "rhs" sphere_scalar8 /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_scalar8_sphere_l2_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/typmod.rs:45
-- vchord::datatype::typmod::_vchord_typmod_in_65535
CREATE  FUNCTION "_vchord_typmod_in_65535"(
    "list" cstring[] /* pgrx::datum::array::Array<&core::ffi::c_str::CStr> */
) RETURNS INT /* i32 */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_typmod_in_65535_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/typmod.rs:63
-- vchord::datatype::typmod::_vchord_typmod_out
CREATE  FUNCTION "_vchord_typmod_out"(
    "typmod" INT /* i32 */
) RETURNS cstring /* alloc::ffi::c_str::CString */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_typmod_out_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/functions_scalar8.rs:8
-- vchord::datatype::functions_scalar8::_vchord_vector_quantize_to_scalar8
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_vector.rs:54
-- vchord::datatype::operators_pgvector_vector::_vchord_vector_sphere_cosine_in
CREATE  FUNCTION "_vchord_vector_sphere_cosine_in"(
    "lhs" vector, /* vchord::datatype::memory_pgvector_vector::PgvectorVectorInput */
    "rhs" sphere_vector /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_vector_sphere_cosine_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_vector.rs:30
-- vchord::datatype::operators_pgvector_vector::_vchord_vector_sphere_ip_in
CREATE  FUNCTION "_vchord_vector_sphere_ip_in"(
    "lhs" vector, /* vchord::datatype::memory_pgvector_vector::PgvectorVectorInput */
    "rhs" sphere_vector /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_vector_sphere_ip_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/datatype/operators_pgvector_vector.rs:6
-- vchord::datatype::operators_pgvector_vector::_vchord_vector_sphere_l2_in
CREATE  FUNCTION "_vchord_vector_sphere_l2_in"(
    "lhs" vector, /* vchord::datatype::memory_pgvector_vector::PgvectorVectorInput */
    "rhs" sphere_vector /* pgrx::heap_tuple::PgHeapTuple<pgrx::pgbox::AllocatedByRust> */
) RETURNS bool /* bool */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchord_vector_sphere_l2_in_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/am.rs:33
-- vchord::index::am::_vchordrq_amhandler
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/functions.rs:12
-- vchord::index::functions::_vchordrq_prewarm
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:26
-- vchord::index::opclass::_vchordrq_support_halfvec_cosine_ops
CREATE  FUNCTION "_vchordrq_support_halfvec_cosine_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_halfvec_cosine_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:21
-- vchord::index::opclass::_vchordrq_support_halfvec_ip_ops
CREATE  FUNCTION "_vchordrq_support_halfvec_ip_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_halfvec_ip_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:16
-- vchord::index::opclass::_vchordrq_support_halfvec_l2_ops
CREATE  FUNCTION "_vchordrq_support_halfvec_l2_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_halfvec_l2_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:11
-- vchord::index::opclass::_vchordrq_support_vector_cosine_ops
CREATE  FUNCTION "_vchordrq_support_vector_cosine_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_vector_cosine_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:6
-- vchord::index::opclass::_vchordrq_support_vector_ip_ops
CREATE  FUNCTION "_vchordrq_support_vector_ip_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_vector_ip_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/index/opclass.rs:1
-- vchord::index::opclass::_vchordrq_support_vector_l2_ops
CREATE  FUNCTION "_vchordrq_support_vector_l2_ops"() RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', '_vchordrq_support_vector_l2_ops_wrapper';
/* </end connected objects> */

/* <begin connected objects> */
-- src/lib.rs:19
-- finalize
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

-- List of access methods

CREATE ACCESS METHOD vchordrq TYPE INDEX HANDLER vchordrq_amhandler;

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordrq;

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
/* </end connected objects> */

