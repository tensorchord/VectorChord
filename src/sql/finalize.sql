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

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

CREATE TYPE sphere_halfvec AS (
    center halfvec,
    radius REAL
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
    PROCEDURE = _vchord_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_halfvec_sphere_l2_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
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
    PROCEDURE = _vchord_vector_sphere_ip_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_halfvec_sphere_ip_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
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
    PROCEDURE = _vchord_vector_sphere_cosine_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_halfvec_sphere_cosine_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec
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
    PROCEDURE = _vchord_vector_operator_maxsim,
    LEFTARG = vector[],
    RIGHTARG = vector[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_halfvec_operator_maxsim,
    LEFTARG = halfvec[],
    RIGHTARG = halfvec[]
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

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_vector';

CREATE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_halfvec';

CREATE FUNCTION sphere(rabitq8, real) RETURNS sphere_rabitq8
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_rabitq8';

CREATE FUNCTION sphere(rabitq4, real) RETURNS sphere_rabitq4
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)::sphere_rabitq4';

CREATE FUNCTION quantize_to_rabitq8(vector) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq8_wrapper';

CREATE FUNCTION quantize_to_rabitq8(halfvec) RETURNS rabitq8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq8_wrapper';

CREATE FUNCTION dequantize_to_vector(rabitq8) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq8) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_halfvec_wrapper';

CREATE FUNCTION quantize_to_rabitq4(vector) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_rabitq4_wrapper';

CREATE FUNCTION quantize_to_rabitq4(halfvec) RETURNS rabitq4
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_rabitq4_wrapper';

CREATE FUNCTION dequantize_to_vector(rabitq4) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq4) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_halfvec_wrapper';

CREATE FUNCTION vchordrq_sampled_values(regclass) RETURNS SETOF TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_sampled_values_wrapper';

CREATE FUNCTION vchordrq_sampled_queries(regclass)
RETURNS TABLE(
    schema_name NAME,
    index_name NAME,
    table_name NAME,
    column_name NAME,
    operator NAME,
    value TEXT
)
STRICT LANGUAGE plpgsql AS $$
DECLARE
    ext_schema TEXT;
    query_text TEXT;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';

    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    query_text := format(
        $q$
        WITH index_metadata AS (
            SELECT
                NS.nspname AS schema_name,
                I.relname AS index_name,
                C.relname AS table_name,
                PA.attname AS column_name,
                OP.oprname AS operator
            FROM
                pg_catalog.pg_index X
            JOIN
                pg_catalog.pg_class C ON C.oid = X.indrelid
            JOIN
                pg_catalog.pg_namespace NS ON C.relnamespace = NS.oid
            JOIN
                pg_catalog.pg_class I ON I.oid = X.indexrelid
            JOIN
                pg_catalog.pg_am A ON A.oid = I.relam
            LEFT JOIN
                pg_catalog.pg_opclass AS OPC ON OPC.oid = X.indclass[0]
            LEFT JOIN
                pg_catalog.pg_amop AO ON OPC.opcfamily = AO.amopfamily
            LEFT JOIN
                pg_catalog.pg_operator OP ON OP.oid = AO.amopopr
            LEFT JOIN
                pg_catalog.pg_attribute PA ON PA.attrelid = X.indrelid AND PA.attnum = X.indkey[0]
            WHERE
                A.amname = 'vchordrq'
                AND AO.amopstrategy = 1
                AND C.relkind = 'r'
                AND X.indnatts = 1
                AND X.indexrelid = %1$s
        )
        SELECT
            im.schema_name,
            im.index_name,
            im.table_name,
            im.column_name,
            im.operator,
            s.value
        FROM
            index_metadata im,
            LATERAL %2$I.vchordrq_sampled_values(%1$s) AS s(value);
        $q$,
        $1::oid,
        ext_schema
    );
    RETURN QUERY EXECUTE query_text;
END;
$$;

CREATE FUNCTION vchordrq_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_amhandler_wrapper';

CREATE FUNCTION vchordrq_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_prewarm_wrapper';

CREATE FUNCTION vchordrq_evaluate_query_recall(
    query text,
    exact_search boolean default false,
    accu_probes TEXT default NULL,
    accu_epsilon real default 1.9
)
RETURNS real
LANGUAGE plpgsql
AS $$
DECLARE
    rough tid[];
    accu tid[];
    match_count integer := 0;
    accu_k integer;
    recall real;
    rough_probes text;
BEGIN
    IF query IS NULL OR exact_search IS NULL OR accu_epsilon IS NULL THEN
        RETURN NULL;
    END IF;
    IF query LIKE '%@#%' AND NOT exact_search THEN
        RAISE EXCEPTION 'MaxSim operator cannot be used for estimated recall evaluation. Please use exact_search => true.';
    END IF;
    IF NOT exact_search THEN
        BEGIN
            rough_probes := current_setting('vchordrq.probes');
        END;
    END IF;

    BEGIN
        EXECUTE
            format('SELECT coalesce(array_agg(id), array[]::tid[]) FROM (%s) AS result(id)', query)
        INTO
            rough;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Error executing ANN query "%": %', query, SQLERRM;
    END;

    BEGIN
        IF exact_search THEN
            SET LOCAL vchordrq.enable_scan = off;
        ELSE
            IF accu_probes IS NULL THEN
                IF rough_probes = '' THEN
                    accu_probes := '';
                ELSIF position(',' in rough_probes) > 0 THEN
                    accu_probes := '65535,65535';
                ELSE
                    accu_probes := '65535';
                END IF;
            END IF;
            EXECUTE format('SET LOCAL "vchordrq.probes" = %L', accu_probes);
            EXECUTE format('SET LOCAL "vchordrq.epsilon" = %L', accu_epsilon);
            SET LOCAL vchordrq.max_scan_tuples = -1;
        END IF;
        EXECUTE
            format('SELECT coalesce(array_agg(id), array[]::tid[]) FROM (%s) AS result(id)', query)
        INTO
            accu;
    EXCEPTION WHEN OTHERS THEN
         RAISE EXCEPTION 'Error executing Ground Truth query "%": %', query, SQLERRM;
    END;
    accu_k := cardinality(accu);
    IF accu_k = 0 THEN
        RAISE WARNING  'Query "%": No results found, returning NaN for recall.', query;
        RETURN 'NaN';
    END IF;
    SELECT COUNT(*) INTO match_count FROM (SELECT unnest(rough) INTERSECT SELECT unnest(accu)) AS tids;
    recall := match_count::real / accu_k::real;
    RETURN recall;
END;
$$;

CREATE FUNCTION vchordg_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_amhandler_wrapper';

CREATE FUNCTION vchordg_prewarm(regclass) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_prewarm_wrapper';

-- List of access methods

CREATE ACCESS METHOD vchordrq TYPE INDEX HANDLER vchordrq_amhandler;
CREATE ACCESS METHOD vchordg TYPE INDEX HANDLER vchordg_amhandler;

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq8_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY rabitq4_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_l2_ops USING vchordg;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordg;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq8_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_l2_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_ip_ops USING vchordg;
CREATE OPERATOR FAMILY rabitq4_cosine_ops USING vchordg;

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

CREATE OPERATOR CLASS vector_maxsim_ops
    FOR TYPE vector[] USING vchordrq FAMILY vector_maxsim_ops AS
    OPERATOR 3 @# (vector[], vector[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_vector_maxsim_ops();

CREATE OPERATOR CLASS halfvec_maxsim_ops
    FOR TYPE halfvec[] USING vchordrq FAMILY halfvec_maxsim_ops AS
    OPERATOR 3 @# (halfvec[], halfvec[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_halfvec_maxsim_ops();

CREATE OPERATOR CLASS rabitq8_maxsim_ops
    FOR TYPE rabitq8[] USING vchordrq FAMILY rabitq8_maxsim_ops AS
    OPERATOR 3 @# (rabitq8[], rabitq8[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq8_maxsim_ops();

CREATE OPERATOR CLASS rabitq4_maxsim_ops
    FOR TYPE rabitq4[] USING vchordrq FAMILY rabitq4_maxsim_ops AS
    OPERATOR 3 @# (rabitq4[], rabitq4[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_rabitq4_maxsim_ops();

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordg FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordg FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordg FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_cosine_ops();

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

-- List of views

CREATE VIEW vchordrq_sampled_queries AS
SELECT
    record.schema_name,
    record.index_name,
    record.table_name,
    record.column_name,
    record.operator,
    record.value
FROM
    (
        SELECT i.oid
        FROM pg_catalog.pg_class AS i
        JOIN pg_catalog.pg_index AS ix ON i.oid = ix.indexrelid
        JOIN pg_catalog.pg_opclass AS opc ON ix.indclass[0] = opc.oid
        JOIN pg_catalog.pg_am AS am ON opc.opcmethod = am.oid
        WHERE am.amname = 'vchordrq'
    ) AS index_oids
CROSS JOIN LATERAL vchordrq_sampled_queries(index_oids.oid::regclass) AS record;

-- Phase 3B tensor-source bindings

CREATE TABLE _vchordrq_maxsim_sources (
    index_oid oid PRIMARY KEY,
    heap_oid oid NOT NULL,
    model_contract_id text NOT NULL
        CHECK (
            model_contract_id OPERATOR(pg_catalog.<>) ''::text
            AND pg_catalog.length(model_contract_id) OPERATOR(pg_catalog.<=) 512
        ),
    storage text NOT NULL CHECK (
        storage OPERATOR(pg_catalog.=) ANY (
            ARRAY['heap_array', 'external_ref', 'external_relation']::text[]
        )
    ),
    model_contract_attnum smallint NOT NULL CHECK (
        model_contract_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    public_id_attnum smallint NOT NULL CHECK (
        public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
    ),
    descriptor_oid oid,
    descriptor_public_id_attnum smallint,
    tensor_ref_attnum smallint,
    tensor_rows_attnum smallint,
    tensor_dim_attnum smallint,
    tensor_dtype_attnum smallint,
    tensor_checksum_attnum smallint,
    registered_by oid NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        (
            storage OPERATOR(pg_catalog.=) 'heap_array'::text
            AND descriptor_oid IS NULL
            AND descriptor_public_id_attnum IS NULL
            AND tensor_ref_attnum IS NULL
            AND tensor_rows_attnum IS NULL
            AND tensor_dim_attnum IS NULL
            AND tensor_dtype_attnum IS NULL
            AND tensor_checksum_attnum IS NULL
        )
        OR
        (
            storage OPERATOR(pg_catalog.=) 'external_ref'::text
            AND descriptor_oid IS NULL
            AND descriptor_public_id_attnum IS NULL
            AND tensor_ref_attnum IS NOT NULL
            AND tensor_ref_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_rows_attnum IS NOT NULL
            AND tensor_rows_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dim_attnum IS NOT NULL
            AND tensor_dim_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dtype_attnum IS NOT NULL
            AND tensor_dtype_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_checksum_attnum IS NOT NULL
            AND tensor_checksum_attnum OPERATOR(pg_catalog.>) 0::smallint
        )
        OR
        (
            storage OPERATOR(pg_catalog.=) 'external_relation'::text
            AND descriptor_oid IS NOT NULL
            AND descriptor_public_id_attnum IS NOT NULL
            AND descriptor_public_id_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_ref_attnum IS NOT NULL
            AND tensor_ref_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_rows_attnum IS NOT NULL
            AND tensor_rows_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dim_attnum IS NOT NULL
            AND tensor_dim_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_dtype_attnum IS NOT NULL
            AND tensor_dtype_attnum OPERATOR(pg_catalog.>) 0::smallint
            AND tensor_checksum_attnum IS NOT NULL
            AND tensor_checksum_attnum OPERATOR(pg_catalog.>) 0::smallint
        )
    )
);

REVOKE ALL ON TABLE _vchordrq_maxsim_sources FROM PUBLIC;

CREATE FUNCTION vchordrq_register_maxsim_source(
    index_relation regclass,
    model_contract_id text,
    storage text,
    model_contract_column name,
    public_id_column name,
    tensor_ref_column name DEFAULT NULL,
    tensor_rows_column name DEFAULT NULL,
    tensor_dim_column name DEFAULT NULL,
    tensor_dtype_column name DEFAULT NULL,
    tensor_checksum_column name DEFAULT NULL,
    descriptor_relation regclass DEFAULT NULL,
    descriptor_public_id_column name DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    heap_oid oid;
    index_owner oid;
    descriptor_oid oid;
    descriptor_owner oid;
    tensor_relation_oid oid;
    normalized_storage text;
    attnum smallint;
    atttypid oid;
    attnotnull boolean;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    descriptor_id_is_unique boolean;
BEGIN
    IF index_relation IS NULL THEN
        RAISE EXCEPTION 'index_relation must not be NULL';
    END IF;
    IF model_contract_id IS NULL
       OR btrim(model_contract_id) = ''
       OR length(model_contract_id) > 512 THEN
        RAISE EXCEPTION 'model_contract_id must contain between 1 and 512 characters';
    END IF;
    model_contract_id := btrim(model_contract_id);
    IF model_contract_column IS NULL OR public_id_column IS NULL THEN
        RAISE EXCEPTION 'model_contract_column and public_id_column must not be NULL';
    END IF;

    normalized_storage := lower(btrim(storage));
    IF normalized_storage IS NULL
       OR normalized_storage NOT IN ('heap_array', 'external_ref', 'external_relation') THEN
        RAISE EXCEPTION 'storage must be heap_array, external_ref, or external_relation';
    END IF;

    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    IF caller_oid IS NULL THEN
        RAISE EXCEPTION 'could not resolve caller role';
    END IF;

    SELECT x.indrelid, i.relowner
    INTO heap_oid, index_owner
    FROM pg_catalog.pg_index AS x
    JOIN pg_catalog.pg_class AS i ON i.oid = x.indexrelid
    JOIN pg_catalog.pg_class AS h ON h.oid = x.indrelid
    JOIN pg_catalog.pg_am AS am ON am.oid = i.relam
    JOIN pg_catalog.pg_opclass AS opc ON opc.oid = x.indclass[0]
    WHERE x.indexrelid = index_relation::oid
      AND i.relkind = 'i'
      AND h.relkind IN ('r', 'm')
      AND am.amname = 'vchordrq'
      AND opc.opcname IN (
          'vector_maxsim_ops',
          'halfvec_maxsim_ops',
          'rabitq8_maxsim_ops',
          'rabitq4_maxsim_ops'
      )
      AND x.indisvalid
      AND x.indisready
      AND x.indnatts = 1
      AND x.indnkeyatts = 1;
    IF heap_oid IS NULL THEN
        RAISE EXCEPTION 'relation % is not a valid single-key vchordrq MaxSim index',
            index_relation;
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the index owner may register its MaxSim tensor source';
    END IF;

    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = heap_oid
      AND a.attname = model_contract_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'model contract column % must be a NOT NULL text column',
            model_contract_column;
    END IF;
    model_contract_attnum := attnum;

    attnum := NULL;
    SELECT a.attnum, a.atttypid, a.attnotnull
    INTO attnum, atttypid, attnotnull
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = heap_oid
      AND a.attname = public_id_column
      AND a.attnum > 0
      AND NOT a.attisdropped;
    IF attnum IS NULL OR atttypid <> 'bigint'::regtype OR NOT attnotnull THEN
        RAISE EXCEPTION 'public ID column % must be a NOT NULL bigint column',
            public_id_column;
    END IF;
    public_id_attnum := attnum;

    IF normalized_storage = 'external_relation' THEN
        IF descriptor_relation IS NULL OR descriptor_public_id_column IS NULL THEN
            RAISE EXCEPTION 'external_relation sources require descriptor_relation and descriptor_public_id_column';
        END IF;
        SELECT c.oid, c.relowner
        INTO descriptor_oid, descriptor_owner
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = descriptor_relation::oid
          AND c.relkind IN ('r', 'm');
        IF descriptor_oid IS NULL THEN
            RAISE EXCEPTION 'descriptor relation % must be a table or materialized view',
                descriptor_relation;
        END IF;
        IF NOT pg_catalog.pg_has_role(caller_oid, descriptor_owner, 'USAGE') THEN
            RAISE EXCEPTION 'only the descriptor relation owner may register it as a MaxSim tensor source';
        END IF;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = descriptor_oid
          AND a.attname = descriptor_public_id_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'bigint'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'descriptor public ID column % must be a NOT NULL bigint column',
                descriptor_public_id_column;
        END IF;
        descriptor_public_id_attnum := attnum;

        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index AS x
            WHERE x.indrelid = descriptor_oid
              AND x.indisunique
              AND x.indisvalid
              AND x.indisready
              AND x.indnkeyatts = 1
              AND x.indkey[0] = descriptor_public_id_attnum
              AND x.indexprs IS NULL
              AND x.indpred IS NULL
        ) INTO descriptor_id_is_unique;
        IF NOT descriptor_id_is_unique THEN
            RAISE EXCEPTION 'descriptor public ID column % must have a non-partial single-key unique index',
                descriptor_public_id_column;
        END IF;
        tensor_relation_oid := descriptor_oid;
    ELSE
        IF descriptor_relation IS NOT NULL OR descriptor_public_id_column IS NOT NULL THEN
            RAISE EXCEPTION '% sources must not specify a descriptor relation', normalized_storage;
        END IF;
        tensor_relation_oid := heap_oid;
    END IF;

    IF normalized_storage = 'heap_array' THEN
        IF tensor_ref_column IS NOT NULL
           OR tensor_rows_column IS NOT NULL
           OR tensor_dim_column IS NOT NULL
           OR tensor_dtype_column IS NOT NULL
           OR tensor_checksum_column IS NOT NULL THEN
            RAISE EXCEPTION 'heap_array sources must not specify external tensor columns';
        END IF;
    ELSE
        IF tensor_ref_column IS NULL
           OR tensor_rows_column IS NULL
           OR tensor_dim_column IS NULL
           OR tensor_dtype_column IS NULL
           OR tensor_checksum_column IS NULL THEN
            RAISE EXCEPTION 'external tensor sources require ref, rows, dim, dtype, and checksum columns';
        END IF;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_ref_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor ref column % must be a NOT NULL text column',
                tensor_ref_column;
        END IF;
        tensor_ref_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_rows_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor rows column % must be a NOT NULL integer column',
                tensor_rows_column;
        END IF;
        tensor_rows_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_dim_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'integer'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor dim column % must be a NOT NULL integer column',
                tensor_dim_column;
        END IF;
        tensor_dim_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_dtype_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor dtype column % must be a NOT NULL text column',
                tensor_dtype_column;
        END IF;
        tensor_dtype_attnum := attnum;

        attnum := NULL;
        SELECT a.attnum, a.atttypid, a.attnotnull
        INTO attnum, atttypid, attnotnull
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = tensor_relation_oid
          AND a.attname = tensor_checksum_column
          AND a.attnum > 0
          AND NOT a.attisdropped;
        IF attnum IS NULL OR atttypid <> 'text'::regtype OR NOT attnotnull THEN
            RAISE EXCEPTION 'tensor checksum column % must be a NOT NULL text column',
                tensor_checksum_column;
        END IF;
        tensor_checksum_attnum := attnum;

        IF (CASE WHEN normalized_storage = 'external_ref' THEN 7 ELSE 6 END) <> (
            SELECT count(DISTINCT u.attnum)
            FROM pg_catalog.unnest(ARRAY[
                CASE WHEN normalized_storage = 'external_ref' THEN model_contract_attnum ELSE descriptor_public_id_attnum END,
                CASE WHEN normalized_storage = 'external_ref' THEN public_id_attnum ELSE NULL END,
                tensor_ref_attnum,
                tensor_rows_attnum,
                tensor_dim_attnum,
                tensor_dtype_attnum,
                tensor_checksum_attnum
            ]) AS u(attnum)
        ) THEN
            RAISE EXCEPTION 'tensor source columns must be distinct';
        END IF;
    END IF;

    EXECUTE pg_catalog.format(
        'INSERT INTO %I._vchordrq_maxsim_sources (
             index_oid, heap_oid, model_contract_id, storage,
             model_contract_attnum, public_id_attnum,
             descriptor_oid, descriptor_public_id_attnum,
             tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
             tensor_dtype_attnum, tensor_checksum_attnum, registered_by
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
         ON CONFLICT (index_oid) DO UPDATE SET
             heap_oid = EXCLUDED.heap_oid,
             model_contract_id = EXCLUDED.model_contract_id,
             storage = EXCLUDED.storage,
             model_contract_attnum = EXCLUDED.model_contract_attnum,
             public_id_attnum = EXCLUDED.public_id_attnum,
             descriptor_oid = EXCLUDED.descriptor_oid,
             descriptor_public_id_attnum = EXCLUDED.descriptor_public_id_attnum,
             tensor_ref_attnum = EXCLUDED.tensor_ref_attnum,
             tensor_rows_attnum = EXCLUDED.tensor_rows_attnum,
             tensor_dim_attnum = EXCLUDED.tensor_dim_attnum,
             tensor_dtype_attnum = EXCLUDED.tensor_dtype_attnum,
             tensor_checksum_attnum = EXCLUDED.tensor_checksum_attnum,
             registered_by = EXCLUDED.registered_by,
             registered_at = pg_catalog.clock_timestamp()',
        ext_schema
    ) USING
        index_relation::oid,
        heap_oid,
        model_contract_id,
        normalized_storage,
        model_contract_attnum,
        public_id_attnum,
        descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum,
        caller_oid;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_register_maxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_register_maxsim_source(
    regclass, text, text, name, name, name, name, name, name, name, regclass, name
) TO PUBLIC;

CREATE FUNCTION vchordrq_unregister_maxsim_source(index_relation regclass)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    index_owner oid;
    removed_count bigint;
BEGIN
    IF index_relation IS NULL THEN
        RETURN false;
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;

    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;
    SELECT c.relowner INTO index_owner
    FROM pg_catalog.pg_class AS c
    WHERE c.oid = index_relation::oid AND c.relkind = 'i';
    IF index_owner IS NULL
       OR NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE') THEN
        RAISE EXCEPTION 'only the index owner may unregister its MaxSim tensor source';
    END IF;

    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_maxsim_sources WHERE index_oid = $1',
        ext_schema
    ) USING index_relation::oid;
    GET DIAGNOSTICS removed_count = ROW_COUNT;
    RETURN removed_count > 0;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_unregister_maxsim_source(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_unregister_maxsim_source(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_maxsim_source_info(index_relation regclass)
RETURNS TABLE(
    registered_index regclass,
    heap_relation regclass,
    model_contract_id text,
    source_storage text,
    model_contract_column name,
    public_id_column name,
    descriptor_relation regclass,
    descriptor_public_id_column name,
    tensor_ref_column name,
    tensor_rows_column name,
    tensor_dim_column name,
    tensor_dtype_column name,
    tensor_checksum_column name
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    caller_oid oid;
    bound_heap_oid oid;
    live_heap_oid oid;
    index_owner oid;
    bound_model_contract_id text;
    bound_storage text;
    bound_descriptor_oid oid;
    tensor_relation_oid oid;
    model_contract_attnum smallint;
    public_id_attnum smallint;
    descriptor_public_id_attnum smallint;
    tensor_ref_attnum smallint;
    tensor_rows_attnum smallint;
    tensor_dim_attnum smallint;
    tensor_dtype_attnum smallint;
    tensor_checksum_attnum smallint;
    valid_columns bigint;
    expected_columns bigint;
    resolved_model_contract_column name;
    resolved_public_id_column name;
    resolved_descriptor_public_id_column name;
    resolved_tensor_ref_column name;
    resolved_tensor_rows_column name;
    resolved_tensor_dim_column name;
    resolved_tensor_dtype_column name;
    resolved_tensor_checksum_column name;
BEGIN
    IF index_relation IS NULL THEN
        RAISE EXCEPTION 'index_relation must not be NULL';
    END IF;
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RAISE EXCEPTION 'vchord is not installed';
    END IF;
    SELECT r.oid INTO caller_oid
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = session_user;

    EXECUTE pg_catalog.format(
        'SELECT heap_oid, model_contract_id, storage,
                model_contract_attnum, public_id_attnum,
                descriptor_oid, descriptor_public_id_attnum,
                tensor_ref_attnum, tensor_rows_attnum, tensor_dim_attnum,
                tensor_dtype_attnum, tensor_checksum_attnum
         FROM %I._vchordrq_maxsim_sources
         WHERE index_oid = $1',
        ext_schema
    ) INTO
        bound_heap_oid,
        bound_model_contract_id,
        bound_storage,
        model_contract_attnum,
        public_id_attnum,
        bound_descriptor_oid,
        descriptor_public_id_attnum,
        tensor_ref_attnum,
        tensor_rows_attnum,
        tensor_dim_attnum,
        tensor_dtype_attnum,
        tensor_checksum_attnum
    USING index_relation::oid;
    IF bound_heap_oid IS NULL THEN
        RAISE EXCEPTION 'MaxSim tensor source is not registered for index %',
            index_relation;
    END IF;

    SELECT x.indrelid, i.relowner
    INTO live_heap_oid, index_owner
    FROM pg_catalog.pg_index AS x
    JOIN pg_catalog.pg_class AS i ON i.oid = x.indexrelid
    JOIN pg_catalog.pg_class AS h ON h.oid = x.indrelid
    JOIN pg_catalog.pg_am AS am ON am.oid = i.relam
    JOIN pg_catalog.pg_opclass AS opc ON opc.oid = x.indclass[0]
    WHERE x.indexrelid = index_relation::oid
      AND i.relkind = 'i'
      AND h.relkind IN ('r', 'm')
      AND am.amname = 'vchordrq'
      AND opc.opcname IN (
          'vector_maxsim_ops',
          'halfvec_maxsim_ops',
          'rabitq8_maxsim_ops',
          'rabitq4_maxsim_ops'
      )
      AND x.indisvalid
      AND x.indisready
      AND x.indnatts = 1
      AND x.indnkeyatts = 1;
    IF live_heap_oid IS NULL OR live_heap_oid <> bound_heap_oid THEN
        RAISE EXCEPTION 'registered MaxSim tensor source is stale or invalid';
    END IF;
    IF NOT pg_catalog.pg_has_role(caller_oid, index_owner, 'USAGE')
       AND NOT pg_catalog.has_table_privilege(caller_oid, live_heap_oid, 'SELECT') THEN
        RAISE EXCEPTION 'permission denied for registered MaxSim tensor source';
    END IF;

    IF bound_storage NOT IN ('heap_array', 'external_ref', 'external_relation') THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid storage';
    END IF;

    SELECT count(*)
    INTO valid_columns
    FROM (
        VALUES
            (model_contract_attnum, 'text'::regtype),
            (public_id_attnum, 'bigint'::regtype)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = bound_heap_oid
     AND a.attnum = expected.attnum
     AND a.atttypid = expected.atttypid
     AND a.attnotnull
     AND NOT a.attisdropped;
    IF valid_columns <> 2 THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid heap columns';
    END IF;

    IF bound_storage = 'heap_array' THEN
        IF bound_descriptor_oid IS NOT NULL
           OR descriptor_public_id_attnum IS NOT NULL
           OR tensor_ref_attnum IS NOT NULL
           OR tensor_rows_attnum IS NOT NULL
           OR tensor_dim_attnum IS NOT NULL
           OR tensor_dtype_attnum IS NOT NULL
           OR tensor_checksum_attnum IS NOT NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid heap_array binding';
        END IF;
        tensor_relation_oid := NULL;
    ELSIF bound_storage = 'external_ref' THEN
        IF bound_descriptor_oid IS NOT NULL OR descriptor_public_id_attnum IS NOT NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid external_ref binding';
        END IF;
        tensor_relation_oid := bound_heap_oid;
    ELSE
        IF bound_descriptor_oid IS NULL OR descriptor_public_id_attnum IS NULL THEN
            RAISE EXCEPTION 'registered MaxSim tensor source has invalid external_relation binding';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_class AS c
        WHERE c.oid = bound_descriptor_oid
          AND c.relkind IN ('r', 'm');
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor relation is stale or invalid';
        END IF;
        IF NOT pg_catalog.has_table_privilege(caller_oid, bound_descriptor_oid, 'SELECT') THEN
            RAISE EXCEPTION 'SELECT privilege on the registered descriptor relation is required';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_attribute AS a
        WHERE a.attrelid = bound_descriptor_oid
          AND a.attnum = descriptor_public_id_attnum
          AND a.atttypid = 'bigint'::regtype
          AND a.attnotnull
          AND NOT a.attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor public ID column is invalid';
        END IF;
        PERFORM 1
        FROM pg_catalog.pg_index AS x
        WHERE x.indrelid = bound_descriptor_oid
          AND x.indisunique
          AND x.indisvalid
          AND x.indisready
          AND x.indnkeyatts = 1
          AND x.indkey[0] = descriptor_public_id_attnum
          AND x.indexprs IS NULL
          AND x.indpred IS NULL;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'registered MaxSim descriptor public ID is no longer unique';
        END IF;
        tensor_relation_oid := bound_descriptor_oid;
    END IF;

    expected_columns := CASE WHEN bound_storage = 'heap_array' THEN 0 ELSE 5 END;
    SELECT count(*)
    INTO valid_columns
    FROM (
        VALUES
            (tensor_ref_attnum, 'text'::regtype),
            (tensor_rows_attnum, 'integer'::regtype),
            (tensor_dim_attnum, 'integer'::regtype),
            (tensor_dtype_attnum, 'text'::regtype),
            (tensor_checksum_attnum, 'text'::regtype)
    ) AS expected(attnum, atttypid)
    JOIN pg_catalog.pg_attribute AS a
      ON a.attrelid = tensor_relation_oid
     AND a.attnum = expected.attnum
     AND a.atttypid = expected.atttypid
     AND a.attnotnull
     AND NOT a.attisdropped
    WHERE expected.attnum IS NOT NULL;
    IF valid_columns <> expected_columns THEN
        RAISE EXCEPTION 'registered MaxSim tensor source has invalid descriptor columns';
    END IF;

    SELECT a.attname INTO resolved_model_contract_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_heap_oid AND a.attnum = model_contract_attnum;
    SELECT a.attname INTO resolved_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_heap_oid AND a.attnum = public_id_attnum;
    SELECT a.attname INTO resolved_descriptor_public_id_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = bound_descriptor_oid AND a.attnum = descriptor_public_id_attnum;
    SELECT a.attname INTO resolved_tensor_ref_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_ref_attnum;
    SELECT a.attname INTO resolved_tensor_rows_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_rows_attnum;
    SELECT a.attname INTO resolved_tensor_dim_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dim_attnum;
    SELECT a.attname INTO resolved_tensor_dtype_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_dtype_attnum;
    SELECT a.attname INTO resolved_tensor_checksum_column
    FROM pg_catalog.pg_attribute AS a
    WHERE a.attrelid = tensor_relation_oid AND a.attnum = tensor_checksum_attnum;

    RETURN QUERY SELECT
        index_relation,
        bound_heap_oid::regclass,
        bound_model_contract_id,
        bound_storage,
        resolved_model_contract_column,
        resolved_public_id_column,
        bound_descriptor_oid::regclass,
        resolved_descriptor_public_id_column,
        resolved_tensor_ref_column,
        resolved_tensor_rows_column,
        resolved_tensor_dim_column,
        resolved_tensor_dtype_column,
        resolved_tensor_checksum_column;
END;
$$;

REVOKE ALL ON FUNCTION vchordrq_maxsim_source_info(regclass) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_maxsim_source_info(regclass) TO PUBLIC;

CREATE FUNCTION vchordrq_maxsim_search(
    index_relation regclass,
    query anyarray,
    candidate_limit integer,
    top_k integer
)
RETURNS TABLE(public_id bigint, similarity real)
STRICT
VOLATILE
PARALLEL UNSAFE
LANGUAGE c
AS 'MODULE_PATHNAME', '_vchordrq_maxsim_search_external_wrapper';

COMMENT ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer)
IS 'Restricted Phase 3B external-tensor MaxSim search; returns exact similarity under caller MVCC, SELECT privileges, and PostgreSQL row visibility.';

REVOKE ALL ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION vchordrq_maxsim_search(regclass, anyarray, integer, integer) TO PUBLIC;

CREATE FUNCTION _vchordrq_maxsim_source_sql_drop()
RETURNS event_trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    ext_schema name;
    registry regclass;
BEGIN
    SELECT n.nspname
    INTO ext_schema
    FROM pg_catalog.pg_extension AS e
    JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace
    WHERE e.extname = 'vchord';
    IF ext_schema IS NULL THEN
        RETURN;
    END IF;
    registry := pg_catalog.to_regclass(
        pg_catalog.format('%I._vchordrq_maxsim_sources', ext_schema)
    );
    IF registry IS NULL THEN
        RETURN;
    END IF;

    EXECUTE pg_catalog.format(
        'DELETE FROM %I._vchordrq_maxsim_sources AS s
         USING pg_catalog.pg_event_trigger_dropped_objects() AS d
         WHERE d.objid = s.index_oid
            OR (
                d.objid = s.heap_oid
                AND (
                    d.objsubid = 0
                    OR d.objsubid = s.model_contract_attnum
                    OR d.objsubid = s.public_id_attnum
                    OR (
                        s.storage = ''external_ref''
                        AND d.objsubid IN (
                            s.tensor_ref_attnum,
                            s.tensor_rows_attnum,
                            s.tensor_dim_attnum,
                            s.tensor_dtype_attnum,
                            s.tensor_checksum_attnum
                        )
                    )
                )
            )
            OR (
                d.objid = s.descriptor_oid
                AND (
                    d.objsubid = 0
                    OR d.objsubid = s.descriptor_public_id_attnum
                    OR d.objsubid IN (
                        s.tensor_ref_attnum,
                        s.tensor_rows_attnum,
                        s.tensor_dim_attnum,
                        s.tensor_dtype_attnum,
                        s.tensor_checksum_attnum
                    )
                )
            )',
        ext_schema
    );
END;
$$;

REVOKE ALL ON FUNCTION _vchordrq_maxsim_source_sql_drop() FROM PUBLIC;

CREATE EVENT TRIGGER _vchordrq_maxsim_source_sql_drop
ON sql_drop
EXECUTE FUNCTION _vchordrq_maxsim_source_sql_drop();
