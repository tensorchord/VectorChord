-- List of functions

CREATE FUNCTION dequantize_to_vector(rabitq8) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq8) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq8_dequantize_to_halfvec_wrapper';

CREATE FUNCTION dequantize_to_vector(rabitq4) RETURNS vector
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_vector_wrapper';

CREATE FUNCTION dequantize_to_halfvec(rabitq4) RETURNS halfvec
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_rabitq4_dequantize_to_halfvec_wrapper';
