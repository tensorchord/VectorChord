// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

// Referenced symbols must exist in the dynamic library when dropping functions.
// So we should never remove symbols used by schema, otherwise there will be errors in upgrade.
// Reference:
// * https://www.postgresql.org/message-id/CACX+KaPOzzRHEt4w_=iqKbTpMKjyrUGVng1C749yP3r6dprtcg@mail.gmail.com
// * https://github.com/tensorchord/pgvecto.rs/issues/397

#[allow(unused_macros)]
macro_rules! symbol {
    ($t:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            #[doc(hidden)]
            #[pgrx::pg_guard]
            extern "C-unwind" fn [<$t _wrapper>](_fcinfo: pgrx::pg_sys::FunctionCallInfo) -> pgrx::pg_sys::Datum {
                pgrx::error!(
                    "the symbol {} is removed in the extension; please run extension update scripts",
                    stringify!($t),
                );
            }
            #[unsafe(no_mangle)]
            #[doc(hidden)]
            pub extern "C" fn [<pg_finfo_ $t _wrapper>]() -> &'static ::pgrx::pg_sys::Pg_finfo_record {
                const V1_API: ::pgrx::pg_sys::Pg_finfo_record = ::pgrx::pg_sys::Pg_finfo_record {
                    api_version: 1,
                };
                &V1_API
            }
        }
    };
}
