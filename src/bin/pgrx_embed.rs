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

#![allow(unsafe_code)]
#![allow(unused_crate_dependencies)]

::pgrx::pgrx_embed!();

#[macro_export]
macro_rules! schema_generation {
    ($($symbol:ident)*) => {
        pub fn main() -> Result<(), Box<dyn std::error::Error>> {
            extern crate vchord as _;

            use ::pgrx::pgrx_sql_entity_graph::ControlFile;
            use ::pgrx::pgrx_sql_entity_graph::PgrxSql;
            use ::pgrx::pgrx_sql_entity_graph::SqlGraphEntity;

            let p = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/vchord.control"));
            let control_file = ControlFile::try_from(p)?;

            unsafe extern "Rust" {
                $(safe fn $symbol() -> SqlGraphEntity;)*
            }

            let mut e = vec![SqlGraphEntity::ExtensionRoot(control_file)];
            $(e.push($symbol());)*

            let pgrx_sql = PgrxSql::build(e.into_iter(), "vchord".to_string(), false)?;
            pgrx_sql.write(&mut std::io::stdout())?;

            Ok(())
        }
    };
}
