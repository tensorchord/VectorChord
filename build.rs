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

use std::collections::HashMap;
use std::env::{var, var_os};
use std::error::Error;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    if var("CARGO_CFG_TARGET_OS")? == "macos" {
        if let Some(path) = var_os("PGRX_PG_CONFIG_PATH") {
            let map = {
                let command_output = Command::new(&path).output()?;
                let command_stdout = String::from_utf8(command_output.stdout)?;
                let mut map = HashMap::new();
                for line in command_stdout.lines() {
                    if let Some((key, value)) = line.split_once(" = ") {
                        map.insert(key.to_string(), value.to_string());
                        eprintln!("Config `{key}`: {value}");
                    }
                }
                map
            };
            let bindir = &map["BINDIR"];
            println!("cargo::rustc-link-arg-cdylib=-Wl,-bundle,-bundle_loader,{bindir}/postgres",);
        } else {
            println!("cargo::rustc-link-arg-cdylib=-Wl,-undefined,dynamic_lookup");
        }
    }
    Ok(())
}
