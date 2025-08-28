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

use std::env::{VarError, var};
use std::error::Error;
use std::ffi::OsString;
use std::path::Path;

fn compiler_version(cc: impl AsRef<Path>) -> Option<u16> {
    let cc = cc.as_ref();
    if let Ok(r) = std::process::Command::new(cc).arg("-dumpversion").output()
        && r.status.success()
        && let Some(major) = r.stdout.split(|c| !c.is_ascii_digit()).next()
        && let Ok(major) = std::str::from_utf8(major)
        && let Ok(major) = major.parse::<u16>()
    {
        return Some(major);
    }
    None
}

fn compiler(host: &str, target: &str, clang_version: u16, gcc_version: u16) -> Option<OsString> {
    let keys = [
        &format!("CC_{target}"),
        &format!("CC_{}", target.replace("-", "_")),
        "TARGET_CC",
        "CC",
    ];
    if keys.iter().any(|key| std::env::var_os(key).is_some()) {
        return None;
    }
    if host == target {
        if let Ok(cc) = which::which("clang")
            && compiler_version(&cc) >= Some(clang_version)
        {
            return Some(cc.into());
        }
        if let Ok(cc) = which::which("gcc")
            && compiler_version(&cc) >= Some(gcc_version)
        {
            return Some(cc.into());
        }
    }
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=cshim");
    let host = var("HOST")?;
    let target = var("TARGET")?;
    let target_arch = var("CARGO_CFG_TARGET_ARCH")?;
    match target_arch.as_str() {
        "aarch64" => {
            let mut build = cc::Build::new();
            if let Some(compiler) = compiler(&host, &target, 16, 14) {
                build.compiler(compiler);
            }
            build.file("./cshim/aarch64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        "powerpc64" => {
            if let Err(VarError::NotPresent) = var("CARGO_FEATURE_EXPERIMENTAL") {
                println!("cargo::error=`experimental` should be enabled on this platform");
            }
        }
        "s390x" => {
            if let Err(VarError::NotPresent) = var("CARGO_FEATURE_EXPERIMENTAL") {
                println!("cargo::error=`experimental` should be enabled on this platform");
            }
        }
        "x86_64" => {
            let mut build = cc::Build::new();
            if let Some(compiler) = compiler(&host, &target, 16, 12) {
                build.compiler(compiler);
            }
            build.file("./cshim/x86_64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        _ => {
            if let Err(VarError::NotPresent) = var("CARGO_FEATURE_EXPERIMENTAL") {
                println!("cargo::error=`experimental` should be enabled on this platform");
            }
            let messages = [
                "This platform has poor SIMD implementation.",
                "Please submit a feature request on https://github.com/tensorchord/VectorChord/issues.",
            ];
            for message in messages {
                println!("cargo::warning={message}");
            }
        }
    }
    Ok(())
}
