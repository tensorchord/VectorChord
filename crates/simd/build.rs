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

use std::env::var;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=cshim");
    let target_arch = var("CARGO_CFG_TARGET_ARCH")?;
    let target_endian = var("CARGO_CFG_TARGET_ENDIAN")?;
    #[allow(clippy::single_match)]
    match target_arch.as_str() {
        "aarch64" => {
            let mut build = cc::Build::new();
            build.file("./cshim/aarch64_byte.c");
            build.file("./cshim/aarch64_halfbyte.c");
            if target_endian == "little" {
                build.file("./cshim/aarch64_sve_fp16.c");
                build.file("./cshim/aarch64_sve_fp32.c");
            }
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        _ => {}
    }
    Ok(())
}
