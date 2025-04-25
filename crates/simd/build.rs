use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo::rerun-if-changed=cshim");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;
    match target_arch.as_str() {
        "aarch64" => {
            let mut build = cc::Build::new();
            build.file("./cshim/aarch64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        "x86_64" => {
            let mut build = cc::Build::new();
            build.file("./cshim/x86_64.c");
            build.opt_level(3);
            build.compile("simd_cshim");
        }
        _ => (),
    }
    Ok(())
}
