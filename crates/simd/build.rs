fn main() {
    println!("cargo::rerun-if-changed=cshim.c");
    let mut build = cc::Build::new();
    build.file("cshim.c");
    build.opt_level(3);
    build.compile("simd_cshim");
}
