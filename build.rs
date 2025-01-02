fn main() {
    println!(r#"cargo::rustc-check-cfg=cfg(pgrx_embed)"#);
    println!(r#"cargo::rustc-check-cfg=cfg(feature, values("pg12"))"#);
}
