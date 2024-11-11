#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_support_vector_l2_ops() -> String {
    "vector_l2_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_support_vector_ip_ops() -> String {
    "vector_ip_ops".to_string()
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _rabbithole_support_vector_cosine_ops() -> String {
    "vector_cosine_ops".to_string()
}
