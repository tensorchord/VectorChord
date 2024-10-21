use base::search::*;
use base::vector::{VectBorrowed, VectorBorrowed};
use pgrx::pg_sys::panic::ErrorReportable;
use pgrx::{error, Spi};

use crate::datatype::memory_pgvector_vector::PgvectorVectorOutput;

pub fn pointer_to_ctid(pointer: Pointer) -> pgrx::pg_sys::ItemPointerData {
    let value = pointer.as_u64();
    pgrx::pg_sys::ItemPointerData {
        ip_blkid: pgrx::pg_sys::BlockIdData {
            bi_hi: ((value >> 32) & 0xffff) as u16,
            bi_lo: ((value >> 16) & 0xffff) as u16,
        },
        ip_posid: (value & 0xffff) as u16,
    }
}

pub fn ctid_to_pointer(ctid: pgrx::pg_sys::ItemPointerData) -> Pointer {
    let mut value = 0;
    value |= (ctid.ip_blkid.bi_hi as u64) << 32;
    value |= (ctid.ip_blkid.bi_lo as u64) << 16;
    value |= ctid.ip_posid as u64;
    Pointer::new(value)
}

pub fn load_table_vectors<F>(
    table_name: &str,
    column_name: &str,
    rows: u32,
    dims: u32,
    preprocess: Option<F>,
) -> Vec<Vec<f32>>
where
    F: Fn(VectBorrowed<f32>) -> Vec<f32>,
{
    let query = format!("SELECT {column_name} FROM {table_name};");
    let mut centroids = Vec::new();

    Spi::connect(|client| {
        let tup_table = client.select(&query, None, None).unwrap_or_report();
        assert_eq!(tup_table.len(), rows as usize);

        for row in tup_table {
            let vector = row[column_name].value::<PgvectorVectorOutput>();
            if let Ok(Some(v)) = vector {
                let borrowed = v.as_borrowed();
                assert_eq!(borrowed.dims(), dims);
                if let Some(ref f) = preprocess{
                    centroids.push(f(borrowed));
                }else{
                    centroids.push(borrowed.slice().to_vec());
                }
            } else {
                error!("load vectors from column is not valid")
            }
        }
        centroids
    })
}
