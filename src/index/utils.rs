use base::search::*;
use base::vector::VectorBorrowed;
use pgrx::pg_sys::panic::ErrorReportable;
use pgrx::{error, Spi};

use crate::algorithm::rabitq;
use crate::datatype::memory_vecf32::Vecf32Output;

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

pub fn load_centroids(table_name: &str, column_name: &str, nlist: u32, dims: u32) -> Vec<Vec<f32>> {
    let query = format!("SELECT {column_name} FROM {table_name};");
    let mut centroids = Vec::new();

    Spi::connect(|client| {
        let tup_table = client.select(&query, None, None).unwrap_or_report();
        assert_eq!(tup_table.len(), nlist as usize);

        for row in tup_table {
            let vector = row[column_name].value::<Vecf32Output>();
            if let Ok(Some(v)) = vector {
                let borrowed = v.as_borrowed();
                assert_eq!(borrowed.dims(), dims);
                let projected_centroids = rabitq::project(borrowed.slice());
                centroids.push(projected_centroids);
            } else {
                error!("centroid vectors is not valid")
            }
        }
        centroids
    })
}
