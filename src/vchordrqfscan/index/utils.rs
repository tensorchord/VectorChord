use base::search::*;

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
