use super::memory_scalar8::{Scalar8Input, Scalar8Output};
use pgrx::datum::Internal;
use pgrx::pg_sys::Oid;
use vector::VectorBorrowed;
use vector::scalar8::Scalar8Borrowed;

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_send(vector: Scalar8Input<'_>) -> Vec<u8> {
    let vector = vector.as_borrowed();
    let mut stream = Vec::<u8>::new();
    stream.extend(vector.dims().to_be_bytes());
    stream.extend(vector.sum_of_x2().to_be_bytes());
    stream.extend(vector.k().to_be_bytes());
    stream.extend(vector.b().to_be_bytes());
    stream.extend(vector.sum_of_code().to_be_bytes());
    for &c in vector.code() {
        stream.extend(c.to_be_bytes());
    }
    stream
}

#[pgrx::pg_extern(immutable, strict, parallel_safe)]
fn _vchord_scalar8_recv(internal: Internal, oid: Oid, typmod: i32) -> Scalar8Output {
    let _ = (oid, typmod);
    let buf = unsafe { internal.get_mut::<pgrx::pg_sys::StringInfoData>().unwrap() };

    let dims = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        u32::from_be_bytes(raw)
    };
    let sum_of_x2 = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let k = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let b = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let sum_of_code = {
        assert!(buf.cursor < i32::MAX - 4 && buf.cursor + 4 <= buf.len);
        let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 4]>().read() };
        buf.cursor += 4;
        f32::from_be_bytes(raw)
    };
    let code = {
        let mut result = Vec::with_capacity(dims as _);
        for _ in 0..dims {
            result.push({
                assert!(buf.cursor < i32::MAX - 1 && buf.cursor + 1 <= buf.len);
                let raw = unsafe { buf.data.add(buf.cursor as _).cast::<[u8; 1]>().read() };
                buf.cursor += 1;
                u8::from_be_bytes(raw)
            });
        }
        result
    };

    if let Some(x) = Scalar8Borrowed::new_checked(sum_of_x2, k, b, sum_of_code, &code) {
        Scalar8Output::new(x)
    } else {
        pgrx::error!("detect data corruption");
    }
}
