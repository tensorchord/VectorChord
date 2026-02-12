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
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha12Rng;
    let mut rng = ChaCha12Rng::from_seed([7; 32]);
    let mut bits = (0..262144).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
    if var("CARGO_CFG_TARGET_ENDIAN")?.as_str() == "big" {
        for chunk in bits.as_chunks_mut::<8>().0 {
            chunk.reverse();
        }
    }
    let out_dir = var("OUT_DIR")?;
    std::fs::write(PathBuf::from(out_dir).join("bits"), bits)?;
    Ok(())
}
