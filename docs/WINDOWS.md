# Building VectorChord on Windows (Native, MSVC)

VectorChord builds and runs natively on Windows via MSVC — no Docker, no WSL2 required. The codebase already lists `x86_64-pc-windows-msvc` as a supported cargo target in `xtask`; this guide documents the toolchain and steps so anyone with PostgreSQL on Windows can produce a working `vchord.dll`.

## Required toolchain

| Component | Minimum | Why |
|---|---|---|
| **Rust** | **1.95.0** stable | The `simd` crate uses AVX-512 FP16 intrinsics (`stdarch_x86_avx512_f16`) which were unstable before Rust 1.95. Earlier versions fail with `E0658`. |
| **Visual Studio 2022 Build Tools** | 17.x | C/C++ compiler + linker (`cl.exe`, `link.exe`) for the `cc` build dependencies and the final cdylib link |
| **LLVM / clang** | 16+ | `pgrx-pg-sys` uses `bindgen`, which requires `libclang.dll`. Set `LIBCLANG_PATH` to the LLVM `bin` directory. |
| **PostgreSQL 17 (or 14–18)** | matching dev install | Provides `pg_config.exe`, `include/server/` headers, and `lib/postgres.lib` import library |
| **cargo-pgrx** | matching `pgrx` pin in `Cargo.toml` (currently 0.17.0) | `cargo install --locked cargo-pgrx@0.17.0` |

## Setup

1. Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload.
2. Install [Rust](https://rustup.rs/) (the installer auto-detects MSVC).
3. Install LLVM — either via [official Windows installer](https://github.com/llvm/llvm-project/releases) or extract the portable `.tar.xz` for `x86_64-pc-windows-msvc`.
4. Install [PostgreSQL 17 from EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
5. Install cargo-pgrx and initialize:

   ```cmd
   cargo install --locked cargo-pgrx@0.17.0
   cargo pgrx init --pg17 "C:\Program Files\PostgreSQL\17\bin\pg_config.exe"
   ```

## Build

From a "x64 Native Tools Command Prompt for VS 2022" (or after running `vcvars64.bat` in your shell):

```cmd
set "PATH=C:\path\to\llvm\bin;C:\Program Files\PostgreSQL\17\bin;%PATH%"
set "LIBCLANG_PATH=C:\path\to\llvm\bin"
set "PG_CONFIG=C:\Program Files\PostgreSQL\17\bin\pg_config.exe"

cargo run -p xtask --release -- build
```

Build output:

- `build\pkglibdir\vchord.dll` — the extension binary
- `build\sharedir\extension\vchord--<version>.sql` — extension SQL definitions
- `build\sharedir\extension\vchord.control` — Postgres control file

The DLL is a normal PE32+ x64 binary that imports symbols from `postgres.exe`, exactly the way pgvector's Windows extension does.

## Install (administrator)

Open a Command Prompt **as administrator**:

```cmd
copy /Y .\build\pkglibdir\vchord.dll                "C:\Program Files\PostgreSQL\17\lib\"
copy /Y .\build\sharedir\extension\vchord--*.sql    "C:\Program Files\PostgreSQL\17\share\extension\"
copy /Y .\build\sharedir\extension\vchord.control   "C:\Program Files\PostgreSQL\17\share\extension\"
```

Edit `C:\Program Files\PostgreSQL\17\data\postgresql.conf` and add:

```
shared_preload_libraries = 'vchord'
```

Restart Postgres and create the extension:

```cmd
net stop postgresql-x64-17
net start postgresql-x64-17
```

```sql
CREATE EXTENSION vchord CASCADE;
```

## Pitfalls

1. **Rust 1.94 and earlier fail to build** with `E0658: use of unstable library feature 'stdarch_x86_avx512_f16'` in `crates/simd/src/floating_f16.rs`. Run `rustup update` to get 1.95+.
2. **`cargo pgrx init` does not auto-download Postgres on Windows.** Install PostgreSQL via the EnterpriseDB installer first, then point `cargo pgrx init --pg17` at its `pg_config.exe`.
3. **`xtask build` requires `PG_CONFIG`** to be set before running — the Linux Makefile sets it implicitly; on Windows you must export it manually.
4. **vcvars64.bat does not propagate environment to bash sessions.** Use the "x64 Native Tools Command Prompt for VS 2022" or wrap your build in a `.cmd` file that calls `vcvars64.bat` first.

## Verification

Run `dumpbin /dependents build\pkglibdir\vchord.dll` from the VS dev prompt — the imports should be `postgres.exe`, `KERNEL32.dll`, `ntdll.dll`, `VCRUNTIME140.dll`, `api-ms-win-crt-*` (UCRT), and `bcryptprimitives.dll`. Anything else (especially Linux `.so` references) means a misconfigured build.
