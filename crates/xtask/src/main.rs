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

use clap::{Args, Parser, Subcommand};
use object::{Object, ObjectSymbol};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Build(BuildArgs),
    Clippy(ClippyArgs),
}

#[derive(Args)]
struct BuildArgs {
    #[arg(short, long, default_value = "./build")]
    output: String,
    #[arg(long, default_value = target_triple::TARGET, env = "TARGET")]
    target: String,
    #[arg(long, default_value = "release", env = "PROFILE")]
    profile: String,
    #[arg(long, env = "RUNNER")]
    runner: Option<String>,
}

#[derive(Args)]
struct ClippyArgs {
    #[arg(long, default_value = target_triple::TARGET, env = "TARGET")]
    target: String,
    #[arg(long, default_value = "release", env = "PROFILE")]
    profile: String,
}

struct RustcCfg {
    is_macos: bool,
    is_windows: bool,
    is_emscripten: bool,
    is_unix: bool,
    is_powerpc64: bool,
}

impl RustcCfg {
    fn dll_prefix(&self) -> Result<&'static str, Box<dyn Error>> {
        if self.is_macos {
            Ok("lib")
        } else if self.is_windows || self.is_emscripten {
            Ok("")
        } else if self.is_unix {
            Ok("lib")
        } else {
            Err("unknown operating system".into())
        }
    }
    fn dll_suffix(&self) -> Result<&'static str, Box<dyn Error>> {
        if self.is_macos {
            Ok(".dylib")
        } else if self.is_windows {
            Ok(".dll")
        } else if self.is_emscripten {
            Ok(".wasm")
        } else if self.is_unix {
            Ok(".so")
        } else {
            Err("unknown operating system".into())
        }
    }
    fn exe_suffix(&self) -> Result<&'static str, Box<dyn Error>> {
        if self.is_macos {
            Ok("")
        } else if self.is_windows {
            Ok(".exe")
        } else if self.is_emscripten {
            Ok(".js")
        } else if self.is_unix {
            Ok("")
        } else {
            Err("unknown operating system".into())
        }
    }
    fn ext_suffix(&self, fork: &str) -> Result<&'static str, Box<dyn Error>> {
        if self.is_macos {
            Ok(if matches!(fork, "pg14" | "pg15") {
                ".so"
            } else {
                ".dylib"
            })
        } else if self.is_windows {
            Ok(".dll")
        } else if self.is_emscripten || self.is_unix {
            Ok(".so")
        } else {
            Err("unknown operating system".into())
        }
    }
}

#[derive(Deserialize)]
struct CargoMetadata {
    target_directory: String,
}

fn pg_config(pg_config: impl AsRef<Path>) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut command = Command::new(pg_config.as_ref());
    command.stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let command_status = command_output.status;
    if !command_status.success() {
        return Err(format!("PostgreSQL failed: {command_status}").into());
    }
    let contents = String::from_utf8(command_output.stdout)?;
    let mut result = HashMap::new();
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once(" = ") {
            result.insert(key.to_string(), value.to_string());
        }
    }
    Ok(result)
}

fn control_file(path: impl AsRef<Path>) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let path = path.as_ref();
    eprintln!("Reading {path:?}");
    let contents = std::fs::read_to_string(path)?;
    let mut result = HashMap::new();
    for line in contents.lines() {
        if let Some((key, prefix_stripped)) = line.split_once(" = '")
            && let Some(value) = prefix_stripped.strip_suffix("'")
        {
            result.insert(key.to_string(), value.to_string());
        }
    }
    Ok(result)
}

fn rustc_cfg(target: &str) -> Result<RustcCfg, Box<dyn Error>> {
    let mut command = Command::new("rustc");
    command
        .args(["--print", "cfg"])
        .args(["--target", target])
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let command_status = command_output.status;
    if !command_status.success() {
        return Err(format!("Rust failed: {command_status}").into());
    }
    let contents = String::from_utf8(command_output.stdout)?;
    let mut cfgs = HashSet::new();
    for line in contents.lines() {
        cfgs.insert(line.to_string());
    }
    Ok(RustcCfg {
        is_macos: cfgs.contains("target_os=\"macos\""),
        is_unix: cfgs.contains("target_family=\"unix\""),
        is_emscripten: cfgs.contains("target_os=\"emscripten\""),
        is_windows: cfgs.contains("target_os=\"windows\""),
        is_powerpc64: cfgs.contains("target_arch=\"powerpc64\""),
    })
}

fn cargo_metadata() -> Result<CargoMetadata, Box<dyn Error>> {
    let mut command = Command::new("cargo");
    command
        .args(["metadata", "--format-version", "1"])
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let command_status = command_output.status;
    if !command_status.success() {
        return Err(format!("Cargo failed: {command_status}").into());
    }
    let contents = String::from_utf8(command_output.stdout)?;
    let cargo_metadata: CargoMetadata = serde_json::from_str(&contents)?;
    Ok(cargo_metadata)
}

fn build(
    pg_config: impl AsRef<Path>,
    pg_version: &str,
    rustc_cfg: &RustcCfg,
    cargo_metadata: &CargoMetadata,
    profile: &str,
    target: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut command = Command::new("cargo");
    command.args(["rustc"]);
    if !matches!(profile, "dev" | "test") {
        command.args(["--crate-type", "cdylib"]);
    }
    command
        .args(["-p", "vchord", "--lib"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features", pg_version])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo failed: {command_status}").into());
    }
    let mut result = PathBuf::from(&cargo_metadata.target_directory);
    result.push(target);
    result.push(match profile {
        "dev" | "test" => "debug",
        "release" | "bench" => "release",
        profile => profile,
    });
    result.push(format!(
        "{}vchord{}",
        rustc_cfg.dll_prefix()?,
        rustc_cfg.dll_suffix()?
    ));
    Ok(result)
}

fn parse(rustc_cfg: &RustcCfg, obj: impl AsRef<Path>) -> Result<Vec<String>, Box<dyn Error>> {
    let obj = obj.as_ref();
    eprintln!("Reading {obj:?}");
    let contents = std::fs::read(obj)?;
    let object = object::File::parse(contents.as_slice())?;
    let exports;
    if rustc_cfg.is_macos {
        exports = object
            .exports()?
            .into_iter()
            .flat_map(|x| std::str::from_utf8(x.name()))
            .flat_map(|x| x.strip_prefix("_"))
            .filter(|x| x.starts_with("__pgrx_internals"))
            .map(str::to_string)
            .collect();
    } else if rustc_cfg.is_emscripten {
        exports = object
            .symbols()
            .flat_map(|x| x.name().ok())
            .filter(|x| x.starts_with("__pgrx_internals"))
            .map(str::to_string)
            .collect();
    } else {
        exports = object
            .exports()?
            .into_iter()
            .flat_map(|x| std::str::from_utf8(x.name()))
            .filter(|x| x.starts_with("__pgrx_internals"))
            .map(str::to_string)
            .collect();
    }
    Ok(exports)
}

fn generate(
    runner: &Option<Vec<String>>,
    pg_config: impl AsRef<Path>,
    pg_version: &str,
    rustc_cfg: &RustcCfg,
    cargo_metadata: &CargoMetadata,
    profile: &str,
    target: &str,
    exports: Vec<String>,
    postmaster: impl AsRef<Path>,
) -> Result<String, Box<dyn Error>> {
    let imports = if rustc_cfg.is_powerpc64 {
        let postmaster = postmaster.as_ref();
        eprintln!("Reading {postmaster:?}");
        let contents = std::fs::read(postmaster)?;
        let object = object::File::parse(contents.as_slice())?;
        object
            .exports()?
            .into_iter()
            .flat_map(|x| std::str::from_utf8(x.name()))
            .filter(|x| !["_start", "_IO_stdin_used", "main"].contains(x))
            .map(str::to_string)
            .collect::<Vec<String>>()
    } else {
        Vec::new()
    };
    let mut pgrx_embed = tempfile::NamedTempFile::new()?;
    eprintln!("Writing {:?}", pgrx_embed.path());
    let contents = format!(
        "crate::schema_generation!({}; {});",
        exports.join(" "),
        imports.join(" ")
    );
    std::io::Write::write_all(pgrx_embed.as_file_mut(), contents.as_bytes())?;
    let mut command = Command::new("cargo");
    command
        .args(["rustc"])
        .args(["-p", "vchord", "--bin", "pgrx_embed_vchord"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features", pg_version])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .args(["--", "-C", "lto=off", "--cfg", "pgrx_embed"])
        .env("PGRX_EMBED", pgrx_embed.path())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo failed: {command_status}").into());
    }
    let mut result = PathBuf::from(&cargo_metadata.target_directory);
    result.push(target);
    result.push(match profile {
        "dev" | "test" => "debug",
        "release" | "bench" => "release",
        profile => profile,
    });
    result.push(format!("pgrx_embed_vchord{}", rustc_cfg.exe_suffix()?));
    let mut command;
    if let Some(runner) = runner {
        command = Command::new(&runner[0]);
        for arg in runner[1..].iter() {
            command.arg(arg);
        }
        command.arg(result);
    } else {
        command = Command::new(result);
    }
    command.stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let command_status = command_output.status;
    if !command_status.success() {
        return Err(format!("Cargo failed: {command_status}").into());
    }
    let command_stdout = String::from_utf8(command_output.stdout)?.replace("\t", "    ");
    Ok(command_stdout)
}

fn install_by_copying(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    #[cfg_attr(not(target_family = "unix"), expect(unused_variables))] is_executable: bool,
) -> Result<(), Box<dyn Error>> {
    eprintln!("Copying {:?} to {:?}", src.as_ref(), dst.as_ref());
    std::fs::copy(src, &dst)?;
    #[cfg(target_family = "unix")]
    {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;
        let perm = Permissions::from_mode(if !is_executable { 0o644 } else { 0o755 });
        std::fs::set_permissions(dst, perm)?;
    }
    Ok(())
}

fn install_by_writing(
    contents: impl AsRef<[u8]>,
    dst: impl AsRef<Path>,
    #[cfg_attr(not(target_family = "unix"), expect(unused_variables))] is_executable: bool,
) -> Result<(), Box<dyn Error>> {
    eprintln!("Writing {:?}", dst.as_ref());
    std::fs::write(&dst, contents)?;
    #[cfg(target_family = "unix")]
    {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;
        let perm = Permissions::from_mode(if !is_executable { 0o644 } else { 0o755 });
        std::fs::set_permissions(dst, perm)?;
    }
    Ok(())
}

fn clippy(
    pg_config: impl AsRef<Path>,
    pg_version: &str,
    profile: &str,
    target: &str,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("cargo");
    command
        .args(["clippy"])
        .args(["--workspace"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features", pg_version])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo failed: {command_status}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("PGRX_PG_CONFIG_PATH");
    }
    let cli = Cli::parse();
    match cli.command {
        Commands::Build(BuildArgs {
            output,
            target,
            profile,
            runner,
        }) => {
            if !std::fs::exists("./vchord.control")? {
                return Err("The script must be run from the VectorChord source directory.".into());
            }
            let vchord_version = control_file("./vchord.control")?["default_version"].clone();
            let runner = runner.and_then(|runner| shlex::split(&runner));
            let path = if let Some(value) = std::env::var_os("PG_CONFIG") {
                PathBuf::from(value)
            } else {
                return Err("Environment variable `PG_CONFIG` is not set.".into());
            };
            let pg_config = pg_config(&path)?;
            let pg_version = {
                let version = pg_config["VERSION"].clone();
                if let Some(prefix_stripped) = version.strip_prefix("PostgreSQL ") {
                    if let Some((stripped, _)) =
                        prefix_stripped.split_once(|c: char| !c.is_ascii_digit())
                    {
                        format!("pg{stripped}",)
                    } else {
                        format!("pg{prefix_stripped}",)
                    }
                } else {
                    return Err("PostgreSQL version is invalid.".into());
                }
            };
            let postmaster = format!("{}/postgres", pg_config["BINDIR"]);
            let rustc_cfg = rustc_cfg(&target)?;
            let cargo_metadata = cargo_metadata()?;
            let obj = build(
                &path,
                &pg_version,
                &rustc_cfg,
                &cargo_metadata,
                &profile,
                &target,
            )?;
            let pkglibdir = format!("{output}/pkglibdir");
            let sharedir = format!("{output}/sharedir");
            let sharedir_extension = format!("{sharedir}/extension");
            if std::fs::exists(&output)? {
                std::fs::remove_dir_all(&output)?;
            }
            std::fs::create_dir_all(&output)?;
            std::fs::create_dir_all(&pkglibdir)?;
            std::fs::create_dir_all(&sharedir)?;
            std::fs::create_dir_all(&sharedir_extension)?;
            install_by_copying(
                &obj,
                format!("{pkglibdir}/vchord{}", rustc_cfg.ext_suffix(&pg_version)?),
                true,
            )?;
            install_by_copying(
                "./vchord.control",
                format!("{sharedir}/extension/vchord.control"),
                false,
            )?;
            if vchord_version != "0.0.0" {
                for e in std::fs::read_dir("./sql/upgrade")?.collect::<Result<Vec<_>, _>>()? {
                    install_by_copying(
                        e.path(),
                        format!("{sharedir}/extension/{}", e.file_name().display()),
                        false,
                    )?;
                }
                install_by_copying(
                    format!("./sql/install/vchord--{vchord_version}.sql"),
                    format!("{sharedir}/extension/vchord--{vchord_version}.sql"),
                    false,
                )?;
            } else {
                let exports = parse(&rustc_cfg, obj)?;
                install_by_writing(
                    generate(
                        &runner,
                        &path,
                        &pg_version,
                        &rustc_cfg,
                        &cargo_metadata,
                        &profile,
                        &target,
                        exports,
                        postmaster,
                    )?,
                    format!("{sharedir_extension}/vchord--0.0.0.sql"),
                    false,
                )?;
            }
        }
        Commands::Clippy(ClippyArgs { target, profile }) => {
            if !std::fs::exists("./vchord.control")? {
                return Err("The script must be run from the VectorChord source directory.".into());
            }
            let path = if let Some(value) = std::env::var_os("PG_CONFIG") {
                PathBuf::from(value)
            } else {
                return Err("Environment variable `PG_CONFIG` is not set.".into());
            };
            let pg_config = pg_config(&path)?;
            let pg_version = {
                let version = pg_config["VERSION"].clone();
                if let Some(prefix_stripped) = version.strip_prefix("PostgreSQL ") {
                    if let Some((stripped, _)) =
                        prefix_stripped.split_once(|c: char| !c.is_ascii_digit())
                    {
                        format!("pg{stripped}",)
                    } else {
                        format!("pg{prefix_stripped}",)
                    }
                } else {
                    return Err("PostgreSQL version is invalid.".into());
                }
            };
            clippy(&path, &pg_version, &profile, &target)?;
        }
    }
    Ok(())
}
