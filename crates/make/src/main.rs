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
use std::collections::{HashMap, HashSet};
use std::env::var_os;
use std::error::Error;
use std::fs::read_dir;
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
}

#[derive(Args)]
struct BuildArgs {
    #[arg(short, long)]
    output: String,
    #[arg(long, default_value = "release")]
    profile: String,
    #[arg(long, default_value = target_triple::TARGET)]
    target: String,
    #[arg(long)]
    runner: Option<String>,
    #[arg(long, action = clap::ArgAction::SetTrue, env = "EXPERIMENTAL", value_parser = clap::builder::FalseyValueParser::new())]
    experimental: bool,
}

struct TargetSpecificInformation {
    is_macos: bool,
    is_windows: bool,
    is_emscripten: bool,
    is_unix: bool,
}

impl TargetSpecificInformation {
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
            Ok(if matches!(fork, "pg13" | "pg14" | "pg15") {
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

fn pg_config(pg_config: impl AsRef<Path>) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut command = Command::new(pg_config.as_ref());
    command.stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
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

fn target_specific_information(target: &str) -> Result<TargetSpecificInformation, Box<dyn Error>> {
    let mut command = Command::new("rustc");
    command
        .args(["--print", "cfg"])
        .args(["--target", target])
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let contents = String::from_utf8(command_output.stdout)?;
    let mut cfgs = HashSet::new();
    for line in contents.lines() {
        cfgs.insert(line.to_string());
    }
    Ok(TargetSpecificInformation {
        is_macos: cfgs.contains("target_os=\"macos\""),
        is_unix: cfgs.contains("target_family=\"unix\""),
        is_emscripten: cfgs.contains("target_os=\"emscripten\""),
        is_windows: cfgs.contains("target_os=\"windows\""),
    })
}

fn build(
    pg_config: impl AsRef<Path>,
    pg_version: &str,
    tsi: &TargetSpecificInformation,
    profile: &str,
    target: &str,
    experimental: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut command = Command::new("cargo");
    command
        .args(["build", "-p", "vchord", "--lib"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features".into(), {
            let mut features = vec![pg_version];
            if experimental {
                features.push("simd/experimental");
            }
            features.join(",")
        }])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo build failed: {command_status}").into());
    }
    let mut result = PathBuf::from("./target");
    result.push(target);
    result.push(match profile {
        "dev" | "test" => "debug",
        "release" | "bench" => "release",
        profile => profile,
    });
    result.push(format!("{}vchord{}", tsi.dll_prefix()?, tsi.dll_suffix()?));
    Ok(result)
}

fn parse(
    tsi: &TargetSpecificInformation,
    obj: impl AsRef<Path>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let obj = obj.as_ref();
    eprintln!("Reading {obj:?}");
    let contents = std::fs::read(obj)?;
    let object = object::File::parse(contents.as_slice())?;
    let exports;
    if tsi.is_macos {
        exports = object
            .exports()?
            .into_iter()
            .flat_map(|x| std::str::from_utf8(x.name()))
            .flat_map(|x| x.strip_prefix("_"))
            .filter(|x| x.starts_with("__pgrx_internals"))
            .map(str::to_string)
            .collect();
    } else if tsi.is_emscripten {
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
    tsi: &TargetSpecificInformation,
    profile: &str,
    target: &str,
    exports: Vec<String>,
    experimental: bool,
) -> Result<String, Box<dyn Error>> {
    let pgrx_embed = std::env::temp_dir().join("VCHORD_PGRX_EMBED");
    eprintln!("Writing {pgrx_embed:?}");
    std::fs::write(
        &pgrx_embed,
        format!("crate::schema_generation!({});", exports.join(" ")),
    )?;
    let mut command = Command::new("cargo");
    command
        .args(["rustc", "-p", "vchord", "--bin", "pgrx_embed_vchord"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features".into(), {
            let mut features = vec![pg_version];
            if experimental {
                features.push("simd/experimental");
            }
            features.join(",")
        }])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .args(["--", "--cfg", "pgrx_embed"])
        .env("PGRX_EMBED", &pgrx_embed)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo build failed: {command_status}").into());
    }
    let mut result = PathBuf::from("./target");
    result.push(target);
    result.push(match profile {
        "dev" | "test" => "debug",
        "release" | "bench" => "release",
        profile => profile,
    });
    result.push(format!("pgrx_embed_vchord{}", tsi.exe_suffix()?));
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    if !std::fs::exists("./vchord.control")? {
        return Err("The script must be run from the VectorChord source directory.".into());
    }
    let vchord_version = control_file("./vchord.control")?["default_version"].clone();
    match cli.command {
        Commands::Build(BuildArgs {
            output,
            profile,
            target,
            runner,
            experimental,
        }) => {
            let runner = runner.and_then(|runner| shlex::split(&runner));
            let path = if let Some(value) = var_os("PGRX_PG_CONFIG_PATH") {
                PathBuf::from(value)
            } else {
                return Err("Environment variable `PGRX_PG_CONFIG_PATH` is not set.".into());
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
            let tsi = target_specific_information(&target)?;
            let obj = build(&path, &pg_version, &tsi, &profile, &target, experimental)?;
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
                format!("{pkglibdir}/vchord{}", tsi.ext_suffix(&pg_version)?),
                true,
            )?;
            install_by_copying(
                "./vchord.control",
                format!("{sharedir}/extension/vchord.control"),
                false,
            )?;
            if vchord_version != "0.0.0" {
                for e in read_dir("./sql/upgrade")?.collect::<Result<Vec<_>, _>>()? {
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
                let exports = parse(&tsi, obj)?;
                install_by_writing(
                    generate(
                        &runner,
                        &path,
                        &pg_version,
                        &tsi,
                        &profile,
                        &target,
                        exports,
                        experimental,
                    )?,
                    format!("{sharedir_extension}/vchord--0.0.0.sql"),
                    false,
                )?;
            }
        }
    }
    Ok(())
}
