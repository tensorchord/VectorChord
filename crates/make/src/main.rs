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

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use object::Object;
use std::collections::HashMap;
use std::env::consts::{DLL_PREFIX, DLL_SUFFIX, OS};
use std::env::var_os;
use std::fs::read_dir;
use std::io::Write;
use std::path::PathBuf;
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !std::fs::exists("vchord.control")? {
        bail!("The script must be run from the VectorChord source directory.")
    }
    let path = if let Some(value) = var_os("PGRX_PG_CONFIG_PATH") {
        eprintln!("Environment variable `PGRX_PG_CONFIG_PATH`: {value:#?}");
        PathBuf::from(value)
    } else {
        if let Ok(path) = which::which("pg_config") {
            eprintln!("Found executable `pg_config` in PATH: {path:#?}");
            eprintln!("Hint: set environment variable `PGRX_PG_CONFIG_PATH` to {path:#?}");
        }
        bail!("Environment variable `PGRX_PG_CONFIG_PATH` is not set.")
    };
    let map = {
        let mut command = Command::new(&path);
        command.stderr(Stdio::inherit());
        let command_output = command.output()?;
        let command_stdout = String::from_utf8(command_output.stdout)?;
        let mut map = HashMap::new();
        for line in command_stdout.lines() {
            if let Some((key, value)) = line.split_once(" = ") {
                map.insert(key.to_string(), value.to_string());
                eprintln!("Config `{key}`: {value}");
            }
        }
        map
    };
    let fork = {
        let version = map["VERSION"].clone();
        let fork = if let Some(prefix_stripped) = version.strip_prefix("PostgreSQL ") {
            if let Some((stripped, _)) = prefix_stripped.split_once(|c: char| !c.is_ascii_digit()) {
                format!("pg{stripped}",)
            } else {
                format!("pg{prefix_stripped}",)
            }
        } else {
            bail!("PostgreSQL version is invalid.")
        };
        eprintln!("Fork: {fork}");
        fork
    };
    let version = 'version: {
        for line in std::fs::read_to_string("./vchord.control")?.lines() {
            if let Some(prefix_stripped) = line.strip_prefix("default_version = '") {
                if let Some(stripped) = prefix_stripped.strip_suffix("'") {
                    eprintln!("VectorChord version: {stripped}");
                    break 'version stripped.to_string();
                }
            }
        }
        bail!("VectorChord version is not defined.")
    };
    let build = || {
        let mut command = Command::new("cargo");
        command
            .args(["build", "--release"])
            .args(["-p", "vchord", "--lib"])
            .args(["--features", fork.as_str()])
            .env("PGRX_PG_CONFIG_PATH", &path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let debug = format!("{command:?}");
        let status = command.spawn()?.wait()?;
        if !status.success() {
            bail!("Cargo build failed: {debug}");
        }
        Ok(())
    };
    let schema = || {
        let object = std::fs::read(format!("./target/release/{DLL_PREFIX}vchord{DLL_SUFFIX}"))?;
        let object = object::File::parse(object.as_slice())?;
        let exports = object
            .exports()?
            .into_iter()
            .flat_map(|x| str::from_utf8(x.name()));
        let exports = if matches!(object.format(), object::BinaryFormat::MachO) {
            exports
                .flat_map(|x| x.strip_prefix("_"))
                .filter(|x| x.starts_with("__pgrx_internals"))
                .collect::<Vec<_>>()
        } else {
            exports
                .filter(|x| x.starts_with("__pgrx_internals"))
                .collect::<Vec<_>>()
        };
        let pushes = exports
            .into_iter()
            .map(|x| {
                format!(
                    r#"
                entities.push(unsafe {{
                    unsafe extern "Rust" {{
                        fn {x}() -> ::pgrx::pgrx_sql_entity_graph::SqlGraphEntity;
                    }}
                    {x}()
                }});
            "#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!(
            r#"
            pub fn main() {{
                extern crate vchord as _;

                let mut entities = Vec::new();
                let control_file = std::fs::read_to_string("./vchord.control").unwrap();
                let control_file = ::pgrx::pgrx_sql_entity_graph::ControlFile::try_from(control_file.as_str()).expect(".control file should properly formatted");
                let control_file_entity = ::pgrx::pgrx_sql_entity_graph::SqlGraphEntity::ExtensionRoot(control_file);

                entities.push(control_file_entity);

                {pushes}

                let pgrx_sql = ::pgrx::pgrx_sql_entity_graph::PgrxSql::build(
                    entities.into_iter(),
                    "vchord".to_string(),
                    false,
                )
                .expect("SQL generation error");

                pgrx_sql
                    .write(&mut std::io::stdout())
                    .expect("Could not write SQL to stdout");
            }}
        "#
        );
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(code.as_bytes())?;
        let pgrx_embed_path = file.into_temp_path();
        let mut command = Command::new("cargo");
        command
            .args(["rustc"])
            .args(["-p", "vchord", "--bin", "pgrx_embed_vchord"])
            .args(["--features", fork.as_str()])
            .args(["--", "--cfg", "pgrx_embed"])
            .env("PGRX_EMBED", &pgrx_embed_path)
            .env("PGRX_PG_CONFIG_PATH", &path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let debug = format!("{command:?}");
        let status = command.spawn()?.wait()?;
        if !status.success() {
            bail!("Cargo build failed: {debug}");
        }
        let mut command = Command::new("./target/debug/pgrx_embed_vchord");
        command.stderr(Stdio::inherit());
        let command_output = command.output()?;
        let command_stdout = String::from_utf8(command_output.stdout)?.replace("\t", "    ");
        Ok(command_stdout)
    };
    let uilts_install = |permission: &str, src: &str, dst: &str| {
        let mut command = Command::new("install");
        command.args(["-m", permission, src, dst]);
        let debug = format!("{command:?}");
        let status = command.spawn()?.wait()?;
        if !status.success() {
            bail!("Command execution failed: {debug}");
        }
        Ok(())
    };
    let dll_suffix = if OS == "macos" && matches!(fork.as_str(), "pg13" | "pg14" | "pg15") {
        ".so"
    } else {
        DLL_SUFFIX
    };
    let install = |pkglibdir, sharedir_extension| -> Result<()> {
        uilts_install(
            "755",
            &format!("./target/release/{DLL_PREFIX}vchord{DLL_SUFFIX}"),
            &format!("{pkglibdir}/vchord{dll_suffix}"),
        )?;
        uilts_install(
            "644",
            "./vchord.control",
            &format!("{sharedir_extension}/vchord.control"),
        )?;
        if version != "0.0.0" {
            for maybe_entry in read_dir("./sql/upgrade")? {
                let path = maybe_entry?.path();
                let name = path.file_name().context("broken assets")?;
                uilts_install(
                    "644",
                    &format!("{}", path.display()),
                    &format!("{sharedir_extension}/{}", name.display()),
                )?;
            }
            uilts_install(
                "644",
                &format!("./sql/install/vchord--{version}.sql"),
                &format!("{sharedir_extension}/vchord--{version}.sql"),
            )?;
        } else {
            let contents = schema()?;
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(contents.as_bytes())?;
            let path = file.into_temp_path();
            uilts_install(
                "644",
                &format!("{}", path.display()),
                &format!("{sharedir_extension}/vchord--0.0.0.sql"),
            )?;
        }
        Ok(())
    };
    match cli.command {
        Commands::Build(BuildArgs { output }) => {
            build()?;
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
            install(pkglibdir, sharedir_extension)?;
        }
    }
    Ok(())
}
