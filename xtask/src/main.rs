use infer_xtask::{
    Result, apply, release_notes, repository_plan, validate_event, validate_promotion,
    validate_versions, version,
};
use std::{env, fs, path::PathBuf};

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    match args.first().map(String::as_str) {
        Some("ci-required") if args.len() == 1 => {
            let required = infer_xtask::ci_required(&root, &env::var("CI_BEFORE")?)?;
            println!("{}", if required { "Source validation required" } else { "Validated release metadata only; skip duplicate source CI/preparation" });
            if let Some(output) = env::var_os("GITHUB_OUTPUT") {
                use std::io::Write;
                writeln!(fs::OpenOptions::new().append(true).open(output)?, "required={required}")?;
            }
        }
        Some("prepare") => {
            let write = args.iter().any(|s| s == "--apply");
            if args.iter().skip(1).any(|s| s != "--apply") { return Err("Usage: prepare [--apply] (default is a read-only dry run)".into()); }
            let plan = repository_plan(&root)?;
            if let Some(plan) = &plan {
                println!("Proposed version: {}\n\n{}", plan.version, plan.notes);
                if write { apply(&root, plan)?; validate_versions(&root)?; }
            } else { println!("No releasable changes; no release PR needed."); }
            if write && let Some(output) = env::var_os("GITHUB_OUTPUT") {
                use std::io::Write;
                writeln!(fs::OpenOptions::new().append(true).open(output)?, "prepared={}", plan.is_some())?;
            }
        }
        Some("notes") => {
            validate_versions(&root)?;
            let notes = release_notes(&fs::read_to_string(root.join("CHANGELOG.md"))?, &version(&root)?)?;
            if args.len() == 3 && args[1] == "--out" { fs::write(&args[2], notes)?; }
            else if args.len() == 1 { print!("{notes}"); }
            else { return Err("Usage: notes [--out FILE]".into()); }
        }
        Some("version") if args.len() == 1 => { validate_versions(&root)?; println!("{}", version(&root)?); }
        Some("verify-promotion") if args.len() == 1 => {
            let pending = validate_promotion(&root)?;
            println!("{}", if pending { "Valid unpublished prepared release" } else { "No pending release; already published version" });
            if let Some(output) = env::var_os("GITHUB_OUTPUT") {
                use std::io::Write;
                writeln!(fs::OpenOptions::new().append(true).open(output)?, "pending={pending}")?;
            }
        }
        Some("verify-event") if args.len() == 1 => {
            let event: serde_json::Value = serde_json::from_slice(&fs::read(env::var_os("GITHUB_EVENT_PATH").ok_or("GITHUB_EVENT_PATH is required")?)?)?;
            validate_event(&root, &event)?;
        }
        Some("example") if args.len() == 1 => {
            let old = semver::Version::parse("0.8.0")?;
            let mut commit = infer_xtask::cocogitto::Commit::new("a".repeat(40), "fix(install): preserve existing configuration during updates (#42)".into());
            commit.author.name = Some("Example contributor".into());
            let (_, notes) = infer_xtask::generated_release(&old, vec![commit])?.unwrap();
            println!("Dry-run Cocogitto example (no files or Git state changed):\n\n{notes}");
        }
        _ => return Err("Usage: cargo run --manifest-path xtask/Cargo.toml -- {prepare [--apply]|notes [--out FILE]|version|verify-event|verify-promotion|example}".into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Release preparation failed: {error}");
        std::process::exit(1);
    }
}
