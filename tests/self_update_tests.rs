use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn invoke(exe: &Path, mirror: &Path, args: &[&str]) -> std::process::Output {
    Command::new(exe)
        .arg("self-update")
        .args(args)
        .env("RUNE_BASE_URL", mirror)
        .env("RUNE_VERSION", "latest")
        .env("RUNE_HOME", mirror.join("must not create"))
        .output()
        .unwrap()
}

fn package(mirror: &Path, name: &str, exe_name: &str) -> String {
    let archive = mirror.join(name);
    if archive.exists() {
        fs::remove_file(&archive).unwrap();
    }
    assert!(
        Command::new("tar")
            .arg(if cfg!(windows) { "-acf" } else { "-czf" })
            .arg(&archive)
            .arg("-C")
            .arg(mirror)
            .arg(exe_name)
            .status()
            .unwrap()
            .success()
    );
    let checksum = format!(
        "{}  {name}\n",
        hex::encode(Sha256::digest(fs::read(archive).unwrap()))
    );
    fs::write(mirror.join(format!("{name}.sha256")), &checksum).unwrap();
    checksum
}

#[test]
fn verified_update_replaces_running_executable_and_preserves_neighbors() {
    let root = std::env::temp_dir().join(format!(
        "rune infer update {} {}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let fixture = Fixture(root);
    let root = &fixture.0;
    let mirror = root.join("mirror");
    fs::create_dir(&mirror).unwrap();
    let exe_name = if cfg!(windows) {
        "rune-infer.exe"
    } else {
        "rune-infer"
    };
    let installed = root.join(exe_name);
    fs::copy(env!("CARGO_BIN_EXE_rune-infer"), &installed).unwrap();
    let before = fs::read(&installed).unwrap();
    for name in [
        "rune-kit",
        "infer/config/catalog.json",
        "infer/.env",
        "tools/plugins/sentinel",
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "keep").unwrap();
    }
    let source = root.join("fixture.rs");
    let candidate = mirror.join(exe_name);
    let os = match std::env::consts::OS {
        "windows" => "windows",
        "macos" => "macos",
        _ => "linux",
    };
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    };
    let archive_name = format!(
        "rune-infer-{os}-{arch}.{}",
        if cfg!(windows) { "zip" } else { "tar.gz" }
    );
    // Missing assets and equal-version repeated updates preserve the installed binary.
    assert!(!invoke(&installed, &mirror, &[]).status.success());
    assert_eq!(fs::read(&installed).unwrap(), before);
    fs::copy(&installed, &candidate).unwrap();
    package(&mirror, &archive_name, exe_name);
    for _ in 0..2 {
        let unchanged = invoke(&installed, &mirror, &[]);
        assert!(unchanged.status.success(), "{unchanged:?}");
        assert!(String::from_utf8_lossy(&unchanged.stdout).contains("Already up to date"));
        assert_eq!(fs::read(&installed).unwrap(), before);
    }
    fs::write(&source, "fn main() { println!(\"rune-infer 9999.0.0\"); }").unwrap();
    assert!(
        Command::new("rustc")
            .arg(&source)
            .arg("--crate-name")
            .arg("update_fixture")
            .arg("-o")
            .arg(&candidate)
            .status()
            .unwrap()
            .success()
    );
    let checksum = mirror.join(format!("{archive_name}.sha256"));
    let valid_checksum = package(&mirror, &archive_name, exe_name);

    let check = invoke(&installed, &mirror, &["--check"]);
    assert!(check.status.success(), "{check:?}");
    assert!(String::from_utf8_lossy(&check.stdout).contains("Update available"));
    assert_eq!(fs::read(&installed).unwrap(), before);
    let mismatch = invoke(&installed, &mirror, &["--version", "9998.0.0"]);
    assert!(!mismatch.status.success());
    assert_eq!(fs::read(&installed).unwrap(), before);

    fs::write(&checksum, format!("{}  {archive_name}\n", "0".repeat(64))).unwrap();
    let corrupt = invoke(&installed, &mirror, &[]);
    assert!(!corrupt.status.success());
    assert!(String::from_utf8_lossy(&corrupt.stderr).contains("SHA-256 mismatch"));
    assert_eq!(fs::read(&installed).unwrap(), before);
    fs::write(&checksum, valid_checksum).unwrap();

    fs::write(root.join(".rune-infer-self-update.lock"), "other updater").unwrap();
    assert!(!invoke(&installed, &mirror, &[]).status.success());
    fs::remove_file(root.join(".rune-infer-self-update.lock")).unwrap();
    let update = invoke(&installed, &mirror, &["--version", "9999.0.0"]);
    assert!(update.status.success(), "{update:?}");
    assert_eq!(fs::read(&installed).unwrap(), fs::read(&candidate).unwrap());
    assert_eq!(
        String::from_utf8(
            Command::new(&installed)
                .arg("--version")
                .output()
                .unwrap()
                .stdout
        )
        .unwrap()
        .trim(),
        "rune-infer 9999.0.0"
    );
    for name in [
        "rune-kit",
        "infer/config/catalog.json",
        "infer/.env",
        "tools/plugins/sentinel",
    ] {
        assert_eq!(fs::read_to_string(root.join(name)).unwrap(), "keep");
    }
    assert!(!mirror.join("must not create").exists());
    assert!(!root.join(".rune-infer-self-update.lock").exists());
}
