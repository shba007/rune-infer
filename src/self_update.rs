//! Update only the invoked executable; never resolve or migrate the Infer home.
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn stable_version(raw: &str) -> Result<(u64, u64, u64)> {
    let raw = raw.strip_prefix('v').unwrap_or(raw);
    let parts: Vec<_> = raw.split('.').collect();
    ensure!(parts.len() == 3, "Expected a stable version such as 0.9.0");
    let mut numbers = [0; 3];
    for (i, part) in parts.iter().enumerate() {
        ensure!(
            !part.is_empty()
                && part.bytes().all(|c| c.is_ascii_digit())
                && (part.len() == 1 || !part.starts_with('0')),
            "Expected a stable version such as 0.9.0"
        );
        numbers[i] = part.parse()?;
    }
    Ok((numbers[0], numbers[1], numbers[2]))
}

fn platform() -> Result<String> {
    let os = match std::env::consts::OS {
        "windows" => "windows",
        "linux" => "linux",
        "macos" => "macos",
        other => bail!("No self-update release for {other}; use your installation manager"),
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => bail!("No self-update release for {other}; use your installation manager"),
    };
    Ok(format!("{os}-{arch}"))
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

async fn fetch(client: &reqwest::Client, base: &str, name: &str, dest: &Path) -> Result<()> {
    if Path::new(base).is_dir() {
        fs::copy(Path::new(base).join(name), dest).context("Cannot read mirror asset")?;
        return Ok(());
    }
    ensure!(
        reqwest::Url::parse(base)?.scheme() == "https",
        "Remote update sources must use HTTPS"
    );
    let mut response = client
        .get(format!("{}/{name}", base.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()
        .context("Release asset unavailable; the release may still be building")?;
    let mut file = fs::File::create(dest)?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(())
}

fn verify_checksum(archive: &Path, checksum: &Path, name: &str) -> Result<()> {
    let text = fs::read_to_string(checksum)?;
    let fields: Vec<_> = text.split_whitespace().collect();
    ensure!(
        fields.len() == 2
            && fields[0].len() == 64
            && fields[0].bytes().all(|c| c.is_ascii_hexdigit())
            && fields[1].trim_start_matches('*') == name,
        "Invalid checksum file for {name}"
    );
    let mut file = fs::File::open(archive)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    ensure!(
        hex::encode(digest.finalize()).eq_ignore_ascii_case(fields[0]),
        "SHA-256 mismatch; executable was not updated"
    );
    Ok(())
}

fn executable_version(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.file_type().is_file(),
        "Executable must be a regular file"
    );
    let result = Command::new(path).arg("--version").output()?;
    ensure!(
        result.status.success(),
        "Executable version verification failed"
    );
    let output = String::from_utf8(result.stdout)?;
    let version = output
        .trim()
        .strip_prefix("rune-infer ")
        .context("Unexpected executable identity")?;
    stable_version(version)?;
    Ok(version.to_owned())
}

fn replace(target: &Path, candidate: &Path, backup: &Path, version: &str) -> Result<()> {
    fs::rename(target, backup).context("Cannot preserve current executable; update aborted")?;
    let installed = (|| -> Result<()> {
        fs::rename(candidate, target)?;
        ensure!(
            executable_version(target)? == version,
            "Installed version validation failed"
        );
        Ok(())
    })();
    if let Err(error) = installed {
        if target.exists() {
            fs::remove_file(target).with_context(|| {
                format!(
                    "Cannot remove failed update; previous executable retained at {}",
                    backup.display()
                )
            })?;
        }
        fs::rename(backup, target).with_context(|| {
            format!(
                "Cannot restore executable; backup retained at {}",
                backup.display()
            )
        })?;
        return Err(error.context("Update failed; previous executable restored"));
    }
    // Windows may keep the running image locked until this process exits.
    if fs::remove_file(backup).is_err() {
        println!(
            "Previous executable retained at {} (remove after this command exits).",
            backup.display()
        );
    }
    Ok(())
}

pub async fn run(check: bool, requested: Option<&str>) -> Result<()> {
    let requested = requested
        .map(str::to_owned)
        .or_else(|| std::env::var("RUNE_VERSION").ok())
        .unwrap_or_else(|| "latest".into());
    if requested != "latest" {
        stable_version(&requested)?;
    }
    let target = std::env::current_exe().context("Cannot locate this executable")?;
    let parent = target
        .parent()
        .context("Executable has no parent directory")?;
    let lock_path = parent.join(".rune-infer-self-update.lock");
    let mut lock_file = OpenOptions::new().write(true).create_new(true).open(&lock_path)
        .context("Cannot lock installation: another update may be running, or the directory is not writable. If an update crashed, remove .rune-infer-self-update.lock after confirming no updater is running.")?;
    let lock = Lock(lock_path);
    writeln!(lock_file, "{}", std::process::id())?;
    drop(lock_file);
    let id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    let stage = parent.join(format!(".rune-infer-update-{id}"));
    fs::create_dir(&stage).context("Cannot stage update alongside executable")?;
    let scratch = Scratch(stage);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&scratch.0, fs::Permissions::from_mode(0o700))?;
    }
    let repo = std::env::var("RUNE_REPO").unwrap_or_else(|_| "shba007/rune-infer".into());
    let base = std::env::var("RUNE_BASE_URL").unwrap_or_else(|_| {
        if requested == "latest" {
            format!("https://github.com/{repo}/releases/latest/download")
        } else {
            format!(
                "https://github.com/{repo}/releases/download/v{}",
                requested.trim_start_matches('v')
            )
        }
    });
    if !Path::new(&base).is_dir() {
        ensure!(
            reqwest::Url::parse(&base)?.scheme() == "https",
            "Remote update sources must use HTTPS"
        );
    }
    let extension = if cfg!(windows) { "zip" } else { "tar.gz" };
    let name = format!("rune-infer-{}.{extension}", platform()?);
    // Check published release metadata without downloading or executing a candidate.
    if check {
        let version = if requested != "latest" {
            requested.trim_start_matches('v').to_owned()
        } else {
            ensure!(
                std::env::var_os("RUNE_BASE_URL").is_none(),
                "For a custom update mirror, use self-update --check --version VERSION; latest version metadata is unavailable"
            );
            ensure!(
                repo.split('/').count() == 2
                    && repo.split('/').all(|part| !part.is_empty()
                        && part
                            .bytes()
                            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))),
                "Invalid update repository"
            );
            let metadata: serde_json::Value = reqwest::Client::builder()
                .https_only(true)
                .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(30))
                .build()?
                .get(format!(
                    "https://api.github.com/repos/{repo}/releases/latest"
                ))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            ensure!(
                metadata["assets"].as_array().is_some_and(|assets| assets
                    .iter()
                    .any(|asset| asset["name"].as_str() == Some(&name))),
                "Latest release has no binary for this platform yet"
            );
            metadata["tag_name"]
                .as_str()
                .context("Release has no version tag")?
                .trim_start_matches('v')
                .to_owned()
        };
        if Path::new(&base).is_dir() {
            ensure!(
                Path::new(&base).join(&name).is_file()
                    && Path::new(&base).join(format!("{name}.sha256")).is_file(),
                "Requested release assets are unavailable in this mirror"
            );
        } else if requested != "latest" {
            reqwest::Client::builder()
                .https_only(true)
                .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(30))
                .build()?
                .head(format!("{}/{name}", base.trim_end_matches('/')))
                .send()
                .await?
                .error_for_status()?;
        }
        let available = stable_version(&version)?;
        let current = stable_version(env!("CARGO_PKG_VERSION"))?;
        if available > current {
            println!(
                "Update available: {} -> {version}. Run rune-infer self-update to install. Availability is based on release metadata; binaries are not downloaded or executed.",
                env!("CARGO_PKG_VERSION")
            );
        } else {
            println!(
                "Already up to date: {} (release {version}).",
                env!("CARGO_PKG_VERSION")
            );
        }
        return Ok(());
    }
    let archive = scratch.0.join(&name);
    let checksum = scratch.0.join(format!("{name}.sha256"));
    let client = reqwest::Client::builder()
        .https_only(true)
        .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(600))
        .build()?;
    println!("Checking {requested} release for {}...", platform()?);
    fetch(&client, &base, &name, &archive).await?;
    fetch(&client, &base, &format!("{name}.sha256"), &checksum).await?;
    verify_checksum(&archive, &checksum, &name)?;
    rune_infer::inference::engines::download::validate_archive(&archive)
        .map_err(|error| anyhow::anyhow!("Unsafe update archive: {error}"))?;
    let executable = if cfg!(windows) {
        "rune-infer.exe"
    } else {
        "rune-infer"
    };
    let unpack = rune_infer::inference::process::system_command("tar")
        .arg(if cfg!(windows) { "-xf" } else { "-xzf" })
        .arg(&archive)
        .arg("-C")
        .arg(&scratch.0)
        .args(["--no-same-owner", "--no-same-permissions"])
        .arg("--")
        .arg(executable)
        .output()
        .context("Cannot unpack release; install the system tar utility")?;
    ensure!(unpack.status.success(), "Cannot unpack release executable");
    let candidate = scratch.0.join(executable);
    ensure!(
        fs::symlink_metadata(&candidate)?.file_type().is_file(),
        "Release executable must be a regular file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o755))?;
    }
    let version = executable_version(&candidate)?;
    if requested != "latest" {
        ensure!(
            version == requested.trim_start_matches('v'),
            "Release executable does not match requested version"
        );
    }
    let current = stable_version(env!("CARGO_PKG_VERSION"))?;
    let available = stable_version(&version)?;
    if available <= current {
        println!(
            "Already up to date: {} (release {version}); downgrades are not installed.",
            env!("CARGO_PKG_VERSION")
        );
        return Ok(());
    }
    let backup = parent.join(format!(
        ".rune-infer-previous-{id}{}",
        if cfg!(windows) { ".exe" } else { "" }
    ));
    replace(&target, &candidate, &backup, &version)?;
    println!(
        "Updated {} to {version}. Restart any running server to use it.",
        target.display()
    );
    drop(scratch);
    drop(lock);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_stable_versions_are_accepted() {
        assert_eq!(stable_version("v0.9.1").unwrap(), (0, 9, 1));
        for value in ["../1.2.3", "1.2", "01.2.3", "1.2.3-beta", "1.2.3/other", ""] {
            assert!(stable_version(value).is_err());
        }
    }

    #[test]
    fn failed_installed_validation_restores_original() {
        let path = std::env::temp_dir().join(format!(
            "rune-infer-rollback-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let scratch = Scratch(path);
        let target = scratch.0.join("installed");
        let candidate = scratch.0.join("invalid");
        let backup = scratch.0.join("previous");
        fs::write(&target, "original executable").unwrap();
        fs::write(&candidate, "invalid executable").unwrap();
        assert!(replace(&target, &candidate, &backup, "9999.0.0").is_err());
        assert_eq!(fs::read_to_string(target).unwrap(), "original executable");
        assert!(!backup.exists());
    }
}
