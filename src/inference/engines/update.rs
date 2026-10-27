use super::{audio::windows_audio_assets, download};
use crate::{config::EngineKind, inference::AppState};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
type UpdateResult<T> = Result<T, String>;
pub(crate) enum UpdateEvent {
    Progress(String),
    Finished(Result<String, String>),
}

pub(crate) fn update_audio(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    let emit = |message: String| {
        let _ = events.send(UpdateEvent::Progress(message));
    };
    if !cfg!(windows) {
        return Err("audio.cpp updates are currently supported on Windows only".into());
    }
    emit("Waiting for models to unload".into());
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::AudioCpp) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let current = PathBuf::from("bin").join("audiocpp");
    let exe_name = if cfg!(windows) {
        "audiocpp_cli.exe"
    } else {
        "audiocpp_cli"
    };
    let current_exe = current.join(exe_name);
    let current_exe = if current_exe.is_file() {
        current_exe
    } else {
        download::find_executable_recursive(&current, exe_name)
            .ok_or("Installed audio.cpp executable not found")?
    };
    let installed_version = candidate_version(&current_exe)
        .map_err(|e| format!("Cannot verify installed audio.cpp version: {e}"))?;
    if version_matches(&installed_version, &expected_tag) {
        return Ok("Already up to date; no download needed".into());
    }
    emit("Checking release assets".into());
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let mut request = client
        .get("https://api.github.com/repos/0xShug0/audio.cpp/releases?per_page=5")
        .header("Accept", "application/vnd.github+json");
    if let Ok(token) = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")) {
        request = request.bearer_auth(token.trim());
    }
    let releases: serde_json::Value = request
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
    let release = releases
        .as_array()
        .and_then(|items| {
            items.iter().find(|r| {
                r["tag_name"].as_str() == Some(expected_tag.as_str())
                    && download::is_stable_release(r)
            })
        })
        .ok_or("The selected release is no longer available; check versions again")?;
    let assets = release["assets"]
        .as_array()
        .ok_or("Release has no assets")?;
    let selected = windows_audio_assets(assets, download::detect_host_cuda_version())
        .map_err(|e| e.to_string())?;
    let packages: Vec<_> = selected
        .into_iter()
        .map(|asset| {
            let (name, url) =
                download::extract_release_asset_info(asset).map_err(|e| e.to_string())?;
            let sha = download::asset_sha256(std::slice::from_ref(asset), &name)
                .map_err(|e| e.to_string())?;
            Ok((name, url, sha))
        })
        .collect::<UpdateResult<_>>()?;
    let bin = PathBuf::from("bin");
    let bin_info = std::fs::symlink_metadata(&bin).map_err(|e| e.to_string())?;
    if !bin_info.is_dir() || bin_info.file_type().is_symlink() {
        return Err("Engine directory is not a regular directory".into());
    }
    let metadata = std::fs::symlink_metadata(&current).map_err(|e| e.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("Managed audio.cpp directory is not a regular directory".into());
    }
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let stage = bin.join(format!(".audio-update-{}-{id}", std::process::id()));
    let backup = bin.join(format!(".audio-backup-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        for (name, url, sha) in packages {
            if stop.load(Ordering::Relaxed) {
                return Err("Update cancelled".into());
            }
            let tx = events.clone();
            download::with_progress(
                move |progress| {
                    let status = if progress.extracting {
                        format!("Extracting {}", progress.filename)
                    } else if let Some(total) = progress.total {
                        format!(
                            "Downloading {} · {} / {}",
                            progress.filename,
                            download::format_bytes(progress.downloaded),
                            download::format_bytes(total)
                        )
                    } else {
                        format!(
                            "Downloading {} · {}",
                            progress.filename,
                            download::format_bytes(progress.downloaded)
                        )
                    };
                    let completed = !progress.extracting
                        && progress
                            .total
                            .is_some_and(|total| progress.downloaded == total);
                    let _ = tx.send(UpdateEvent::Progress(status));
                    if completed {
                        let _ = tx.send(UpdateEvent::Progress("Verifying checksum".into()));
                    }
                },
                || {
                    download::download_and_extract(
                        &client,
                        &url,
                        &stage,
                        &name,
                        "[EngineUpdate]",
                        &sha,
                    )
                },
            )
            .map_err(|e| e.to_string())?;
        }
        download::flatten_dlls(&stage);
        let exe = if cfg!(windows) {
            "audiocpp_cli.exe"
        } else {
            "audiocpp_cli"
        };
        let staged_exe = stage.join(exe);
        if !staged_exe.is_file() {
            let nested = download::find_executable_recursive(&stage, exe)
                .ok_or("Verified archive does not contain the expected engine executable")?;
            std::fs::copy(nested, &staged_exe).map_err(|e| e.to_string())?;
        }
        download::make_executable(&staged_exe).map_err(|e| e.to_string())?;
        preserve_extras(&current, &stage)?;
        emit("Checking replacement engine".into());
        let actual = candidate_version(&staged_exe)?;
        if !version_matches(&actual, &expected_tag) {
            return Err(format!(
                "Replacement version does not match release {expected_tag}"
            ));
        }
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        emit("Installing verified engine".into());
        publish(&current, &stage, &backup)?;
        match cleanup_backup(&backup, &current) {
            Ok(true) => Ok(format!(
                "Updated to {expected_tag}; old engine files removed"
            )),
            Ok(false) => Ok(format!(
                "Updated to {expected_tag}; extra files preserved in {}",
                backup.display()
            )),
            Err(error) => Ok(format!(
                "Updated to {expected_tag}; cleanup failed: {error}. Backup: {}",
                backup.display()
            )),
        }
    })();
    if stage.exists() {
        let parent = std::fs::canonicalize(&bin);
        let resolved = std::fs::canonicalize(&stage);
        if let (Ok(parent), Ok(resolved)) = (parent, resolved) {
            if resolved.parent() == Some(parent.as_path()) {
                let _ = std::fs::remove_dir_all(resolved);
            }
        }
    }
    result
}
pub(crate) fn update_whisper(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    let emit = |message: String| {
        let _ = events.send(UpdateEvent::Progress(message));
    };
    if !cfg!(windows) {
        return Err("whisper.cpp updates are currently supported on Windows only".into());
    }
    emit("Waiting for models to unload".into());
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::WhisperCpp) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let bin = PathBuf::from("bin");
    let current = bin.join("whisper-cpp");
    for path in [&bin, &current] {
        let meta = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine directory is not a regular directory".into());
        }
    }
    let exe = "whisper-cli.exe";
    let direct = current.join(exe);
    let current_exe = if direct.is_file() {
        direct
    } else {
        download::find_executable_recursive(&current, exe)
            .ok_or("Installed whisper.cpp executable not found")?
    };
    let installed = candidate_version(&current_exe)
        .map_err(|e| format!("Cannot verify installed whisper.cpp version: {e}"))?;
    if !installed.contains("whisper.cpp version:") {
        return Err(
            "Cannot determine installed whisper.cpp version; use a supported managed build".into(),
        );
    }
    if version_matches(&installed, &expected_tag) {
        return Ok("Already up to date; no download needed".into());
    }
    emit("Checking release assets".into());
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let mut request = client
        .get("https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=100")
        .header("Accept", "application/vnd.github+json");
    if let Ok(token) = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")) {
        request = request.bearer_auth(token.trim());
    }
    let releases: serde_json::Value = request
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?
        .json()
        .map_err(|e| e.to_string())?;
    let (tag, asset) = super::whisper::stable_asset(&releases)
        .ok_or("No verified stable whisper.cpp release is available")?;
    if tag != expected_tag {
        return Err("The selected release has changed; check versions again".into());
    }
    let (name, url) = download::extract_release_asset_info(asset).map_err(|e| e.to_string())?;
    let digest =
        download::asset_sha256(std::slice::from_ref(asset), &name).map_err(|e| e.to_string())?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let stage = bin.join(format!(".whisper-update-{}-{id}", std::process::id()));
    let backup = bin.join(format!(".whisper-backup-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        let tx = events.clone();
        download::with_progress(
            move |progress| {
                let status = if progress.extracting {
                    format!("Extracting {}", progress.filename)
                } else if let Some(total) = progress.total {
                    format!(
                        "Downloading {} · {} / {}",
                        progress.filename,
                        download::format_bytes(progress.downloaded),
                        download::format_bytes(total)
                    )
                } else {
                    format!(
                        "Downloading {} · {}",
                        progress.filename,
                        download::format_bytes(progress.downloaded)
                    )
                };
                let complete = !progress.extracting
                    && progress
                        .total
                        .is_some_and(|total| progress.downloaded == total);
                let _ = tx.send(UpdateEvent::Progress(status));
                if complete {
                    let _ = tx.send(UpdateEvent::Progress("Verifying checksum".into()));
                }
            },
            || {
                download::download_and_extract(
                    &client,
                    &url,
                    &stage,
                    &name,
                    "[WhisperUpdate]",
                    &digest,
                )
            },
        )
        .map_err(|e| e.to_string())?;
        let staged = download::find_executable_recursive(&stage, exe)
            .ok_or("Verified archive does not contain whisper-cli.exe")?;
        download::make_executable(&staged).map_err(|e| e.to_string())?;
        emit("Checking replacement engine".into());
        let actual = candidate_version(&staged)?;
        if !actual.contains("whisper.cpp version:") || !version_matches(&actual, &expected_tag) {
            return Err(format!(
                "Replacement version does not match release {expected_tag}"
            ));
        }
        preserve_extras(&current, &stage)?;
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        emit("Installing verified engine".into());
        publish(&current, &stage, &backup)?;
        match cleanup_backup_for(&backup, &current, true) {
            Ok(true) => Ok(format!(
                "Updated to {expected_tag}; old engine files removed"
            )),
            Ok(false) => Ok(format!(
                "Updated to {expected_tag}; extra files preserved in {}",
                backup.display()
            )),
            Err(error) => Ok(format!(
                "Updated to {expected_tag}; cleanup failed: {error}. Backup: {}",
                backup.display()
            )),
        }
    })();
    if stage.exists() {
        if let (Ok(parent), Ok(resolved)) =
            (std::fs::canonicalize(&bin), std::fs::canonicalize(&stage))
        {
            if resolved.parent() == Some(parent.as_path()) {
                let _ = std::fs::remove_dir_all(resolved);
            }
        }
    }
    result
}

fn version_matches(output: &str, tag: &str) -> bool {
    let version = tag.trim_start_matches('v');
    !version.is_empty() && output.split_whitespace().any(|part| part == version)
}

// Delete only known old audio.cpp runtime binaries and exact duplicate files.
// Differing extra files stay in the backup instead of being silently discarded.
fn cleanup_backup(backup: &Path, current: &Path) -> UpdateResult<bool> {
    cleanup_backup_for(backup, current, false)
}

fn cleanup_backup_for(backup: &Path, current: &Path, whisper: bool) -> UpdateResult<bool> {
    for path in [backup, current] {
        let metadata = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err("Engine and backup must be regular directories".into());
        }
    }
    let backup = std::fs::canonicalize(backup).map_err(|e| e.to_string())?;
    let current = std::fs::canonicalize(current).map_err(|e| e.to_string())?;
    if backup.parent() != current.parent()
        || !backup
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| {
                n.starts_with(if whisper {
                    ".whisper-backup-"
                } else {
                    ".audio-backup-"
                })
            })
    {
        return Err("Backup path is outside the managed engine directory".into());
    }
    fn clean(
        directory: &Path,
        current: &Path,
        root: &Path,
        current_root: &Path,
        whisper: bool,
    ) -> UpdateResult<()> {
        for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            let checked = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
            if !checked.starts_with(root) {
                return Err("Backup entry escaped its directory".into());
            }
            let counterpart = current.join(entry.file_name());
            if kind.is_dir() {
                clean(&checked, &counterpart, root, current_root, whisper)?;
                if std::fs::read_dir(&checked)
                    .map_err(|e| e.to_string())?
                    .next()
                    .is_none()
                {
                    std::fs::remove_dir(&checked).map_err(|e| e.to_string())?;
                }
            } else if kind.is_file() {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let known_executable = if whisper {
                    matches!(name.as_str(), "whisper-cli.exe" | "whisper-server.exe")
                } else {
                    matches!(
                        name.as_str(),
                        "audiocpp_cli.exe"
                            | "audiocpp_gguf.exe"
                            | "audiocpp_server.exe"
                            | "audiocpp_cli"
                            | "audiocpp_gguf"
                            | "audiocpp_server"
                    )
                };
                let known_dll = (whisper && name == "whisper.dll")
                    || matches!(
                        name.as_str(),
                        "ggml.dll"
                            | "ggml-base.dll"
                            | "ggml-cuda.dll"
                            | "ggml-cpu.dll"
                            | "ggml-cpu-alderlake.dll"
                            | "ggml-cpu-cannonlake.dll"
                            | "ggml-cpu-cascadelake.dll"
                            | "ggml-cpu-haswell.dll"
                            | "ggml-cpu-icelake.dll"
                            | "ggml-cpu-sandybridge.dll"
                            | "ggml-cpu-skylakex.dll"
                            | "ggml-cpu-sse42.dll"
                            | "ggml-cpu-x64.dll"
                            | "cublas64_12.dll"
                            | "cublas64_13.dll"
                            | "cublaslt64_12.dll"
                            | "cublaslt64_13.dll"
                            | "cufft64_11.dll"
                            | "cufft64_12.dll"
                            | "cudart64_12.dll"
                            | "cudart64_13.dll"
                            | "msvcp140.dll"
                            | "msvcp140_1.dll"
                            | "msvcp140_2.dll"
                            | "msvcp140_atomic_wait.dll"
                            | "msvcp140_codecvt_ids.dll"
                            | "vcruntime140.dll"
                            | "vcruntime140_1.dll"
                            | "vcruntime140_threads.dll"
                    );
                let duplicate = if !known_executable
                    && !known_dll
                    && std::fs::symlink_metadata(&counterpart)
                        .is_ok_and(|m| m.is_file() && !m.file_type().is_symlink())
                {
                    let target = std::fs::canonicalize(&counterpart).map_err(|e| e.to_string())?;
                    target.starts_with(current_root)
                        && download::file_sha256(&checked).map_err(|e| e.to_string())?
                            == download::file_sha256(&target).map_err(|e| e.to_string())?
                } else {
                    false
                };
                if known_executable || known_dll || duplicate {
                    std::fs::remove_file(&checked).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    clean(&backup, &current, &backup, &current, whisper)?;
    if std::fs::read_dir(&backup)
        .map_err(|e| e.to_string())?
        .next()
        .is_none()
    {
        std::fs::remove_dir(&backup).map_err(|e| e.to_string())?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn candidate_version(path: &Path) -> UpdateResult<String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let mut command = Command::new(path);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                let mut output = String::new();
                child
                    .stdout
                    .take()
                    .ok_or("Missing version output")?
                    .take(4096)
                    .read_to_string(&mut output)
                    .map_err(|e| e.to_string())?;
                child
                    .stderr
                    .take()
                    .ok_or("Missing version diagnostics")?
                    .take(4096)
                    .read_to_string(&mut output)
                    .map_err(|e| e.to_string())?;
                return Ok(output);
            }
            Ok(Some(_)) => return Err("Replacement engine failed its startup check".into()),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.to_string());
            }
            Ok(None) if started.elapsed() >= Duration::from_secs(5) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Replacement engine startup check timed out".into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(40)),
        }
    }
}

fn preserve_extras(source: &Path, target: &Path) -> UpdateResult<()> {
    for entry in std::fs::read_dir(source).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        let destination = target.join(entry.file_name());
        if kind.is_symlink() {
            return Err("Engine directory contains a symbolic link; update refused".into());
        }
        if kind.is_dir() {
            if !destination.exists() {
                std::fs::create_dir(&destination).map_err(|e| e.to_string())?;
            }
            if !destination.is_dir() {
                return Err("Engine update has conflicting files".into());
            }
            preserve_extras(&entry.path(), &destination)?;
        } else if kind.is_file() {
            let is_old_runtime = entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "dll" | "exe" | "so" | "dylib"
                    )
                });
            if is_old_runtime && !destination.exists() {
                continue;
            }
            if !destination.exists() {
                std::fs::copy(entry.path(), destination).map_err(|e| e.to_string())?;
            } else if !destination.is_file() {
                return Err("Engine update has conflicting files".into());
            }
        } else {
            return Err("Engine directory has a special file; update refused".into());
        }
    }
    Ok(())
}
fn publish(current: &Path, staged: &Path, backup: &Path) -> UpdateResult<()> {
    if backup.exists() {
        return Err("Backup destination already exists".into());
    }
    std::fs::rename(current, backup).map_err(|e| format!("Cannot back up current engine: {e}"))?;
    if let Err(error) = std::fs::rename(staged, current) {
        return match std::fs::rename(backup, current) {
            Ok(()) => Err(format!("Cannot install update; original restored: {error}")),
            Err(restore) => Err(format!(
                "Cannot install update: {error}. Restore original from {}: {restore}",
                backup.display()
            )),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rune-engine-update-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        root
    }
    #[test]
    fn installed_version_matches_exact_release_without_network() {
        assert!(version_matches("audio.cpp 0.9.3\ngit: abc", "v0.9.3"));
        assert!(!version_matches("audio.cpp 0.9.30", "v0.9.3"));
        assert!(!version_matches("audio.cpp 0.9.2", "v0.9.3"));
        assert!(!version_matches("", ""));
    }

    #[test]
    fn backup_cleanup_removes_old_engine_files_and_preserves_changed_extras() {
        let root = root();
        let current = root.join("audiocpp");
        let backup = root.join(".audio-backup-fixture");
        std::fs::create_dir(&current).unwrap();
        std::fs::create_dir(&backup).unwrap();
        std::fs::write(current.join("audiocpp_cli.exe"), "new").unwrap();
        std::fs::write(current.join("config.json"), "new config").unwrap();
        std::fs::write(current.join("same.txt"), "same").unwrap();
        std::fs::write(backup.join("audiocpp_cli.exe"), "old").unwrap();
        std::fs::write(backup.join("ggml-cuda.dll"), "old library").unwrap();
        std::fs::write(backup.join("config.json"), "custom old config").unwrap();
        std::fs::write(backup.join("ggml-user.dll"), "custom library").unwrap();
        std::fs::write(backup.join("same.txt"), "same").unwrap();
        assert!(!cleanup_backup(&backup, &current).unwrap());
        assert!(!backup.join("audiocpp_cli.exe").exists());
        assert!(!backup.join("ggml-cuda.dll").exists());
        assert!(!backup.join("same.txt").exists());
        assert_eq!(
            std::fs::read_to_string(backup.join("config.json")).unwrap(),
            "custom old config"
        );
        assert_eq!(
            std::fs::read_to_string(current.join("audiocpp_cli.exe")).unwrap(),
            "new"
        );
        assert!(backup.join("ggml-user.dll").exists());
        std::fs::remove_file(backup.join("config.json")).unwrap();
        std::fs::remove_file(backup.join("ggml-user.dll")).unwrap();
        assert!(cleanup_backup(&backup, &current).unwrap());
        assert!(!backup.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn whisper_update_removes_old_engine_and_retains_custom_files_and_backup_on_conflict() {
        let root = root();
        let current = root.join("whisper-cpp");
        let stage = root.join("stage");
        let backup = root.join(".whisper-backup-fixture");
        std::fs::create_dir(&current).unwrap();
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(current.join("whisper-cli.exe"), "old").unwrap();
        std::fs::write(current.join("whisper.dll"), "old library").unwrap();
        std::fs::write(current.join("user.txt"), "custom").unwrap();
        std::fs::write(current.join("custom.dll"), "custom library").unwrap();
        std::fs::write(stage.join("whisper-cli.exe"), "new").unwrap();
        std::fs::write(stage.join("whisper.dll"), "new library").unwrap();
        preserve_extras(&current, &stage).unwrap();
        publish(&current, &stage, &backup).unwrap();
        assert!(!cleanup_backup_for(&backup, &current, true).unwrap());
        assert_eq!(
            std::fs::read_to_string(current.join("whisper-cli.exe")).unwrap(),
            "new"
        );
        assert_eq!(
            std::fs::read_to_string(current.join("user.txt")).unwrap(),
            "custom"
        );
        assert!(!backup.join("whisper-cli.exe").exists());
        assert!(!backup.join("whisper.dll").exists());
        assert!(backup.join("custom.dll").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn version_matching_is_exact_for_whisper() {
        assert!(version_matches("whisper.cpp version: 1.9.2", "v1.9.2"));
        assert!(!version_matches("whisper.cpp version: 1.9.20", "v1.9.2"));
        assert!(!version_matches("whisper.cpp version: 1.9.2-dev", "v1.9.2"));
    }

    #[test]
    fn publishing_preserves_backup_and_restores_after_failure() {
        let root = root();
        let old = root.join("old");
        let staged = root.join("stage");
        let backup = root.join("backup");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("engine"), "old").unwrap();
        assert!(publish(&old, &staged, &backup).is_err());
        assert_eq!(std::fs::read_to_string(old.join("engine")).unwrap(), "old");
        std::fs::create_dir(&staged).unwrap();
        std::fs::write(staged.join("engine"), "new").unwrap();
        std::fs::write(old.join("custom"), "keep").unwrap();
        std::fs::write(old.join("old-runtime.dll"), "old library").unwrap();
        preserve_extras(&old, &staged).unwrap();
        publish(&old, &staged, &backup).unwrap();
        assert_eq!(std::fs::read_to_string(old.join("engine")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(old.join("custom")).unwrap(), "keep");
        assert_eq!(
            std::fs::read_to_string(backup.join("engine")).unwrap(),
            "old"
        );
        assert!(!old.join("old-runtime.dll").exists());
        assert!(backup.join("old-runtime.dll").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
