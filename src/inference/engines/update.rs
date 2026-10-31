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
    Download(download::DownloadProgress),
    Finished(Result<String, String>),
}

pub(crate) fn install_engine(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    kind: EngineKind,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    let directory = match kind {
        EngineKind::LlamaServer => "llama-upstream",
        EngineKind::LlamaServerPrism => "llama-prism",
        EngineKind::AudioCpp => "audiocpp",
        EngineKind::SdServer => "sd-server",
        EngineKind::Crispasr => "crispasr",
        EngineKind::NemoSpeech => "nemo-speech",
        EngineKind::WhisperCpp => "whisper-cpp",
        EngineKind::Needle => "needle",
        EngineKind::Encoder => return Err("Encoder is built in and uses synthetic output".into()),
        EngineKind::Litert => "litert",
    };
    if !cfg!(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "macos"
    )) {
        return Err(
            "Atomic no-overwrite engine installation is not available on this platform".into(),
        );
    }
    let emit = |message: String| {
        let _ = events.send(UpdateEvent::Progress(message));
    };
    emit("Waiting for models to unload".into());
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Installation cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(kind) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    stage_engine_install(
        PathBuf::from("bin"),
        directory,
        kind,
        &stop,
        &events,
        |stage| download::with_install_bin(stage, || crate::setup::install_engine(kind)),
        |path| validate_install_binary(kind, path),
    )
}

fn stage_engine_install(
    bin: PathBuf,
    directory: &str,
    kind: EngineKind,
    stop: &AtomicBool,
    events: &std::sync::mpsc::Sender<UpdateEvent>,
    installer: impl FnOnce(PathBuf) -> UpdateResult<PathBuf>,
    validate: impl FnOnce(&Path) -> UpdateResult<()>,
) -> UpdateResult<String> {
    let emit = |message: String| {
        let _ = events.send(UpdateEvent::Progress(message));
    };
    std::fs::create_dir_all(&bin).map_err(|e| e.to_string())?;
    let meta = std::fs::symlink_metadata(&bin).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err("Engine directory is not a regular directory".into());
    }
    let bin = std::fs::canonicalize(&bin).map_err(|e| e.to_string())?;
    let destination = bin.join(directory);
    let incomplete = incomplete_installation(&destination, kind)?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let stage = bin.join(format!(".engine-install-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        emit(
            match kind {
                EngineKind::Needle => "Checking approved Needle distribution and SHA-256",
                EngineKind::Litert => "Checking pinned LiteRT 2.3.0 CPU runtime and SHA-256",
                _ => "Checking release · stable preferred",
            }
            .into(),
        );
        let tx = events.clone();
        let path = download::with_progress(
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
                let _ = tx.send(UpdateEvent::Progress(status));
                let _ = tx.send(UpdateEvent::Download(progress));
            },
            || installer(stage.clone()),
        )?;
        let engine_dir = stage.join(directory);
        let resolved_engine = std::fs::canonicalize(&engine_dir)
            .map_err(|_| "Installer did not produce a managed engine in the staging directory")?;
        let resolved_binary = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
        if resolved_engine.parent() != Some(stage.as_path())
            || !resolved_binary.starts_with(&resolved_engine)
            || !resolved_binary.is_file()
        {
            return Err(
                "Installer resolved an external executable; no managed installation was made"
                    .into(),
            );
        }
        ensure_no_links(&engine_dir)?;
        validate(&resolved_binary)?;
        if stop.load(Ordering::Relaxed) {
            return Err("Installation cancelled".into());
        }
        emit("Installing verified engine".into());
        if incomplete {
            if !incomplete_installation(&destination, kind)? {
                return Err(
                    "Incomplete engine directory changed during installation; retry".into(),
                );
            }
            let backup = bin.join(format!(
                ".{directory}-incomplete-{}-{id}",
                std::process::id()
            ));
            publish_repaired_engine(&engine_dir, &destination, &backup)?;
            Ok(format!(
                "Installed {kind}; previous files preserved at {}",
                backup.display()
            ))
        } else {
            publish_new_engine(&engine_dir, &destination).map_err(|e| {
                format!("Cannot publish installation without overwriting existing files: {e}")
            })?;
            Ok(format!("Installed {kind}"))
        }
    })();
    if stage.exists() {
        if let Ok(resolved) = std::fs::canonicalize(&stage) {
            if resolved.parent() == Some(bin.as_path()) {
                let _ = std::fs::remove_dir_all(resolved);
            }
        }
    }
    result
}

// Only a regular, link-free directory without an engine executable is eligible
// for install recovery. Never turn installation into an implicit update.
fn incomplete_installation(destination: &Path, kind: EngineKind) -> UpdateResult<bool> {
    let meta = match std::fs::symlink_metadata(destination) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("Cannot inspect engine directory: {error}")),
    };
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err(
            "Existing engine location is not a regular directory; installation refused".into(),
        );
    }
    ensure_no_links(destination)?;
    let executable = match kind {
        EngineKind::LlamaServer | EngineKind::LlamaServerPrism => "llama-server",
        EngineKind::AudioCpp => "audiocpp_cli",
        EngineKind::SdServer => "sd-server",
        EngineKind::Crispasr => "crispasr",
        EngineKind::NemoSpeech => "nemo-speech",
        EngineKind::WhisperCpp => "whisper-cli",
        EngineKind::Needle => "needle",
        EngineKind::Encoder => return Err("Encoder is built in".into()),
        EngineKind::Litert => "libLiteRt.dll",
    };
    if [executable.to_string(), format!("{executable}.exe")]
        .iter()
        .any(|name| download::find_executable_recursive(destination, name).is_some())
    {
        return Err("Engine executable already exists; use update instead of install".into());
    }
    Ok(true)
}

fn publish_repaired_engine(source: &Path, destination: &Path, backup: &Path) -> UpdateResult<()> {
    publish_new_engine(destination, backup)
        .map_err(|e| format!("Cannot preserve previous engine files: {e}"))?;
    if let Err(error) = publish_new_engine(source, destination) {
        return match publish_new_engine(backup, destination) {
            Ok(()) => Err(format!(
                "Cannot install engine; previous files restored: {error}"
            )),
            Err(restore) => Err(format!(
                "Cannot install engine: {error}. Previous files preserved at {} (restore failed: {restore})",
                backup.display()
            )),
        };
    }
    Ok(())
}

fn publish_new_engine(source: &Path, destination: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn MoveFileW(existing: *const u16, new: *const u16) -> i32;
        }
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let destination: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        let result = unsafe { MoveFileW(source.as_ptr(), destination.as_ptr()) };
        if result != 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())
            .map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let destination = std::ffi::CString::new(destination.as_os_str().as_bytes())
            .map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())
            .map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let destination = std::ffi::CString::new(destination.as_os_str().as_bytes())
            .map_err(|_| std::io::ErrorKind::InvalidInput)?;
        let result =
            unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
    #[cfg(not(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "macos"
    )))]
    {
        let _ = (source, destination);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Atomic no-overwrite engine installation is not available on this platform",
        ))
    }
}

fn validate_install_binary(kind: EngineKind, path: &Path) -> UpdateResult<()> {
    if kind == EngineKind::Litert {
        return super::litert::validate_runtime(path).map_err(|e| e.to_string());
    }
    if kind == EngineKind::Needle {
        let output = candidate_output(path, "--help")?;
        if !output.contains("--model")
            || !output.contains("--tools")
            || !output.contains("--prompt")
        {
            return Err("Needle startup check returned an unexpected CLI interface".into());
        }
        return Ok(());
    }
    let version =
        candidate_version(path).map_err(|e| format!("Cannot verify engine startup: {e}"))?;
    if kind == EngineKind::SdServer && super::sd_server::reported_version(&version).is_none() {
        return Err("Cannot verify SD-server commit identity from its version output".into());
    }
    if kind == EngineKind::WhisperCpp && !version.contains("whisper.cpp version:") {
        return Err("Cannot verify whisper.cpp version".into());
    }
    if kind == EngineKind::Crispasr && crispasr_version(&version).is_none() {
        return Err("Cannot verify CrispASR version".into());
    }
    Ok(())
}

pub(crate) fn update_llama(
    state: Arc<AppState>,
    kind: EngineKind,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    expected_build: Option<String>,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("llama-server updates are currently supported on Windows x64 only".into());
    }
    let (directory, release_url) = match kind {
        EngineKind::LlamaServer => (
            "llama-upstream",
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
        ),
        EngineKind::LlamaServerPrism => (
            "llama-prism",
            "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases",
        ),
        _ => return Err("Unsupported llama-server engine kind".into()),
    };
    let _ = events.send(UpdateEvent::Progress("Waiting for models to unload".into()));
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(kind) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let release = download::resolve_release(&client, release_url).map_err(|e| e.to_string())?;
    let (build, commit) = if kind == EngineKind::LlamaServerPrism {
        let (build, commit) =
            checked_prism_build(&release, &expected_tag, expected_build.as_deref())?;
        (build, Some(commit))
    } else {
        (
            checked_llama_build(&release, &expected_tag, expected_build.as_deref())?,
            None,
        )
    };
    let current = PathBuf::from("bin").join(directory);
    for directory in [Path::new("bin"), current.as_path()] {
        let meta = std::fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine location is not a regular directory".into());
        }
    }
    ensure_no_links(&current)?;
    let exe = download::find_executable_recursive(&current, "llama-server.exe")
        .ok_or("Managed llama-server executable not found")?;
    let installed_output = candidate_version(&exe)?;
    let installed = llama_binary_build(&installed_output)
        .ok_or("Cannot verify installed llama-server build")?;
    if installed > build || llama_version_matches(&installed_output, build, commit.as_deref()) {
        return Ok("Installed build is current or newer; no download needed".into());
    }
    stage_engine_update(
        PathBuf::from("bin"),
        directory,
        &stop,
        &events,
        |stage| {
            super::llama_server::install::install_release(&client, &release, stage)
                .map_err(|e| e.to_string())
        },
        |path| {
            let output = candidate_version(path)?;
            if !llama_version_matches(&output, build, commit.as_deref()) {
                return Err(format!(
                    "Replacement does not match selected llama-server release {expected_tag} (build b{build})"
                ));
            }
            Ok(())
        },
    )
}

fn checked_llama_build(
    release: &serde_json::Value,
    tag: &str,
    build_tag: Option<&str>,
) -> UpdateResult<u64> {
    if release["tag_name"].as_str() != Some(tag) || release["rune_build_tag"].as_str() != build_tag
    {
        return Err(
            "The selected release or binary build has changed; check versions again".into(),
        );
    }
    let build = build_tag
        .unwrap_or(tag)
        .strip_prefix('b')
        .filter(|value| !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()))
        .and_then(|value| value.parse().ok())
        .ok_or("Release has no verifiable llama-server build identity; update refused")?;
    Ok(build)
}

fn checked_prism_build(
    release: &serde_json::Value,
    tag: &str,
    build_tag: Option<&str>,
) -> UpdateResult<(u64, String)> {
    if release["tag_name"].as_str() != Some(tag) || release["rune_build_tag"].as_str() != build_tag
    {
        return Err("The selected Prism release has changed; check versions again".into());
    }
    let (build, commit) = tag
        .strip_prefix("prism-b")
        .and_then(|value| value.split_once('-'))
        .ok_or("Prism release has no verifiable build and commit identity")?;
    if build.is_empty()
        || !build.bytes().all(|c| c.is_ascii_digit())
        || !(7..=40).contains(&commit.len())
        || !commit.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err("Prism release has an invalid build or commit identity".into());
    }
    Ok((
        build.parse().map_err(|_| "Invalid Prism build number")?,
        commit.to_ascii_lowercase(),
    ))
}

fn llama_version_matches(output: &str, build: u64, commit: Option<&str>) -> bool {
    if llama_binary_build(output) != Some(build) {
        return false;
    }
    commit.is_none_or(|expected| {
        let actual = output
            .split_once("commit ")
            .map(|(_, tail)| {
                tail.split(|c: char| !c.is_ascii_hexdigit())
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase()
            })
            .unwrap_or_default();
        actual.len() >= 7 && (actual.starts_with(expected) || expected.starts_with(&actual))
    })
}

fn llama_binary_build(output: &str) -> Option<u64> {
    let tail = output.split("build ").nth(1)?;
    tail.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn stage_engine_update(
    bin: PathBuf,
    directory: &str,
    stop: &AtomicBool,
    events: &std::sync::mpsc::Sender<UpdateEvent>,
    installer: impl FnOnce(PathBuf) -> UpdateResult<PathBuf>,
    validate: impl FnOnce(&Path) -> UpdateResult<()>,
) -> UpdateResult<String> {
    let meta = std::fs::symlink_metadata(&bin).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err("Engine directory is not a regular directory".into());
    }
    let bin = std::fs::canonicalize(bin).map_err(|e| e.to_string())?;
    let current = bin.join(directory);
    let meta = std::fs::symlink_metadata(&current).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err("Managed engine directory is not a regular directory".into());
    }
    ensure_no_links(&current)?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let stage = bin.join(format!(".{directory}-update-{}-{id}", std::process::id()));
    let backup = bin.join(format!(".{directory}-backup-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        let _ = events.send(UpdateEvent::Progress(
            "Downloading verified release packages".into(),
        ));
        let tx = events.clone();
        let path = download::with_progress(
            move |progress| {
                let _ = tx.send(UpdateEvent::Download(progress));
            },
            || installer(stage.clone()),
        )?;
        let path = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
        if !path.starts_with(&stage) || !path.is_file() {
            return Err("Installer returned an external executable; update refused".into());
        }
        ensure_no_links(&stage)?;
        let _ = events.send(UpdateEvent::Progress("Checking replacement engine".into()));
        validate(&path)?;
        preserve_extras(&current, &stage)?;
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        let _ = events.send(UpdateEvent::Progress("Installing verified engine".into()));
        publish_repaired_engine(&stage, &current, &backup)?;
        Ok(format!(
            "Updated {directory}; previous files preserved at {}",
            backup.display()
        ))
    })();
    if let Ok(resolved) = std::fs::canonicalize(&stage) {
        if resolved.parent() == Some(bin.as_path()) {
            let _ = std::fs::remove_dir_all(resolved);
        }
    }
    result
}

pub(crate) fn update_nemo(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("NeMo Speech updates are currently supported on Windows x64 only".into());
    }
    let _ = events.send(UpdateEvent::Progress("Waiting for models to unload".into()));
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::NemoSpeech) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let release = download::resolve_release(
        &client,
        "https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases",
    )
    .map_err(|e| e.to_string())?;
    let expected = checked_nemo_release(&release, &expected_tag)?;
    let current = Path::new("bin/nemo-speech");
    for directory in [Path::new("bin"), current] {
        let meta = std::fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine location is not a regular directory".into());
        }
    }
    ensure_no_links(current)?;
    let exe = download::find_executable_recursive(current, "nemo-speech.exe")
        .ok_or("Managed NeMo Speech executable not found")?;
    let installed = nemo_binary_version(&candidate_version(&exe)?)
        .ok_or("Cannot verify installed NeMo Speech version")?;
    if installed >= expected {
        return Ok("Installed version is current or newer; no download needed".into());
    }
    stage_engine_update(
        PathBuf::from("bin"),
        "nemo-speech",
        &stop,
        &events,
        |stage| {
            super::nemo_speech::NemoSpeechEngine::install_release(&client, &release, stage)
                .map_err(|e| e.to_string())
        },
        |path| {
            if nemo_binary_version(&candidate_version(path)?) != Some(expected) {
                return Err(format!(
                    "Replacement does not match selected NeMo Speech release {expected_tag}"
                ));
            }
            Ok(())
        },
    )
}

fn checked_nemo_release(release: &serde_json::Value, tag: &str) -> UpdateResult<(u64, u64, u64)> {
    if release["tag_name"].as_str() != Some(tag) {
        return Err("The selected NeMo Speech release has changed; check versions again".into());
    }
    if tag.split_whitespace().count() != 1 {
        return Err("NeMo Speech release has no verifiable version".into());
    }
    semantic_version(tag).ok_or("NeMo Speech release has no verifiable version".into())
}

fn nemo_binary_version(output: &str) -> Option<(u64, u64, u64)> {
    let version = output
        .trim()
        .strip_prefix("nemo-speech ")?
        .split_whitespace()
        .next()?;
    semantic_version(version)
}

pub(crate) fn update_sd(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("SD-server updates are currently supported on Windows x64 only".into());
    }
    let _ = events.send(UpdateEvent::Progress("Waiting for models to unload".into()));
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::SdServer) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let release = download::resolve_release(
        &client,
        "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases",
    )
    .map_err(|e| e.to_string())?;
    if release["tag_name"].as_str() != Some(expected_tag.as_str()) {
        return Err("The selected SD-server release has changed; check versions again".into());
    }
    let selected = super::sd_server::release_commit(&expected_tag)
        .ok_or("Selected SD-server release has no verifiable commit identity")?;
    let current = Path::new("bin/sd-server");
    for directory in [Path::new("bin"), current] {
        let meta = std::fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine location is not a regular directory".into());
        }
    }
    ensure_no_links(current)?;
    let exe = download::find_executable_recursive(current, "sd-server.exe")
        .ok_or("Managed SD-server executable not found")?;
    let installed = super::sd_server::reported_version(&candidate_version(&exe)?)
        .ok_or("Cannot verify installed SD-server commit")?;
    if super::sd_server::revision_matches(&installed, &expected_tag) == Some(true) {
        return Ok("Already up to date; no download needed".into());
    }
    let installed_commit = installed
        .strip_prefix("commit ")
        .ok_or("Missing installed commit")?;
    let _ = events.send(UpdateEvent::Progress(
        "Verifying selected release ancestry".into(),
    ));
    let comparison = download::compare_sd_commits(&client, installed_commit, selected)
        .map_err(|e| format!("Cannot verify SD-server release order; update refused: {e}"))?;
    if !sd_update_is_forward(installed_commit, &comparison)? {
        return Ok("Already up to date; no download needed".into());
    }
    stage_engine_update(
        PathBuf::from("bin"),
        "sd-server",
        &stop,
        &events,
        |stage| {
            super::sd_server::SdServerEngine::install_release(&client, &release, stage)
                .map_err(|e| e.to_string())
        },
        |path| {
            let version = super::sd_server::reported_version(&candidate_version(path)?)
                .ok_or("Cannot verify replacement SD-server commit")?;
            if super::sd_server::revision_matches(&version, &expected_tag) != Some(true) {
                return Err("Replacement SD-server commit does not match selected release".into());
            }
            Ok(())
        },
    )
}

fn sd_update_is_forward(installed: &str, comparison: &serde_json::Value) -> UpdateResult<bool> {
    if !(7..=40).contains(&installed.len()) || !installed.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid installed SD-server commit identity".into());
    }
    let base = comparison["base_commit"]["sha"].as_str().unwrap_or("");
    if base.len() != 40
        || !base.bytes().all(|c| c.is_ascii_hexdigit())
        || !base
            .to_ascii_lowercase()
            .starts_with(&installed.to_ascii_lowercase())
    {
        return Err(
            "SD-server comparison does not identify the installed commit; update refused".into(),
        );
    }
    let merge = comparison["merge_base_commit"]["sha"]
        .as_str()
        .unwrap_or("");
    match (
        comparison["status"].as_str(),
        comparison["ahead_by"].as_u64(),
        comparison["behind_by"].as_u64(),
    ) {
        (Some("ahead"), Some(ahead), Some(0)) if ahead > 0 && merge.eq_ignore_ascii_case(base) => {
            Ok(true)
        }
        (Some("identical"), Some(0), Some(0)) if merge.eq_ignore_ascii_case(base) => Ok(false),
        (Some("behind"), _, _) => {
            Err("Selected SD-server release is older than installed; downgrade refused".into())
        }
        _ => Err(
            "Selected SD-server release is divergent or its ancestry is unknown; update refused"
                .into(),
        ),
    }
}

pub(crate) fn update_needle(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    use super::needle;
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("Needle updates are currently supported on Windows x64 only".into());
    }
    let selected = expected
        .strip_prefix("revision ")
        .filter(|value| value.len() == 40 && value.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or("Selected Needle revision is invalid; check versions again")?;
    let _ = events.send(UpdateEvent::Progress("Waiting for models to unload".into()));
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::Needle) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let current = Path::new("bin/needle");
    for directory in [Path::new("bin"), current] {
        let meta = std::fs::symlink_metadata(directory).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine location is not a regular directory".into());
        }
    }
    ensure_no_links(current)?;
    let installed = needle::installed_revision(&current.join("needle.exe"))
        .ok_or("Cannot verify installed Needle receipt and binary checksum; update refused")?;
    if installed.eq_ignore_ascii_case(&expected) {
        return Ok("Already up to date; no download needed".into());
    }
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let metadata = needle::latest_metadata(&client).map_err(|e| e.to_string())?;
    if metadata["sha"].as_str() != Some(selected) {
        return Err("Selected Needle revision has changed; check versions again".into());
    }
    let _ = events.send(UpdateEvent::Progress(
        "Verifying selected revision ancestry".into(),
    ));
    needle::verify_forward_update(&client, installed.trim_start_matches("revision "), selected)
        .map_err(|e| e.to_string())?;
    stage_engine_update(
        PathBuf::from("bin"),
        "needle",
        &stop,
        &events,
        |stage| {
            needle::NeedleEngine::install_release(&client, &metadata, stage)
                .map_err(|e| e.to_string())
        },
        |path| {
            validate_install_binary(EngineKind::Needle, path)?;
            if needle::installed_revision(path).as_deref() != Some(expected.as_str()) {
                return Err("Replacement Needle receipt or binary checksum does not match selected revision".into());
            }
            Ok(())
        },
    )
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
    let release = download::resolve_release(
        &client,
        "https://api.github.com/repos/0xShug0/audio.cpp/releases",
    )
    .map_err(|e| e.to_string())?;
    if release["tag_name"].as_str() != Some(expected_tag.as_str()) {
        return Err("The selected release has changed; check versions again".into());
    }
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
                    let _ = tx.send(UpdateEvent::Download(progress));
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
    ensure_no_links(&current)?;
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
    if let (Some(installed), Some(available)) = (
        semantic_version(&installed),
        semantic_version(&expected_tag),
    ) {
        if installed > available {
            return Ok(
                "Installed version is newer than the available release; no download needed".into(),
            );
        }
    }
    emit("Checking release assets".into());
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let release = super::whisper::resolve_release(&client).map_err(|e| e.to_string())?;
    let asset = super::whisper::platform_asset(&release)
        .ok_or("No verified compatible whisper.cpp release is available")?;
    if release["tag_name"].as_str() != Some(expected_tag.as_str()) {
        return Err("The selected release has changed; check versions again".into());
    }
    let (name, url) = download::extract_release_asset_info(asset).map_err(|e| e.to_string())?;
    let digest =
        download::asset_sha256(std::slice::from_ref(asset), &name).map_err(|e| e.to_string())?;
    let package = super::whisper::PackageIdentity::from_asset(asset).map_err(|e| e.to_string())?;
    if whisper_package_is_current(&installed, &expected_tag, &current, &package) {
        return Ok("Already up to date; no download needed".into());
    }
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
                let _ = tx.send(UpdateEvent::Download(progress));
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
        super::whisper::record_installation(&stage, &package).map_err(|e| e.to_string())?;
        preserve_extras(&current, &stage)?;
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        emit("Installing verified engine".into());
        publish(&current, &stage, &backup)?;
        match cleanup_backup_for(&backup, &current, BackupKind::Whisper) {
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

pub(crate) fn update_crispasr(
    state: Arc<AppState>,
    stop: Arc<AtomicBool>,
    expected_tag: String,
    events: std::sync::mpsc::Sender<UpdateEvent>,
) -> UpdateResult<String> {
    let emit = |message: String| {
        let _ = events.send(UpdateEvent::Progress(message));
    };
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("CrispASR updates are currently supported on Windows x86-64 only".into());
    }
    emit("Waiting for models to unload".into());
    let _reservation = loop {
        if stop.load(Ordering::Relaxed) {
            return Err("Update cancelled".into());
        }
        if let Some(guard) = state.try_reserve_engine_update(EngineKind::Crispasr) {
            break guard;
        }
        std::thread::sleep(Duration::from_millis(300));
    };
    let bin = PathBuf::from("bin");
    let current = bin.join("crispasr");
    for path in [&bin, &current] {
        let meta = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("Managed engine directory is not a regular directory".into());
        }
    }
    ensure_no_links(&current)?;
    let exe = "crispasr.exe";
    let direct = current.join(exe);
    let current_exe = if direct.is_file() {
        direct
    } else {
        download::find_executable_recursive(&current, exe)
            .ok_or("Installed CrispASR executable not found")?
    };
    let installed = candidate_version(&current_exe)
        .map_err(|e| format!("Cannot verify installed CrispASR version: {e}"))?;
    let installed_version = crispasr_version(&installed)
        .ok_or("Cannot determine installed CrispASR version; use a supported managed build")?;
    if version_matches(installed_version, &expected_tag) {
        return Ok("Already up to date; no download needed".into());
    }
    if let (Some(installed), Some(available)) = (
        semantic_version(installed_version),
        semantic_version(&expected_tag),
    ) {
        if installed > available {
            return Ok(
                "Installed version is newer than the available release; no download needed".into(),
            );
        }
    }
    let asset_name = crispasr_asset_name(&installed).unwrap_or_else(|_| crispasr_hardware_asset());
    emit(format!("Selected hardware build: {asset_name}"));
    emit("Checking release assets".into());
    let client = download::create_download_client().map_err(|e| e.to_string())?;
    let release = download::resolve_release(
        &client,
        "https://api.github.com/repos/CrispStrobe/CrispASR/releases",
    )
    .map_err(|e| e.to_string())?;
    if release["tag_name"].as_str() != Some(expected_tag.as_str()) {
        return Err("The selected release has changed; check versions again".into());
    }
    let asset = release["assets"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .find(|asset| asset["name"].as_str() == Some(asset_name))
        })
        .ok_or(
            "The selected release has no verified asset matching the installed CrispASR backend",
        )?;
    let (name, url) = download::extract_release_asset_info(asset).map_err(|e| e.to_string())?;
    let digest =
        download::asset_sha256(std::slice::from_ref(asset), &name).map_err(|e| e.to_string())?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let stage = bin.join(format!(".crispasr-update-{}-{id}", std::process::id()));
    let backup = bin.join(format!(".crispasr-backup-{}-{id}", std::process::id()));
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
                let _ = tx.send(UpdateEvent::Download(progress));
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
                    "[CrispAsrUpdate]",
                    &digest,
                )
            },
        )
        .map_err(|e| e.to_string())?;
        let staged = download::find_executable_recursive(&stage, exe)
            .ok_or("Verified archive does not contain crispasr.exe")?;
        download::make_executable(&staged).map_err(|e| e.to_string())?;
        emit("Checking replacement engine".into());
        let actual = candidate_version(&staged)?;
        if !crispasr_version(&actual).is_some_and(|version| version_matches(version, &expected_tag))
            || crispasr_asset_name(&actual)? != asset_name
        {
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
        match cleanup_backup_for(&backup, &current, BackupKind::Crispasr) {
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

fn whisper_package_is_current(
    installed: &str,
    tag: &str,
    directory: &Path,
    package: &super::whisper::PackageIdentity,
) -> bool {
    version_matches(installed, tag)
        && (!package.is_gpu() || super::whisper::installation_matches(directory, package))
}

fn crispasr_version(output: &str) -> Option<&str> {
    output.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(':')?;
        (name.trim().eq_ignore_ascii_case("version") && semantic_version(value.trim()).is_some())
            .then_some(value.trim())
    })
}

fn crispasr_asset_name(version: &str) -> UpdateResult<&'static str> {
    let field = |key: &str| {
        version.lines().find_map(|line| {
            let (name, value) = line.trim().split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case(key)
                .then_some(value.trim())
        })
    };
    let backends = field("ggml backends").ok_or(
        "Cannot identify the installed CrispASR backend; select a compatible build manually",
    )?;
    let backends: Vec<_> = backends.split(',').map(str::trim).collect();
    if backends.contains(&"cuda") {
        return match field("cuda toolkit").and_then(|value| value.split_whitespace().next()) {
            Some(value) if value.starts_with("13.") => Ok("crispasr-windows-x86_64-cuda13.zip"),
            Some(value) if value.starts_with("12.8.") => Ok("crispasr-windows-x86_64-cuda.zip"),
            Some(value) if value.starts_with("12.6.") => Ok("crispasr-windows-x86_64-cuda126.zip"),
            _ => Err("Cannot identify the installed CrispASR CUDA toolkit".into()),
        };
    }
    if backends.contains(&"vulkan") {
        return Ok("crispasr-windows-x86_64-vulkan.zip");
    }
    if backends == ["cpu"] {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Ok("crispasr-windows-x86_64-cpu.zip");
        }
        return Ok("crispasr-windows-x86_64-cpu-legacy.zip");
    }
    Err("Cannot identify the installed CrispASR backend; select a compatible build manually".into())
}

fn crispasr_hardware_asset() -> &'static str {
    let cuda = download::detect_host_cuda_version();
    let capability = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| {
            text.lines()
                .next()?
                .trim()
                .split_once('.')
                .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
        });
    crispasr_hardware_asset_for(cuda, capability, cpu_has_avx2())
}

fn cpu_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn crispasr_hardware_asset_for(
    cuda: Option<(u32, u32)>,
    capability: Option<(u32, u32)>,
    avx2: bool,
) -> &'static str {
    if avx2 {
        if let (Some(cuda), Some(cap)) = (cuda, capability) {
            if cuda >= (13, 0) && cap >= (7, 5) {
                return "crispasr-windows-x86_64-cuda13.zip";
            }
            if cuda >= (12, 8) && cap >= (7, 5) {
                return "crispasr-windows-x86_64-cuda.zip";
            }
            if cuda >= (12, 6) && cap >= (6, 0) {
                return "crispasr-windows-x86_64-cuda126.zip";
            }
        }
    }
    if avx2 {
        "crispasr-windows-x86_64-cpu.zip"
    } else {
        "crispasr-windows-x86_64-cpu-legacy.zip"
    }
}

fn semantic_version(text: &str) -> Option<(u64, u64, u64)> {
    text.split_whitespace().find_map(|word| {
        let mut parts = word.trim_start_matches('v').split('.');
        let version = (
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
            parts.next()?.parse().ok()?,
        );
        if parts.next().is_some() {
            None
        } else {
            Some(version)
        }
    })
}

fn ensure_no_links(directory: &Path) -> UpdateResult<()> {
    for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_symlink() {
            return Err("Engine directory contains a symbolic link; update refused".into());
        }
        if kind.is_dir() {
            ensure_no_links(&entry.path())?;
        } else if !kind.is_file() {
            return Err("Engine directory has a special file; update refused".into());
        }
    }
    Ok(())
}

fn version_matches(output: &str, tag: &str) -> bool {
    let version = tag.trim_start_matches('v');
    !version.is_empty() && output.split_whitespace().any(|part| part == version)
}

// Delete only known old audio.cpp runtime binaries and exact duplicate files.
// Differing extra files stay in the backup instead of being silently discarded.
fn cleanup_backup(backup: &Path, current: &Path) -> UpdateResult<bool> {
    cleanup_backup_for(backup, current, BackupKind::Audio)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackupKind {
    Audio,
    Whisper,
    Crispasr,
}

fn cleanup_backup_for(backup: &Path, current: &Path, kind: BackupKind) -> UpdateResult<bool> {
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
                n.starts_with(match kind {
                    BackupKind::Whisper => ".whisper-backup-",
                    BackupKind::Audio => ".audio-backup-",
                    BackupKind::Crispasr => ".crispasr-backup-",
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
        backup_kind: BackupKind,
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
                clean(&checked, &counterpart, root, current_root, backup_kind)?;
                if std::fs::read_dir(&checked)
                    .map_err(|e| e.to_string())?
                    .next()
                    .is_none()
                {
                    std::fs::remove_dir(&checked).map_err(|e| e.to_string())?;
                }
            } else if kind.is_file() {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let known_executable = if backup_kind == BackupKind::Crispasr {
                    matches!(name.as_str(), "crispasr.exe" | "crispasr-quantize.exe")
                } else if backup_kind == BackupKind::Whisper {
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
                let known_dll = (backup_kind != BackupKind::Audio && name == "whisper.dll")
                    || (backup_kind == BackupKind::Crispasr
                        && matches!(
                            name.as_str(),
                            "crispasr.dll" | "openblas.dll" | "vcomp140.dll"
                        ))
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
                let managed_receipt =
                    backup_kind == BackupKind::Whisper && name == super::whisper::INSTALL_RECEIPT;
                if known_executable || known_dll || duplicate || managed_receipt {
                    std::fs::remove_file(&checked).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    clean(&backup, &current, &backup, &current, kind)?;
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
    candidate_output(path, "--version")
}

fn candidate_output(path: &Path, argument: &str) -> UpdateResult<String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let mut command = Command::new(path);
    command
        .arg(argument)
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
    #[test]
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[ignore = "downloads the official GPU package into target, runs --version and verifies its receipt; active engines are unchanged"]
    fn live_whisper_gpu_package_refresh_receipt() {
        let client = download::create_download_client().unwrap();
        let release = super::super::whisper::resolve_release(&client).unwrap();
        let asset = super::super::whisper::platform_asset(&release).unwrap();
        let package = super::super::whisper::PackageIdentity::from_asset(asset).unwrap();
        assert!(
            package.is_gpu(),
            "Requires a compatible NVIDIA driver and published GPU package"
        );
        let tag = release["tag_name"].as_str().unwrap();
        let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
        let root = target.join(format!(
            "whisper-gpu-live-{}",
            crate::setup::generate_api_key()
        ));
        let bin = root.join("bin");
        let installed = format!("whisper.cpp version: {}", tag.trim_start_matches('v'));
        assert!(!whisper_package_is_current(
            &installed,
            tag,
            &bin.join("whisper-cpp"),
            &package
        ));
        let (tx, _rx) = std::sync::mpsc::channel();
        let stop = AtomicBool::new(false);
        stage_engine_install(
            bin.clone(),
            "whisper-cpp",
            EngineKind::WhisperCpp,
            &stop,
            &tx,
            |stage| {
                super::super::whisper::WhisperCppEngine::install_release(
                    &client,
                    &release,
                    stage.join("whisper-cpp"),
                )
                .map_err(|e| e.to_string())
            },
            |binary| validate_install_binary(EngineKind::WhisperCpp, binary),
        )
        .unwrap();
        let current = bin.join("whisper-cpp");
        let binary = download::find_executable_recursive(&current, "whisper-cli.exe").unwrap();
        let version = candidate_version(&binary).unwrap();
        assert!(whisper_package_is_current(
            &version, tag, &current, &package
        ));
        assert!(super::super::whisper::installation_matches(
            &current, &package
        ));
        println!(
            "Verified GPU package {} · {} · staged install {}",
            package.archive,
            package.sha256,
            current.display()
        );
        // Retain the isolated package for review/reuse, not in the active bin directory.
    }

    fn root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rune-engine-update-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        root
    }
    #[test]
    fn litert_install_requires_a_verified_dll_and_preserves_existing_runtime_files() {
        let root = root();
        let dll = root.join("libLiteRt.dll");
        assert!(incomplete_installation(&root, EngineKind::Litert).unwrap());
        std::fs::write(&dll, "unverified runtime").unwrap();
        assert!(
            validate_install_binary(EngineKind::Litert, &dll)
                .unwrap_err()
                .contains("checksum mismatch")
        );
        assert!(incomplete_installation(&root, EngineKind::Litert).is_err());
        assert_eq!(std::fs::read_to_string(&dll).unwrap(), "unverified runtime");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[ignore = "Downloads the pinned official Windows x64 LiteRT runtime into a temporary fixture"]
    fn live_litert_install_is_staged_verified_and_reusable() {
        assert!(
            std::env::var_os("LITERT_LIBRARY_PATH").is_none(),
            "Run without a runtime override to check managed installation"
        );
        let root = root();
        let bin = root.join("bin");
        let (tx, _) = std::sync::mpsc::channel();
        let result = stage_engine_install(
            bin.clone(),
            "litert",
            EngineKind::Litert,
            &AtomicBool::new(false),
            &tx,
            |stage| {
                download::with_install_bin(stage, || {
                    crate::setup::install_engine(EngineKind::Litert)
                })
            },
            |path| validate_install_binary(EngineKind::Litert, path),
        )
        .unwrap();
        assert!(result.contains("Installed litert"));
        let dll = bin.join("litert/libLiteRt.dll");
        assert_eq!(
            super::super::litert::runtime_version(&dll).as_deref(),
            Some("LiteRT 2.3.0 CPU")
        );
        let reused = download::with_install_bin(bin.clone(), || {
            crate::setup::install_engine(EngineKind::Litert)
        })
        .unwrap();
        assert_eq!(reused, std::fs::canonicalize(&dll).unwrap());
        assert_eq!(
            std::fs::read_dir(&bin).unwrap().count(),
            1,
            "No staging directory left behind"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installation_publish_refuses_existing_empty_directory() {
        let root = root();
        let source = root.join("stage");
        let destination = root.join("existing");
        std::fs::create_dir(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(source.join("engine"), "new").unwrap();
        assert!(publish_new_engine(&source, &destination).is_err());
        assert!(source.join("engine").exists());
        assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn install_failure_cleans_staging_retry_succeeds_and_existing_files_survive() {
        let root = root();
        let bin = root.join("bin");
        let stop = AtomicBool::new(false);
        let (tx, _) = std::sync::mpsc::channel();
        let install = |stage: PathBuf| {
            let dir = stage.join("whisper-cpp");
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join("whisper-cli.exe"), "fixture").unwrap();
            Ok(dir.join("whisper-cli.exe"))
        };
        let failed = stage_engine_install(
            bin.clone(),
            "whisper-cpp",
            EngineKind::WhisperCpp,
            &stop,
            &tx,
            |stage| {
                install(stage)?;
                Err("download failed".into())
            },
            |_| Ok(()),
        );
        assert!(failed.is_err());
        assert_eq!(std::fs::read_dir(&bin).unwrap().count(), 0);
        assert!(
            stage_engine_install(
                bin.clone(),
                "whisper-cpp",
                EngineKind::WhisperCpp,
                &stop,
                &tx,
                install,
                |_| Ok(())
            )
            .is_ok()
        );
        assert_eq!(
            std::fs::read_to_string(bin.join("whisper-cpp/whisper-cli.exe")).unwrap(),
            "fixture"
        );
        assert!(
            stage_engine_install(
                bin.clone(),
                "whisper-cpp",
                EngineKind::WhisperCpp,
                &stop,
                &tx,
                |_| panic!("must not overwrite"),
                |_| Ok(())
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(bin.join("whisper-cpp/whisper-cli.exe")).unwrap(),
            "fixture"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_install_retries_preserve_files_and_only_publish_validated_packages() {
        let root = root();
        let bin = root.join("bin");
        let current = bin.join("whisper-cpp");
        std::fs::create_dir_all(current.join("nested")).unwrap();
        std::fs::write(current.join("nested/custom.txt"), "keep").unwrap();
        std::fs::write(current.join("old.dll"), "incomplete runtime").unwrap();
        let stop = AtomicBool::new(false);
        let (tx, _) = std::sync::mpsc::channel();
        let install = |stage: PathBuf| {
            let dir = stage.join("whisper-cpp");
            std::fs::create_dir(&dir).unwrap();
            let path = dir.join("whisper-cli.exe");
            std::fs::write(&path, "verified fixture").unwrap();
            Ok(path)
        };
        for failure in ["download", "validation", "cancel"] {
            stop.store(false, Ordering::Relaxed);
            let result = stage_engine_install(
                bin.clone(),
                "whisper-cpp",
                EngineKind::WhisperCpp,
                &stop,
                &tx,
                |stage| {
                    let path = install(stage)?;
                    if failure == "download" {
                        return Err("fixture download failure".into());
                    }
                    Ok(path)
                },
                |_| {
                    if failure == "validation" {
                        return Err("fixture validation failure".into());
                    }
                    if failure == "cancel" {
                        stop.store(true, Ordering::Relaxed);
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{failure}");
            assert_eq!(
                std::fs::read_to_string(current.join("nested/custom.txt")).unwrap(),
                "keep"
            );
            assert_eq!(std::fs::read_dir(&bin).unwrap().count(), 1);
        }
        stop.store(false, Ordering::Relaxed);
        let result = stage_engine_install(
            bin.clone(),
            "whisper-cpp",
            EngineKind::WhisperCpp,
            &stop,
            &tx,
            install,
            |_| Ok(()),
        )
        .unwrap();
        let backup = std::fs::read_dir(&bin)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(".whisper-cpp-incomplete-")
            })
            .unwrap();
        assert!(result.contains(&backup.display().to_string()));
        assert_eq!(
            std::fs::read_to_string(backup.join("nested/custom.txt")).unwrap(),
            "keep"
        );
        assert_eq!(
            std::fs::read_to_string(backup.join("old.dll")).unwrap(),
            "incomplete runtime"
        );
        assert!(!current.join("old.dll").exists());
        assert!(current.join("whisper-cli.exe").exists());
        // An empty folder is eligible, but a nested executable is an installed engine.
        let empty = bin.join("empty");
        std::fs::create_dir(&empty).unwrap();
        assert!(incomplete_installation(&empty, EngineKind::WhisperCpp).unwrap());
        std::fs::create_dir(empty.join("nested")).unwrap();
        std::fs::write(empty.join("nested/whisper-cli"), "existing").unwrap();
        assert!(incomplete_installation(&empty, EngineKind::WhisperCpp).is_err());
        let file = bin.join("file");
        std::fs::write(&file, "keep").unwrap();
        assert!(incomplete_installation(&file, EngineKind::WhisperCpp).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_install_publish_restores_original_and_never_overwrites_backup() {
        let root = root();
        let current = root.join("current");
        let source = root.join("missing-stage");
        let backup = root.join("backup");
        std::fs::create_dir(&current).unwrap();
        std::fs::write(current.join("custom.txt"), "keep").unwrap();
        assert!(
            publish_repaired_engine(&source, &current, &backup)
                .unwrap_err()
                .contains("restored")
        );
        assert!(current.join("custom.txt").exists());
        assert!(!backup.exists());
        std::fs::create_dir(&backup).unwrap();
        std::fs::write(backup.join("other.txt"), "other").unwrap();
        assert!(publish_repaired_engine(&source, &current, &backup).is_err());
        assert!(current.join("custom.txt").exists());
        assert!(backup.join("other.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn llama_update_requires_the_selected_release_and_exact_binary_build() {
        let stable = serde_json::json!({"tag_name":"v0.8.0", "rune_build_tag":"b12100"});
        assert_eq!(
            checked_llama_build(&stable, "v0.8.0", Some("b12100")).unwrap(),
            12100
        );
        for (tag, build) in [
            ("v0.7.0", Some("b12100")),
            ("v0.8.0", Some("b12101")),
            ("v0.8.0", None),
        ] {
            assert!(checked_llama_build(&stable, tag, build).is_err());
        }
        assert_eq!(
            checked_llama_build(&serde_json::json!({"tag_name":"b12100"}), "b12100", None).unwrap(),
            12100
        );
        assert!(
            checked_llama_build(&serde_json::json!({"tag_name":"v0.8.0"}), "v0.8.0", None).is_err()
        );
        for (output, build) in [
            (
                "version: 0.7.0-dev (build 11669, commit 620d2225c)",
                Some(11669),
            ),
            (
                "version: 0.8.0-dev (build 12100, commit abcdef123)",
                Some(12100),
            ),
            (
                "version: 0.8.0-dev (build 121000, commit abcdef123)",
                Some(121000),
            ),
            ("version: 0.8.0", None),
            ("build invalid", None),
        ] {
            assert_eq!(llama_binary_build(output), build);
        }
    }

    #[test]
    fn managed_updates_validate_preserve_extras_and_leave_original_on_failure() {
        for directory in [
            "llama-upstream",
            "llama-prism",
            "nemo-speech",
            "sd-server",
            "needle",
        ] {
            let executable = match directory {
                "nemo-speech" => "nemo-speech.exe",
                "sd-server" => "sd-server.exe",
                "needle" => "needle.exe",
                _ => "llama-server.exe",
            };
            let root = root();
            let bin = root.join("bin");
            let current = bin.join(directory);
            std::fs::create_dir_all(&current).unwrap();
            std::fs::write(current.join(executable), "old engine").unwrap();
            std::fs::write(current.join("ggml-old.dll"), "old runtime").unwrap();
            std::fs::write(current.join("custom.json"), "custom configuration").unwrap();
            let sibling = bin.join(if directory == "llama-prism" {
                "llama-upstream"
            } else {
                "llama-prism"
            });
            std::fs::create_dir(&sibling).unwrap();
            std::fs::write(sibling.join("llama-server.exe"), "untouched sibling").unwrap();
            if directory == "needle" {
                std::fs::write(current.join(".needle-install.json"), "old receipt").unwrap();
            }
            let (tx, _) = std::sync::mpsc::channel();
            let stop = AtomicBool::new(false);
            for failure in ["download", "validation", "external", "cancel", "none"] {
                stop.store(false, Ordering::Relaxed);
                let result = stage_engine_update(
                    bin.clone(),
                    directory,
                    &stop,
                    &tx,
                    |stage| {
                        if directory == "needle" {
                            std::fs::write(stage.join(".needle-install.json"), "new receipt")
                                .unwrap();
                        }
                        let path = stage.join(executable);
                        std::fs::write(&path, "new engine").unwrap();
                        std::fs::write(stage.join("ggml-new.dll"), "new runtime").unwrap();
                        match failure {
                            "download" => Err("fixture download failure".into()),
                            "external" => Ok(current.join(executable)),
                            _ => Ok(path),
                        }
                    },
                    |_| {
                        if failure == "validation" {
                            return Err("wrong build".into());
                        }
                        if failure == "cancel" {
                            stop.store(true, Ordering::Relaxed);
                        }
                        Ok(())
                    },
                );
                if failure != "none" {
                    assert!(result.is_err(), "{failure}");
                    assert_eq!(
                        std::fs::read_to_string(current.join(executable)).unwrap(),
                        "old engine"
                    );
                    assert_eq!(std::fs::read_dir(&bin).unwrap().count(), 2);
                } else {
                    let message = result.unwrap();
                    let backup = std::fs::read_dir(&bin)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .find(|path| {
                            path.file_name()
                                .unwrap()
                                .to_string_lossy()
                                .starts_with(&format!(".{directory}-backup-"))
                        })
                        .unwrap();
                    assert!(message.contains(&backup.display().to_string()));
                    assert_eq!(
                        std::fs::read_to_string(current.join(executable)).unwrap(),
                        "new engine"
                    );
                    assert_eq!(
                        std::fs::read_to_string(current.join("custom.json")).unwrap(),
                        "custom configuration"
                    );
                    assert!(current.join("ggml-new.dll").exists());
                    assert!(!current.join("ggml-old.dll").exists());
                    assert!(backup.join(executable).exists());
                    if directory == "needle" {
                        assert_eq!(
                            std::fs::read_to_string(current.join(".needle-install.json")).unwrap(),
                            "new receipt"
                        );
                        assert_eq!(
                            std::fs::read_to_string(backup.join(".needle-install.json")).unwrap(),
                            "old receipt"
                        );
                    }
                    assert!(backup.join("ggml-old.dll").exists());
                }
            }
            assert_eq!(
                std::fs::read_to_string(sibling.join("llama-server.exe")).unwrap(),
                "untouched sibling"
            );
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn prism_updates_require_matching_release_build_and_commit() {
        let tag = "prism-b12128-d7b2896";
        let release = serde_json::json!({"tag_name":tag});
        let (build, commit) = checked_prism_build(&release, tag, None).unwrap();
        assert_eq!(build, 12128);
        assert_eq!(commit, "d7b2896");
        assert!(checked_prism_build(&release, "prism-b12129-d7b2896", None).is_err());
        assert!(checked_prism_build(&release, tag, Some("b12128")).is_err());
        for invalid in [
            "b12128",
            "prism-b12128",
            "prism-b12128-bad",
            "prism-b1-zzzzzzz",
            "prism-b-abcdef1",
            "prism-b1-abcdef1/extra",
            "prism-b18446744073709551616-abcdef1",
        ] {
            assert!(
                checked_prism_build(&serde_json::json!({"tag_name":invalid}), invalid, None)
                    .is_err(),
                "{invalid}"
            );
        }
        for (output, matches) in [
            ("0.2.0-dev (build 12128, commit d7b2896aa)", true),
            ("0.2.0-dev (build 12128, commit D7B2896)", true),
            ("0.2.0-dev (build 12129, commit d7b2896)", false),
            ("0.2.0-dev (build 12128, commit c9c34bc9c)", false),
            ("0.2.0-dev (build 12128)", false),
            ("0.2.0-dev (build 12128, commit d7b)", false),
        ] {
            assert_eq!(
                llama_version_matches(output, build, Some(&commit)),
                matches,
                "{output}"
            );
        }
    }

    #[test]
    fn nemo_updates_check_release_identity_and_exact_engine_version() {
        let release = serde_json::json!({"tag_name":"v0.3.0"});
        assert_eq!(checked_nemo_release(&release, "v0.3.0").unwrap(), (0, 3, 0));
        assert!(checked_nemo_release(&release, "v0.4.0").is_err());
        for invalid in [
            "",
            "nightly",
            "v0.3",
            "v0.3.0-dev",
            "v0.3.0.1",
            "release v0.3.0",
        ] {
            assert!(
                checked_nemo_release(&serde_json::json!({"tag_name":invalid}), invalid).is_err()
            );
        }
        for (output, expected) in [
            ("nemo-speech 0.3.0\n", Some((0, 3, 0))),
            ("nemo-speech 0.2.0", Some((0, 2, 0))),
            ("nemo-speech 0.3.1", Some((0, 3, 1))),
            ("nemo-speech 0.3.00", Some((0, 3, 0))),
            ("other-engine 0.3.0", None),
            ("nemo-speech 0.3.0-dev", None),
            ("nemo-speech 0.3.0.1", None),
            ("nemo-speech unrelated 0.3.0", None),
        ] {
            assert_eq!(nemo_binary_version(output), expected, "{output}");
        }
    }

    #[test]
    fn sd_updates_require_verified_forward_ancestry() {
        let base = "a".repeat(40);
        let fixture = serde_json::json!({
            "base_commit":{"sha":base}, "merge_base_commit":{"sha":base},
            "status":"ahead", "ahead_by":2, "behind_by":0,
        });
        for (status, ahead, behind, expected) in [
            ("ahead", 2, 0, Some(true)),
            ("identical", 0, 0, Some(false)),
            ("behind", 0, 2, None),
            ("diverged", 2, 1, None),
            ("ahead", 2, 1, None),
            ("ahead", 0, 0, None),
            ("identical", 1, 0, None),
            ("unknown", 2, 0, None),
        ] {
            let mut value = fixture.clone();
            value["status"] = serde_json::json!(status);
            value["ahead_by"] = serde_json::json!(ahead);
            value["behind_by"] = serde_json::json!(behind);
            assert_eq!(
                sd_update_is_forward("aaaaaaa", &value).ok(),
                expected,
                "{status}/{ahead}/{behind}"
            );
        }
        for (field, value) in [
            ("base_commit", serde_json::json!({"sha":"b".repeat(40)})),
            (
                "merge_base_commit",
                serde_json::json!({"sha":"b".repeat(40)}),
            ),
            ("base_commit", serde_json::json!({"sha":"aaaaaaa"})),
            ("ahead_by", serde_json::Value::Null),
            ("behind_by", serde_json::Value::Null),
        ] {
            let mut response = fixture.clone();
            response[field] = value;
            assert!(
                sd_update_is_forward("aaaaaaa", &response).is_err(),
                "{field}"
            );
        }
        assert!(sd_update_is_forward("", &fixture).is_err());
        assert!(sd_update_is_forward("aaaaaaa", &serde_json::json!({})).is_err());
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
        std::fs::write(
            current.join(super::super::whisper::INSTALL_RECEIPT),
            "old package receipt",
        )
        .unwrap();
        std::fs::write(stage.join("whisper-cli.exe"), "new").unwrap();
        std::fs::write(stage.join("whisper.dll"), "new library").unwrap();
        std::fs::write(
            stage.join(super::super::whisper::INSTALL_RECEIPT),
            "new package receipt",
        )
        .unwrap();
        preserve_extras(&current, &stage).unwrap();
        publish(&current, &stage, &backup).unwrap();
        assert!(!cleanup_backup_for(&backup, &current, BackupKind::Whisper).unwrap());
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
        assert!(!backup.join(super::super::whisper::INSTALL_RECEIPT).exists());
        assert_eq!(
            std::fs::read_to_string(current.join(super::super::whisper::INSTALL_RECEIPT)).unwrap(),
            "new package receipt"
        );
        assert!(backup.join("custom.dll").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn version_matching_is_exact_for_whisper() {
        assert!(version_matches("whisper.cpp version: 1.9.2", "v1.9.2"));
        assert!(!version_matches("whisper.cpp version: 1.9.20", "v1.9.2"));
        assert!(!version_matches("whisper.cpp version: 1.9.2-dev", "v1.9.2"));
        assert!(semantic_version("whisper.cpp version: 1.9.3") > semantic_version("v1.9.2"));
        assert_eq!(semantic_version("whisper.cpp version: 1.9.2-dev"), None);
    }

    #[test]
    fn crispasr_selects_only_matching_known_backends() {
        let cuda = "version : 0.8.43\nggml backends : cpu,cuda\ncuda toolkit : 13.0.48 (build/link target)\n";
        assert_eq!(crispasr_version(cuda), Some("0.8.43"));
        assert_eq!(
            crispasr_asset_name(cuda).unwrap(),
            "crispasr-windows-x86_64-cuda13.zip"
        );
        assert_eq!(
            crispasr_asset_name("ggml backends : cpu,vulkan").unwrap(),
            "crispasr-windows-x86_64-vulkan.zip"
        );
        assert!(crispasr_asset_name("ggml backends : cpu,cuda\ncuda toolkit: unknown").is_err());
        assert!(crispasr_asset_name("version : 0.8.43").is_err());
        assert!(crispasr_version("version : malformed").is_none());
        assert_eq!(
            crispasr_hardware_asset_for(Some((13, 0)), Some((8, 9)), true),
            "crispasr-windows-x86_64-cuda13.zip"
        );
        assert_eq!(
            crispasr_hardware_asset_for(Some((13, 0)), Some((6, 1)), true),
            "crispasr-windows-x86_64-cuda126.zip"
        );
        assert_eq!(
            crispasr_hardware_asset_for(Some((12, 8)), Some((8, 0)), true),
            "crispasr-windows-x86_64-cuda.zip"
        );
        assert_eq!(
            crispasr_hardware_asset_for(None, None, false),
            "crispasr-windows-x86_64-cpu-legacy.zip"
        );
    }

    #[test]
    fn crispasr_replacement_preserves_custom_files_and_cleans_known_runtime() {
        let root = root();
        let current = root.join("crispasr");
        let stage = root.join("stage");
        let backup = root.join(".crispasr-backup-fixture");
        std::fs::create_dir(&current).unwrap();
        std::fs::create_dir(&stage).unwrap();
        std::fs::write(current.join("crispasr.exe"), "old").unwrap();
        std::fs::write(current.join("crispasr.dll"), "old library").unwrap();
        std::fs::write(current.join("config.json"), "user config").unwrap();
        std::fs::write(current.join("custom.dll"), "user library").unwrap();
        std::fs::write(stage.join("crispasr.exe"), "new").unwrap();
        std::fs::write(stage.join("crispasr.dll"), "new library").unwrap();
        preserve_extras(&current, &stage).unwrap();
        publish(&current, &stage, &backup).unwrap();
        assert!(!cleanup_backup_for(&backup, &current, BackupKind::Crispasr).unwrap());
        assert_eq!(
            std::fs::read_to_string(current.join("config.json")).unwrap(),
            "user config"
        );
        assert_eq!(
            std::fs::read_to_string(current.join("crispasr.exe")).unwrap(),
            "new"
        );
        assert!(!backup.join("crispasr.exe").exists());
        assert!(!backup.join("crispasr.dll").exists());
        assert!(backup.join("custom.dll").exists());
        std::fs::remove_dir_all(root).unwrap();
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
