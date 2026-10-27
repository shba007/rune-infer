use sha2::{Digest, Sha256};
use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub filename: String,
    pub extracting: bool,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub bytes_per_second: f64,
}

thread_local! {
    static INSTALL_BIN: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn engine_bin() -> PathBuf {
    INSTALL_BIN.with(|root| {
        root.borrow()
            .clone()
            .unwrap_or_else(|| PathBuf::from("bin"))
    })
}

pub(crate) fn with_install_bin<T>(path: PathBuf, work: impl FnOnce() -> T) -> T {
    struct Restore(Option<PathBuf>);
    impl Drop for Restore {
        fn drop(&mut self) {
            INSTALL_BIN.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let _restore = Restore(INSTALL_BIN.with(|slot| slot.replace(Some(path))));
    work()
}

type ProgressCallback = Box<dyn FnMut(DownloadProgress)>;
thread_local! {
    static PROGRESS: std::cell::RefCell<Option<ProgressCallback>> = const { std::cell::RefCell::new(None) };
}

/// Scope download reporting to this setup worker; runtime downloads remain independent.
pub fn with_progress<T>(
    callback: impl FnMut(DownloadProgress) + 'static,
    work: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<ProgressCallback>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRESS.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let _restore = Restore(PROGRESS.with(|slot| slot.replace(Some(Box::new(callback)))));
    work()
}

fn report(progress: DownloadProgress) {
    PROGRESS.with(|slot| {
        if let Some(callback) = slot.borrow_mut().as_mut() {
            callback(progress);
        }
    });
}

pub fn create_download_client() -> Result<reqwest::blocking::Client, Box<dyn Error>> {
    let mut builder = reqwest::blocking::Client::builder()
        .https_only(true)
        .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(15));

    if let Ok(v4) = "0.0.0.0".parse::<std::net::IpAddr>() {
        builder = builder.local_address(v4);
    }

    Ok(builder.build()?)
}

/// Only stable semantic-versioned GitHub releases are eligible for automatic installation.
pub fn is_stable_release(release: &serde_json::Value) -> bool {
    if release["draft"].as_bool() != Some(false) || release["prerelease"].as_bool() != Some(false) {
        return false;
    }
    let Some(tag) = release["tag_name"].as_str() else {
        return false;
    };
    let tag = tag.strip_prefix('v').unwrap_or(tag);
    let pieces: Vec<_> = tag.split('.').collect();
    pieces.len() == 3
        && pieces
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

pub fn stable_release(value: &serde_json::Value) -> Result<&serde_json::Value, Box<dyn Error>> {
    let release = if let Some(releases) = value.as_array() {
        releases.iter().find(|r| is_stable_release(r))
    } else if is_stable_release(value) {
        Some(value)
    } else {
        None
    };
    release.ok_or_else(|| "No verified stable production release was found; install a verified stable engine manually or retry later.".into())
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn download_to_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest_path: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = dest_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("file");
    let filename = filename.split(".part-").next().unwrap_or(filename);

    let mut resp = client.get(url).send()?.error_for_status()?;
    let total_size = resp.content_length().unwrap_or(0);

    tracing::info!(
        "{} Downloading {} ({})...",
        prefix,
        filename,
        if total_size > 0 {
            format_bytes(total_size)
        } else {
            "unknown size".to_string()
        }
    );

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(dest_path)?;
        let mut buffer = [0u8; 65536];
        let mut downloaded = 0u64;
        let started = Instant::now();
        let mut last_update = Instant::now();
        let update = |downloaded| {
            report(DownloadProgress {
                filename: filename.to_owned(),
                extracting: false,
                downloaded,
                total: (total_size > 0).then_some(total_size),
                bytes_per_second: downloaded as f64 / started.elapsed().as_secs_f64().max(0.001),
            })
        };
        update(0);
        loop {
            let n = resp.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            downloaded += n as u64;
            if last_update.elapsed() >= Duration::from_millis(100) {
                update(downloaded);
                last_update = Instant::now();
            }
        }
        if total_size > 0 && downloaded != total_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Incomplete engine download; retry.",
            ));
        }
        file.flush()?;
        update(downloaded);
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(dest_path);
    }
    result?;

    Ok(())
}

pub fn extract_archive(
    archive_file: &Path,
    dest_dir: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = archive_file
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    report(DownloadProgress {
        filename: filename.into(),
        extracting: true,
        downloaded: 0,
        total: None,
        bytes_per_second: 0.0,
    });
    tracing::info!("{} Extracting {}...", prefix, filename);
    validate_archive(archive_file)?;
    let stage = dest_dir.join(format!(
        ".extract-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&stage)?;
    let result = (|| -> Result<(), Box<dyn Error>> {
        let output = crate::inference::process::system_command("tar")
            .arg("-xf")
            .arg(archive_file)
            .arg("-C")
            .arg(&stage)
            .args(["--no-same-owner", "--no-same-permissions"])
            .output()?;
        if !output.status.success() {
            return Err("Engine archive extraction failed.".into());
        }
        publish_extracted(&stage, dest_dir)?;
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&stage);
    result
}

#[doc(hidden)]
pub fn validate_archive(archive_file: &Path) -> Result<(), Box<dyn Error>> {
    // tar is invoked directly, without a shell. No archive Rust dependency exists.
    let names = crate::inference::process::system_command("tar")
        .arg("-tf")
        .arg(archive_file)
        .output()?;
    if !names.status.success() {
        return Err("Cannot list engine archive; install the system tar utility.".into());
    }
    let names = String::from_utf8(names.stdout)?;
    for entry in names.lines() {
        validate_archive_path(entry)?;
    }
    let types = crate::inference::process::system_command("tar")
        .arg("-tvf")
        .arg(archive_file)
        .output()?;
    if !types.status.success() {
        return Err("Cannot inspect archive entry types.".into());
    }
    let types = String::from_utf8(types.stdout)?;
    if types.lines().count() != names.lines().count()
        || types
            .lines()
            .any(|line| !matches!(line.as_bytes().first(), Some(b'-' | b'd')))
    {
        return Err("Archive contains links, special files, or ambiguous entry names; installation refused.".into());
    }
    Ok(())
}

pub fn download_and_extract(
    client: &reqwest::blocking::Client,
    url: &str,
    dest_dir: &Path,
    archive_name: &str,
    prefix: &str,
    expected_sha256: &str,
) -> Result<(), Box<dyn Error>> {
    validate_archive_path(archive_name)?;
    if archive_name.contains('/') || matches!(archive_name, "." | "..") {
        return Err("Invalid archive filename.".into());
    }
    let temp_archive = dest_dir.join(archive_name);
    download_verified(client, url, &temp_archive, prefix, expected_sha256)?;
    let result = extract_archive(&temp_archive, dest_dir, prefix);
    let _ = std::fs::remove_file(&temp_archive);
    result
}

fn validate_archive_path(name: &str) -> Result<(), Box<dyn Error>> {
    if name.is_empty()
        || name.starts_with('/')
        || name.contains(['\\', ':'])
        || name.chars().any(|c| c.is_control())
        || name.split('/').any(|part| part == "..")
    {
        return Err("Unsafe archive entry path; installation refused.".into());
    }
    Ok(())
}

fn publish_extracted(source: &Path, target: &Path) -> Result<(), Box<dyn Error>> {
    fn collect(
        source: &Path,
        target: &Path,
        files: &mut Vec<(PathBuf, PathBuf)>,
    ) -> Result<(), Box<dyn Error>> {
        if std::fs::symlink_metadata(target)
            .is_ok_and(|m| !m.is_dir() || m.file_type().is_symlink())
        {
            return Err("Installation directory is not a regular directory.".into());
        }
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            let dest = target.join(entry.file_name());
            let kind = entry.file_type()?;
            if std::fs::symlink_metadata(&dest).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err(
                    "Installation path contains a symbolic link; refusing to overwrite.".into(),
                );
            }
            if kind.is_dir() {
                collect(&entry.path(), &dest, files)?;
            } else if kind.is_file() {
                if dest.exists() {
                    if !dest.is_file() || file_sha256(&dest)? != file_sha256(&entry.path())? {
                        return Err(format!("Conflicting installed file {}; preserve it or use an empty engine directory.", dest.display()).into());
                    }
                } else {
                    files.push((entry.path(), dest));
                }
            } else {
                return Err("Extracted archive contains a non-regular entry.".into());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    collect(source, target, &mut files)?;
    let mut installed = Vec::new();
    for (source, dest) in files {
        let result = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(dest.parent().expect("joined path has parent"))?;
            std::fs::rename(source, &dest)
        })();
        if let Err(error) = result {
            for path in installed {
                let _ = std::fs::remove_file(path);
            }
            return Err(error.into());
        }
        installed.push(dest);
    }
    Ok(())
}

pub fn file_sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub fn download_verified(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    prefix: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Missing or invalid published SHA-256; verified installation required.".into());
    }
    if reqwest::Url::parse(url)?.scheme() != "https" {
        return Err("Engine downloads require HTTPS.".into());
    }
    let part = dest.with_file_name(format!(
        "{}.part-{}-{}",
        dest.file_name()
            .ok_or("Download destination has no filename")?
            .to_string_lossy(),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let result = (|| -> Result<(), Box<dyn Error>> {
        download_to_file(client, url, &part, prefix)?;
        verify_sha256(&part, expected)?;
        if dest.exists() {
            return Err(format!(
                "Download destination {} already exists; refusing to overwrite.",
                dest.display()
            )
            .into());
        }
        std::fs::rename(&part, dest)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(part);
    result
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), Box<dyn Error>> {
    if !file_sha256(path)?.eq_ignore_ascii_case(expected) {
        return Err("Engine SHA-256 mismatch; installation refused.".into());
    }
    Ok(())
}

pub fn asset_sha256(assets: &[serde_json::Value], name: &str) -> Result<String, Box<dyn Error>> {
    let digest = assets.iter().find(|a| a["name"].as_str() == Some(name))
        .and_then(|a| a["digest"].as_str()).and_then(|s| s.strip_prefix("sha256:"))
        .filter(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or("Release asset has no published SHA-256; retry a newer release or explicitly configure your own engine binary.")?;
    Ok(digest.to_owned())
}

pub fn find_executable_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().map(|n| n == name).unwrap_or(false) {
                return Some(path);
            } else if path.is_dir() {
                if let Some(found) = find_executable_recursive(&path, name) {
                    return Some(found);
                }
            }
        }
    }
    None
}

pub fn flatten_dlls(base_dir: &Path) {
    #[cfg(windows)]
    {
        if let Ok(entries) = std::fs::read_dir(base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(sub_entries) = std::fs::read_dir(&path) {
                        for sub in sub_entries.flatten() {
                            let sub_path = sub.path();
                            if sub_path.is_file() {
                                if let Some(ext) = sub_path.extension() {
                                    if ext.eq_ignore_ascii_case("dll") {
                                        if let Some(fname) = sub_path.file_name() {
                                            let dest = base_dir.join(fname);
                                            if !dest.exists() {
                                                let _ = std::fs::copy(&sub_path, &dest);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = base_dir;
}

pub fn make_executable(path: &Path) -> Result<(), Box<dyn Error>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(path, perms)?;
        }
    }
    let _ = path;
    Ok(())
}

pub fn detect_host_cuda_version() -> Option<(u32, u32)> {
    let output = Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let idx = text
        .find("CUDA UMD Version:")
        .map(|i| i + "CUDA UMD Version:".len())
        .or_else(|| {
            text.find("CUDA Version:")
                .map(|i| i + "CUDA Version:".len())
        })?;
    let rest = &text[idx..];
    let ver_str = rest.split_whitespace().next()?;
    parse_cuda_version(ver_str)
}

pub fn parse_cuda_version(s: &str) -> Option<(u32, u32)> {
    let clean = s.trim_start_matches("cu").trim_start_matches('v');
    let mut parts = clean.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

pub fn matches_arch(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    if cfg!(target_arch = "x86_64") {
        (lower.contains("x64") || lower.contains("x86_64") || lower.contains("amd64"))
            && !lower.contains("arm64")
    } else if cfg!(target_arch = "aarch64") {
        lower.contains("arm64") || lower.contains("aarch64")
    } else {
        true
    }
}

pub fn extract_cuda_version(filename: &str) -> Option<(u32, u32)> {
    let lower = filename.to_lowercase();
    for (idx, _) in lower.match_indices("cuda") {
        let slice = &lower[idx + "cuda".len()..];
        let clean = slice
            .trim_start_matches('-')
            .trim_start_matches("cu")
            .trim_start_matches('_');
        if let Some(ver_part) = clean.split(|c: char| !c.is_numeric() && c != '.').next() {
            if !ver_part.is_empty() {
                let mut parts = ver_part.split('.');
                if let Some(major_str) = parts.next() {
                    if let Ok(major) = major_str.parse::<u32>() {
                        let minor: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        return Some((major, minor));
                    }
                }
            }
        }
    }
    None
}

pub fn extract_release_asset_info(
    val: &serde_json::Value,
) -> Result<(String, String), Box<dyn Error>> {
    let name = val["name"]
        .as_str()
        .ok_or("GitHub release asset is missing 'name' field")?
        .to_string();
    let url = val["browser_download_url"]
        .as_str()
        .ok_or("GitHub release asset is missing 'browser_download_url' field")?
        .to_string();
    Ok((name, url))
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn serve(response: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/asset", listener.local_addr().unwrap());
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).unwrap();
            assert!(count > 0);
            stream.write_all(response).unwrap();
        });
        (url, worker)
    }

    #[test]
    fn production_release_selection_rejects_nightly_development_and_unverified_tags() {
        let releases = serde_json::json!([
            {"tag_name":"b12345","draft":false,"prerelease":false},
            {"tag_name":"v1.0.0-nightly","draft":false,"prerelease":false},
            {"tag_name":"v1.2.3","draft":false,"prerelease":true},
            {"tag_name":"v1.0.0","draft":false,"prerelease":false}
        ]);
        assert_eq!(stable_release(&releases).unwrap()["tag_name"], "v1.0.0");
        assert!(
            stable_release(
                &serde_json::json!([{"tag_name":"b123","draft":false,"prerelease":false}])
            )
            .is_err()
        );
        assert!(!is_stable_release(
            &serde_json::json!({"tag_name":"v1.0.0"})
        ));
    }

    #[test]
    fn digest_metadata_and_archive_paths_fail_closed() {
        use serde_json::json;
        assert!(asset_sha256(&[json!({"name":"engine.zip"})], "engine.zip").is_err());
        let digest = "a".repeat(64);
        assert_eq!(
            asset_sha256(
                &[json!({"name":"engine.zip", "digest":format!("sha256:{digest}")})],
                "engine.zip"
            )
            .unwrap(),
            digest
        );
        for path in [
            "../escape",
            "x/../escape",
            "/absolute",
            "C:/absolute",
            "x\\escape",
            "x\nescape",
        ] {
            assert!(validate_archive_path(path).is_err(), "{path}");
        }
        assert!(validate_archive_path("./bin/engine.exe").is_ok());
    }

    #[test]
    fn extraction_preserves_conflicts_and_checks_hashes() {
        let root =
            std::env::temp_dir().join(format!("rune-extract-{}", crate::setup::generate_api_key()));
        let source = root.join("source");
        let target = root.join("target");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(source.join("engine"), b"test").unwrap();
        assert!(verify_sha256(&source.join("engine"), &"0".repeat(64)).is_err());
        verify_sha256(
            &source.join("engine"),
            &file_sha256(&source.join("engine")).unwrap(),
        )
        .unwrap();
        std::fs::write(source.join("new"), b"new").unwrap();
        std::fs::write(target.join("engine"), b"preserve").unwrap();
        assert!(publish_extracted(&source, &target).is_err());
        assert!(!target.join("new").exists());
        assert_eq!(std::fs::read(target.join("engine")).unwrap(), b"preserve");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn extraction_uses_literal_paths_and_publishes_regular_files() {
        let root = std::env::temp_dir().join(format!(
            "rune's-extract-{}",
            crate::setup::generate_api_key()
        ));
        let source = root.join("source");
        let target = root.join("target");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(source.join("asset.txt"), "verified fixture").unwrap();
        let archive = root.join("fixture.tar");
        assert!(
            crate::inference::process::system_command("tar")
                .arg("-cf")
                .arg(&archive)
                .arg("-C")
                .arg(&source)
                .arg("asset.txt")
                .output()
                .unwrap()
                .status
                .success()
        );
        extract_archive(&archive, &target, "test").unwrap();
        assert_eq!(
            std::fs::read_to_string(target.join("asset.txt")).unwrap(),
            "verified fixture"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn archive_links_and_traversal_are_rejected_before_extraction() {
        let root = std::env::temp_dir().join(format!(
            "rune-unsafe-archive-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let archive = root.join("unsafe.tar");
        for (name, kind) in [("../escape", b'0'), ("link", b'2'), ("hardlink", b'1')] {
            let mut header = [0u8; 512];
            header[..name.len()].copy_from_slice(name.as_bytes());
            header[100..108].copy_from_slice(b"0000644\0");
            header[108..116].copy_from_slice(b"0000000\0");
            header[116..124].copy_from_slice(b"0000000\0");
            header[124..136].copy_from_slice(b"00000000000\0");
            header[136..148].copy_from_slice(b"00000000000\0");
            header[148..156].fill(b' ');
            header[156] = kind;
            if kind != b'0' {
                header[157..166].copy_from_slice(b"../escape");
            }
            header[257..263].copy_from_slice(b"ustar\0");
            header[263..265].copy_from_slice(b"00");
            let checksum: u32 = header.iter().map(|b| u32::from(*b)).sum();
            header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
            let mut bytes = header.to_vec();
            bytes.extend([0u8; 1024]);
            std::fs::write(&archive, bytes).unwrap();
            assert!(extract_archive(&archive, &root, "test").is_err(), "{name}");
            assert!(!root.join("link").exists() && !root.join("hardlink").exists());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_bytes_and_rejects_incomplete_transfers() {
        let root = std::env::temp_dir().join(format!(
            "rune-download-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dest = root.join("engine.zip");
        let progress = Rc::new(RefCell::new(Vec::new()));
        let capture = progress.clone();
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap();
        let (url, worker) =
            serve(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ntest");
        with_progress(
            move |update| capture.borrow_mut().push(update),
            || download_to_file(&client, &url, &dest, "test"),
        )
        .unwrap();
        worker.join().unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"test");
        let updates = progress.borrow();
        assert_eq!(updates.first().unwrap().downloaded, 0);
        let done = updates.last().unwrap();
        assert_eq!(done.filename, "engine.zip");
        assert_eq!(done.downloaded, 4);
        assert_eq!(done.total, Some(4));
        assert!(done.bytes_per_second.is_finite() && done.bytes_per_second > 0.0);
        drop(updates);
        let count = progress.borrow().len();
        let (url, worker) =
            serve(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\nConnection: close\r\n\r\nshort");
        assert!(download_to_file(&client, &url, &dest, "test").is_err());
        worker.join().unwrap();
        assert!(!dest.exists());
        assert_eq!(
            progress.borrow().len(),
            count,
            "callback must be scoped to setup worker"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
