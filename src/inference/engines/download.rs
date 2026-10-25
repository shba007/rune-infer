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
        .user_agent("rune-infer/0.1.0 (Windows NT 10.0; Win64; x64)")
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(15));

    if let Ok(v4) = "0.0.0.0".parse::<std::net::IpAddr>() {
        builder = builder.local_address(v4);
    }

    Ok(builder.build()?)
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

pub fn download_to_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest_path: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = dest_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("file");

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
    let is_tar = filename.ends_with(".tar.gz") || filename.ends_with(".tgz");
    let is_zip = filename.ends_with(".zip");

    report(DownloadProgress {
        filename: filename.into(),
        extracting: true,
        downloaded: 0,
        total: None,
        bytes_per_second: 0.0,
    });
    tracing::info!("{} Extracting {}...", prefix, filename);

    let tar_flag = if is_tar { "-xzf" } else { "-xf" };
    if let Ok(status) = Command::new("tar")
        .args([
            tar_flag,
            archive_file.to_str().unwrap(),
            "-C",
            dest_dir.to_str().unwrap(),
        ])
        .output()
    {
        if status.status.success() {
            return Ok(());
        }
    }

    if is_zip {
        #[cfg(windows)]
        {
            let ps_cmd = format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                archive_file.display(),
                dest_dir.display()
            );
            let status = Command::new("powershell")
                .args(["-NoProfile", "-Command", &ps_cmd])
                .output();
            if let Ok(s) = status {
                if s.status.success() {
                    return Ok(());
                }
            }
        }

        #[cfg(unix)]
        {
            let status = Command::new("unzip")
                .args([
                    "-q",
                    "-o",
                    archive_file.to_str().unwrap(),
                    "-d",
                    dest_dir.to_str().unwrap(),
                ])
                .output();
            if let Ok(s) = status {
                if s.status.success() {
                    return Ok(());
                }
            }
        }
    }

    Err(format!("Failed to extract archive: {}", archive_file.display()).into())
}

pub fn download_and_extract(
    client: &reqwest::blocking::Client,
    url: &str,
    dest_dir: &Path,
    archive_name: &str,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let temp_archive = dest_dir.join(archive_name);
    download_to_file(client, url, &temp_archive, prefix)?;
    extract_archive(&temp_archive, dest_dir, prefix)?;
    let _ = std::fs::remove_file(&temp_archive);
    Ok(())
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
