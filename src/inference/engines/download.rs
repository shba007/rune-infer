use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Creates a reusable HTTP client configured for engine binary downloads.
/// Forces IPv4 to eliminate 40-60 second IPv6 resolution timeouts on Windows.
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

/// Downloads a remote URL to a local destination with real-time progress logging.
/// Explicitly flushes and closes the file handle so no locks remain on Windows.
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

    println!(
        "{} Downloading {} ({})...",
        prefix,
        filename,
        if total_size > 0 {
            format_bytes(total_size)
        } else {
            "unknown size".to_string()
        }
    );

    {
        let mut file = std::fs::File::create(dest_path)?;
        let mut buffer = [0u8; 65536];
        let mut downloaded: u64 = 0;
        let mut last_log = Instant::now();

        while let Ok(n) = resp.read(&mut buffer) {
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            downloaded += n as u64;

            if last_log.elapsed() >= Duration::from_secs(3) {
                if total_size > 0 {
                    let pct = (downloaded as f64 / total_size as f64) * 100.0;
                    println!(
                        "{}   -> {} / {} ({:.1}%)",
                        prefix,
                        format_bytes(downloaded),
                        format_bytes(total_size),
                        pct
                    );
                } else {
                    println!("{}   -> {}", prefix, format_bytes(downloaded));
                }
                last_log = Instant::now();
            }
        }
        file.flush()?;
    } // `file` handle is dropped here, releasing the Windows file lock

    Ok(())
}

/// Unified extraction mechanism supporting .zip, .tar.gz, and .tgz.
/// Prefers `tar` if present, with fallbacks to PowerShell Expand-Archive (Windows) or unzip (Unix).
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

    println!("{} Extracting {}...", prefix, filename);

    let tar_flag = if is_tar { "-xzf" } else { "-xf" };
    if let Ok(status) = Command::new("tar")
        .args([
            tar_flag,
            archive_file.to_str().unwrap(),
            "-C",
            dest_dir.to_str().unwrap(),
        ])
        .status()
    {
        if status.success() {
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
                .status();
            if let Ok(s) = status {
                if s.success() {
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
                .status();
            if let Ok(s) = status {
                if s.success() {
                    return Ok(());
                }
            }
        }
    }

    Err(format!("Failed to extract archive: {}", archive_file.display()).into())
}

/// Downloads an archive to `dest_dir`, extracts it, and cleans up the temporary archive.
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

/// Recursively searches for an executable inside a directory.
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

/// On Windows, moves any DLL files located in immediate subdirectories up into `base_dir`.
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
}

/// Marks a binary as executable on Unix systems (chmod 0755).
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

/// Detects the host's maximum CUDA version via `nvidia-smi`.
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
