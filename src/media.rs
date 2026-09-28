use base64::Engine;
use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use crate::types::{ApiError, ErrorResponse};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone)]
pub struct S3Config {
    pub endpoint: String,
    pub access_key: String,
    pub secret_key: String,
    pub region: String,
}

impl S3Config {
    pub fn from_env() -> Option<Self> {
        let access_key = std::env::var("S3_ACCESS_KEY")
            .or_else(|_| std::env::var("RUSTFS_ACCESS_KEY"))
            .or_else(|_| std::env::var("AWS_ACCESS_KEY_ID"))
            .ok()?;
        let secret_key = std::env::var("S3_SECRET_KEY")
            .or_else(|_| std::env::var("RUSTFS_SECRET_KEY"))
            .or_else(|_| std::env::var("AWS_SECRET_ACCESS_KEY"))
            .ok()?;
        let endpoint = std::env::var("S3_ENDPOINT")
            .or_else(|_| std::env::var("RUSTFS_ENDPOINT"))
            .unwrap_or_else(|_| "http://127.0.0.1:9000".to_string());
        let region = std::env::var("S3_REGION")
            .or_else(|_| std::env::var("AWS_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        Some(Self {
            endpoint,
            access_key,
            secret_key,
            region,
        })
    }
}

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC initialization failed");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

pub fn format_iso8601_date(now: std::time::SystemTime) -> (String, String) {
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = secs % 60;
    let mins = secs / 60;
    let min = mins % 60;
    let hours = mins / 60;
    let hour = hours % 24;
    let days = hours / 24;

    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let date_str = format!("{:04}{:02}{:02}", y, m, d);
    let date_time_str = format!("{:04}{:02}{:02}T{:02}{:02}{:02}Z", y, m, d, hour, min, sec);
    (date_str, date_time_str)
}

pub fn resolve_media_source(raw_url: &str) -> String {
    let s3_conf = S3Config::from_env();
    if let Some(rest) = raw_url.strip_prefix("s3://") {
        let ep = s3_conf
            .as_ref()
            .map(|c| c.endpoint.as_str())
            .unwrap_or("http://127.0.0.1:9000");
        format!(
            "{}/{}",
            ep.trim_end_matches('/'),
            rest.trim_start_matches('/')
        )
    } else {
        raw_url.to_string()
    }
}

pub fn ensure_temp_dir(sub: &str) -> Result<PathBuf, ErrorResponse> {
    let dir = std::env::temp_dir().join("rune_infer").join(sub);
    std::fs::create_dir_all(&dir).map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Failed to create temporary directory: {e}"))
            .with_type("server_error"),
    })?;
    Ok(dir)
}

pub fn build_authenticated_get(
    client: &reqwest::blocking::Client,
    url_str: &str,
) -> Result<reqwest::blocking::RequestBuilder, ErrorResponse> {
    let req_builder = client.get(url_str);
    let s3_conf = match S3Config::from_env() {
        Some(c) => c,
        None => return Ok(req_builder),
    };

    let parsed_url = reqwest::Url::parse(url_str).map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Invalid URL '{url_str}': {e}")),
    })?;

    let is_s3 = url_str.starts_with(s3_conf.endpoint.trim_end_matches('/'))
        || url_str.contains(":9000/")
        || parsed_url
            .host_str()
            .map(|h| h.contains("s3") || h.contains("r2"))
            .unwrap_or(false);

    if !is_s3 || url_str.contains("X-Amz-Signature=") {
        return Ok(req_builder);
    }

    let host = match parsed_url.port() {
        Some(p) => format!("{}:{}", parsed_url.host_str().unwrap_or(""), p),
        None => parsed_url.host_str().unwrap_or("").to_string(),
    };
    let path = parsed_url.path();
    let query = parsed_url.query().unwrap_or("");

    let (date_stamp, amz_date) = format_iso8601_date(std::time::SystemTime::now());
    let empty_payload_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    let canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{empty_payload_hash}\nx-amz-date:{amz_date}\n");
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";

    let canonical_request = format!(
        "GET\n{path}\n{query}\n{canonical_headers}\n{signed_headers}\n{empty_payload_hash}"
    );
    let canonical_request_hash = sha256_hex(canonical_request.as_bytes());

    let credential_scope = format!("{date_stamp}/{}/s3/aws4_request", s3_conf.region);
    let string_to_sign =
        format!("AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{canonical_request_hash}");

    let k_secret = format!("AWS4{}", s3_conf.secret_key);
    let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, s3_conf.region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");

    let signature_bytes = hmac_sha256(&k_signing, string_to_sign.as_bytes());
    let signature: String = signature_bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    let auth_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
        s3_conf.access_key
    );

    Ok(req_builder
        .header("host", host)
        .header("x-amz-date", amz_date)
        .header("x-amz-content-sha256", empty_payload_hash)
        .header("Authorization", auth_header))
}

pub fn is_video_format(url_or_path: &str) -> bool {
    let lower = url_or_path.to_lowercase();
    let path_part = lower.split('?').next().unwrap_or(&lower);
    lower.starts_with("data:video/")
        || path_part.ends_with(".mp4")
        || path_part.ends_with(".mkv")
        || path_part.ends_with(".mov")
        || path_part.ends_with(".webm")
        || path_part.ends_with(".avi")
}

pub fn download_remote_to_temp_file(url: &str) -> Result<PathBuf, ErrorResponse> {
    let target_url = resolve_media_source(url);
    let temp_dir = ensure_temp_dir("downloads")?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_file = temp_dir.join(format!("rune_dl_{}_{}.mp4", std::process::id(), now));

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!("Failed to build HTTP client: {e}"))
                .with_type("server_error"),
        })?;

    let req = build_authenticated_get(&client, &target_url)?;
    let mut resp = req
        .send()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!(
                "Failed to fetch video from URL '{target_url}': {e}"
            ))
            .with_type("invalid_request_error")
            .with_code("media_fetch_error"),
        })?
        .error_for_status()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!("Remote server returned HTTP error for URL: {e}"))
                .with_type("invalid_request_error")
                .with_code("remote_media_error"),
        })?;

    let mut file = std::fs::File::create(&temp_file).map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Failed to create temporary file for download: {e}"))
            .with_type("server_error"),
    })?;

    std::io::copy(&mut resp, &mut file).map_err(|e| {
        let _ = std::fs::remove_file(&temp_file);
        ErrorResponse {
            error: ApiError::new(format!("Failed to stream media to disk: {e}"))
                .with_type("server_error"),
        }
    })?;

    Ok(temp_file)
}

pub fn download_remote_bytes(url: &str) -> Result<Vec<u8>, ErrorResponse> {
    let target_url = resolve_media_source(url);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!("Failed to build HTTP client: {e}"))
                .with_type("server_error"),
        })?;

    let req = build_authenticated_get(&client, &target_url)?;
    let resp = req
        .send()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!(
                "Failed to fetch image from URL '{target_url}': {e}"
            ))
            .with_type("invalid_request_error")
            .with_code("media_fetch_error"),
        })?
        .error_for_status()
        .map_err(|e| ErrorResponse {
            error: ApiError::new(format!("Remote server returned error for image URL: {e}"))
                .with_type("invalid_request_error")
                .with_code("remote_media_error"),
        })?;

    let bytes = resp.bytes().map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Failed to read image bytes: {e}")).with_type("server_error"),
    })?;

    Ok(bytes.to_vec())
}

pub fn process_image_bytes(
    bytes: &[u8],
    max_dims: (u32, u32),
) -> Result<(Vec<u8>, u32, u32), ErrorResponse> {
    let img = image::load_from_memory(bytes).map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Invalid or unreadable image data: {e}"))
            .with_type("invalid_request_error")
            .with_code("invalid_image"),
    })?;

    let (mut w, mut h) = (img.width(), img.height());
    let (max_w, max_h) = max_dims;

    if w > max_w || h > max_h {
        let resized = img.resize(max_w, max_h, image::imageops::FilterType::Triangle);
        w = resized.width();
        h = resized.height();
        let mut buf = std::io::Cursor::new(Vec::new());
        resized
            .write_to(&mut buf, image::ImageFormat::Jpeg)
            .map_err(|e| ErrorResponse {
                error: ApiError::new(format!("Failed to encode resized image: {e}"))
                    .with_type("server_error")
                    .with_code("image_processing_error"),
            })?;
        Ok((buf.into_inner(), w, h))
    } else {
        Ok((bytes.to_vec(), w, h))
    }
}

pub fn extract_video_frames(
    video_path: &Path,
    max_dims: (u32, u32),
) -> Result<Vec<(Vec<u8>, u32, u32)>, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join("rune_infer").join(format!(
        "vid_frames_{}_{}",
        std::process::id(),
        now
    ));
    std::fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let out_pattern = temp_dir.join("frame_%04d.jpg");
    let output = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-vf")
        .arg("fps=2,scale='min(768,iw)':-2")
        .arg("-vframes")
        .arg("16")
        .arg(&out_pattern)
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let mut frames = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&temp_dir) {
                let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort();
                for p in paths {
                    if let Ok(bytes) = std::fs::read(&p) {
                        if let Ok(processed) = process_image_bytes(&bytes, max_dims) {
                            frames.push(processed);
                        }
                    }
                }
            }
            let _ = std::fs::remove_dir_all(&temp_dir);
            if frames.is_empty() {
                Err("FFmpeg extracted 0 frames from video".to_string())
            } else {
                Ok(frames)
            }
        }
        Ok(out) => {
            let _ = std::fs::remove_dir_all(&temp_dir);
            let stderr = String::from_utf8_lossy(&out.stderr);
            Err(format!(
                "FFmpeg failed with {}: {}",
                out.status,
                stderr.trim()
            ))
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&temp_dir);
            Err(format!("Failed to spawn ffmpeg: {e}. Is ffmpeg in PATH?"))
        }
    }
}

pub fn decode_media(
    url: &str,
    max_dims: (u32, u32),
) -> Result<Vec<(Vec<u8>, u32, u32)>, ErrorResponse> {
    let is_remote =
        url.starts_with("http://") || url.starts_with("https://") || url.starts_with("s3://");

    if is_video_format(url) {
        if is_remote {
            let temp_video = download_remote_to_temp_file(url)?;
            let frames = extract_video_frames(&temp_video, max_dims).map_err(|e| ErrorResponse {
                error: ApiError::new(format!("Video processing error: {e}"))
                    .with_type("video_processing_error")
                    .with_code("ffmpeg_error"),
            });
            let _ = std::fs::remove_file(&temp_video);
            frames
        } else if let Some(rest) = url.strip_prefix("data:") {
            let (_header, b64_data) = rest.split_once(',').ok_or_else(|| ErrorResponse {
                error: ApiError::new("Invalid video data URL: missing comma separator")
                    .with_type("invalid_request_error")
                    .with_code("invalid_video"),
            })?;

            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64_data.trim())
                .map_err(|e| ErrorResponse {
                    error: ApiError::new(&format!("Invalid base64 video payload: {e}"))
                        .with_type("invalid_request_error")
                        .with_code("invalid_video"),
                })?;

            let temp_dir = ensure_temp_dir("uploads")?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let temp_video = temp_dir.join(format!("rune_tmp_{}_{}.mp4", std::process::id(), now));
            std::fs::write(&temp_video, &bytes).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Failed to write temporary video: {e}")),
            })?;

            let frames = extract_video_frames(&temp_video, max_dims).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Video processing error: {e}"))
                    .with_type("video_processing_error")
                    .with_code("ffmpeg_error"),
            });
            let _ = std::fs::remove_file(&temp_video);
            frames
        } else {
            let path = Path::new(url);
            if !path.exists() {
                return Err(ErrorResponse {
                    error: ApiError::new(&format!("Video file not found at '{}'", url))
                        .with_type("invalid_request_error")
                        .with_code("file_not_found"),
                });
            }
            extract_video_frames(path, max_dims).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Video processing error: {e}"))
                    .with_type("video_processing_error")
                    .with_code("ffmpeg_error"),
            })
        }
    } else if is_remote {
        let bytes = download_remote_bytes(url)?;
        let processed = process_image_bytes(&bytes, max_dims)?;
        Ok(vec![processed])
    } else if let Some(rest) = url.strip_prefix("data:") {
        let (header, b64_data) = rest.split_once(',').ok_or_else(|| ErrorResponse {
            error: ApiError::new("Invalid image data URL: missing comma separator")
                .with_type("invalid_request_error")
                .with_code("invalid_image"),
        })?;

        let header_lower = header.to_lowercase();
        if !header_lower.starts_with("image/") || !header_lower.contains(";base64") {
            return Err(ErrorResponse {
                error: ApiError::new("Invalid image data URL (expected `image/*;base64,<data>`)")
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            });
        }

        let clean_b64 = b64_data.trim();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(clean_b64)
            .map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Invalid base64 payload: {e}"))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            })?;

        let processed = process_image_bytes(&bytes, max_dims)?;
        Ok(vec![processed])
    } else {
        match std::fs::read(url) {
            Ok(bytes) => {
                let processed = process_image_bytes(&bytes, max_dims)?;
                Ok(vec![processed])
            }
            Err(e) => Err(ErrorResponse {
                error: ApiError::new(&format!("Could not read file '{}': {e}", url))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            }),
        }
    }
}
