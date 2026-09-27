use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::{
    Json, Router,
    extract::{Query, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use hmac::{Hmac, KeyInit, Mac};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::config::ModelRegistry;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceDelta, ErrorResponse,
    HealthResponse, MediaItem, ModelInfo, ModelsResponse, ResponseMessage, ToolCall, ToolCallChunk,
    Usage,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub config: ModelRegistry,
}

pub fn create_router(state: AppState) -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/inference", post(inference_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::new(state))
}

#[derive(Debug, Clone)]
struct S3Config {
    endpoint: String,
    access_key: String,
    secret_key: String,
    region: String,
}

impl S3Config {
    fn from_env() -> Option<Self> {
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

type HmacSha256 = Hmac<Sha256>;

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC initialization failed");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn format_iso8601_date(now: std::time::SystemTime) -> (String, String) {
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

fn resolve_media_source(raw_url: &str) -> String {
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

fn ensure_temp_dir(sub: &str) -> Result<std::path::PathBuf, ErrorResponse> {
    let dir = std::env::temp_dir().join("rune_infer").join(sub);
    std::fs::create_dir_all(&dir).map_err(|e| ErrorResponse {
        error: ApiError::new(format!("Failed to create temporary directory: {e}"))
            .with_type("server_error"),
    })?;
    Ok(dir)
}

fn build_authenticated_get(
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

fn is_video_format(url_or_path: &str) -> bool {
    let lower = url_or_path.to_lowercase();
    let path_part = lower.split('?').next().unwrap_or(&lower);
    lower.starts_with("data:video/")
        || path_part.ends_with(".mp4")
        || path_part.ends_with(".mkv")
        || path_part.ends_with(".mov")
        || path_part.ends_with(".webm")
        || path_part.ends_with(".avi")
}

fn download_remote_to_temp_file(url: &str) -> Result<std::path::PathBuf, ErrorResponse> {
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

fn download_remote_bytes(url: &str) -> Result<Vec<u8>, ErrorResponse> {
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

fn process_image_bytes(
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

fn extract_video_frames(
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

fn decode_media(
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

#[axum::debug_handler]
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded_models = state.inference.loaded_models();
    Json(HealthResponse::new().with_loaded_models(loaded_models))
}

#[axum::debug_handler]
async fn models_handler(State(state): State<Arc<AppState>>) -> Response {
    let mut mtp_models = Vec::new();
    let data = state
        .config
        .models
        .iter()
        .map(|m| {
            let heads = m.mtp_heads();
            if m.has_mtp() {
                mtp_models.push(m.id.clone());
            }
            ModelInfo::new(&m.id)
                .with_ownership("Rune Infer".to_string())
                .with_mtp_heads(heads)
                .with_capabilities(m.resolved_capabilities())
        })
        .collect();

    let mut headers = axum::http::HeaderMap::new();
    if !mtp_models.is_empty() {
        headers.insert(
            "x-mtp-available",
            axum::http::HeaderValue::from_static("true"),
        );
        if let Ok(val) = axum::http::HeaderValue::from_str(&mtp_models.join(", ")) {
            headers.insert("x-mtp-models", val);
        }
    }

    (
        headers,
        Json(ModelsResponse {
            object: "list".to_string(),
            data,
        }),
    )
        .into_response()
}

#[derive(Deserialize, Default)]
struct InferenceQuery {
    engine: Option<String>,
}

#[axum::debug_handler]
async fn inference_handler(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InferenceQuery>,
    Json(task): Json<InferenceTaskRequest>,
) -> Response {
    let start_time = std::time::Instant::now();
    let engine_id = match q.engine.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(id) => id.to_string(),
        None => {
            tracing::warn!(target: "audit", status = 400, "Inference call missing 'engine' param");
            return (
                StatusCode::BAD_REQUEST,
                Json(InferenceTaskResponse::Error(
                    "Missing required query parameter: 'engine'".into(),
                )),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let res =
        tokio::task::spawn_blocking(move || inference.execute_task(&engine_id, &task, None)).await;

    match res {
        Ok(Ok(output)) => {
            tracing::info!(
                target: "audit",
                status = 200,
                latency_ms = start_time.elapsed().as_millis(),
                tokens = output.usage.total_tokens,
                "Inference task completed"
            );
            Json(output).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(
                target: "audit",
                status = 404,
                latency_ms = start_time.elapsed().as_millis(),
                error = %e,
                "Inference task failed: engine not found or execution error"
            );
            (StatusCode::NOT_FOUND, Json(InferenceTaskResponse::Error(e))).into_response()
        }
        Err(join_err) => {
            tracing::error!(
                target: "audit",
                status = 500,
                latency_ms = start_time.elapsed().as_millis(),
                error = %join_err,
                "Inference task thread join error"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(InferenceTaskResponse::Error(format!(
                    "Task failed: {join_err}"
                ))),
            )
                .into_response()
        }
    }
}

fn parse_single_tool_call(val: &serde_json::Value, id: String) -> Option<ToolCall> {
    let obj = val.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id,
        r#type: "function".to_string(),
        function: crate::types::FunctionCall {
            name,
            arguments: args,
        },
    })
}

fn convert_to_tool_calls(value: &serde_json::Value, created: u64) -> Option<Vec<ToolCall>> {
    match value {
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            let mut calls = Vec::new();
            for (idx, item) in arr.iter().enumerate() {
                if let Some(call) =
                    parse_single_tool_call(item, format!("call_{}_{}", created, idx))
                {
                    calls.push(call);
                }
            }
            if calls.is_empty() { None } else { Some(calls) }
        }
        serde_json::Value::Object(_) => {
            parse_single_tool_call(value, format!("call_{}_0", created)).map(|c| vec![c])
        }
        _ => None,
    }
}

#[axum::debug_handler]
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let start_time = std::time::Instant::now();

    if request.model.trim().is_empty() {
        tracing::warn!(target: "audit", status = 400, "Chat completion rejected: missing 'model'");
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing required field: 'model'")
                    .with_type("invalid_request_error")
                    .with_code("model_missing"),
            }),
        )
            .into_response();
    }

    let model_config = match state.config.find(&request.model) {
        Some(c) => c.clone(),
        None => {
            tracing::warn!(
                target: "audit",
                status = 404,
                model = %request.model,
                "Chat completion rejected: model not found in catalog"
            );
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found", request.model))
                        .with_type("invalid_request_error")
                        .with_code("model_not_found"),
                }),
            )
                .into_response();
        }
    };

    let mut has_video = false;
    let mut has_media = false;

    for msg in &request.messages {
        let (_, media) = msg.split_text_and_media();
        if !media.is_empty() {
            has_media = true;
        }
        if media.iter().any(|m| matches!(m, MediaItem::Video(_))) {
            has_video = true;
        }
    }

    if has_media
        && !model_config.vision
        && model_config.modality != crate::config::Modality::VisionText
    {
        tracing::warn!(
            target: "audit",
            status = 400,
            model = %request.model,
            "Chat completion rejected: media provided to text-only model"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support vision/media inputs.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("model_vision_unsupported"),
            }),
        )
            .into_response();
    }

    let is_structured_mode = match &request.response_format {
        Some(rf) => rf.r#type == "json_object" || rf.r#type == "json_schema",
        None => false,
    };

    if is_structured_mode && !model_config.supports_structured_output() {
        tracing::warn!(
            target: "audit",
            status = 400,
            model = %request.model,
            "Chat completion rejected: structured JSON mode requested on unsupported model"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support structured JSON mode (capability 'Structured Output' is missing in models.json).",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("unsupported_response_format"),
            }),
        )
            .into_response();
    }

    let model_id = request.model.clone();
    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        Ok(Err(err)) => {
            tracing::error!(
                target: "audit",
                status = 404,
                model = %request.model,
                error = %err,
                "Failed to acquire engine for model"
            );
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(&format!(
                        "Model '{}' not found or failed to load: {err}",
                        request.model
                    ))
                    .with_type("invalid_request_error")
                    .with_code("model_not_found"),
                }),
            )
                .into_response();
        }
        Err(join_err) => {
            tracing::error!(
                target: "audit",
                status = 500,
                error = %join_err,
                "Engine spawn_blocking task join error"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(&format!("Thread execution error: {join_err}")),
                }),
            )
                .into_response();
        }
    };

    let tools_value = request.tools.clone().unwrap_or(serde_json::Value::Null);
    let has_tools = match &tools_value {
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
        _ => false,
    };

    let structured_schema = if is_structured_mode {
        if let Some(ref rf) = request.response_format {
            if let Some(ref js) = rf.json_schema {
                js.schema
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({ "type": "object" }))
            } else {
                serde_json::json!({ "type": "object" })
            }
        } else {
            serde_json::json!({})
        }
    } else {
        serde_json::json!({})
    };

    let default_tool_system = if is_structured_mode {
        format!(
            "You are a helpful assistant.\nYou must respond ONLY with a valid JSON object matching this schema:\n{}",
            serde_json::to_string_pretty(&structured_schema).unwrap_or_default()
        )
    } else if has_tools {
        let tools_json = serde_json::to_string_pretty(&tools_value).unwrap_or_default();
        format!(
            "You are a helpful assistant with access to tools.\n\
            When calling a tool, reply ONLY with a <tool_call> block containing a JSON array:\n\
            <tool_call>\n\
            [{{\"name\": \"tool_name\", \"arguments\": {{...}}}}]\n\
            </tool_call>\n\n\
            Available Tools:\n\
            {}",
            tools_json
        )
    } else if has_video {
        "You are a helpful assistant with native video understanding capabilities. Analyze the video sequence directly, observing temporal motion, continuity, actions, and timestamps.".to_string()
    } else {
        "You are a helpful assistant.".to_string()
    };

    let max_dims = model_config.max_dimensions();
    let mut total_media_tokens = 0u32;
    let mut image_count = 0usize;
    let mut video_count = 0usize;

    let (prompt, images) = {
        let mut prompt = String::new();
        let mut images = Vec::new();

        let has_system = request.messages.iter().any(|m| m.role == "system");
        if !has_system {
            prompt.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n",
                default_tool_system
            ));
        }

        for msg in &request.messages {
            let (mut text, media_items) = msg.split_text_and_media();
            let role = if msg.role.trim().is_empty() {
                "user"
            } else {
                msg.role.trim()
            };

            if role == "system" && (has_tools || is_structured_mode) {
                text = format!("{}\n\n{}", text, default_tool_system);
            }

            let mut media_blocks = String::new();
            for item in &media_items {
                match item {
                    MediaItem::Video(url) => {
                        video_count += 1;
                        match decode_media(url, max_dims) {
                            Ok(frames) => {
                                let (w, h) =
                                    frames.first().map(|f| (f.1, f.2)).unwrap_or((768, 768));
                                let patch_w = (w + 27) / 28;
                                let patch_h = (h + 27) / 28;
                                let tubelet_slices = ((frames.len() + 1) / 2) as u32;
                                total_media_tokens = total_media_tokens
                                    .saturating_add((tubelet_slices * patch_w * patch_h) + 32);

                                media_blocks
                                    .push_str(&format!("\nVideo {video_count}:\n<|video_start|>"));
                                for (frame_bytes, _, _) in frames {
                                    images.push(frame_bytes);
                                    media_blocks.push_str("<__media__>");
                                }
                                media_blocks.push_str("<|video_end|>\n");
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "audit",
                                    status = 400,
                                    error = %e.error.message,
                                    "Video decoding failed"
                                );
                                return (StatusCode::BAD_REQUEST, Json(e)).into_response();
                            }
                        }
                    }
                    MediaItem::Image(url) => {
                        image_count += 1;
                        match decode_media(url, max_dims) {
                            Ok(frames) => {
                                for (frame_bytes, w, h) in frames {
                                    let patch_w = (w + 27) / 28;
                                    let patch_h = (h + 27) / 28;
                                    total_media_tokens =
                                        total_media_tokens.saturating_add((patch_w * patch_h) + 32);

                                    images.push(frame_bytes);
                                    media_blocks.push_str(&format!(
                                        "\nPicture {image_count}:\n<__media__>\n"
                                    ));
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "audit",
                                    status = 400,
                                    error = %e.error.message,
                                    "Image decoding failed"
                                );
                                return (StatusCode::BAD_REQUEST, Json(e)).into_response();
                            }
                        }
                    }
                }
            }

            text.push_str(&media_blocks);
            prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text.trim()));
        }
        prompt.push_str("<|im_start|>assistant\n");
        (prompt, images)
    };

    let est_text_tokens = ((prompt.len() / 4).max(1)) as u32;
    let reserved_tokens = request.max_tokens.unwrap_or(2048) as u32;
    let total_required = est_text_tokens
        .saturating_add(total_media_tokens)
        .saturating_add(reserved_tokens);
    let context_limit = model_config.effective_context_limit();

    if total_required > context_limit {
        let err_msg = format!(
            "Request token budget estimated at ~{} tokens (text: {}, media: {}, max_tokens: {}) exceeds model '{}' context limit of {} tokens.",
            total_required,
            est_text_tokens,
            total_media_tokens,
            reserved_tokens,
            model_config.id,
            context_limit
        );
        tracing::warn!(
            target: "audit",
            status = 413,
            model = %request.model,
            error = %err_msg,
            "Context window exceeded"
        );
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(err_msg)
                    .with_type("invalid_request_error")
                    .with_code("context_length_exceeded"),
            }),
        )
            .into_response();
    }

    let task_schema = if is_structured_mode {
        structured_schema
    } else if has_tools {
        tools_value
    } else {
        serde_json::json!({})
    };

    let task = InferenceTaskRequest::ToolCall {
        prompt,
        schema: task_schema,
        images,
        messages: request.messages.clone(),
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let task_clone = task.clone();
        let engine_clone = engine.clone();
        let model_id = request.model.clone();

        tokio::task::spawn_blocking(move || {
            if has_tools && !is_structured_mode {
                match engine_clone.execute(&task_clone, None) {
                    Ok(output) => match output.response {
                        InferenceTaskResponse::ToolCall(val) => {
                            if let Some(tool_calls) = convert_to_tool_calls(&val, created) {
                                let tool_chunks = tool_calls
                                    .into_iter()
                                    .enumerate()
                                    .map(|(idx, tc)| ToolCallChunk {
                                        index: idx,
                                        id: Some(tc.id),
                                        r#type: Some(tc.r#type),
                                        function: Some(crate::types::FunctionCallChunk {
                                            name: Some(tc.function.name),
                                            arguments: Some(tc.function.arguments),
                                        }),
                                    })
                                    .collect();

                                let chunk = ChatCompletionResponse {
                                    id: format!("chatcmpl-{created}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id.clone(),
                                    choices: vec![Choice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChoiceDelta {
                                            content: None,
                                            role: Some("assistant".to_string()),
                                            tool_calls: Some(tool_chunks),
                                        }),
                                        finish_reason: Some("tool_calls".to_string()),
                                    }],
                                    usage: output.usage,
                                };
                                let _ = tx.send(
                                    Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap_or_default()),
                                );
                            } else {
                                let chunk = ChatCompletionResponse {
                                    id: format!("chatcmpl-{created}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id.clone(),
                                    choices: vec![Choice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChoiceDelta {
                                            content: Some(val.to_string()),
                                            role: Some("assistant".to_string()),
                                            tool_calls: None,
                                        }),
                                        finish_reason: Some("stop".to_string()),
                                    }],
                                    usage: output.usage,
                                };
                                let _ = tx.send(
                                    Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap_or_default()),
                                );
                            }
                        }
                        InferenceTaskResponse::Text(text) => {
                            let chunk = ChatCompletionResponse {
                                id: format!("chatcmpl-{created}"),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id.clone(),
                                choices: vec![Choice {
                                    index: 0,
                                    message: None,
                                    delta: Some(ChoiceDelta {
                                        content: Some(text),
                                        role: Some("assistant".to_string()),
                                        tool_calls: None,
                                    }),
                                    finish_reason: Some("stop".to_string()),
                                }],
                                usage: output.usage,
                            };
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&chunk).unwrap_or_default()),
                            );
                        }
                        InferenceTaskResponse::Error(other) => {
                            let err_chunk = serde_json::json!({
                                "error": {
                                    "message": other,
                                    "type": "server_error",
                                    "code": "inference_error"
                                }
                            });
                            let _ = tx
                                .send(Event::default().event("error").data(err_chunk.to_string()));
                        }
                    },
                    Err(e) => {
                        let err_chunk = serde_json::json!({
                            "error": {
                                "message": e.to_string(),
                                "type": "server_error",
                                "code": "engine_execution_failed"
                            }
                        });
                        let _ =
                            tx.send(Event::default().event("error").data(err_chunk.to_string()));
                    }
                }
            } else {
                let tx_clone = tx.clone();
                let model_id_clone = model_id.clone();
                let mut completion_tokens = 0u32;
                let mut on_token = move |piece: &str| -> bool {
                    completion_tokens += 1;
                    let chunk = ChatCompletionResponse {
                        id: format!("chatcmpl-{created}"),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![Choice {
                            index: 0,
                            message: None,
                            delta: Some(ChoiceDelta {
                                content: Some(piece.to_string()),
                                role: Some("assistant".to_string()),
                                tool_calls: None,
                            }),
                            finish_reason: None,
                        }],
                        usage: Usage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                        },
                    };
                    tx_clone
                        .send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_ok()
                };

                match engine_clone.execute(&task_clone, Some(&mut on_token)) {
                    Ok(final_output) => {
                        let final_chunk = ChatCompletionResponse {
                            id: format!("chatcmpl-{created}"),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id.clone(),
                            choices: vec![Choice {
                                index: 0,
                                message: None,
                                delta: Some(ChoiceDelta {
                                    content: None,
                                    role: None,
                                    tool_calls: None,
                                }),
                                finish_reason: Some("stop".to_string()),
                            }],
                            usage: final_output.usage,
                        };
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                        );
                    }
                    Err(e) => {
                        let err_chunk = serde_json::json!({
                            "error": {
                                "message": e.to_string(),
                                "type": "server_error",
                                "code": "engine_execution_failed"
                            }
                        });
                        let _ =
                            tx.send(Event::default().event("error").data(err_chunk.to_string()));
                    }
                }
            }
        });

        tracing::info!(
            target: "audit",
            status = 200,
            latency_ms = start_time.elapsed().as_millis(),
            model = %request.model,
            media = %format!("images: {image_count}, videos: {video_count}"),
            "SSE stream started"
        );

        let stream = UnboundedReceiverStream::new(rx);
        let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));
        let sse_stream = stream.map(Ok::<_, Infallible>).chain(done_stream);
        let mut resp = Sse::new(sse_stream).into_response();
        if let Some(heads) = model_config.mtp_heads() {
            if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                resp.headers_mut().insert("x-mtp-heads", val);
            }
        }
        return resp;
    }

    let task_clone = task.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::ToolCall(val) => {
                tracing::info!(
                    target: "audit",
                    status = 200,
                    latency_ms = start_time.elapsed().as_millis(),
                    model = %request.model,
                    prompt_tokens = output.usage.prompt_tokens,
                    completion_tokens = output.usage.completion_tokens,
                    media = %format!("images: {image_count}, videos: {video_count}"),
                    "Chat completion successful (structured/tool_call)"
                );
                let json_resp = if is_structured_mode {
                    let content_str = match &val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string_pretty(other).unwrap_or_default(),
                    };
                    let resp = chat_completion_response(
                        &request.model,
                        Some(content_str),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else if let Some(tool_calls) = convert_to_tool_calls(&val, created) {
                    let resp = chat_completion_response(
                        &request.model,
                        None,
                        Some(tool_calls),
                        "tool_calls",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else {
                    let resp = chat_completion_response(
                        &request.model,
                        Some(val.to_string()),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                };

                let mut final_resp = json_resp;
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Text(text) => {
                tracing::info!(
                    target: "audit",
                    status = 200,
                    latency_ms = start_time.elapsed().as_millis(),
                    model = %request.model,
                    prompt_tokens = output.usage.prompt_tokens,
                    completion_tokens = output.usage.completion_tokens,
                    media = %format!("images: {image_count}, videos: {video_count}"),
                    "Chat completion successful (text)"
                );
                let resp = chat_completion_response(
                    &request.model,
                    Some(text),
                    None,
                    "stop",
                    created,
                    output.usage,
                );
                let mut final_resp = Json(resp).into_response();
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Error(e) => {
                tracing::error!(
                    target: "audit",
                    status = 500,
                    latency_ms = start_time.elapsed().as_millis(),
                    model = %request.model,
                    error = %e,
                    "Inference returned internal error"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(e)
                            .with_type("server_error")
                            .with_code("inference_error"),
                    }),
                )
                    .into_response()
            }
        },
        Ok(Err(e)) => {
            let status = if e.contains("context") || e.contains("tokens") || e.contains("limit") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            tracing::warn!(
                target: "audit",
                status = status.as_u16(),
                latency_ms = start_time.elapsed().as_millis(),
                model = %request.model,
                error = %e,
                "Engine execution returned error"
            );
            (
                status,
                Json(ErrorResponse {
                    error: ApiError::new(e)
                        .with_type("invalid_request_error")
                        .with_code("inference_failure"),
                }),
            )
                .into_response()
        }
        Err(join_err) => {
            tracing::error!(
                target: "audit",
                status = 500,
                latency_ms = start_time.elapsed().as_millis(),
                model = %request.model,
                error = %join_err,
                "Engine thread join error"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Thread execution failure: {join_err}"))
                        .with_type("server_error")
                        .with_code("internal_error"),
                }),
            )
                .into_response()
        }
    }
}

fn chat_completion_response(
    model: &str,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Some(ResponseMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            }),
            delta: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage,
    }
}
