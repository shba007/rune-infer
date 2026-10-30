use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
mod common;
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::media::{decode_media, validate_url_for_ssrf};
use rune_infer::schema::ModelEntry;
use tower::ServiceExt;

#[test]
fn test_ssrf_protection_blocks_private_and_metadata_addresses() {
    // 1. Loopback IPv4 & IPv6
    assert!(validate_url_for_ssrf("http://127.0.0.1:8080/image.png").is_err());
    assert!(validate_url_for_ssrf("http://localhost:8080/image.png").is_err());
    assert!(validate_url_for_ssrf("http://[::1]:8080/image.png").is_err());

    // 2. Cloud metadata IP (AWS, GCP, Azure IMDS)
    let meta_err = validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/").unwrap_err();
    assert_eq!(meta_err.error.code.as_deref(), Some("ssrf_blocked"));

    // 3. RFC1918 Private networks
    assert!(validate_url_for_ssrf("http://10.0.0.1:9000/internal.mp4").is_err());
    assert!(validate_url_for_ssrf("http://172.16.5.10/video.mp4").is_err());
    assert!(validate_url_for_ssrf("http://192.168.1.100/snapshot.jpg").is_err());

    // 4. GCP internal metadata domain
    assert!(validate_url_for_ssrf("http://metadata.google.internal/computeMetadata/v1/").is_err());

    // 5. Valid public URL passes SSRF validation
    assert!(validate_url_for_ssrf("https://example.com/sample.png").is_ok());
}

#[test]
fn test_local_file_ingestion_blocked_by_default() {
    // Ensure environment flag is unset
    unsafe {
        std::env::remove_var("RUNE_ALLOW_LOCAL_FILES");
    }

    // Direct local path should be blocked
    let res = decode_media("Cargo.toml", (768, 768));
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.error.code.as_deref(), Some("local_files_disabled"));

    // Windows / Unix absolute paths should also be blocked
    let res_abs = decode_media("C:/Windows/win.ini", (768, 768));
    assert!(res_abs.is_err());
    assert_eq!(
        res_abs.unwrap_err().error.code.as_deref(),
        Some("local_files_disabled")
    );
}

#[tokio::test]
async fn test_audio_upload_filename_path_traversal_sanitized() {
    let mut catalog = ModelRegistry {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            max_loaded_models: 1,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.95,
        },
        ..Default::default()
    };

    let raw = r#"{
        "id": "audio8-asr-infinite",
        "name": "Edge0 Audio8 ASR Infinite",
        "modality": "SpeechToText",
        "tasks": ["audio-to-text"],
        "capabilities": ["Speech-to-Text"],
        "engine": { "supported": ["audio-cpp"], "preferred": "audio-cpp" },
        "sampling": {},
        "local": {
            "path": "Cargo.toml",
            "context_length": -1,
            "gpu_layers": 99,
            "args": ""
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    catalog
        .models
        .push(ModelConfig::try_from(entry).expect("valid model config"));

    let app = common::create_mock_app(catalog);

    let boundary = "------------------------boundaryTraversal123";
    let wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio8-asr-infinite\r\n").as_bytes());
    // Malicious traversal filename attempt
    body_bytes.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"../../../../etc/cron.d/malicious.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body_bytes.extend_from_slice(wav_bytes);
    body_bytes.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body_bytes))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // Must succeed without crashing or writing out of temp directory
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["text"], "malicious.wav");
}

#[test]
fn test_default_models_config_binds_to_localhost() {
    let server_path = if std::path::Path::new("config/server.json").exists() {
        "config/server.json"
    } else {
        "config/models.json"
    };
    let raw = std::fs::read_to_string(server_path).expect("server configuration exists");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
    let host = json["server"]["host"].as_str().expect("host string");
    assert_eq!(
        host, "127.0.0.1",
        "Default server host must bind to localhost (127.0.0.1) for security"
    );
}
