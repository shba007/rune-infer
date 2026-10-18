use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use rune_infer::api::{AppState, create_router};
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::inference::engines::download::extract_release_asset_info;
use rune_infer::inference::process::kill_pid;
use rune_infer::proxy::ProxyService;
use rune_infer::schema::ModelEntry;
use rune_infer::types::{HealthResponse, LivezResponse, ReadyzResponse};
use std::sync::Arc;
use tower::ServiceExt;

fn sample_model_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-27b",
        "name": "Qwen3.8 27B",
        "modality": "VisionText",
        "capabilities": ["Chat", "Vision"],
        "engine": { "supported": ["llama-server"] },
        "max_context_length": 262144,
        "sampling": { "temperature": 0.3, "max_tokens": 1024 },
        "local": {
            "path": "D:/Models/qwen3.8-27b.gguf",
            "params": 27000000000,
            "bits_per_weight": 3.44,
            "context_length": 131072,
            "gpu_layers": 99,
            "args": "-fa on"
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

fn create_test_app(api_key: Option<String>) -> axum::Router {
    let mut catalog = ModelRegistry {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key,
            max_loaded_models: 1,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.95,
        },
        ..Default::default()
    };
    catalog.models.push(sample_model_config());

    let inference = Arc::new(InferenceAppState::new(&catalog));
    let proxy = Arc::new(ProxyService::new(&catalog));
    let state = AppState::new(inference, proxy, catalog);
    create_router(state)
}

#[tokio::test]
async fn test_livez_and_readyz_probes_unauthenticated() {
    // Setup app with strict API key required for normal endpoints
    let app = create_test_app(Some("sk-strict-key-12345".to_string()));

    // 1. /livez probe must return 200 OK without requiring authentication
    let livez_req = Request::builder()
        .method("GET")
        .uri("/livez")
        .body(Body::empty())
        .unwrap();

    let livez_resp = app.clone().oneshot(livez_req).await.unwrap();
    assert_eq!(livez_resp.status(), StatusCode::OK);
    let body = to_bytes(livez_resp.into_body(), usize::MAX).await.unwrap();
    let livez_data: LivezResponse = serde_json::from_slice(&body).expect("valid LivezResponse");
    assert_eq!(livez_data.status, "alive");
    assert_eq!(livez_data.service, "rune-infer");

    // 2. /readyz probe must return 200 OK and describe model catalog readiness
    let readyz_req = Request::builder()
        .method("GET")
        .uri("/readyz")
        .body(Body::empty())
        .unwrap();

    let readyz_resp = app.clone().oneshot(readyz_req).await.unwrap();
    assert_eq!(readyz_resp.status(), StatusCode::OK);
    let body = to_bytes(readyz_resp.into_body(), usize::MAX).await.unwrap();
    let readyz_data: ReadyzResponse = serde_json::from_slice(&body).expect("valid ReadyzResponse");
    assert_eq!(readyz_data.status, "ready");
    assert_eq!(readyz_data.service, "rune-infer");
    assert_eq!(readyz_data.models_configured, 1);
    assert!(readyz_data.models_loaded.is_empty());

    // 3. /health probe must also succeed without authentication
    let health_req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let health_resp = app.oneshot(health_req).await.unwrap();
    assert_eq!(health_resp.status(), StatusCode::OK);
    let body = to_bytes(health_resp.into_body(), usize::MAX).await.unwrap();
    let health_data: HealthResponse = serde_json::from_slice(&body).expect("valid HealthResponse");
    assert_eq!(health_data.status, "ok");
    assert_eq!(health_data.service, "rune-infer");
}

#[test]
fn test_defensive_release_asset_extraction() {
    // 1. Valid GitHub release asset object
    let valid_asset = serde_json::json!({
        "name": "llama-b3500-bin-win-cuda-cu12.4-x64.zip",
        "browser_download_url": "https://github.com/ggml-org/llama.cpp/releases/download/b3500/llama-b3500-bin-win-cuda-cu12.4-x64.zip"
    });
    let (name, url) = extract_release_asset_info(&valid_asset).expect("should extract valid asset");
    assert_eq!(name, "llama-b3500-bin-win-cuda-cu12.4-x64.zip");
    assert!(url.starts_with("https://github.com"));

    // 2. GitHub API rate limit response object (no name/browser_download_url)
    let rate_limit_payload = serde_json::json!({
        "message": "API rate limit exceeded for 192.0.2.1. (But here's the good news: Authenticated requests get a higher rate limit.)",
        "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"
    });
    assert!(extract_release_asset_info(&rate_limit_payload).is_err());

    // 3. Asset missing browser_download_url
    let partial_asset = serde_json::json!({
        "name": "archive.tar.gz"
    });
    assert!(extract_release_asset_info(&partial_asset).is_err());
}

#[test]
fn test_process_kill_pid_safety_boundary() {
    // Calling kill_pid on 0 or 1 should safely no-op without killing host or panic
    kill_pid(0);
    kill_pid(1);
}
