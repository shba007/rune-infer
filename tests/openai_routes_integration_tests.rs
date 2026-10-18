use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use rune_infer::api::{AppState, create_router};
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::proxy::ProxyService;
use rune_infer::schema::ModelEntry;
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
async fn test_openai_auth_rejection_and_acceptance() {
    let app = create_test_app(Some("sk-rune-test-secret-key".to_string()));

    // 1. Missing Authorization header -> 401
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "authentication_error");
    assert_eq!(json["error"]["code"], "invalid_api_key");

    // 2. Wrong Bearer token -> 401
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer wrong-key")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // 3. Valid Bearer token with non-existent model -> Passes auth, returns 404 (model_not_found)
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-rune-test-secret-key")
        .body(Body::from(
            r#"{"model":"unknown-model-xyz","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn test_openai_chat_validation_errors() {
    let app = create_test_app(None);

    // 1. Invalid JSON body -> 400
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("invalid json payload"))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // 2. Empty messages array -> 400
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"qwen3.8-27b","messages":[]}"#))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "empty_array");

    // 3. Missing model parameter -> 400
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "model_missing");
}

#[tokio::test]
async fn test_openai_models_endpoints() {
    let app = create_test_app(None);

    // 1. GET /v1/models -> 200 list
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    let models = json["data"].as_array().expect("array of models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["id"], "qwen3.8-27b");

    // 2. GET /v1/models/qwen3.8-27b -> 200
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models/qwen3.8-27b")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], "qwen3.8-27b");

    // 3. GET /v1/models/non-existent-model -> 404
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models/non-existent-model")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_anthropic_routes_are_disabled() {
    let app = create_test_app(None);

    // 1. POST /v1/messages should be 404 (disabled)
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // 2. POST /messages should be 404 (disabled)
    let req = Request::builder()
        .method("POST")
        .uri("/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // 3. POST /anthropic/v1/messages should be 404 (disabled)
    let req = Request::builder()
        .method("POST")
        .uri("/anthropic/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_audio_speech_route() {
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
        "id": "audio8-tts-preview-0.6b",
        "name": "Audio8 TTS Preview 0.6B",
        "modality": "TextToSpeech",
        "capabilities": ["Text-to-Speech"],
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

    let inference = Arc::new(InferenceAppState::new(&catalog));
    let proxy = Arc::new(ProxyService::new(&catalog));
    let state = AppState::new(inference, proxy, catalog);
    let app = create_router(state);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"audio8-tts-preview-0.6b","input":"Testing speech synthesis in Rune Infer","voice":"alloy","response_format":"wav"}"#,
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("content-type").unwrap(), "audio/wav");

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    assert!(body.len() > 44);
    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(&body[8..12], b"WAVE");
}

#[tokio::test]
async fn test_audio_transcription_route() {
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

    let inference = Arc::new(InferenceAppState::new(&catalog));
    let proxy = Arc::new(ProxyService::new(&catalog));
    let state = AppState::new(inference, proxy, catalog);
    let app = create_router(state);

    let boundary = "------------------------boundary123";
    let wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio8-asr-infinite\r\n").as_bytes());
    body_bytes.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
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
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("text").is_some());
}
