use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use base64::Engine;
use rune_infer::api::{AppState, create_router};
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::proxy::ProxyService;
use rune_infer::schema::ModelEntry;
use std::sync::Arc;
use tower::ServiceExt;
mod common;

fn sample_model_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-27b",
        "name": "Qwen3.8 27B",
        "modality": "VisionText",
        "tasks": ["text-image-to-text"],
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
async fn image_generation_validates_dimensions_before_model_dispatch() {
    let app = create_test_app(None);
    for (size, expected_code) in [
        ("1080x1080", "model_task_unsupported"),
        ("720x720", "model_task_unsupported"),
        ("1280x720", "model_task_unsupported"),
        ("0x720", "invalid_parameter_value"),
        ("999999999x720", "invalid_parameter_value"),
    ] {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/images/generations")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "model":"qwen3.8-27b", "prompt":"A lighthouse", "size":size
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 16384).await.unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["code"], expected_code, "{size}: {error}");
    }
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
async fn test_audio_speech_route_refuses_unverified_model() {
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
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(error["error"]["code"], "model_tasks_unverified");
}

#[tokio::test]
async fn test_audio_transcription_route_refuses_unverified_model() {
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
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "model_tasks_unverified");
}

fn sample_multimodal_encoder_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-encoder",
        "name": "Qwen3.8 Multimodal Encoder",
        "modality": "VisionText",
        "tasks": ["text-image-audio-to-text"],
        "capabilities": ["Chat", "Vision", "Audio", "Tool Call"],
        "engine": { "supported": ["encoder"], "preferred": "encoder" },
        "max_context_length": 262144,
        "sampling": { "temperature": 0.3, "max_tokens": 1024 },
        "local": {
            "path": "Cargo.toml",
            "params": 27000000000,
            "bits_per_weight": 3.44,
            "context_length": 131072,
            "gpu_layers": 99,
            "args": ""
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

fn sample_gemma4_omni_encoder_config() -> ModelConfig {
    let raw = r#"{
        "id": "gemma-4-12b-it",
        "name": "Gemma 4 12B IT",
        "modality": "Omni",
        "tasks": ["text-image-audio-video-to-text"],
        "capabilities": ["Chat", "Vision", "Audio", "Video", "Omni", "Tool Call"],
        "engine": { "supported": ["encoder"], "preferred": "encoder" },
        "max_context_length": 262144,
        "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 40, "max_tokens": 4096 },
        "local": {
            "path": "Cargo.toml",
            "params": 12000000000,
            "bits_per_weight": 4.5,
            "context_length": 262144,
            "gpu_layers": 99,
            "args": ""
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

/// Sees images but declares no audio support (like MiniCPM-o with only its vision encoder).
fn sample_vision_only_config() -> ModelConfig {
    let raw = r#"{
        "id": "vision-only",
        "name": "Vision Only",
        "modality": "VisionText",
        "tasks": ["text-image-to-text"],
        "capabilities": ["Chat", "Vision", "Tool Call"],
        "engine": { "supported": ["encoder"], "preferred": "encoder" },
        "max_context_length": 8192,
        "sampling": {},
        "local": {
            "path": "Cargo.toml",
            "params": 1000000000,
            "bits_per_weight": 8.0,
            "context_length": 4096,
            "gpu_layers": 99,
            "args": ""
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

fn create_multimodal_test_app() -> axum::Router {
    let mut catalog = ModelRegistry {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            max_loaded_models: 2,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.95,
        },
        ..Default::default()
    };
    catalog.models.push(sample_multimodal_encoder_config());
    catalog.models.push(sample_gemma4_omni_encoder_config());
    catalog.models.push(sample_vision_only_config());
    common::create_mock_app(catalog)
}

fn sample_png_base64() -> String {
    let png_bytes = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
    format!("data:image/png;base64,{b64}")
}

fn sample_wav_base64() -> String {
    let wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00";
    base64::engine::general_purpose::STANDARD.encode(wav_bytes)
}

#[tokio::test]
async fn test_vision_only_model_rejects_audio_input_with_a_clear_error() {
    // A vision model is not an audio model: this used to be forwarded to the engine and
    // come back as a bare 500 ("audio input is not supported").
    let app = create_multimodal_test_app();
    let payload = serde_json::json!({
        "model": "vision-only",
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "What did they say?" },
                { "type": "input_audio", "input_audio": { "data": sample_wav_base64(), "format": "wav" } }
            ]
        }]
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "model_task_unsupported");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("vision-only")
    );
}

#[tokio::test]
async fn test_openai_chat_input_audio_with_screenshot_and_tools() {
    let app = create_multimodal_test_app();
    let img_uri = sample_png_base64();
    let wav_b64 = sample_wav_base64();

    let payload = serde_json::json!({
        "model": "qwen3.8-encoder",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Click the button based on this screenshot and voice command." },
                    { "type": "image_url", "image_url": { "url": img_uri } },
                    { "type": "input_audio", "input_audio": { "data": wav_b64, "format": "wav" } }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "click_coordinate",
                    "description": "Click screen coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": { "type": "number" },
                            "y": { "type": "number" }
                        },
                        "required": ["x", "y"]
                    }
                }
            }
        ],
        "tool_choice": "required"
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let tool_calls = res["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls");
    assert_eq!(tool_calls[0]["function"]["name"], "click_coordinate");
}

#[tokio::test]
async fn test_openai_chat_input_audio_with_screenshot_text_response() {
    let app = create_multimodal_test_app();
    let img_uri = sample_png_base64();
    let wav_b64 = sample_wav_base64();

    let payload = serde_json::json!({
        "model": "qwen3.8-encoder",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Analyze the UI screenshot and transcribed voice prompt." },
                    { "type": "image_url", "image_url": { "url": img_uri } },
                    { "type": "input_audio", "input_audio": { "data": wav_b64, "format": "wav" } }
                ]
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let content = res["choices"][0]["message"]["content"]
        .as_str()
        .expect("message content");
    assert!(content.contains("1 image(s) and 1 audio(s)"));
}

#[tokio::test]
async fn test_openai_chat_input_audio_unsupported_format() {
    let app = create_multimodal_test_app();
    let wav_b64 = sample_wav_base64();

    let payload = serde_json::json!({
        "model": "qwen3.8-encoder",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": wav_b64, "format": "flac" } }
                ]
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(res["error"]["code"], "unsupported_audio_format");
}

#[tokio::test]
async fn test_openai_chat_input_audio_invalid_base64() {
    let app = create_multimodal_test_app();

    let payload = serde_json::json!({
        "model": "qwen3.8-encoder",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": "!!!not_base64!!!", "format": "wav" } }
                ]
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(res["error"]["code"], "invalid_audio_format");
}

#[tokio::test]
async fn test_gemma4_omni_model_catalog_resolution() {
    let app = create_multimodal_test_app();

    // 1. Exact model ID via GET /v1/models/{model}
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models/gemma-4-12b-it")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], "gemma-4-12b-it");
    assert_eq!(
        json["tasks"],
        serde_json::json!(["text-image-audio-video-to-text"])
    );
    assert_eq!(
        json["modalities_in"],
        serde_json::json!(["text", "image", "audio", "video"])
    );
    assert_eq!(json["modalities_out"], serde_json::json!(["text"]));

    // 2. Chat completion with full Hugging Face repository/GGUF slug resolves and succeeds
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"unsloth/gemma-4-12b-it-GGUF","messages":[{"role":"user","content":"Hello Gemma 4"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-model-served")
            .unwrap()
            .to_str()
            .unwrap(),
        "gemma-4-12b-it"
    );
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model"], "gemma-4-12b-it");
    assert!(json["choices"][0]["message"]["content"].as_str().is_some());

    // 3. Chat completion with exact ID
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"Hello Gemma 4"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["model"], "gemma-4-12b-it");
    assert!(json["choices"][0]["message"]["content"].as_str().is_some());
}

#[tokio::test]
async fn test_gemma4_omni_chat_completion_all_modalities() {
    let app = create_multimodal_test_app();
    let img_uri = sample_png_base64();
    let wav_b64 = sample_wav_base64();

    let payload = serde_json::json!({
        "model": "unsloth/gemma-4-12b-it-GGUF",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Analyze the omni inputs including image screenshot and speech audio." },
                    { "type": "image_url", "image_url": { "url": img_uri } },
                    { "type": "input_audio", "input_audio": { "data": wav_b64, "format": "wav" } }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "omni_action",
                    "description": "Trigger omni multimodal action",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_type": { "type": "string" }
                        },
                        "required": ["action_type"]
                    }
                }
            }
        ],
        "tool_choice": "none"
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-model-served")
            .unwrap()
            .to_str()
            .unwrap(),
        "gemma-4-12b-it"
    );

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(res["model"], "gemma-4-12b-it");
    let content = res["choices"][0]["message"]["content"]
        .as_str()
        .expect("message content");
    assert!(content.contains("1 image(s) and 1 audio(s)"));
}
