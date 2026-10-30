use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use base64::Engine;
mod common;
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::schema::ModelEntry;
use tower::ServiceExt;

fn sample_qwen_vision_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-27b",
        "name": "Qwen3.8 27B",
        "modality": "VisionText",
        "tasks": ["text-image-to-text", "text-image-to-detection"],
        "capabilities": ["Chat", "Vision", "Tool Call", "Reasoning"],
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

fn create_test_app() -> axum::Router {
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
    catalog.models.push(sample_qwen_vision_config());

    common::create_mock_app(catalog)
}

fn sample_png_base64() -> String {
    // 1x1 red PNG
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

#[tokio::test]
async fn test_multimodal_vision_chat_with_several_base64_images() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    // Payload with two screenshots in the user message
    let payload = serde_json::json!({
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Compare these two screenshots and describe UI differences." },
                    { "type": "image_url", "image_url": { "url": img_uri } },
                    { "type": "image_url", "image_url": { "url": img_uri } }
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
    assert!(res["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn test_vision_plus_tools_with_tool_choice_required() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    // Request combining vision screenshot + tools + tool_choice: "required"
    let payload = serde_json::json!({
        "model": "qwen3.8-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Click the Save button." },
                    { "type": "image_url", "image_url": { "url": img_uri } }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "click_coordinate",
                    "description": "Click on specific screen coordinates",
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
    assert!(res["choices"][0]["message"].get("content").is_some());
}

#[tokio::test]
async fn test_encoder_grounding_refuses_fabricated_coordinates() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    let payload = serde_json::json!({
        "model": "qwen3.8-27b",
        "image": img_uri,
        "query": "Save button"
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/images/grounding")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(resp.headers()["x-model-served"], "qwen3.8-27b");
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unsupported")
    );
    assert!(response.get("point").is_none());
}
