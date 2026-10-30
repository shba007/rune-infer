use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use base64::Engine;
mod common;
use rune_infer::config::{EngineKind, Modality, ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::engines::encoder::EncoderEngine;
use rune_infer::inference::traits::InferenceEngine;
use rune_infer::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use rune_infer::schema::ModelEntry;
use rune_infer::types::{DecisionAnswer, DecisionQuestion, DecisionResponse};
use std::path::Path;
use tower::ServiceExt;

fn sample_liquid_d1_config() -> ModelConfig {
    let raw = r#"{
        "id": "liquid-d1-3b",
        "name": "Liquid AI d1-3B",
        "description": "Multimodal System 1 decision model",
        "modality": "Decision",
        "tasks": ["text-image-to-decision"],
        "capabilities": ["Decision", "System1", "Classification", "Scoring", "Vision"],
        "engine": { "supported": ["encoder"], "preferred": "encoder" },
        "max_context_length": 32768,
        "sampling": {},
        "local": {
            "path": "Cargo.toml",
            "params": 3120000000,
            "bits_per_weight": 4.5,
            "context_length": 32768,
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
    catalog.models.push(sample_liquid_d1_config());

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

#[test]
fn test_liquid_d1_catalog_configuration() {
    let server_path = Path::new("config/server.json");
    let catalog_path = Path::new("config/catalog.json");
    let overrides = vec![
        ("GEMINI_API_KEY".to_string(), "test-gemini-key".to_string()),
        (
            "OPENROUTER_API_KEY".to_string(),
            "test-openrouter-key".to_string(),
        ),
        ("RUNE_API_KEY".to_string(), "test-rune-key".to_string()),
    ];
    let registry = ModelRegistry::from_files_with_env(server_path, catalog_path, &overrides)
        .expect("Failed to parse server and catalog config");

    let model = registry
        .find("liquid-d1-3b")
        .expect("Model 'liquid-d1-3b' must be present in catalog");

    assert_eq!(model.id, "liquid-d1-3b");
    assert_eq!(model.name, "Liquid AI d1-3B");
    assert_eq!(model.modality, Modality::Decision);
    assert!(model.is_decision());
    assert!(!model.is_vision());
    assert_eq!(model.tasks, ["text-to-decision"]);

    let engine = model.engine.as_ref().expect("Local model must have engine");
    assert!(engine.supported.contains(&EngineKind::LlamaServer));
    assert_eq!(model.engine_plan().first(), Some(&EngineKind::LlamaServer));
    assert_eq!(model.max_context_length, Some(32768));

    // Case-insensitive and short-name lookup tests
    assert_eq!(
        registry.find("LIQUID-D1-3B").map(|m| m.id.as_str()),
        Some("liquid-d1-3b")
    );
    assert_eq!(
        registry.find("d1-3b").map(|m| m.id.as_str()),
        Some("liquid-d1-3b")
    );
    assert_eq!(
        registry.find("Liquid AI d1-3B").map(|m| m.id.as_str()),
        Some("liquid-d1-3b")
    );
}

#[test]
fn test_encoder_engine_decision_execution() {
    let config = sample_liquid_d1_config();
    let engine = EncoderEngine::new(&config).expect("failed to create encoder engine");

    let questions = vec![
        DecisionQuestion {
            name: "routing_mode".to_string(),
            r#type: "choice".to_string(),
            instruction: "Select optimal routing".to_string(),
            criteria: Some(serde_json::json!(["fast", "balanced", "thorough"])),
        },
        DecisionQuestion {
            name: "requires_human_approval".to_string(),
            r#type: "noul".to_string(),
            instruction: "Does action require human approval?".to_string(),
            criteria: None,
        },
        DecisionQuestion {
            name: "risk_assessment".to_string(),
            r#type: "score".to_string(),
            instruction: "Score potential operation risk".to_string(),
            criteria: None,
        },
    ];

    let task = InferenceTaskRequest::Decision {
        model: Some("liquid-d1-3b".to_string()),
        text: "User is requesting a system kernel update in production environment.".to_string(),
        images: Vec::new(),
        questions,
    };

    let result = engine.execute(&task, None).expect("execution succeeded");
    match result.response {
        InferenceTaskResponse::Decision(resp) => {
            assert_eq!(resp.object, "decision");
            assert_eq!(resp.model, "liquid-d1-3b");
            // Zero-token output contract check
            assert_eq!(resp.usage.completion_tokens, 0);

            // Choice assertion
            if let Some(DecisionAnswer::Choice {
                decision,
                confidence,
                probabilities,
            }) = resp.answers.get("routing_mode")
            {
                assert!(!decision.is_empty());
                assert!(*confidence > 0.0);
                assert_eq!(probabilities.len(), 3);
            } else {
                panic!("Expected DecisionAnswer::Choice");
            }

            // Noul assertion
            if let Some(DecisionAnswer::Noul {
                decision: _,
                probability,
            }) = resp.answers.get("requires_human_approval")
            {
                assert!(*probability >= 0.0 && *probability <= 1.0);
            } else {
                panic!("Expected DecisionAnswer::Noul");
            }

            // Score assertion
            if let Some(DecisionAnswer::Score {
                level,
                score,
                probabilities,
            }) = resp.answers.get("risk_assessment")
            {
                assert!(!level.is_empty());
                assert!(*score >= 0.0 && *score <= 1.0);
                assert!(!probabilities.is_empty());
            } else {
                panic!("Expected DecisionAnswer::Score");
            }
        }
        other => panic!("Unexpected response variant: {other:?}"),
    }
}

#[tokio::test]
async fn test_api_v1_decisions_choice_and_noul() {
    let app = create_test_app();

    let payload = serde_json::json!({
        "model": "liquid-d1-3b",
        "state": "System memory usage at 94%, incoming high-priority transaction batch.",
        "questions": [
            {
                "name": "action",
                "type": "choice",
                "instruction": "Select resource allocation action",
                "criteria": ["scale_up", "throttle_batch", "shed_load"]
            },
            {
                "name": "is_critical",
                "type": "noul",
                "instruction": "Is immediate system intervention needed?"
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: DecisionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(res.object, "decision");
    assert_eq!(res.model, "liquid-d1-3b");
    assert_eq!(res.usage.completion_tokens, 0);
    assert!(res.answers.contains_key("action"));
    assert!(res.answers.contains_key("is_critical"));
}

#[tokio::test]
async fn test_api_v1_decisions_multimodal_vision_state() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    let payload = serde_json::json!({
        "model": "liquid-d1-3b",
        "state": {
            "text": "Inspect UI frame for pop-up modal and security alert confirmation button.",
            "images": [img_uri]
        },
        "questions": [
            {
                "name": "modal_detected",
                "type": "noul",
                "instruction": "Is an active dialogue modal visible in the screen frame?"
            },
            {
                "name": "intent",
                "type": "choice",
                "instruction": "Identify user interface element to interact with",
                "criteria": ["click_confirm", "click_cancel", "dismiss_modal", "ignore"]
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: DecisionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(res.object, "decision");
    assert_eq!(res.usage.completion_tokens, 0);
    assert!(res.answers.contains_key("modal_detected"));
    assert!(res.answers.contains_key("intent"));
}

#[tokio::test]
async fn test_api_v1_decisions_empty_questions_validation() {
    let app = create_test_app();

    let payload = serde_json::json!({
        "model": "liquid-d1-3b",
        "state": "Some state description",
        "questions": []
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(res["error"]["code"], "empty_questions");
}

#[tokio::test]
async fn test_api_v1_decisions_empty_state_validation() {
    let app = create_test_app();

    let payload = serde_json::json!({
        "model": "liquid-d1-3b",
        "state": "   ",
        "questions": [
            {
                "name": "q1",
                "type": "noul",
                "instruction": "Is it valid?"
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(res["error"]["code"], "empty_state");
}

#[tokio::test]
async fn test_api_v1_decisions_top_level_images_and_screenshot() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    let payload = serde_json::json!({
        "model": "liquid-d1-3b",
        "state": "Evaluate current application UI screenshot.",
        "images": [img_uri.clone()],
        "questions": [
            {
                "name": "dialog_open",
                "type": "noul",
                "instruction": "Is an active dialogue modal visible?"
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: DecisionResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(res.model, "liquid-d1-3b");
    assert!(res.answers.contains_key("dialog_open"));
    // Verify prompt tokens account for image frames (> 256 tokens)
    assert!(res.usage.prompt_tokens >= 256);
}

#[tokio::test]
async fn test_api_v1_decisions_state_only_images_no_text() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    let payload = serde_json::json!({
        "model": "d1-3b",
        "state": {
            "images": [img_uri]
        },
        "questions": [
            {
                "name": "is_alert_visible",
                "type": "noul",
                "instructions": "Is an alert popup visible in the image?"
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: DecisionResponse = serde_json::from_slice(&body).unwrap();

    // Canonical model ID should be liquid-d1-3b
    assert_eq!(res.model, "liquid-d1-3b");
    assert!(res.answers.contains_key("is_alert_visible"));
    assert!(res.usage.prompt_tokens >= 256);
}

#[tokio::test]
async fn test_api_v1_decisions_questions_as_map_with_instructions_plural() {
    let app = create_test_app();
    let img_uri = sample_png_base64();

    let payload = serde_json::json!({
        "model": "d1-3b-q8-0",
        "state": {
            "screenshot": img_uri
        },
        "questions": {
            "popup_detected": {
                "type": "noul",
                "instructions": "Is a popup modal displayed?"
            },
            "suggested_action": {
                "type": "choice",
                "instructions": "Select next best automated action",
                "criteria": ["wait_for_load", "dismiss_popup", "proceed"]
            }
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let res: DecisionResponse = serde_json::from_slice(&body).unwrap();

    // Canonical model ID should resolve from alias "d1-3b-q8-0" to "liquid-d1-3b"
    assert_eq!(res.model, "liquid-d1-3b");
    assert!(res.answers.contains_key("popup_detected"));
    assert!(res.answers.contains_key("suggested_action"));
}
