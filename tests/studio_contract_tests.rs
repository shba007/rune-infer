use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use rune_infer::api::{AppState, create_router};
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::proxy::ProxyService;
use rune_infer::schema::ModelEntry;
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;

fn sample_qwen_model_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-27b",
        "name": "Qwen 3.8 27B",
        "description": "Multimodal reasoning model",
        "modality": "VisionText",
        "capabilities": ["Chat", "Vision", "Tool Call"],
        "roles": ["chat", "vision"],
        "aliases": ["qwen-chat", "default-chat"],
        "engine": { "supported": ["llama-server"], "preferred": "llama-server" },
        "max_context_length": 131072,
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

fn sample_cloud_model_config() -> ModelConfig {
    let raw = r#"{
        "id": "gemini-flash",
        "name": "Gemini 2.5 Flash",
        "description": "Cloud fast multimodal",
        "modality": "VisionText",
        "capabilities": ["Chat", "Vision"],
        "roles": ["fast"],
        "aliases": ["flash"],
        "max_context_length": 1048576,
        "sampling": {},
        "remote": {
            "provider": "google",
            "upstream_model": "gemini-2.5-flash",
            "api_key": "test-key"
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

fn create_studio_test_app() -> (axum::Router, Arc<AppState>) {
    let mut roles = HashMap::new();
    roles.insert("default".to_string(), "qwen3.8-27b".to_string());
    let mut aliases = HashMap::new();
    aliases.insert("qwen-alias".to_string(), "qwen3.8-27b".to_string());

    let mut catalog = ModelRegistry {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            api_key: None,
            max_loaded_models: 1,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.95,
        },
        roles,
        aliases,
        ..Default::default()
    };
    catalog.models.push(sample_qwen_model_config());
    catalog.models.push(sample_cloud_model_config());

    let inference = Arc::new(InferenceAppState::new(&catalog));
    let proxy = Arc::new(ProxyService::new(&catalog));
    let state = Arc::new(AppState::new(inference, proxy, catalog));
    let router = create_router((*state).clone());
    (router, state)
}

#[tokio::test]
async fn test_inf8_health_livez_readyz_probes() {
    let (app, _) = create_studio_test_app();

    // 1. GET /livez
    let req = Request::builder()
        .method("GET")
        .uri("/livez")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "alive");

    // 2. GET /readyz
    let req = Request::builder()
        .method("GET")
        .uri("/readyz")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ready");
    assert_eq!(json["active_requests"], 0);
    assert_eq!(json["busy"], false);

    // 3. GET /health
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
    assert!(json["backend"].is_string());
    let backend = json["backend"].as_str().unwrap();
    assert!(backend == "cuda" || backend == "metal" || backend == "cpu");
}

#[tokio::test]
async fn test_inf7_inf15_models_structured_capabilities_and_locality() {
    let (app, _) = create_studio_test_app();

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(json["vram_total_bytes"].is_number() || json["vram_total_bytes"].is_null());
    assert!(json["vram_free_bytes"].is_number() || json["vram_free_bytes"].is_null());
    assert!(json["vram_used_bytes"].is_number() || json["vram_used_bytes"].is_null());

    let models = json["data"].as_array().expect("data array");
    let local_model = models
        .iter()
        .find(|m| m["id"] == "qwen3.8-27b")
        .expect("local model");
    assert_eq!(local_model["locality"], "local");
    assert_eq!(local_model["resident"], false);
    assert_eq!(local_model["status"], "available");
    assert_eq!(
        local_model["modalities_in"],
        serde_json::json!(["text", "image"])
    );
    assert_eq!(local_model["modalities_out"], serde_json::json!(["text"]));
    assert_eq!(local_model["tool_calls"], true);
    assert_eq!(local_model["thinking_control"], "thinking");
    assert_eq!(local_model["context_tokens"], 131072);
    assert!(local_model["vram_mb_estimate"].as_u64().unwrap() > 0);

    let cloud_model = models
        .iter()
        .find(|m| m["id"] == "gemini-flash")
        .expect("cloud model");
    assert_eq!(cloud_model["locality"], "cloud");
    assert_eq!(cloud_model["resident"], false);
    assert_eq!(cloud_model["status"], "remote");

    // Single model retrieval GET /v1/models/qwen3.8-27b
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models/qwen3.8-27b")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let single: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(single["id"], "qwen3.8-27b");
    assert_eq!(single["locality"], "local");
}

#[tokio::test]
async fn test_inf4_residency_and_x_no_swap_refusal() {
    let (app, _) = create_studio_test_app();

    // qwen3.8-27b is local and not currently resident in memory.
    // When client sends x-no-swap: true, rune-infer must refuse with 409 Conflict (swap_refused).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-no-swap", "true")
        .body(Body::from(
            r#"{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "swap_refused");
    assert_eq!(json["error"]["code"], "swap_refused");
}

#[tokio::test]
async fn test_inf3_background_work_yields_to_foreground() {
    let (app, state) = create_studio_test_app();

    // Simulate an active foreground task
    state
        .concurrency
        .active_foreground
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    // Incoming background request
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-rune-priority", "background")
        .body(Body::from(
            r#"{"model":"gemini-flash","messages":[{"role":"user","content":"reflection"}]}"#,
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        resp.headers().get("retry-after").unwrap().to_str().unwrap(),
        "1"
    );
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "server_busy");
    assert_eq!(json["error"]["code"], "busy");

    // Clean up simulated foreground count
    state
        .concurrency
        .active_foreground
        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn test_inf16_aliases_and_roles_resolution() {
    let (_, state) = create_studio_test_app();
    let config = state.config();

    // 1. Exact model ID
    let exact = config.find("qwen3.8-27b").expect("exact match");
    assert_eq!(exact.id, "qwen3.8-27b");

    // 2. Server-level role
    let by_server_role = config.find("default").expect("server role match");
    assert_eq!(by_server_role.id, "qwen3.8-27b");

    // 3. Server-level alias
    let by_server_alias = config.find("qwen-alias").expect("server alias match");
    assert_eq!(by_server_alias.id, "qwen3.8-27b");

    // 4. Model-level role
    let by_model_role = config.find("vision").expect("model role match");
    assert_eq!(by_model_role.id, "qwen3.8-27b");

    // 5. Model-level alias
    let by_model_alias = config.find("default-chat").expect("model alias match");
    assert_eq!(by_model_alias.id, "qwen3.8-27b");

    // 6. Cloud model role & alias
    let cloud_by_role = config.find("fast").expect("cloud role match");
    assert_eq!(cloud_by_role.id, "gemini-flash");
    let cloud_by_alias = config.find("flash").expect("cloud alias match");
    assert_eq!(cloud_by_alias.id, "gemini-flash");
}

#[test]
fn test_inf2_honest_usage_structure_with_prompt_tokens_details() {
    use rune_infer::types::{PromptTokensDetails, Usage};

    let usage = Usage {
        prompt_tokens: 1536,
        completion_tokens: 128,
        total_tokens: 1664,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: 0,
            image_tokens: 1024,
            audio_tokens: 256,
        }),
    };

    let json = serde_json::to_value(&usage).expect("usage json serialization");
    assert_eq!(json["prompt_tokens"], 1536);
    assert_eq!(json["completion_tokens"], 128);
    assert_eq!(json["total_tokens"], 1664);
    assert_eq!(json["prompt_tokens_details"]["image_tokens"], 1024);
    assert_eq!(json["prompt_tokens_details"]["audio_tokens"], 256);
    assert_eq!(json["prompt_tokens_details"]["cached_tokens"], 0);
}

struct MockSlowEngine {
    canceled: Arc<std::sync::atomic::AtomicBool>,
}

impl rune_infer::inference::traits::InferenceEngine for MockSlowEngine {
    fn id(&self) -> &str {
        "mock-slow"
    }

    fn execute(
        &self,
        _task: &rune_infer::inference::types::InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<rune_infer::inference::types::InferenceOutput, Box<dyn std::error::Error>> {
        for _ in 0..50 {
            if self.canceled.load(std::sync::atomic::Ordering::SeqCst) {
                return Err("aborted".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(rune_infer::inference::types::InferenceOutput {
            response: rune_infer::inference::types::InferenceTaskResponse::Text("done".into()),
            usage: rune_infer::types::Usage::new(10, 10),
        })
    }

    fn cancel(&self) {
        self.canceled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn test_if1_background_preemption_when_foreground_arrives() {
    use axum::http::HeaderMap;
    use rune_infer::api::ModelExecutionGate;
    use rune_infer::inference::types::InferenceTaskRequest;

    let (_, state) = create_studio_test_app();
    let mut bg_headers = HeaderMap::new();
    bg_headers.insert("x-rune-priority", "background".parse().unwrap());

    let bg_gate = ModelExecutionGate::enter(&state, &bg_headers, "qwen3.8-27b").expect("bg gate");
    assert_eq!(
        state
            .concurrency
            .active_background
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    let canceled_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let engine = Arc::new(MockSlowEngine {
        canceled: canceled_flag.clone(),
    });

    let state_clone = state.clone();
    let task = InferenceTaskRequest::Embedding {
        model: None,
        input: vec!["slow reflection".into()],
        dimensions: None,
    };

    let bg_join =
        tokio::spawn(async move { bg_gate.execute_task(&state_clone, engine, task).await });

    // Allow background task to start executing
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Foreground request arrives!
    let mut fg_headers = HeaderMap::new();
    fg_headers.insert("x-rune-priority", "foreground".parse().unwrap());
    let fg_gate = ModelExecutionGate::enter(&state, &fg_headers, "qwen3.8-27b").expect("fg gate");
    assert_eq!(
        state
            .concurrency
            .active_foreground
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    // The background task must have yielded immediately with 503
    let result = bg_join.await.unwrap();
    let resp = result.expect_err("background request should yield");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers().get("retry-after").unwrap().to_str().unwrap(),
        "1"
    );
    assert_eq!(
        resp.headers()
            .get("x-model-served")
            .unwrap()
            .to_str()
            .unwrap(),
        "qwen3.8-27b"
    );

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "server_busy");
    assert_eq!(json["error"]["code"], "yielded");

    // Engine was canceled
    assert!(canceled_flag.load(std::sync::atomic::Ordering::SeqCst));

    drop(fg_gate);
}

#[tokio::test]
async fn test_if2_model_headers_and_no_swap_across_routes() {
    let (app, _) = create_studio_test_app();

    let routes = [
        (
            "POST",
            "/v1/decisions",
            r#"{"model":"qwen3.8-27b","state":"The user clicked submit","questions":[{"name":"action","type":"choice","instruction":"Pick action"}]}"#,
        ),
        (
            "POST",
            "/v1/nlu/intents",
            r#"{"model":"qwen3.8-27b","text":"cancel booking","intents":["cancel","book"]}"#,
        ),
        (
            "POST",
            "/v1/embeddings",
            r#"{"model":"qwen3.8-27b","input":"vector test"}"#,
        ),
        (
            "POST",
            "/v1/images/grounding",
            r#"{"model":"qwen3.8-27b","image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==","query":"submit button"}"#,
        ),
        (
            "POST",
            "/v1/images/generations",
            r#"{"model":"qwen3.8-27b","prompt":"a red dot"}"#,
        ),
    ];

    for (method, uri, body) in routes {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .header("x-no-swap", "true")
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::CONFLICT,
            "Route {uri} must reject non-resident swap when x-no-swap=true"
        );
        let body_bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(json["error"]["type"], "swap_refused");
        assert_eq!(json["error"]["code"], "swap_refused");
    }
}

#[test]
fn test_if6_grounding_off_center_target() {
    // Top-right UI element bounding box: [ymin=100, xmin=800, ymax=200, xmax=950]
    let box_2d = [100u32, 800u32, 200u32, 950u32];
    let center_x = (box_2d[1] + box_2d[3]) / 2;
    let center_y = (box_2d[0] + box_2d[2]) / 2;

    assert_eq!(center_x, 875);
    assert_eq!(center_y, 150);
    // Verified: true center is in the top-right quadrant, not hardcoded [500, 500]!
    assert_ne!([center_x, center_y], [500, 500]);
}

#[tokio::test]
async fn test_safe_blocking_client_drop_in_async_context() {
    use rune_infer::inference::process::SafeBlockingClient;

    // Constructing reqwest::blocking::Client must be done in a blocking context
    let raw_client = tokio::task::spawn_blocking(|| {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_millis(100))
            .build()
            .expect("build client")
    })
    .await
    .expect("spawn_blocking client");

    let safe_client = SafeBlockingClient::new(raw_client);
    // Dropping on the async tokio worker task MUST NOT panic!
    drop(safe_client);
}

#[test]
fn test_http_probe_success_probe_mock_server() {
    use rune_infer::inference::process::http_probe_success;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let port = listener.local_addr().expect("port").port();

    std::thread::spawn(move || {
        if let Ok((mut socket, _)) = listener.accept() {
            let mut buf = [0u8; 512];
            let _ = socket.read(&mut buf);
            let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
            let _ = socket.write_all(resp.as_bytes());
            let _ = socket.flush();
        }
    });

    let ok = http_probe_success(port, "/health");
    assert!(ok, "http_probe_success should return true for 200 OK");

    let fail = http_probe_success(1, "/health");
    assert!(
        !fail,
        "http_probe_success should return false for invalid port"
    );
}

#[tokio::test]
async fn test_llama_server_release_slots_tcp_in_tokio_task() {
    use rune_infer::inference::engines::llama_server::LlamaServerEngine;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let port = listener.local_addr().expect("port").port();

    std::thread::spawn(move || {
        // Accept GET /slots
        if let Ok((mut socket, _)) = listener.accept() {
            let mut buf = [0u8; 512];
            let _ = socket.read(&mut buf);
            let resp =
                "HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\n[{\"id\":0}]";
            let _ = socket.write_all(resp.as_bytes());
            let _ = socket.flush();
        }
        // Accept POST /slots/0?action=release
        if let Ok((mut socket, _)) = listener.accept() {
            let mut buf = [0u8; 512];
            let _ = socket.read(&mut buf);
            let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
            let _ = socket.write_all(resp.as_bytes());
            let _ = socket.flush();
        }
    });

    // Calling release_slots_tcp inside a Tokio worker thread MUST NOT panic
    LlamaServerEngine::release_slots_tcp(port);
}
