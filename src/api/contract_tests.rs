//! Transport contracts use injected engines and never require weights or a GPU.
use super::*;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::InferenceOutput;
use axum::body::{Body, to_bytes};
use base64::Engine;
use tower::ServiceExt;

struct FailingEngine;
impl InferenceEngine for FailingEngine {
    fn id(&self) -> &str {
        "fixture"
    }
    fn execute(
        &self,
        _: &InferenceTaskRequest,
        _: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn std::error::Error>> {
        Err("fixture execution failure".into())
    }
}

fn app(engine: Option<Arc<dyn InferenceEngine>>) -> (Router, Arc<AppState>) {
    let model: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
        "id": "fixture", "name": "Fixture", "modality": "VisionText",
        "engine": {"supported": ["encoder"]},
        "local": {"path": "Cargo.toml"}, "extra": {}
    }))
    .unwrap();
    let mut config = ModelRegistry::default();
    config
        .models
        .push(crate::config::ModelConfig::try_from(model).unwrap());
    let inference = Arc::new(crate::inference::AppState::new(&config));
    if let Some(engine) = engine {
        inference
            .registry
            .lock()
            .unwrap()
            .engines
            .insert("fixture".into(), engine);
    }
    let state = Arc::new(AppState::new(
        inference,
        Arc::new(ProxyService::new(&config)),
        config,
    ));
    (create_router((*state).clone()), state)
}

fn json_routes() -> Vec<(&'static str, serde_json::Value)> {
    let image = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("../../tests/fixtures/studio/off-center.png"))
    );
    vec![
        (
            "/v1/decisions",
            serde_json::json!({"model":"fixture","state":"inspect","images":[image],"questions":[{"name":"red","type":"noul","instruction":"Is red visible?"}]}),
        ),
        (
            "/v1/nlu/intents",
            serde_json::json!({"model":"fixture","text":"cancel","intents":["cancel","book"]}),
        ),
        (
            "/v1/embeddings",
            serde_json::json!({"model":"fixture","input":"text"}),
        ),
        (
            "/v1/ocr",
            serde_json::json!({"model":"fixture","document":{"image_url":image}}),
        ),
        (
            "/v1/images/grounding",
            serde_json::json!({"model":"fixture","image":image,"query":"red rectangle"}),
        ),
        (
            "/v1/images/generations",
            serde_json::json!({"model":"fixture","prompt":"red rectangle"}),
        ),
        (
            "/v1/images/captions",
            serde_json::json!({"model":"fixture","image":image}),
        ),
        (
            "/v1/images/detections",
            serde_json::json!({"model":"fixture","image":image}),
        ),
        (
            "/v1/images/embeddings",
            serde_json::json!({"model":"fixture","image":image}),
        ),
        (
            "/v1/images/recognize",
            serde_json::json!({"model":"fixture","image":image,"targets":[]}),
        ),
        (
            "/v1/images/upscales",
            serde_json::json!({"model":"fixture","image":image}),
        ),
        (
            "/v1/images/restorations",
            serde_json::json!({"model":"fixture","image":image}),
        ),
        (
            "/v1/images/style-transfers",
            serde_json::json!({"model":"fixture","content_image":image,"style_image":image}),
        ),
        (
            "/v1/moderations",
            serde_json::json!({"model":"fixture","input":"text"}),
        ),
        (
            "/v1/audio/speech",
            serde_json::json!({"model":"fixture","input":"hello","voice":"default"}),
        ),
        (
            "/v1/inference?engine=fixture",
            serde_json::json!({"type":"embedding","model":"fixture","input":["text"],"dimensions":null}),
        ),
    ]
}

async fn verify_route(uri: &str, content_type: &str, body: Vec<u8>) {
    let (router, state) = app(None);
    let request = |header: &str, value: &str| {
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", content_type)
            .header(header, value)
            .body(Body::from(body.clone()))
            .unwrap()
    };
    let response = router
        .clone()
        .oneshot(request("x-no-swap", "true"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT, "{uri}");
    state
        .inference
        .registry
        .lock()
        .unwrap()
        .engines
        .insert("fixture".into(), Arc::new(FailingEngine));
    let response = router
        .clone()
        .oneshot(request("x-no-swap", "true"))
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "{uri}"
    );
    assert_eq!(response.headers()["x-model-served"], "fixture", "{uri}");
    assert_eq!(response.headers()["x-model-locality"], "local", "{uri}");
    assert_eq!(response.headers()["x-model-resident"], "true", "{uri}");
    state
        .concurrency
        .active_foreground
        .store(1, std::sync::atomic::Ordering::SeqCst);
    let response = router
        .oneshot(request("x-rune-priority", "background"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS, "{uri}");
    assert_eq!(response.headers()["retry-after"], "1");
}

macro_rules! json_route_contract {
    ($name:ident, $uri:literal) => {
        #[tokio::test]
        async fn $name() {
            let (_, body) = json_routes()
                .into_iter()
                .find(|(uri, _)| *uri == $uri)
                .unwrap();
            verify_route($uri, "application/json", serde_json::to_vec(&body).unwrap()).await;
        }
    };
}
json_route_contract!(decisions_contract, "/v1/decisions");
json_route_contract!(intents_contract, "/v1/nlu/intents");
json_route_contract!(embeddings_contract, "/v1/embeddings");
json_route_contract!(ocr_contract, "/v1/ocr");
json_route_contract!(grounding_contract, "/v1/images/grounding");
json_route_contract!(image_generation_contract, "/v1/images/generations");
json_route_contract!(caption_contract, "/v1/images/captions");
json_route_contract!(detection_contract, "/v1/images/detections");
json_route_contract!(image_embedding_contract, "/v1/images/embeddings");
json_route_contract!(recognition_contract, "/v1/images/recognize");
json_route_contract!(upscale_contract, "/v1/images/upscales");
json_route_contract!(restoration_contract, "/v1/images/restorations");
json_route_contract!(style_transfer_contract, "/v1/images/style-transfers");
json_route_contract!(moderation_contract, "/v1/moderations");
json_route_contract!(speech_contract, "/v1/audio/speech");
json_route_contract!(inference_contract, "/v1/inference?engine=fixture");

async fn multipart_contract(uri: &str, field: &str) {
    let mut body = format!("--fixture-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nfixture\r\n--fixture-boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nred rectangle\r\n--fixture-boundary\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"fixture.png\"\r\nContent-Type: application/octet-stream\r\n\r\n").into_bytes();
    body.extend_from_slice(include_bytes!("../../tests/fixtures/studio/off-center.png"));
    body.extend_from_slice(b"\r\n--fixture-boundary--\r\n");
    verify_route(uri, "multipart/form-data; boundary=fixture-boundary", body).await;
}
macro_rules! multipart_route_contract {
    ($name:ident, $uri:literal, $field:literal) => {
        #[tokio::test]
        async fn $name() {
            multipart_contract($uri, $field).await;
        }
    };
}
multipart_route_contract!(transcriptions_contract, "/v1/audio/transcriptions", "file");
multipart_route_contract!(translations_contract, "/v1/audio/translations", "file");
multipart_route_contract!(diarizations_contract, "/v1/audio/diarizations", "file");
multipart_route_contract!(
    speech_to_speech_contract,
    "/v1/audio/speech-to-speech",
    "file"
);
multipart_route_contract!(image_edit_contract, "/v1/images/edits", "image");

#[tokio::test]
async fn preemption_is_not_lost_before_background_registers_waiter() {
    let (_, state) = app(None);
    let mut headers = HeaderMap::new();
    headers.insert("x-rune-priority", "background".parse().unwrap());
    let background = ModelExecutionGate::enter(&state, &headers, "fixture").unwrap();
    let foreground = ModelExecutionGate::enter(&state, &HeaderMap::new(), "fixture").unwrap();
    drop(foreground); // Even an already completed foreground arrival must invalidate background work.
    let response = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        background.execute_task(
            &state,
            Arc::new(FailingEngine),
            InferenceTaskRequest::Embedding {
                model: None,
                input: vec!["text".into()],
                dimensions: None,
            },
        ),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["error"]["code"],
        "yielded"
    );
}

struct BlockingEngine {
    started: tokio::sync::Notify,
    canceled: std::sync::Mutex<bool>,
    released: std::sync::Condvar,
}

impl InferenceEngine for BlockingEngine {
    fn id(&self) -> &str {
        "fixture"
    }
    fn supports_cancellation(&self) -> bool {
        true
    }
    fn cancel(&self) {
        *self.canceled.lock().unwrap() = true;
        self.released.notify_all();
    }
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn std::error::Error>> {
        let mut canceled = self.canceled.lock().unwrap();
        if matches!(task, InferenceTaskRequest::Embedding { input, .. } if input[0] == "background")
            || matches!(task, InferenceTaskRequest::ToolCall { prompt, .. } if prompt.contains("background"))
        {
            self.started.notify_one();
        }
        // Both requests share the occupied slot. Foreground execution cannot
        // complete until cancel() actually releases the background slot.
        while !*canceled {
            let (next, timeout) = self
                .released
                .wait_timeout(canceled, std::time::Duration::from_secs(2))
                .unwrap();
            canceled = next;
            if timeout.timed_out() {
                return Err("slot release timeout".into());
            }
        }
        Err("fixture finished".into())
    }
}

#[tokio::test]
async fn foreground_http_request_starts_within_250ms_and_background_returns_503() {
    let engine = Arc::new(BlockingEngine {
        started: tokio::sync::Notify::new(),
        canceled: std::sync::Mutex::new(false),
        released: std::sync::Condvar::new(),
    });
    let (router, _) = app(Some(engine.clone()));
    let request = |priority: &str| {
        axum::http::Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .header("x-rune-priority", priority)
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({"model":"fixture","input":priority}))
                    .unwrap(),
            ))
            .unwrap()
    };
    let background = tokio::spawn(router.clone().oneshot(request("background")));
    tokio::time::timeout(std::time::Duration::from_secs(1), engine.started.notified())
        .await
        .unwrap();
    let started = std::time::Instant::now();
    let foreground = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        router.oneshot(request("foreground")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(foreground.status(), StatusCode::INTERNAL_SERVER_ERROR); // Mock intentionally returns an error after execution.
    let background = tokio::time::timeout(std::time::Duration::from_millis(250), background)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(background.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(background.headers()["retry-after"], "1");
    assert!(*engine.canceled.lock().unwrap());
    assert!(started.elapsed() < std::time::Duration::from_millis(250));
    let body = to_bytes(background.into_body(), 4096).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["error"]["code"],
        "yielded"
    );
}

#[test]
fn absent_vram_is_unknown_and_never_serialized_as_zero_or_null() {
    let response = ModelsResponse {
        object: "list".into(),
        data: vec![],
        vram_total_bytes: None,
        vram_used_bytes: None,
        vram_free_bytes: None,
    };
    let json = serde_json::to_value(response).unwrap();
    for field in ["vram_total_bytes", "vram_used_bytes", "vram_free_bytes"] {
        assert!(json.get(field).is_none(), "{field}");
    }
}

#[tokio::test]
async fn no_swap_rechecks_residency_without_starting_a_load() {
    let (_, state) = app(Some(Arc::new(FailingEngine)));
    let mut headers = HeaderMap::new();
    headers.insert("x-no-swap", "true".parse().unwrap());
    let gate = ModelExecutionGate::enter(&state, &headers, "fixture").unwrap();
    state.inference.registry.lock().unwrap().engines.clear();
    let response = match gate.load_engine(&state).await {
        Ok(_) => panic!("must refuse a cold load"),
        Err(response) => response,
    };
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(state.inference.registry.lock().unwrap().loading.is_empty());
}

#[tokio::test]
async fn background_stream_holds_concurrency_and_reports_yield_after_http_200() {
    let engine = Arc::new(BlockingEngine {
        started: tokio::sync::Notify::new(),
        canceled: std::sync::Mutex::new(false),
        released: std::sync::Condvar::new(),
    });
    let (router, state) = app(Some(engine.clone()));
    let request = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json").header("x-rune-priority", "background")
        .body(Body::from(r#"{"model":"fixture","stream":true,"messages":[{"role":"user","content":"background"}]}"#)).unwrap();
    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        state
            .concurrency
            .active_background
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let body = tokio::spawn(to_bytes(response.into_body(), 100_000));
    tokio::time::timeout(std::time::Duration::from_secs(1), engine.started.notified())
        .await
        .unwrap();
    let foreground = ModelExecutionGate::enter(&state, &HeaderMap::new(), "fixture").unwrap();
    let bytes = tokio::time::timeout(std::time::Duration::from_millis(250), body)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("yielded"));
    assert_eq!(
        state
            .concurrency
            .active_background
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    drop(foreground);
}

#[tokio::test]
async fn completed_background_stream_does_not_wait_for_a_future_foreground() {
    struct ImmediateEngine;
    impl InferenceEngine for ImmediateEngine {
        fn id(&self) -> &str {
            "fixture"
        }
        fn supports_cancellation(&self) -> bool {
            true
        }
        fn execute(
            &self,
            _: &InferenceTaskRequest,
            _: Option<&mut dyn FnMut(&str) -> bool>,
        ) -> Result<InferenceOutput, Box<dyn std::error::Error>> {
            Err("fixture finished".into())
        }
    }
    let (router, _) = app(Some(Arc::new(ImmediateEngine)));
    let request = axum::http::Request::builder().method("POST").uri("/v1/chat/completions")
        .header("content-type", "application/json").header("x-rune-priority", "background")
        .body(Body::from(r#"{"model":"fixture","stream":true,"messages":[{"role":"user","content":"background"}]}"#)).unwrap();
    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = tokio::time::timeout(
        std::time::Duration::from_millis(250),
        to_bytes(response.into_body(), 100_000),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
}

struct StructuredEngine(&'static str);
impl InferenceEngine for StructuredEngine {
    fn id(&self) -> &str {
        "fixture"
    }
    fn execute(
        &self,
        _: &InferenceTaskRequest,
        _: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn std::error::Error>> {
        Ok(InferenceOutput {
            response: crate::inference::types::InferenceTaskResponse::Text(self.0.into()),
            usage: crate::types::Usage::new(10, 20),
        })
    }
}

#[tokio::test]
async fn structured_chat_preserves_root_and_rejects_invalid_completions() {
    for (content, expected) in [
        (r#"{"answer":"ok","findings":[]}"#, StatusCode::OK),
        (
            r#"[{"answer":"ok","findings":[]}]"#,
            StatusCode::BAD_GATEWAY,
        ),
        (r#"{"findings":[]}"#, StatusCode::BAD_GATEWAY),
        ("{", StatusCode::BAD_GATEWAY),
    ] {
        let (router, state) = app(Some(Arc::new(StructuredEngine(content))));
        state.config.write().unwrap().models[0].capabilities = Some(vec![
            "Chat".into(),
            "Vision".into(),
            "Structured Output".into(),
        ]);
        let body = serde_json::json!({"model":"fixture","messages":[{"role":"user","content":"describe"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"report","strict":true,
                "schema":{"type":"object","additionalProperties":false,"required":["answer","findings"],
                "properties":{"answer":{"type":"string"},"findings":{"type":"array"}}}}}});
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        if expected == StatusCode::OK {
            assert_eq!(value["choices"][0]["message"]["content"], content);
            assert!(value["choices"][0]["message"]["tool_calls"].is_null());
        } else {
            assert_eq!(value["error"]["code"], "structured_output_invalid");
        }
    }
}

#[tokio::test]
async fn structured_chat_rejects_streaming_and_invalid_schema_before_execution() {
    for (format, stream, code) in [
        (
            serde_json::json!({"type":"json_schema","json_schema":{"name":"fixture_schema","schema":{"type":"object"}}}),
            true,
            "unsupported_response_format",
        ),
        (
            serde_json::json!({"type":"json_schema","json_schema":{"name":"fixture_schema","schema":{"type":"invalid"}}}),
            false,
            "invalid_json_schema",
        ),
        (
            serde_json::json!({"type":"json_schema"}),
            false,
            "invalid_json_schema",
        ),
    ] {
        let (router, _) = app(Some(Arc::new(FailingEngine)));
        let body = serde_json::json!({"model":"fixture","messages":[{"role":"user","content":"test"}],"response_format":format,"stream":stream});
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(value["error"]["code"], code);
    }
}
