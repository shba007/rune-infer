//! Transport contracts use injected engines and never require weights or a GPU.
use super::*;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::InferenceOutput;
use axum::body::{Body, to_bytes};
use base64::Engine;
use tower::ServiceExt;

#[tokio::test]
async fn local_execution_contract_refuses_proxy_and_unsupported_routes() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let received = Arc::new(AtomicUsize::new(0));
    let counter = received.clone();
    let upstream = Router::new().fallback(axum::routing::any(move || {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "mock cloud received request",
            )
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let (_, state) = app(None);
    let mut config = state.config().clone();
    let discovered_local = config.models[0].clone();
    let model = &mut config.models[0];
    model.provider = crate::config::ProviderType::OpenAi;
    model.engine = None;
    model.base_url = Some(url);
    model.rate_limit = Some(crate::config::RateLimitConfig {
        max_retries: 0,
        ..Default::default()
    });
    state.reload_catalog(config.clone());
    // Simulate a reload after discovery/resolution but before engine acquisition.
    assert!(
        state
            .inference
            .get_local_engine(&discovered_local, false)
            .is_err()
    );
    assert_eq!(received.load(Ordering::SeqCst), 0);
    let router = create_router(AppState::new(
        state.inference.clone(),
        Arc::new(ProxyService::new(&config)),
        config,
    ));
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let capabilities: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
    assert_eq!(capabilities["local_execution"]["version"], 1);
    assert_eq!(
        capabilities["local_execution"]["endpoints"],
        serde_json::json!([{"method":"POST","path":"/v1/chat/completions"}])
    );
    for (uri, id, value, duplicate, code) in [
        (
            "/v1/chat/completions",
            "fixture",
            "1",
            false,
            "local_execution_required",
        ),
        (
            "/v1/chat/completions",
            "missing",
            "1",
            false,
            "model_not_found",
        ),
        (
            "/v1/chat/completions",
            "fixture",
            "true",
            false,
            "invalid_local_execution_constraint",
        ),
        (
            "/v1/chat/completions",
            "fixture",
            "1",
            true,
            "invalid_local_execution_constraint",
        ),
        (
            "/v1/embeddings",
            "fixture",
            "1",
            false,
            "local_execution_unsupported",
        ),
        (
            "/v1/responses",
            "fixture",
            "1",
            false,
            "local_execution_unsupported",
        ),
    ] {
        let mut request = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("x-rune-local-only", value);
        if duplicate {
            request = request.header("x-rune-local-only", "1");
        }
        let response = router
            .clone()
            .oneshot(
                request
                    .body(Body::from(
                        serde_json::json!({
                            "model":id,"messages":[{"role":"user","content":"LOCAL_ONLY_SENTINEL"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_client_error(), "{uri}");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], code, "{uri}");
        assert_eq!(received.load(Ordering::SeqCst), 0);
    }
    // Positive control: the same cloud route is reachable without the constraint.
    let control = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"fixture","messages":[{"role":"user","content":"control"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let control_body = to_bytes(control.into_body(), 4096).await.unwrap();
    assert_eq!(
        received.load(Ordering::SeqCst),
        1,
        "{}",
        String::from_utf8_lossy(&control_body)
    );
    server.abort();
}

#[tokio::test]
async fn local_execution_pins_configuration_and_preserves_residency_policy() {
    let (router, state) = app(Some(Arc::new(StructuredEngine("READY"))));
    let mut config = state.config().clone();
    config.models[0].engine = Some(crate::config::EngineConfig {
        supported: vec![crate::config::EngineKind::LlamaServer],
        preferred: None,
    });
    state.reload_catalog(config);
    let expected = state.config().models[0].clone();
    state
        .inference
        .registry
        .lock()
        .unwrap()
        .engine_configs
        .insert(expected.id.clone(), expected.clone());
    let request = |id: &str, no_swap: bool| {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-rune-local-only", "1");
        if no_swap {
            builder = builder.header("x-no-swap", "1");
        }
        builder
            .body(Body::from(
                serde_json::json!({"model":id,
            "messages":[{"role":"user","content":"LOCAL_ONLY_SENTINEL"}]})
                .to_string(),
            ))
            .unwrap()
    };
    let response = router
        .clone()
        .oneshot(request("fixture", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-rune-local-only"], "1");
    let pinned = state.inference.get_local_engine(&expected, false).unwrap();
    let mut changed = state.config().clone();
    changed.models[0].model_path = "different.gguf".into();
    state.reload_catalog(changed.clone());
    assert!(state.inference.get_local_engine(&expected, false).is_err());
    // A resident engine from an older snapshot is also refused for a new snapshot.
    assert!(
        state
            .inference
            .get_local_engine(&changed.models[0], false)
            .is_err()
    );
    let response = router
        .clone()
        .oneshot(request("fixture", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    changed.models[0].provider = crate::config::ProviderType::OpenAi;
    state.reload_catalog(changed);
    assert!(state.inference.get_local_engine(&expected, false).is_err());
    // An acquired local engine remains pinned even after a local-to-cloud reload.
    assert!(Arc::ptr_eq(
        &pinned,
        &state.inference.registry.lock().unwrap().engines["fixture"]
    ));
    let mut restored = state.config().clone();
    restored.models[0] = expected.clone();
    restored.models[0].aliases.push("alias".into());
    state.reload_catalog(restored);
    let response = router
        .clone()
        .oneshot(request("alias", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    state.inference.registry.lock().unwrap().engines.clear();
    let response = router.oneshot(request("fixture", true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn unverified_models_are_refused_before_execution() {
    let (router, state) = app(Some(Arc::new(FailingEngine)));
    state.config.write().unwrap().models[0].tasks.clear();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"fixture","input":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["error"]["code"], "model_tasks_unverified");
}

#[tokio::test]
async fn task_combinations_are_checked_before_loading_or_proxying() {
    let image = serde_json::json!({"type":"image_url","image_url":{"url":"invalid-image"}});
    let audio = serde_json::json!({"type":"input_audio","input_audio":{"data":"invalid-audio","format":"wav"}});
    for (uri, tasks, body, code) in [
        (
            "/v1/chat/completions",
            vec!["text-to-text"],
            serde_json::json!({"messages":[{"role":"user","content":"hello"}],"modalities":["text","audio"]}),
            "unsupported_output_modality",
        ),
        (
            "/v1/chat/completions",
            vec![],
            serde_json::json!({"messages":[{"role":"user","content":"hello"}]}),
            "model_tasks_unverified",
        ),
        (
            "/v1/chat/completions",
            vec!["text-to-embedding"],
            serde_json::json!({"messages":[{"role":"user","content":"hello"}]}),
            "model_task_unsupported",
        ),
        (
            "/v1/chat/completions",
            vec!["text-image-to-text", "audio-to-text"],
            serde_json::json!({"messages":[{"role":"user","content":[image.clone(),audio.clone()]}]}),
            "model_task_unsupported",
        ),
        (
            "/v1/inference?engine=fixture",
            vec!["text-to-text"],
            serde_json::json!({"type":"embedding","input":["hello"]}),
            "model_task_unsupported",
        ),
        (
            "/v1/inference?engine=fixture",
            vec!["text-image-to-text", "audio-to-text"],
            serde_json::json!({"type":"tool_call","prompt":"hello","schema":[],"images":[[1]],"audios":[[2]]}),
            "model_task_unsupported",
        ),
        (
            "/v1/inference?engine=fixture",
            vec!["text-to-image"],
            serde_json::json!({"type":"image_generation","prompt":"square"}),
            "engine_task_unsupported",
        ),
    ] {
        for cloud in [false, true] {
            if cloud && code == "engine_task_unsupported" {
                continue;
            }
            let (router, state) = app(None);
            let mut body = body.clone();
            body["model"] = "fixture".into();
            {
                let mut config = state.config.write().unwrap();
                let model = &mut config.models[0];
                model.tasks = tasks.iter().map(|task| (*task).to_string()).collect();
                if cloud {
                    model.provider = crate::config::ProviderType::Google;
                    model.engine = None;
                }
            }
            let response = router
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                if code == "model_tasks_unverified" {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::BAD_REQUEST
                },
                "{uri}, cloud={cloud}"
            );
            let value: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap())
                    .unwrap();
            assert_eq!(value["error"]["code"], code, "{uri}, cloud={cloud}");
            let registry = state.inference.registry.lock().unwrap();
            assert!(registry.loaded_models().is_empty() && registry.loading_models().is_empty());
        }
    }
}

#[test]
fn task_subsets_require_one_matching_claim_and_implemented_engine() {
    let (_, state) = app(None);
    let mut model = state.config().models[0].clone();
    model.tasks = vec!["text-image-to-text".into()];
    for (inputs, output, engine, expected) in [
        (vec!["text"], "text", None, true),
        (vec!["image"], "text", None, true),
        (vec!["text", "image"], "text", None, true),
        (vec!["text", "audio"], "text", None, false),
        (vec!["text"], "embedding", None, false),
        (
            vec!["image"],
            "text",
            Some(crate::config::EngineKind::Needle),
            false,
        ),
        (
            vec!["text"],
            "text",
            Some(crate::config::EngineKind::SdServer),
            false,
        ),
        (
            vec!["image"],
            "text",
            Some(crate::config::EngineKind::LlamaServer),
            false,
        ),
    ] {
        assert_eq!(
            super::task::Requirements::new(&inputs, output, "chat")
                .validate(&model, engine)
                .is_ok(),
            expected
        );
    }
    model.tasks = vec!["image-to-text".into()];
    let request = serde_json::from_value(serde_json::json!({"model":"fixture","messages":[
        {"role":"user","content":[{"type":"image_url","image_url":{"url":"image"}}]}
    ]}))
    .unwrap();
    assert!(
        super::task::Requirements::chat(&request)
            .validate(&model, None)
            .is_ok()
    );

    let root = std::env::temp_dir().join(format!(
        "rune-task-projector-{}",
        crate::setup::generate_api_key()
    ));
    let projector = root.join("projector.gguf");
    model.mmproj_path = Some(projector.to_string_lossy().into_owned());
    model.tasks = vec!["text-image-audio-to-text".into()];
    for (vision, audio, inputs, expected) in [
        (true, false, vec!["image"], true),
        (true, false, vec!["audio"], false),
        (false, true, vec!["image"], false),
        (true, true, vec!["image", "audio"], true),
    ] {
        crate::detect::testing::Gguf::new("clip")
            .bool_kv("clip.has_vision_encoder", vision)
            .bool_kv("clip.has_audio_encoder", audio)
            .write(&projector);
        assert_eq!(
            super::task::Requirements::new(&inputs, "text", "chat")
                .validate(&model, Some(crate::config::EngineKind::LlamaServer))
                .is_ok(),
            expected
        );
    }
    std::fs::remove_file(projector).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[tokio::test]
#[ignore = "requires the configured MiniCPM weights, projector and installed llama-server; runs real GPU inference"]
async fn live_minicpm_task_combinations() {
    let catalog: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("config/catalog.json").unwrap()).unwrap();
    let entry: crate::schema::ModelEntry = serde_json::from_value(
        catalog["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "minicpm-o-4-5-q8-0")
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut model = crate::config::ModelConfig::try_from(entry).unwrap();
    model.runtime.as_mut().unwrap().context_length = 4096;
    let mut config = ModelRegistry::default();
    config.models.push(model);
    let inference = Arc::new(crate::inference::AppState::new(&config));
    let state = AppState::new(
        inference.clone(),
        Arc::new(ProxyService::new(&config)),
        config,
    );
    let router = create_router(state);
    let image = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("../../tests/fixtures/studio/off-center.png"))
    );
    for (task, content, expected) in [
        (
            "text-to-text",
            serde_json::json!("Reply with READY only."),
            vec!["ready"],
        ),
        (
            "text-image-to-text",
            serde_json::json!([
                {"type":"text","text":"Describe the colored shape and its position in one sentence."},
                {"type":"image_url","image_url":{"url":image}}
            ]),
            vec!["red", "square"],
        ),
    ] {
        let body = serde_json::json!({"model":"minicpm-o-4-5-q8-0","messages":[{"role":"user","content":content}],"thinking":false,"reasoning_effort":"none","temperature":0,"max_tokens":64});
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(240),
            router.clone().oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("x-rune-local-only", "1")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        let status = response.status();
        assert_eq!(response.headers()["x-rune-local-only"], "1");
        assert_eq!(response.headers()["x-model-locality"], "local");
        assert_eq!(
            response.headers()["x-model-resident"],
            if task == "text-to-text" {
                "false"
            } else {
                "true"
            }
        );
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, StatusCode::OK, "{task}: {value}");
        let answer = value["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(
            expected
                .iter()
                .all(|word| answer.to_ascii_lowercase().contains(word)),
            "{task}: {answer}"
        );
        println!("{task}: {answer}");
    }
    drop(router);
    drop(inference);
}

#[test]
fn primitive_catalog_roundtrip_keeps_tasks_and_engine_order() {
    let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
        "id":"fixture","name":"Fixture","tasks":["text-to-text", "text-image-to-text"],
        "engine":{"supported":["llama-server","llama-server-prism"],"preferred":"llama-server-prism"},
        "local":{"path":"model.gguf"}
    })).unwrap();
    let value = serde_json::to_value(&entry).unwrap();
    assert!(value.get("modality").is_none());
    assert!(value.get("sampling").is_none());
    assert!(value.get("extra").is_none());
    assert_eq!(
        value["engine"],
        serde_json::json!(["llama-server-prism", "llama-server"])
    );
    let restored: crate::schema::ModelEntry = serde_json::from_value(value).unwrap();
    let model = crate::config::ModelConfig::try_from(restored).unwrap();
    assert!(model.is_vision());
    assert!(!model.supports_audio());
    assert!(!model.supports_tool_calls());
}

#[tokio::test]
async fn models_filter_matches_explicit_metadata_and_rejects_invalid_bounds() {
    let (router, state) = app(None);
    let mut model = state.config().models[0].clone();
    model.tasks.clear();
    let mut known = model.clone();
    known.capabilities = Some(vec!["Uncensored".into(), "Text-to-Image".into()]);
    known.tasks = vec!["text-to-image".into()];
    assert_eq!(known.modalities_in(), vec!["text"]);
    assert_eq!(known.modalities_out(), vec!["image"]);
    let mut multimodal = known.clone();
    multimodal.tasks = vec!["text-to-text".into(), "text-image-to-text".into()];
    assert_eq!(multimodal.modalities_in(), vec!["text", "image"]);
    assert_eq!(multimodal.modalities_out(), vec!["text"]);
    assert!(model.modalities_in().is_empty());
    assert!(model.modalities_out().is_empty());
    known.total_params = Some(7_000_000_000);
    known.model_path = "model-Q8_0.gguf".into();
    assert!(
        ModelFilters {
            censorship: Some("uncensored".into()),
            modality: Some("text_to_image".into()),
            min_params: Some(7_000_000_000),
            max_params: Some(7_000_000_000),
            quantization: Some("q8_0".into()),
        }
        .matches(&known)
    );
    assert!(
        ModelFilters {
            censorship: Some("censored".into()),
            ..Default::default()
        }
        .matches(&model)
    );
    assert!(
        !ModelFilters {
            censorship: Some("uncensored".into()),
            ..Default::default()
        }
        .matches(&model)
    );
    assert!(
        !ModelFilters {
            min_params: Some(1),
            ..Default::default()
        }
        .matches(&model)
    );
    assert!(
        !ModelFilters {
            modality: Some("text-to-image".into()),
            ..Default::default()
        }
        .matches(&model)
    );
    for (uri, status, count) in [
        ("/v1/models?censorship=censored", StatusCode::OK, Some(1)),
        ("/v1/models?censorship=uncensored", StatusCode::OK, Some(0)),
        ("/v1/models?uncensored=true", StatusCode::BAD_REQUEST, None),
        ("/v1/models?modality=vision", StatusCode::BAD_REQUEST, None),
        (
            "/v1/models?min_params=10&max_params=1",
            StatusCode::BAD_REQUEST,
            None,
        ),
    ] {
        let response = router
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        if let Some(count) = count {
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["data"].as_array().unwrap().len(), count);
        }
    }
}

#[test]
fn task_metadata_requires_explicit_unique_input_output_types() {
    for task in [
        "text-to-text",
        "text-image-to-image",
        "text-image-audio-video-to-text",
        "audio-to-embedding",
        "text-image-to-decision",
    ] {
        assert!(crate::schema::valid_task(task), "{task}");
    }
    for task in [
        "vision",
        "omni",
        "text-to-",
        "-to-text",
        "text-text-to-image",
        "text-to-audio-audio",
        "text-to-unknown",
    ] {
        assert!(!crate::schema::valid_task(task), "{task}");
    }
}

#[test]
fn catalog_vision_models_do_not_match_unrelated_filters() {
    let catalog: crate::schema::CatalogFile =
        serde_json::from_str(include_str!("../../config/catalog.json")).unwrap();
    for id in [
        "gemma-4-12b-it",
        "minicpm-o-4-5-q8-0",
        "qwen3.8-27b",
        "ternary-bonsai-2-27b-pq2-0",
        "ornith-1.5-9b",
    ] {
        let entry = catalog
            .models
            .iter()
            .find(|entry| entry.id == id)
            .unwrap()
            .clone();
        let model = crate::config::ModelConfig::try_from(entry).unwrap();
        assert!(
            !ModelFilters {
                censorship: Some("uncensored".into()),
                ..Default::default()
            }
            .matches(&model),
            "{id}"
        );
        assert!(
            !ModelFilters {
                modality: Some("text-to-image".into()),
                ..Default::default()
            }
            .matches(&model),
            "{id}"
        );
    }
}

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
        "tasks": ["text-image-to-text", "text-to-image", "image-to-image", "audio-to-text",
            "text-to-audio", "audio-to-audio", "audio-to-diarization", "text-to-embedding",
            "image-to-embedding", "text-to-moderation", "text-image-to-detection", "text-image-to-decision"],
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
