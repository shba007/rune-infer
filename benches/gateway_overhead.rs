use divan::black_box;
use rune_infer::config::{ModelConfig, ModelRegistry, ServerConfig};
use rune_infer::inference::budget::{
    calculate_budget, estimate_bytes_per_token, estimate_total_layers,
};
use rune_infer::schema::ModelEntry;

fn main() {
    divan::main();
}

fn sample_qwen_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen3.8-27b",
        "name": "Qwen3.8 27B",
        "modality": "VisionText",
        "capabilities": ["Chat", "Vision"],
        "engine": { "supported": ["llama-server"] },
        "max_context_length": 262144,
        "sampling": { "temperature": 0.3, "max_tokens": 1024 },
        "local": {
            "path": "D:/Models/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_S.gguf",
            "params": 27000000000,
            "bits_per_weight": 3.44,
            "context_length": 131072,
            "gpu_layers": 99,
            "args": "-fa on --cache-type-k q4_0 --cache-type-v q4_0"
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

fn sample_server_config() -> ServerConfig {
    ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        api_key: None,
        max_loaded_models: 1,
        idle_unload_seconds: 300,
        vram_budget_ratio: 0.95,
    }
}

#[divan::bench]
fn vram_preflight_budget_calculation(bencher: divan::Bencher) {
    let model = sample_qwen_config();
    let server = sample_server_config();

    bencher.bench_local(|| {
        let plan = calculate_budget(black_box(&model), black_box(&server));
        black_box(plan)
    });
}

#[divan::bench]
fn kv_cache_bytes_estimation(bencher: divan::Bencher) {
    let model = sample_qwen_config();

    bencher.bench_local(|| {
        let bytes = estimate_bytes_per_token(black_box(&model));
        black_box(bytes)
    });
}

#[divan::bench]
fn model_layer_estimation(bencher: divan::Bencher) {
    let model = sample_qwen_config();

    bencher.bench_local(|| {
        let layers = estimate_total_layers(black_box(&model));
        black_box(layers)
    });
}

#[divan::bench]
fn model_registry_find_lookup(bencher: divan::Bencher) {
    let mut catalog = ModelRegistry::default();
    catalog.models.push(sample_qwen_config());

    bencher.bench_local(|| {
        let found = catalog.find(black_box("qwen3.8-27b"));
        black_box(found)
    });
}

#[divan::bench]
fn router_chat_request_in_memory(bencher: divan::Bencher) {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use rune_infer::api::{AppState, create_router};
    use rune_infer::inference::AppState as InferenceAppState;
    use rune_infer::proxy::ProxyService;
    use std::sync::Arc;
    use tower::ServiceExt;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut catalog = ModelRegistry {
        server: sample_server_config(),
        ..Default::default()
    };
    catalog.models.push(sample_qwen_config());

    let inference = Arc::new(InferenceAppState::new(&catalog));
    let proxy = Arc::new(ProxyService::new(&catalog));
    let state = AppState::new(inference, proxy, catalog);
    let app = create_router(state);

    bencher.bench_local(|| {
        rt.block_on(async {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"unknown-model-fast-reject","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap();

            let response = app.clone().oneshot(req).await.unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            black_box(response)
        })
    });
}
