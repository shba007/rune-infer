use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use rune_infer::{
    api::{AppState, create_router},
    config::{ModelConfig, ModelRegistry},
    inference::AppState as InferenceState,
    proxy::ProxyService,
    schema::ModelEntry,
};
use std::sync::Arc;
use tower::ServiceExt;

fn model(path: &str, tasks: &[&str]) -> ModelConfig {
    let entry: ModelEntry = serde_json::from_value(serde_json::json!({
        "id":"depth-anything-v2-small-fp32", "name":"Depth Anything V2 Small FP32",
        "engine":["litert"], "tasks":tasks,
        "local":{"path":path,"params":null,"bits_per_weight":32,"gpu_layers":0}
    }))
    .unwrap();
    ModelConfig::try_from(entry).unwrap()
}
fn router(model: ModelConfig) -> axum::Router {
    let config = ModelRegistry {
        models: vec![model],
        ..Default::default()
    };
    create_router(AppState::new(
        Arc::new(InferenceState::new(&config)),
        Arc::new(ProxyService::new(&config)),
        config,
    ))
}

#[tokio::test]
async fn depth_requires_verified_tasks_and_refuses_incompatible_engines_before_loading() {
    for (tasks, engine, expected_status, expected_code) in [
        (
            &[][..],
            "litert",
            StatusCode::SERVICE_UNAVAILABLE,
            "model_tasks_unverified",
        ),
        (
            &["image-to-image"][..],
            "litert",
            StatusCode::BAD_REQUEST,
            "model_task_unsupported",
        ),
        (
            &["image-to-depth"][..],
            "needle",
            StatusCode::BAD_REQUEST,
            "engine_task_unsupported",
        ),
    ] {
        let mut config = model("missing.tflite", tasks);
        config.engine = Some(serde_json::from_value(serde_json::json!([engine])).unwrap());
        let app = router(config);
        let response = app
            .oneshot(
                Request::post("/v1/inference?engine=depth-anything-v2-small-fp32")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"type":"image_depth","image_bytes":[1,2,3]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected_status);
        let body = String::from_utf8(
            to_bytes(response.into_body(), 65536)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(expected_code), "{body}");
    }
}

#[tokio::test]
#[ignore = "Requires pinned LiteRT 2.3.0 Windows x64 runtime, reviewed model and a public image fixture"]
async fn live_native_cpu_depth_through_primitive_api() {
    use base64::Engine;
    let model_path = std::env::var("LITERT_TEST_MODEL").expect("LITERT_TEST_MODEL");
    let image =
        std::fs::read(std::env::var("LITERT_TEST_IMAGE").expect("LITERT_TEST_IMAGE")).unwrap();
    let app = router(model(&model_path, &["image-to-depth"]));
    let response = app.oneshot(Request::post("/v1/inference?engine=depth-anything-v2-small-fp32")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&serde_json::json!({"type":"image_depth", "image_bytes":image,"preview":true})).unwrap())).unwrap()).await.unwrap();
    let status = response.status();
    let data: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), 32 * 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(status, StatusCode::OK, "{data}");
    let depth = &data["response"];
    assert_eq!(depth["width"], 686);
    assert_eq!(depth["height"], 518);
    assert_eq!(depth["units"], "relative");
    assert_eq!(depth["layout"], "row-major");
    let values = depth["values"].as_array().unwrap();
    assert_eq!(values.len(), 686 * 518);
    assert!(
        values
            .iter()
            .all(|value| value.as_f64().is_some_and(f64::is_finite))
    );
    assert!(depth["max"].as_f64().unwrap() > depth["min"].as_f64().unwrap());
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(depth["preview_png_base64"].as_str().unwrap())
        .unwrap();
    let preview = image::load_from_memory(&bytes).unwrap();
    assert_eq!((preview.width(), preview.height()), (686, 518));
    assert_eq!(preview.color(), image::ColorType::L16);
    if let Ok(path) = std::env::var("LITERT_TEST_PREVIEW") {
        std::fs::write(path, bytes).unwrap();
    }
    eprintln!(
        "Native CPU depth: {} values, range {}..{}, source {}x{}",
        values.len(),
        depth["min"],
        depth["max"],
        depth["source_width"],
        depth["source_height"]
    );
}
