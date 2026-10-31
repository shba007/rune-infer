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
#[ignore = "Requires pinned LiteRT runtime/helper, reviewed model and image; checks setup saving and real loopback HTTP"]
async fn live_native_cpu_depth_through_primitive_api() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("rune_infer::inference::engines::litert=info")
        .with_test_writer()
        .try_init();
    use base64::Engine;
    use std::path::{Path, PathBuf};
    let model_path = std::env::var("LITERT_TEST_MODEL").expect("LITERT_TEST_MODEL");
    let image =
        std::fs::read(std::env::var("LITERT_TEST_IMAGE").expect("LITERT_TEST_IMAGE")).unwrap();
    let id = std::env::var("LITERT_TEST_MODEL_ID").unwrap_or("depth-anything-v2-small-fp32".into());
    let expected_name =
        std::env::var("LITERT_TEST_MODEL_NAME").unwrap_or("Depth Anything V2 Small FP32".into());
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let root = target.join(format!(
        "depth-acceptance-{}",
        rune_infer::setup::generate_api_key()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let artifact = root.join(format!("{id}.tflite"));
    std::fs::hard_link(&model_path, &artifact).unwrap();
    let found = rune_infer::setup::discover_models(&artifact).unwrap();
    assert_eq!(found.models.len(), 1);
    assert_eq!(found.models[0].display_name, expected_name);
    let plan = rune_infer::setup::SetupPlan {
        exposure: rune_infer::setup::Exposure::Local,
        port: 8080,
        server_api_key: None,
        cloud: Vec::new(),
        local: found.models,
        gpu_available: true,
    };
    let server_path = root.join("config/server.json");
    let catalog_path = root.join("config/catalog.json");
    let env_path = root.join(".env");
    assert!(
        rune_infer::setup::write_verified_setup(
            &plan,
            None,
            &server_path,
            &catalog_path,
            &env_path
        )
        .is_err()
    );
    assert!(!catalog_path.exists());
    let proof =
        rune_infer::setup::prepare_local_models(&plan.local, plan.gpu_available, |message| {
            println!("{message}")
        })
        .unwrap();
    rune_infer::setup::write_verified_setup(
        &plan,
        Some(&proof),
        &server_path,
        &catalog_path,
        &env_path,
    )
    .unwrap();
    let saved = ModelRegistry::from_files(&server_path, &catalog_path).unwrap();
    assert_eq!(saved.models[0].id, id);
    assert_eq!(saved.models[0].tasks, ["image-to-depth"]);
    assert_eq!(saved.models[0].bits_per_weight, Some(32.0));
    assert_eq!(saved.models[0].runtime.as_ref().unwrap().gpu_layers, 99);
    let app = router(saved.models[0].clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap();
    let listing: serde_json::Value = client
        .get(format!("http://{address}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listing["data"][0]["id"], id);
    assert_eq!(
        listing["data"][0]["tasks"],
        serde_json::json!(["image-to-depth"])
    );
    let definition = include_str!(
        "../postman/collections/Rune Infer REST API/Images/Estimate Depth.request.yaml"
    )
    .replace("\r\n", "\n");
    assert!(definition.contains("    value: '{{depth_model}}'\n    disabled: false"));
    let body = definition
        .split("  content: |-\n")
        .nth(1)
        .unwrap()
        .split("\nscripts:")
        .next()
        .unwrap();
    let mut task: serde_json::Value = serde_json::from_str(body).unwrap();
    task["image_bytes"] = serde_json::to_value(image).unwrap();
    let missing = client
        .post(format!("http://{address}/v1/inference"))
        .json(&task)
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status().as_u16(), 400);
    assert!(
        missing
            .text()
            .await
            .unwrap()
            .contains("Missing required query parameter")
    );
    let mut url = reqwest::Url::parse(&format!("http://{address}/v1/inference")).unwrap();
    url.query_pairs_mut().append_pair("engine", &id);
    let mut cases = vec![(
        task.clone(),
        std::env::var("LITERT_TEST_REFERENCE_OUTPUT").ok(),
    )];
    if let Ok(extra) = std::env::var("LITERT_TEST_EXTRA_CASES") {
        for (image, reference) in serde_json::from_str::<Vec<(String, String)>>(&extra).unwrap() {
            let mut extra_task = task.clone();
            extra_task["image_bytes"] =
                serde_json::to_value(std::fs::read(image).unwrap()).unwrap();
            extra_task["preview"] = false.into();
            cases.push((extra_task, Some(reference)));
        }
    }
    for (task, reference) in cases {
        let response = client.post(url.clone()).json(&task).send().await.unwrap();
        let status = response.status();
        let data: serde_json::Value = response.json().await.unwrap();
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
        if let Some(path) = reference {
            let reference = std::fs::read(path).unwrap();
            assert_eq!(reference.len(), values.len() * 4);
            let mut maximum = 0_f64;
            for (actual, reference) in values.iter().zip(reference.as_chunks::<4>().0) {
                let reference = f64::from(f32::from_le_bytes(*reference));
                let error = (actual.as_f64().unwrap() - reference).abs();
                maximum = maximum.max(error);
                assert!(
                    error <= 0.001 + 0.001 * reference.abs(),
                    "API depth differs from reference: {error}"
                );
            }
            println!("HTTP API/reference maximum absolute error={maximum}");
        }
        if task["preview"] == true {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(depth["preview_png_base64"].as_str().unwrap())
                .unwrap();
            let preview = image::load_from_memory(&bytes).unwrap();
            assert_eq!((preview.width(), preview.height()), (686, 518));
            assert_eq!(preview.color(), image::ColorType::L16);
            if let Ok(path) = std::env::var("LITERT_TEST_PREVIEW") {
                std::fs::write(path, bytes).unwrap();
            }
        } else {
            assert!(depth.get("preview_png_base64").is_none());
        }
        eprintln!(
            "Native depth: {} values, range {}..{}, source {}x{}",
            values.len(),
            depth["min"],
            depth["max"],
            depth["source_width"],
            depth["source_height"]
        );
    }
    server.abort();
    let _ = server.await;
    assert!(
        root.canonicalize()
            .unwrap()
            .starts_with(target.canonicalize().unwrap())
    );
    std::fs::remove_dir_all(&root).unwrap();
    assert!(Path::new(&model_path).is_file());
}
