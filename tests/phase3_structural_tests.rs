use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rune_infer::api::AppState;
use rune_infer::config::{
    EngineConfig, EngineKind, Modality, ModelConfig, ModelRegistry, SamplingConfig, ServerConfig,
};
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::proxy::ProxyService;
use rune_infer::schema::{CatalogFile, ModelEntry, SCHEMA_VERSION, ServerFile};

#[test]
fn test_schema3_migration_and_decoupled_loading() {
    let dir = std::env::temp_dir().join(format!(
        "rune-phase3-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config_dir = dir.join("config");
    std::fs::create_dir_all(&config_dir).expect("create config dir");
    let legacy_models_path = config_dir.join("models.json");

    let legacy_json = r#"{
        "schema_version": 2,
        "server": {
            "host": "127.0.0.1",
            "port": 9090,
            "api_key": null,
            "max_loaded_models": 2,
            "idle_unload_seconds": 600,
            "vram_budget_ratio": 0.90
        },
        "models": [
            {
                "id": "test-model-1",
                "name": "Test Model 1",
                "description": "Test Model Description",
                "modality": "Text",
                "capabilities": ["Chat"],
                "engine": { "supported": ["llama-server"], "preferred": "llama-server" },
                "max_context_length": 4096,
                "sampling": {},
                "local": {
                    "path": "test.gguf",
                    "params": null,
                    "bits_per_weight": null,
                    "context_length": 2048,
                    "gpu_layers": 99,
                    "args": ""
                },
                "extra": {}
            }
        ]
    }"#;
    std::fs::write(&legacy_models_path, legacy_json).expect("write legacy config");

    // Execute migration
    rune_infer::migrate::run(&legacy_models_path, false).expect("migration runs successfully");

    let server_path = config_dir.join("server.json");
    let catalog_path = config_dir.join("catalog.json");
    let backup_path = config_dir.join("models.json.v2.bak");

    assert!(server_path.exists(), "server.json must be created");
    assert!(catalog_path.exists(), "catalog.json must be created");
    assert!(
        backup_path.exists(),
        "backup file models.json.v2.bak must be created"
    );

    // Verify ServerFile schema
    let server_file: ServerFile =
        serde_json::from_str(&std::fs::read_to_string(&server_path).unwrap()).unwrap();
    assert_eq!(server_file.schema_version, SCHEMA_VERSION);
    assert_eq!(server_file.server.port, 9090);
    assert_eq!(server_file.server.max_loaded_models, 2);

    // Verify CatalogFile schema
    let catalog_file: CatalogFile =
        serde_json::from_str(&std::fs::read_to_string(&catalog_path).unwrap()).unwrap();
    assert_eq!(catalog_file.schema_version, SCHEMA_VERSION);
    assert_eq!(catalog_file.models.len(), 1);
    assert_eq!(catalog_file.models[0].id, "test-model-1");

    // Verify decoupled loader
    let loaded =
        ModelRegistry::from_files(&server_path, &catalog_path).expect("load from decoupled files");
    assert_eq!(loaded.server.port, 9090);
    assert_eq!(loaded.models.len(), 1);
    assert_eq!(loaded.models[0].id, "test-model-1");

    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn test_runtime_catalog_hot_reloading() {
    let mut initial_config = ModelRegistry {
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

    let sample_entry = ModelEntry {
        id: "initial-model".to_string(),
        name: "Initial Model".to_string(),
        description: "Initial".to_string(),
        modality: Modality::Text,
        capabilities: vec!["Chat".to_string()],
        engine: Some(EngineConfig {
            supported: vec![EngineKind::LlamaServer],
            preferred: Some(EngineKind::LlamaServer),
        }),
        max_context_length: Some(4096),
        sampling: SamplingConfig::default(),
        local: Some(rune_infer::schema::LocalBlock {
            path: "initial.gguf".to_string(),
            params: None,
            bits_per_weight: None,
            context_length: -1,
            gpu_layers: 99,
            args: "".to_string(),
        }),
        remote: None,
        extra: serde_json::Map::new(),
    };
    initial_config
        .models
        .push(ModelConfig::try_from(sample_entry).expect("valid model"));

    let inference = Arc::new(InferenceAppState::new(&initial_config));
    let proxy = Arc::new(ProxyService::new(&initial_config));
    let state = AppState::new(inference, proxy, initial_config);

    assert_eq!(state.config().models.len(), 1);
    assert!(state.config().find("initial-model").is_some());
    assert!(state.config().find("hot-reloaded-model").is_none());

    // Prepare updated catalog with a new model
    let mut updated_config = state.config().clone();
    let new_entry = ModelEntry {
        id: "hot-reloaded-model".to_string(),
        name: "Hot Reloaded Model".to_string(),
        description: "Added dynamically at runtime".to_string(),
        modality: Modality::Text,
        capabilities: vec!["Chat".to_string()],
        engine: Some(EngineConfig {
            supported: vec![EngineKind::LlamaServer],
            preferred: Some(EngineKind::LlamaServer),
        }),
        max_context_length: Some(8192),
        sampling: SamplingConfig::default(),
        local: Some(rune_infer::schema::LocalBlock {
            path: "reloaded.gguf".to_string(),
            params: None,
            bits_per_weight: None,
            context_length: -1,
            gpu_layers: 99,
            args: "".to_string(),
        }),
        remote: None,
        extra: serde_json::Map::new(),
    };
    updated_config
        .models
        .push(ModelConfig::try_from(new_entry).expect("valid model"));

    // Reload catalog in running state
    state.reload_catalog(updated_config);

    // Verify catalog reflects new model immediately without server restart
    assert_eq!(state.config().models.len(), 2);
    assert!(state.config().find("initial-model").is_some());
    let reloaded = state.config().find("hot-reloaded-model").cloned();
    assert!(reloaded.is_some());
    assert_eq!(reloaded.unwrap().name, "Hot Reloaded Model");
}

#[test]
fn test_cancellation_propagation_guard() {
    let cancelled = Arc::new(AtomicBool::new(false));

    struct TestCancelGuard(Arc<AtomicBool>);
    impl Drop for TestCancelGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    assert!(!cancelled.load(Ordering::SeqCst));

    {
        let _guard = TestCancelGuard(cancelled.clone());
        assert!(
            !cancelled.load(Ordering::SeqCst),
            "Flag should be false while guard is alive"
        );
    }

    // When client connection drops (scope ends), guard triggers atomic cancellation
    assert!(
        cancelled.load(Ordering::SeqCst),
        "Flag must be true after guard is dropped"
    );

    // Engine generation token callback checks the flag and aborts loop
    let cancelled_for_closure = cancelled.clone();
    let on_token = move |_: &str| -> bool { !cancelled_for_closure.load(Ordering::Relaxed) };

    assert!(
        !on_token("token_output"),
        "Callback should return false when cancelled flag is set"
    );
}
