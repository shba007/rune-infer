use rune_infer::config::{ModelRegistry, ProviderType, RateLimitConfig, interpolate_env_vars};
use rune_infer::proxy::controller::RemoteModelController;

#[test]
fn test_template_env_interpolation() {
    unsafe {
        std::env::set_var("TEST_UPSTREAM_KEY", "sk-test-secret-999");
    }

    let input = r#"{"api_key": "{{TEST_UPSTREAM_KEY}}", "model": "claude-3-7-sonnet"}"#;
    let interpolated = interpolate_env_vars(input).expect("Interpolation failed");

    assert!(interpolated.contains("sk-test-secret-999"));
    assert!(!interpolated.contains("{{TEST_UPSTREAM_KEY}}"));
}

#[test]
fn test_missing_env_fails_fast() {
    let input = r#"{"api_key": "{{DEFINITELY_NON_EXISTENT_VAR_12345}}"}"#;
    let err = interpolate_env_vars(input).unwrap_err();
    assert!(
        err.to_string()
            .contains("DEFINITELY_NON_EXISTENT_VAR_12345")
    );
}

#[test]
fn test_openrouter_apodex_configuration_parsing() {
    unsafe {
        std::env::set_var("TEST_OPENROUTER_KEY", "sk-or-v1-mocked-credential-987");
    }

    let raw_config = r#"{
        "schema_version": 1,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "max_loaded_models": 1,
            "idle_unload_seconds": 300,
            "vram_budget_ratio": 0.95
        },
        "models": [
            {
                "id": "apodex-1.1-mini",
                "name": "Apodex 1.1 Mini (Free)",
                "provider": "openrouter",
                "modality": "Text",
                "upstream_model": "apodex/apodex-1.1-mini:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "{{TEST_OPENROUTER_KEY}}",
                "max_context_length": 262144
            }
        ]
    }"#;

    let interpolated = interpolate_env_vars(raw_config).expect("Interpolation failed");
    let registry: ModelRegistry =
        serde_json::from_str(&interpolated).expect("Failed to parse registry");
    let model = registry
        .find("apodex-1.1-mini")
        .expect("Model not found in catalog");

    assert_eq!(model.provider, ProviderType::OpenRouter);
    assert_eq!(
        model.upstream_model.as_deref(),
        Some("apodex/apodex-1.1-mini:free")
    );
    assert_eq!(
        model.api_key.as_deref(),
        Some("sk-or-v1-mocked-credential-987")
    );
    assert_eq!(model.max_context_length, Some(262144));
}

#[test]
fn test_openrouter_dots3_note_configuration_parsing() {
    unsafe {
        std::env::set_var("TEST_OPENROUTER_KEY", "sk-or-v1-mocked-credential-987");
    }

    let raw_config = r#"{
        "schema_version": 1,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "max_loaded_models": 1,
            "idle_unload_seconds": 300,
            "vram_budget_ratio": 0.95
        },
        "models": [
            {
                "id": "dots-3-note-preview",
                "name": "Dots3-Note Preview (Free)",
                "provider": "openrouter",
                "modality": "VisionText",
                "vision": true,
                "upstream_model": "dots-studio/dots-3-note-preview:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "{{TEST_OPENROUTER_KEY}}",
                "max_context_length": 512000
            }
        ]
    }"#;

    let interpolated = interpolate_env_vars(raw_config).expect("Interpolation failed");
    let registry: ModelRegistry =
        serde_json::from_str(&interpolated).expect("Failed to parse registry");
    let model = registry
        .find("dots-3-note-preview")
        .expect("Model not found in catalog");

    assert_eq!(model.provider, ProviderType::OpenRouter);
    assert_eq!(
        model.upstream_model.as_deref(),
        Some("dots-studio/dots-3-note-preview:free")
    );
    assert_eq!(model.max_context_length, Some(512000));
    assert!(model.vision);
}

#[tokio::test]
async fn test_rate_limit_controller_concurrency() {
    let config = RateLimitConfig {
        requests_per_minute: 60,
        tokens_per_minute: None,
        requests_per_day: None,
        max_concurrent: 2,
        queue_timeout_seconds: 1,
        max_retries: 5,
        retry_delay_seconds: None,
        retry_delay_ms: 1000,
        retry_timeout_seconds: 30,
    };

    let controller = RemoteModelController::new("test-claude".to_string(), Some(&config));

    // Acquire first two permits
    let permit1 = controller.acquire_permit().await;
    let permit2 = controller.acquire_permit().await;
    assert!(permit1.is_ok());
    assert!(permit2.is_ok());

    // Third acquire should time out because max_concurrent = 2
    let permit3 = controller.acquire_permit().await;
    assert!(permit3.is_err());
    let err = permit3.unwrap_err();
    assert_eq!(err.code.as_deref(), Some("rate_limit_exceeded"));

    // Release permit 1 and retry
    drop(permit1);
    let permit4 = controller.acquire_permit().await;
    assert!(permit4.is_ok());
}
