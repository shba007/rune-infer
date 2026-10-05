use rune_infer::config::{RateLimitConfig, interpolate_env_vars};
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

#[tokio::test]
async fn test_rate_limit_controller_concurrency() {
    let config = RateLimitConfig {
        requests_per_minute: 60,
        tokens_per_minute: None,
        max_concurrent: 2,
        queue_timeout_seconds: 1,
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
