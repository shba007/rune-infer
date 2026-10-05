pub mod adapters;
pub mod controller;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::config::{ModelConfig, ModelRegistry};
use crate::types::{ApiError, ChatCompletionRequest, ErrorResponse};
use controller::RemoteModelController;

#[derive(Clone)]
pub struct ProxyService {
    client: reqwest::Client,
    controllers: HashMap<String, Arc<RemoteModelController>>,
}

fn is_retryable_error(err: &ApiError) -> bool {
    if let Some(ref code) = err.code {
        match code.as_str() {
            "503" | "429" | "502" | "504" | "500" => return true,
            _ => {}
        }
    }
    let lower = err.message.to_lowercase();
    lower.contains("503")
        || lower.contains("high demand")
        || lower.contains("service unavailable")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("temporarily unavailable")
        || lower.contains("connection reset")
        || lower.contains("timeout")
}

impl ProxyService {
    pub fn new(registry: &ModelRegistry) -> Self {
        let client = reqwest::Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_nodelay(true)
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to initialize proxy HTTP client");

        let mut controllers = HashMap::new();
        for model in &registry.models {
            if !model.is_local() {
                let controller = Arc::new(RemoteModelController::new(
                    model.id.clone(),
                    model.rate_limit.as_ref(),
                ));
                controllers.insert(model.id.clone(), controller);
            }
        }

        Self {
            client,
            controllers,
        }
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    pub fn get_controller(&self, model_id: &str) -> Option<Arc<RemoteModelController>> {
        self.controllers.get(model_id).cloned()
    }

    /// Asynchronously executes a chat completion request through the cloud provider proxy.
    /// Handles concurrency limits, sliding-window rate throttling, and automatic in-flight retries
    /// using parameters configured directly in models.json without dropping the client request.
    pub async fn execute_chat(
        &self,
        config: &ModelConfig,
        request: &ChatCompletionRequest,
    ) -> Response {
        let controller = self.get_controller(&config.id);

        // 1. Acquire queue and rate limit permit (asynchronous wait)
        let _permit = if let Some(ref c) = controller {
            match c.acquire_permit().await {
                Ok(p) => Some(p),
                Err(api_err) => {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        Json(ErrorResponse { error: api_err }),
                    )
                        .into_response();
                }
            }
        } else {
            None
        };

        // 2. Load retry configuration from models.json (with sensible defaults)
        let max_retries = config
            .rate_limit
            .as_ref()
            .map(|r| r.max_retries)
            .unwrap_or(5);

        let retry_timeout = Duration::from_secs(
            config
                .rate_limit
                .as_ref()
                .map(|r| r.retry_timeout_seconds)
                .unwrap_or(30),
        );

        let base_retry_delay_ms = config
            .rate_limit
            .as_ref()
            .map(|r| r.effective_retry_delay_ms())
            .unwrap_or(1000);

        let overall_deadline = Instant::now() + retry_timeout;
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            let start_time = Instant::now();

            match adapters::dispatch_chat(&self.client, config, request).await {
                Ok(resp) => {
                    tracing::info!(
                        target: "audit",
                        status = 200,
                        latency_ms = start_time.elapsed().as_millis(),
                        model = %config.id,
                        provider = %config.provider.as_str(),
                        attempts = attempt,
                        "Proxy chat completion dispatched successfully"
                    );
                    return resp;
                }
                Err(err) => {
                    let retryable = is_retryable_error(&err);
                    let now = Instant::now();

                    if !retryable || attempt > max_retries || now >= overall_deadline {
                        tracing::warn!(
                            target: "audit",
                            status = err.code.as_deref().unwrap_or("500"),
                            latency_ms = start_time.elapsed().as_millis(),
                            model = %config.id,
                            provider = %config.provider.as_str(),
                            attempts = attempt,
                            error = %err.message,
                            "Proxy chat completion failed (unretryable or retry budget exhausted)"
                        );

                        if err.code.as_deref() == Some("429") {
                            if let Some(ref c) = controller {
                                c.record_rate_limit_backoff(None).await;
                            }
                        }

                        let status = match err.code.as_deref() {
                            Some("401") => StatusCode::UNAUTHORIZED,
                            Some("403") => StatusCode::FORBIDDEN,
                            Some("404") => StatusCode::NOT_FOUND,
                            Some("429") => StatusCode::TOO_MANY_REQUESTS,
                            Some("400") => StatusCode::BAD_REQUEST,
                            Some("503") => StatusCode::SERVICE_UNAVAILABLE,
                            _ => StatusCode::INTERNAL_SERVER_ERROR,
                        };

                        return (status, Json(ErrorResponse { error: err })).into_response();
                    }

                    // Calculate backoff using base delay from models.json + exponential factor + jitter
                    let factor = 1u64 << (attempt - 1).min(3);
                    let base_delay_ms = base_retry_delay_ms.saturating_mul(factor);
                    let jitter_ms = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_millis() as u64)
                        % 500;
                    let delay = Duration::from_millis(base_delay_ms + jitter_ms);

                    let time_left = overall_deadline.saturating_duration_since(now);
                    let sleep_dur = delay.min(time_left);

                    tracing::warn!(
                        target: "audit",
                        status = err.code.as_deref().unwrap_or("503"),
                        attempt = attempt,
                        max_retries = max_retries,
                        retry_in_ms = sleep_dur.as_millis(),
                        model = %config.id,
                        "Upstream returned transient capacity error; retrying in-flight without dropping request..."
                    );

                    tokio::time::sleep(sleep_dur).await;
                }
            }
        }
    }
}
