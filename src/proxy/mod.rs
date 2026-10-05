pub mod adapters;
pub mod controller;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::config::{ModelConfig, ModelRegistry};
use crate::types::{ChatCompletionRequest, ErrorResponse};
use controller::RemoteModelController;

#[derive(Clone)]
pub struct ProxyService {
    client: reqwest::Client,
    controllers: HashMap<String, Arc<RemoteModelController>>,
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
    /// Handles concurrency limits, sliding-window rate throttling, and reactive 429 backoff.
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

        // 2. Dispatch request to cloud provider adapter
        let start_time = std::time::Instant::now();
        match adapters::dispatch_chat(&self.client, config, request).await {
            Ok(resp) => {
                tracing::info!(
                    target: "audit",
                    status = 200,
                    latency_ms = start_time.elapsed().as_millis(),
                    model = %config.id,
                    provider = %config.provider.as_str(),
                    "Proxy chat completion dispatched successfully"
                );
                resp
            }
            Err(err) => {
                tracing::warn!(
                    target: "audit",
                    status = err.code.as_deref().unwrap_or("500"),
                    latency_ms = start_time.elapsed().as_millis(),
                    model = %config.id,
                    provider = %config.provider.as_str(),
                    error = %err.message,
                    "Proxy chat completion failed"
                );

                // If upstream returned 429, trigger backoff on the local controller queue
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
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };

                (status, Json(ErrorResponse { error: err })).into_response()
            }
        }
    }
}
