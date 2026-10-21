use axum::Json;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::api::AppState;
use crate::config::ModelConfig;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest};
use crate::types::{ApiError, ErrorResponse};

pub struct ConcurrencyGuard {
    pub concurrency: Arc<crate::api::RequestConcurrencyState>,
    pub is_background: bool,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        if self.is_background {
            self.concurrency
                .active_background
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        } else {
            self.concurrency
                .active_foreground
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

pub struct ModelExecutionGate {
    pub model_config: ModelConfig,
    pub is_background: bool,
    pub was_resident: bool,
    pub _guard: ConcurrencyGuard,
}

impl ModelExecutionGate {
    #[allow(clippy::result_large_err)]
    pub fn enter(
        state: &Arc<AppState>,
        headers: &HeaderMap,
        model_name_or_id: &str,
    ) -> Result<Self, Response> {
        let is_background = headers
            .get("x-rune-priority")
            .or_else(|| headers.get("x-priority"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.eq_ignore_ascii_case("background") || s.eq_ignore_ascii_case("low"))
            .unwrap_or(false);

        if is_background
            && state
                .concurrency
                .active_foreground
                .load(std::sync::atomic::Ordering::SeqCst)
                > 0
        {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                [(header::RETRY_AFTER, "1")],
                Json(ErrorResponse {
                    error: ApiError::new("Server busy with foreground request")
                        .with_type("server_busy")
                        .with_code("busy"),
                }),
            )
                .into_response());
        }

        let model_config = match state.config().find(model_name_or_id) {
            Some(c) => c.clone(),
            None => {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model '{model_name_or_id}' not found"))
                            .with_type("invalid_request_error")
                            .with_code("model_not_found"),
                    }),
                )
                    .into_response());
            }
        };

        let no_swap = headers
            .get("x-no-swap")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
            .unwrap_or(false);

        let was_resident = if model_config.is_local() {
            state
                .inference
                .registry
                .lock()
                .map(|r| r.is_model_loaded(&model_config.id))
                .unwrap_or(false)
        } else {
            false
        };

        if no_swap && !was_resident && model_config.is_local() {
            return Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: ApiError::new(format!(
                        "Model '{}' is not currently resident and x-no-swap was requested",
                        model_config.id
                    ))
                    .with_type("swap_refused")
                    .with_code("swap_refused"),
                }),
            )
                .into_response());
        }

        if is_background {
            state
                .concurrency
                .active_background
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        } else {
            state
                .concurrency
                .active_foreground
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            state.concurrency.yield_notify.notify_waiters();
        }

        let guard = ConcurrencyGuard {
            concurrency: state.concurrency.clone(),
            is_background,
        };

        Ok(Self {
            model_config,
            is_background,
            was_resident,
            _guard: guard,
        })
    }

    pub fn attach_headers(&self, state: &Arc<AppState>, mut resp: Response) -> Response {
        let hdrs = resp.headers_mut();
        if let Ok(val) = HeaderValue::from_str(&self.model_config.id) {
            hdrs.insert("x-model-served", val);
        }
        hdrs.insert(
            "x-model-locality",
            HeaderValue::from_static(self.model_config.locality()),
        );
        hdrs.insert(
            "x-model-resident",
            HeaderValue::from_static(if self.was_resident { "true" } else { "false" }),
        );
        if !self.was_resident && self.model_config.is_local() {
            if let Some(dur) = state
                .inference
                .registry
                .lock()
                .ok()
                .and_then(|r| r.get_load_duration_ms(&self.model_config.id))
            {
                if let Ok(val) = HeaderValue::from_str(&dur.to_string()) {
                    hdrs.insert("x-model-load-duration-ms", val);
                }
            }
            if let Some(evicted) = state
                .inference
                .registry
                .lock()
                .ok()
                .and_then(|r| r.last_evicted())
            {
                if let Ok(val) = HeaderValue::from_str(&evicted) {
                    hdrs.insert("x-model-evicted", val);
                }
            }
        }
        if let Some(heads) = self.model_config.mtp_heads() {
            if let Ok(val) = HeaderValue::from_str(&heads.to_string()) {
                hdrs.insert("x-mtp-heads", val);
            }
        }
        resp
    }

    #[allow(clippy::result_large_err)]
    pub async fn execute_task(
        &self,
        state: &Arc<AppState>,
        engine: Arc<dyn InferenceEngine>,
        task: InferenceTaskRequest,
    ) -> Result<InferenceOutput, Response> {
        let engine_clone = engine.clone();
        let task_clone = task.clone();
        let start_time = std::time::Instant::now();
        let model_id = self.model_config.id.clone();

        if self.is_background {
            let exec_handle = tokio::task::spawn_blocking(move || {
                let gen_start = std::time::Instant::now();
                let res = engine_clone
                    .execute(&task_clone, None)
                    .map_err(|e| e.to_string());
                (res, gen_start.elapsed())
            });
            let yield_notified = state.concurrency.yield_notify.notified();
            tokio::select! {
                join_res = exec_handle => {
                    match join_res {
                        Ok((Ok(out), gen_dur)) => {
                            crate::telemetry::record_completion(
                                &model_id,
                                start_time.elapsed().as_millis(),
                                out.usage.prompt_tokens,
                                out.usage.completion_tokens,
                                gen_dur.as_millis(),
                            );
                            Ok(out)
                        }
                        Ok((Err(err), _)) => {
                            crate::telemetry::record_failure(
                                &model_id,
                                500,
                                start_time.elapsed().as_millis(),
                                &err,
                            );
                            Err(self.attach_headers(state, (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse { error: ApiError::new(err) }),
                            ).into_response()))
                        }
                        Err(join_err) => {
                            Err(self.attach_headers(state, (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse { error: ApiError::new(format!("Execution error: {join_err}")) }),
                            ).into_response()))
                        }
                    }
                }
                _ = yield_notified => {
                    engine.cancel();
                    crate::telemetry::record_failure(
                        &model_id,
                        503,
                        start_time.elapsed().as_millis(),
                        "background task yielded to foreground request",
                    );
                    Err(self.attach_headers(state, (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [(header::RETRY_AFTER, "1")],
                        Json(ErrorResponse {
                            error: ApiError::new("Background request yielded to foreground request")
                                .with_type("server_busy")
                                .with_code("yielded"),
                        }),
                    ).into_response()))
                }
            }
        } else {
            let join_res = tokio::task::spawn_blocking(move || {
                let gen_start = std::time::Instant::now();
                let res = engine_clone
                    .execute(&task_clone, None)
                    .map_err(|e| e.to_string());
                (res, gen_start.elapsed())
            })
            .await;
            match join_res {
                Ok((Ok(out), gen_dur)) => {
                    crate::telemetry::record_completion(
                        &model_id,
                        start_time.elapsed().as_millis(),
                        out.usage.prompt_tokens,
                        out.usage.completion_tokens,
                        gen_dur.as_millis(),
                    );
                    Ok(out)
                }
                Ok((Err(err), _)) => {
                    crate::telemetry::record_failure(
                        &model_id,
                        500,
                        start_time.elapsed().as_millis(),
                        &err,
                    );
                    Err(self.attach_headers(
                        state,
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: ApiError::new(err),
                            }),
                        )
                            .into_response(),
                    ))
                }
                Err(join_err) => Err(self.attach_headers(
                    state,
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: ApiError::new(format!("Execution error: {join_err}")),
                        }),
                    )
                        .into_response(),
                )),
            }
        }
    }
}
