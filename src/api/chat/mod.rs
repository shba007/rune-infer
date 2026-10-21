pub mod prompt;
pub mod response;
pub mod stream;
pub mod tools;

pub use prompt::*;
pub use response::*;
pub use stream::*;
pub use tools::*;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::InferenceTaskResponse;
use crate::types::{ApiError, ChatCompletionRequest, ErrorResponse};

#[axum::debug_handler]
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    bytes: Bytes,
) -> Response {
    let start_time = std::time::Instant::now();

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
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, "1")],
            Json(ErrorResponse {
                error: ApiError::new("Server busy with foreground request")
                    .with_type("server_busy")
                    .with_code("busy"),
            }),
        )
            .into_response();
    }

    struct ConcurrencyGuard {
        concurrency: Arc<crate::api::RequestConcurrencyState>,
        is_background: bool,
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

    let _concurrency_guard = ConcurrencyGuard {
        concurrency: state.concurrency.clone(),
        is_background,
    };

    let body_val: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Invalid JSON payload: {e}"))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

    match body_val.get("messages") {
        None | Some(serde_json::Value::Null) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' field must be an array of message objects.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("invalid_type"),
                }),
            )
                .into_response();
        }
        Some(val) if !val.is_array() => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' field must be an array of message objects.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("invalid_type"),
                }),
            )
                .into_response();
        }
        Some(serde_json::Value::Array(arr)) if arr.is_empty() => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' array cannot be empty.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("empty_array"),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    match body_val.get("model") {
        None | Some(serde_json::Value::Null) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("Missing required field: 'model'")
                        .with_type("invalid_request_error")
                        .with_param("model")
                        .with_code("model_missing"),
                }),
            )
                .into_response();
        }
        Some(val) if val.as_str().map(|s| s.trim().is_empty()).unwrap_or(true) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("Missing required field: 'model'")
                        .with_type("invalid_request_error")
                        .with_param("model")
                        .with_code("model_missing"),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    if let Err(err_resp) = tools::validate_tools(&body_val) {
        return err_resp.into_response();
    }

    let request: ChatCompletionRequest = match serde_json::from_value(body_val) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Invalid request structure: {e}"))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

    if request.model.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing required field: 'model'")
                    .with_type("invalid_request_error")
                    .with_code("model_missing"),
            }),
        )
            .into_response();
    }

    let model_config = match state.config().find(&request.model) {
        Some(c) => c.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found", request.model))
                        .with_type("invalid_request_error")
                        .with_code("model_not_found"),
                }),
            )
                .into_response();
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
        true
    };

    if no_swap && !was_resident {
        return (
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
            .into_response();
    }

    if !model_config.is_local() {
        let mut proxy_resp = state.proxy.execute_chat(&model_config, &request).await;
        let hdrs = proxy_resp.headers_mut();
        if let Ok(val) = axum::http::HeaderValue::from_str(&model_config.id) {
            hdrs.insert("x-model-served", val);
        }
        hdrs.insert(
            "x-model-locality",
            axum::http::HeaderValue::from_static("cloud"),
        );
        hdrs.insert(
            "x-model-resident",
            axum::http::HeaderValue::from_static("true"),
        );
        return proxy_resp;
    }

    let req_for_prep = request.clone();
    let cfg_for_prep = model_config.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        prompt::prepare_prompt(&req_for_prep, &cfg_for_prep)
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(err_resp)) => return err_resp.into_response(),
        Err(join_err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Media preparation thread error: {join_err}")),
                }),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let model_target = request.model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_target)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                tracing::error!(
                    target: "audit",
                    model = %request.model,
                    error = %err,
                    "Failed to load model runtime engine"
                );
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!(
                            "Model '{}' not found or failed to load: {err}",
                            request.model
                        ))
                        .with_type("invalid_request_error")
                        .with_code("model_not_found"),
                    }),
                )
                    .into_response();
            }
            Err(join_err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Thread execution error: {join_err}")),
                    }),
                )
                    .into_response();
            }
        };

    let load_duration_ms = state
        .inference
        .registry
        .lock()
        .map(|r| r.get_load_duration_ms(&model_config.id))
        .unwrap_or(None);
    let last_evicted = state
        .inference
        .registry
        .lock()
        .map(|r| r.last_evicted())
        .unwrap_or(None);

    let attach_attribution_headers = |mut resp: Response| -> Response {
        let hdrs = resp.headers_mut();
        if let Ok(val) = axum::http::HeaderValue::from_str(&model_config.id) {
            hdrs.insert("x-model-served", val);
        }
        hdrs.insert(
            "x-model-locality",
            axum::http::HeaderValue::from_static("local"),
        );
        hdrs.insert(
            "x-model-resident",
            axum::http::HeaderValue::from_static(if was_resident { "true" } else { "false" }),
        );
        if !was_resident {
            if let Some(dur) = load_duration_ms {
                if let Ok(val) = axum::http::HeaderValue::from_str(&dur.to_string()) {
                    hdrs.insert("x-model-load-duration-ms", val);
                }
            }
            if let Some(ref evicted) = last_evicted {
                if let Ok(val) = axum::http::HeaderValue::from_str(evicted) {
                    hdrs.insert("x-model-evicted", val);
                }
            }
        }
        if let Some(heads) = model_config.mtp_heads() {
            if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                hdrs.insert("x-mtp-heads", val);
            }
        }
        resp
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let include_usage = request
        .stream_options
        .as_ref()
        .map(|s| s.include_usage)
        .unwrap_or(false);

    if request.stream {
        let yield_notify = if is_background {
            Some(Arc::new(tokio::sync::Notify::new()))
        } else {
            None
        };
        if is_background {
            let concurrency = state.concurrency.clone();
            let yn_clone = yield_notify.clone().unwrap();
            tokio::spawn(async move {
                concurrency.yield_notify.notified().await;
                yn_clone.notify_waiters();
            });
        }
        let stream_resp = stream::handle_chat_stream(
            prepared.task,
            engine,
            model_config.clone(),
            created,
            start_time,
            prepared.image_count,
            prepared.video_count,
            include_usage,
            yield_notify,
        );
        return attach_attribution_headers(stream_resp);
    }

    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancel_flag_clone = cancel_flag.clone();
    let is_completed = Arc::new(std::sync::atomic::AtomicBool::new(false));

    struct CancelOnDrop {
        completed: Arc<std::sync::atomic::AtomicBool>,
        flag: Arc<std::sync::atomic::AtomicBool>,
        engine: Arc<dyn crate::inference::traits::InferenceEngine>,
    }
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            if !self.completed.load(std::sync::atomic::Ordering::Relaxed) {
                self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
                self.engine.cancel();
            }
        }
    }
    let _cancel_guard = CancelOnDrop {
        completed: is_completed.clone(),
        flag: cancel_flag.clone(),
        engine: engine.clone(),
    };

    let task_clone = prepared.task.clone();
    let model_for_stats = model_config.id.clone();
    let engine_for_yield = engine.clone();
    let cancel_flag_for_yield = cancel_flag.clone();

    let exec_handle = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let mut on_token =
            |_: &str| -> bool { !cancel_flag_clone.load(std::sync::atomic::Ordering::Relaxed) };
        let res = engine
            .execute(&task_clone, Some(&mut on_token))
            .map_err(|e| e.to_string());
        (res, gen_started.elapsed())
    });

    let (res, gen_elapsed) = if is_background {
        let yield_notified = state.concurrency.yield_notify.notified();
        tokio::select! {
            join_res = exec_handle => {
                match join_res {
                    Ok(pair) => pair,
                    Err(e) => (Err(format!("Thread execution error: {e}")), std::time::Duration::ZERO),
                }
            }
            _ = yield_notified => {
                cancel_flag_for_yield.store(true, std::sync::atomic::Ordering::Relaxed);
                engine_for_yield.cancel();
                crate::telemetry::record_failure(
                    &model_for_stats,
                    503,
                    start_time.elapsed().as_millis(),
                    "background task yielded to foreground request",
                );
                return attach_attribution_headers((
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(axum::http::header::RETRY_AFTER, "1")],
                    Json(ErrorResponse {
                        error: ApiError::new("Background request yielded to foreground request")
                            .with_type("server_busy")
                            .with_code("yielded"),
                    }),
                ).into_response());
            }
        }
    } else {
        match exec_handle.await {
            Ok(pair) => pair,
            Err(e) => (
                Err(format!("Thread execution error: {e}")),
                std::time::Duration::ZERO,
            ),
        }
    };
    is_completed.store(true, std::sync::atomic::Ordering::Relaxed);

    if cancel_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::telemetry::record_failure(
            &model_for_stats,
            499,
            start_time.elapsed().as_millis(),
            "client disconnected during execution",
        );
    } else {
        match &res {
            Ok(out) => crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                out.usage.prompt_tokens,
                out.usage.completion_tokens,
                gen_elapsed.as_millis(),
            ),
            Err(e) => crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                e,
            ),
        }
    }

    match res {
        Ok(output) => match output.response {
            InferenceTaskResponse::ToolCall {
                content,
                tool_calls,
            } => {
                let json_resp = if prepared.is_structured_mode {
                    let content_str = match &tool_calls {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string_pretty(other).unwrap_or_default(),
                    };
                    let resp = response::chat_completion_response(
                        &model_config.id,
                        Some(content_str),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else if let Some(parsed_calls) =
                    tools::convert_to_tool_calls(&tool_calls, created)
                {
                    let resp = response::chat_completion_response(
                        &model_config.id,
                        content,
                        Some(parsed_calls),
                        "tool_calls",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else {
                    let resp = response::chat_completion_response(
                        &model_config.id,
                        Some(tool_calls.to_string()),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                };

                attach_attribution_headers(json_resp)
            }
            InferenceTaskResponse::Text(text) => {
                let resp = response::chat_completion_response(
                    &model_config.id,
                    Some(text),
                    None,
                    "stop",
                    created,
                    output.usage,
                );
                attach_attribution_headers(Json(resp).into_response())
            }
            InferenceTaskResponse::Error(e) => attach_attribution_headers(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(e)
                            .with_type("server_error")
                            .with_code("inference_error"),
                    }),
                )
                    .into_response(),
            ),
            _ => attach_attribution_headers(
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new("Unexpected response type from text inference")
                            .with_type("server_error"),
                    }),
                )
                    .into_response(),
            ),
        },
        Err(e) => {
            let status = if e.contains("context") || e.contains("tokens") || e.contains("limit") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            attach_attribution_headers(
                (
                    status,
                    Json(ErrorResponse {
                        error: ApiError::new(e)
                            .with_type("invalid_request_error")
                            .with_code("inference_failure"),
                    }),
                )
                    .into_response(),
            )
        }
    }
}
