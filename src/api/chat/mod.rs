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
    bytes: Bytes,
) -> Response {
    let start_time = std::time::Instant::now();

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

    let model_config = match state.config.find(&request.model) {
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

    if !model_config.is_local() {
        return state.proxy.execute_chat(&model_config, &request).await;
    }

    let prepared = match prompt::prepare_prompt(&request, &model_config) {
        Ok(p) => p,
        Err(err_resp) => return err_resp.into_response(),
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

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if request.stream {
        return stream::handle_chat_stream(
            prepared.task,
            engine,
            model_config,
            created,
            start_time,
            prepared.image_count,
            prepared.video_count,
        );
    }

    let task_clone = prepared.task.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(output)) => match output.response {
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
                        &request.model,
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
                        &request.model,
                        content,
                        Some(parsed_calls),
                        "tool_calls",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else {
                    let resp = response::chat_completion_response(
                        &request.model,
                        Some(tool_calls.to_string()),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                };

                let mut final_resp = json_resp;
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Text(text) => {
                let resp = response::chat_completion_response(
                    &request.model,
                    Some(text),
                    None,
                    "stop",
                    created,
                    output.usage,
                );
                let mut final_resp = Json(resp).into_response();
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Error(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e)
                        .with_type("server_error")
                        .with_code("inference_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type from text inference")
                        .with_type("server_error"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => {
            let status = if e.contains("context") || e.contains("tokens") || e.contains("limit") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(ErrorResponse {
                    error: ApiError::new(e)
                        .with_type("invalid_request_error")
                        .with_code("inference_failure"),
                }),
            )
                .into_response()
        }
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(format!("Thread execution failure: {join_err}"))
                    .with_type("server_error")
                    .with_code("internal_error"),
            }),
        )
            .into_response(),
    }
}
