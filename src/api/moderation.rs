use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, ErrorResponse, ModerationRequest};

#[axum::debug_handler]
pub async fn moderations_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ModerationRequest>,
) -> Response {
    let inputs: Vec<String> = match request.input {
        Some(serde_json::Value::String(s)) => {
            if s.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new("'input' field cannot be an empty string.")
                            .with_type("invalid_request_error")
                            .with_param("input")
                            .with_code("empty_input"),
                    }),
                )
                    .into_response();
            }
            vec![s]
        }
        Some(serde_json::Value::Array(arr)) => {
            if arr.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new("'input' field cannot be an empty array.")
                            .with_type("invalid_request_error")
                            .with_param("input")
                            .with_code("empty_input"),
                    }),
                )
                    .into_response();
            }
            let mut items = Vec::new();
            for val in arr {
                if let Some(s) = val.as_str() {
                    items.push(s.to_string());
                }
            }
            items
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'input' is a required string or array field.")
                        .with_type("invalid_request_error")
                        .with_param("input")
                        .with_code("missing_required_parameter"),
                }),
            )
                .into_response();
        }
    };

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::Moderation
                || m.id.contains("guard")
                || m.id.contains("moderation")
        }) {
            Some(m) => m.id.clone(),
            None => "gliguard-LLMGuardrails-300M".to_string(),
        },
    };

    let inference = state.inference.clone();
    let target_model = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&target_model)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Moderation model failed to load: {err}"))
                            .with_type("invalid_request_error")
                            .with_param("model")
                            .with_code("model_not_found"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Execution thread error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let task = InferenceTaskRequest::Moderation {
        model: Some(model_id),
        input: inputs,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Moderation(mod_resp) => Json(mod_resp).into_response(),
            InferenceTaskResponse::Error(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(err).with_type("server_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response variant from moderation engine"),
                }),
            )
                .into_response(),
        },
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err).with_type("server_error"),
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err.to_string()).with_type("server_error"),
            }),
        )
            .into_response(),
    }
}
