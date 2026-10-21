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
    headers: axum::http::HeaderMap,
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
        _ => match state.config().models.iter().find(|m| {
            m.modality == crate::config::Modality::Moderation
                || m.id.contains("guard")
                || m.id.contains("moderation")
        }) {
            Some(m) => m.id.clone(),
            None => "gliguard-LLMGuardrails-300M".to_string(),
        },
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };
    let engine = match gate.load_engine(&state).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };
    let task = InferenceTaskRequest::Moderation {
        model: Some(model_id),
        input: inputs,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(output) => output,
        Err(resp) => return resp,
    };
    gate.attach_headers(
        &state,
        match output.response {
            InferenceTaskResponse::Moderation(response) => Json(response).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
    )
}
