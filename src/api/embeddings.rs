use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, CreateEmbeddingRequest, ErrorResponse};

#[axum::debug_handler]
pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<CreateEmbeddingRequest>,
) -> Response {
    let inputs: Vec<String> = match request.input {
        Some(serde_json::Value::String(s)) => {
            if s.trim().is_empty() {
                vec![]
            } else {
                vec![s]
            }
        }
        Some(serde_json::Value::Array(arr)) => {
            let mut items = Vec::new();
            for val in arr {
                if let Some(s) = val.as_str() {
                    if !s.trim().is_empty() {
                        items.push(s.to_string());
                    }
                }
            }
            items
        }
        _ => Vec::new(),
    };

    if inputs.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'input' is a required string or array of strings parameter.")
                    .with_type("invalid_request_error")
                    .with_param("input")
                    .with_code("missing_required_parameter"),
            }),
        )
            .into_response();
    }

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config().models.iter().find(|m| {
            m.modality == crate::config::Modality::Embedding
                || m.id.contains("embedding")
                || m.name.to_lowercase().contains("embedding")
        }) {
            Some(m) => m.id.clone(),
            None => "granite-embedding-311m-multilingual-r2".to_string(),
        },
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(g) => g,
        Err(resp) => return resp,
    };

    let inference = state.inference.clone();
    let target_model = gate.model_config.id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&target_model)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return gate.attach_headers(
                    &state,
                    (
                        StatusCode::NOT_FOUND,
                        Json(ErrorResponse {
                            error: ApiError::new(format!("Model failed to load: {err}"))
                                .with_type("invalid_request_error")
                                .with_param("model")
                                .with_code("model_not_found"),
                        }),
                    )
                        .into_response(),
                );
            }
            Err(e) => {
                return gate.attach_headers(
                    &state,
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: ApiError::new(format!("Execution thread error: {e}")),
                        }),
                    )
                        .into_response(),
                );
            }
        };

    let task = InferenceTaskRequest::Embedding {
        model: Some(gate.model_config.id.clone()),
        input: inputs,
        dimensions: request.dimensions,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(out) => out,
        Err(err_resp) => return err_resp,
    };

    match output.response {
        InferenceTaskResponse::Embedding(embed_resp) => {
            gate.attach_headers(&state, Json(embed_resp).into_response())
        }
        InferenceTaskResponse::Error(err) => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(err).with_type("server_error"),
                }),
            )
                .into_response(),
        ),
        _ => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response variant from embedding engine"),
                }),
            )
                .into_response(),
        ),
    }
}
