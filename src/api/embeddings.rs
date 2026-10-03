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
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::Embedding
                || m.id.contains("embedding")
                || m.name.to_lowercase().contains("embedding")
        }) {
            Some(m) => m.id.clone(),
            None => "granite-embedding-311m-multilingual-r2".to_string(),
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
                        error: ApiError::new(format!("Model failed to load: {err}"))
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

    let task = InferenceTaskRequest::Embedding {
        model: Some(model_id),
        input: inputs,
        dimensions: request.dimensions,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Embedding(embed_resp) => Json(embed_resp).into_response(),
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
                    error: ApiError::new("Unexpected response variant from embedding engine"),
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
