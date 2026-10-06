use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, ErrorResponse, NluIntentRequest};

#[axum::debug_handler]
pub async fn nlu_intents_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<NluIntentRequest>,
) -> Response {
    let text = match request.text {
        Some(ref t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'text' is a required non-empty string parameter.")
                        .with_type("invalid_request_error")
                        .with_param("text")
                        .with_code("empty_input"),
                }),
            )
                .into_response();
        }
    };

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::Nlu
                || m.id.contains("nlu")
                || m.id.contains("intent")
                || m.id.contains("gliner")
        }) {
            Some(m) => m.id.clone(),
            None => "mmbert32k-intent-classifier-merged".to_string(),
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
                        error: ApiError::new(format!("NLU model failed to load: {err}"))
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

    let task = InferenceTaskRequest::NluIntent {
        model: Some(model_id),
        text,
        candidate_intents: request.candidate_intents,
        candidate_entities: request.candidate_entities,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::NluIntent(nlu_resp) => Json(nlu_resp).into_response(),
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
                    error: ApiError::new("Unexpected response variant from NLU engine"),
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
