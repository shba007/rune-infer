use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{ApiError, DecisionRequest, ErrorResponse};

#[axum::debug_handler]
pub async fn decisions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DecisionRequest>,
) -> Response {
    if request.questions.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(
                    "'questions' is a required non-empty list of decision questions.",
                )
                .with_type("invalid_request_error")
                .with_param("questions")
                .with_code("empty_questions"),
            }),
        )
            .into_response();
    }

    let (text, image_uris) = request.state.split_text_and_images();
    if text.trim().is_empty() && image_uris.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'state' must contain text or at least one image frame.")
                    .with_type("invalid_request_error")
                    .with_param("state")
                    .with_code("empty_state"),
            }),
        )
            .into_response();
    }

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config().models.iter().find(|m| {
            m.modality == crate::config::Modality::Decision
                || m.is_decision()
                || m.id.contains("decision")
                || m.id.contains("d1-3b")
                || m.id.contains("liquid")
        }) {
            Some(m) => m.id.clone(),
            None => "liquid-d1-3b".to_string(),
        },
    };

    let model_config = state.config().find(&model_id).cloned();
    let max_dims = model_config
        .as_ref()
        .map(|c| c.max_dimensions())
        .unwrap_or((4096, 4096));

    let mut decoded_images = Vec::new();
    for uri in &image_uris {
        let uri_for_decode = uri.clone();
        let decoded_frames = match tokio::task::spawn_blocking(move || {
            decode_media(&uri_for_decode, max_dims)
        })
        .await
        {
            Ok(Ok(frames)) if !frames.is_empty() => frames,
            Ok(Ok(_)) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new(
                            "Failed to parse image frames from decision state image.",
                        )
                        .with_type("invalid_request_error")
                        .with_param("state.images")
                        .with_code("invalid_state_image"),
                    }),
                )
                    .into_response();
            }
            Ok(Err(err_resp)) => return (StatusCode::BAD_REQUEST, Json(err_resp)).into_response(),
            Err(join_err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!(
                            "State image decode thread error: {join_err}"
                        )),
                    }),
                )
                    .into_response();
            }
        };

        for (bytes, _w, _h) in decoded_frames {
            decoded_images.push(bytes);
        }
    }

    let inference = state.inference.clone();
    let target_model = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&target_model)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Decision model failed to load: {err}"))
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

    let task = InferenceTaskRequest::Decision {
        model: Some(model_id),
        text,
        images: decoded_images,
        questions: request.questions,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Decision(decision_resp) => Json(decision_resp).into_response(),
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
                    error: ApiError::new("Unexpected response variant from decision engine"),
                }),
            )
                .into_response(),
        },
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(format!("Decision inference failed: {err}")),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(format!("Execution thread error: {e}")),
            }),
        )
            .into_response(),
    }
}
