use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{ApiError, DocumentOcrRequest, ErrorResponse};

#[axum::debug_handler]
pub async fn document_ocr_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DocumentOcrRequest>,
) -> Response {
    let doc_url = match request.document {
        Some(ref doc) => doc
            .image_url
            .as_deref()
            .or(doc.url.as_deref())
            .filter(|u| !u.trim().is_empty())
            .map(|u| u.to_string()),
        None => None,
    };

    let image_uri = match doc_url {
        Some(url) => url,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'document' is a required object containing a document/image source.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("document")
                    .with_code("missing_required_document"),
                }),
            )
                .into_response();
        }
    };

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::Ocr
                || m.id.contains("ocr")
                || m.name.to_lowercase().contains("ocr")
        }) {
            Some(m) => m.id.clone(),
            None => "got-ocr-2.0".to_string(),
        },
    };

    let model_config = state.config.find(&model_id).cloned();
    let max_dims = model_config
        .as_ref()
        .map(|c| c.max_dimensions())
        .unwrap_or((4096, 4096));

    let decoded_frames = match decode_media(&image_uri, max_dims) {
        Ok(frames) if !frames.is_empty() => frames,
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("Failed to parse image frames from document source.")
                        .with_type("invalid_request_error")
                        .with_param("document")
                        .with_code("invalid_document_image"),
                }),
            )
                .into_response();
        }
        Err(err_resp) => return (StatusCode::BAD_REQUEST, Json(err_resp)).into_response(),
    };

    let (image_bytes, width, height) = decoded_frames.into_iter().next().unwrap();

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("OCR model failed to load: {err}"))
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

    let features = request
        .features
        .unwrap_or_else(|| vec!["text".to_string(), "selection_marks".to_string()]);

    let task = InferenceTaskRequest::DocumentOcr {
        model: Some(model_id),
        image_bytes,
        width,
        height,
        features,
        response_format: request.response_format,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Ocr(ocr_resp) => Json(ocr_resp).into_response(),
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
                    error: ApiError::new("Unexpected response variant from OCR engine"),
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
