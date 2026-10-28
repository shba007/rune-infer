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
    headers: axum::http::HeaderMap,
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
        _ => match state.config().models.iter().find(|m| {
            m.modality == crate::config::Modality::Ocr
                || m.id.contains("ocr")
                || m.name.to_lowercase().contains("ocr")
        }) {
            Some(m) => m.id.clone(),
            None => "got-ocr-2.0".to_string(),
        },
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(g) => g,
        Err(resp) => return resp,
    };

    let max_dims = gate.model_config.max_dimensions();

    let uri_for_decode = image_uri.clone();
    let decoded_frames = match tokio::task::spawn_blocking(move || {
        decode_media(&uri_for_decode, max_dims)
    })
    .await
    {
        Ok(Ok(frames)) if !frames.is_empty() => frames,
        Ok(Ok(_)) => {
            return gate.attach_headers(
                &state,
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new("Failed to parse image frames from document source.")
                            .with_type("invalid_request_error")
                            .with_param("document")
                            .with_code("invalid_document_image"),
                    }),
                )
                    .into_response(),
            );
        }
        Ok(Err(err_resp)) => {
            return gate.attach_headers(
                &state,
                (StatusCode::BAD_REQUEST, Json(err_resp)).into_response(),
            );
        }
        Err(join_err) => {
            return gate.attach_headers(
                &state,
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Document decode thread error: {join_err}")),
                    }),
                )
                    .into_response(),
            );
        }
    };

    let (image_bytes, width, height) = decoded_frames.into_iter().next().unwrap();

    let features = request
        .features
        .unwrap_or_else(|| vec!["text".to_string(), "selection_marks".to_string()]);

    let task = InferenceTaskRequest::DocumentOcr {
        model: Some(gate.model_config.id.clone()),
        image_bytes,
        width,
        height,
        features,
        response_format: request.response_format,
    };

    let engine = match gate.load_task_engine(&state, &task).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(out) => out,
        Err(err_resp) => return err_resp,
    };

    match output.response {
        InferenceTaskResponse::Ocr(ocr_resp) => {
            gate.attach_headers(&state, Json(ocr_resp).into_response())
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
                    error: ApiError::new("Unexpected response variant from OCR engine"),
                }),
            )
                .into_response(),
        ),
    }
}
