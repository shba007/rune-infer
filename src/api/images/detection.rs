use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{
    ApiError, DetectObjectsRequest, ErrorResponse, ImageEmbeddingsRequest, ImageGroundingRequest,
    RecognizeObjectsRequest,
};

#[axum::debug_handler]
pub async fn detect_objects_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<DetectObjectsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'image' is a required string field containing a URL or base64 data URI.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("image")
                    .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "yolo-11x-detection".to_string());

    let uri_for_decode = image_uri.clone();
    let decoded_frames = match tokio::task::spawn_blocking(move || {
        decode_media(&uri_for_decode, (2048, 2048))
    })
    .await
    {
        Ok(Ok(frames)) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::DetectObjects {
        model: Some(model_id.clone()),
        image_bytes,
        prompt: request.prompt,
        categories: request.categories,
        confidence_threshold: request.confidence_threshold,
        iou_threshold: request.iou_threshold,
        features: request.features.unwrap_or_default(),
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };
    let engine = match gate.load_engine(&state).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(output) => output,
        Err(resp) => return resp,
    };
    gate.attach_headers(
        &state,
        match output.response {
            InferenceTaskResponse::Detection(det) => Json(det).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
    )
}

#[axum::debug_handler]
pub async fn image_embeddings_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ImageEmbeddingsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'image' is a required string field.")
                        .with_type("invalid_request_error")
                        .with_param("image")
                        .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    if let Some(ref regs) = request.regions {
        for (idx, reg) in regs.iter().enumerate() {
            for &coord in &reg.box_ {
                if coord < 0.0 || coord > 1.0 {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: ApiError::new("Region bounding box coordinates must be normalized between 0.0 and 1.0.")
                                .with_type("invalid_request_error")
                                .with_param(format!("regions[{}].box", idx))
                                .with_code("invalid_box_coordinates"),
                        }),
                    )
                        .into_response();
                }
            }
        }
    }

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "clip-vit-large-patch14".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageEmbedding {
        model: Some(model_id.clone()),
        image_bytes,
        regions: request.regions,
        encoding_format: request.encoding_format,
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };
    let engine = match gate.load_engine(&state).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(output) => output,
        Err(resp) => return resp,
    };
    gate.attach_headers(
        &state,
        match output.response {
            InferenceTaskResponse::ImageEmbedding(emb) => Json(emb).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
    )
}

#[axum::debug_handler]
pub async fn recognize_objects_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<RecognizeObjectsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "clip-vit-large-patch14".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::RecognizeObjects {
        model: Some(model_id.clone()),
        image_bytes,
        top_k: request.top_k.unwrap_or(3),
        targets: request.targets,
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };
    let engine = match gate.load_engine(&state).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(output) => output,
        Err(resp) => return resp,
    };
    gate.attach_headers(
        &state,
        match output.response {
            InferenceTaskResponse::Recognition(rec) => Json(rec).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
    )
}

#[axum::debug_handler]
pub async fn image_grounding_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<ImageGroundingRequest>,
) -> Response {
    if request.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(
                    "'query' (or 'target') is a required non-empty string parameter specifying what UI element or object to locate.",
                )
                .with_type("invalid_request_error")
                .with_param("query")
                .with_code("missing_query"),
            }),
        )
            .into_response();
    }

    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'image' is a required string field containing an image URL or base64 data URI.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("image")
                    .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config().models.iter().find(|m| m.is_vision()) {
            Some(m) => m.id.clone(),
            None => "qwen3.8-27b".to_string(),
        },
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(g) => g,
        Err(resp) => return resp,
    };

    let max_dims = gate.model_config.max_dimensions();

    let uri_for_decode = image_uri.clone();
    let decoded_frames =
        match tokio::task::spawn_blocking(move || decode_media(&uri_for_decode, max_dims)).await {
            Ok(Ok(frames)) if !frames.is_empty() => frames,
            Ok(Err(err_resp)) => {
                return gate.attach_headers(
                    &state,
                    (StatusCode::BAD_REQUEST, Json(err_resp)).into_response(),
                );
            }
            _ => {
                return gate.attach_headers(
                    &state,
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: ApiError::new(
                                "Failed to decode provided image for visual grounding.",
                            )
                            .with_type("invalid_request_error"),
                        }),
                    )
                        .into_response(),
                );
            }
        };

    let (image_bytes, w, h) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageGrounding {
        model: Some(gate.model_config.id.clone()),
        image_bytes,
        query: request.query,
        width: w,
        height: h,
    };

    let engine = match gate.load_engine(&state).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(out) => out,
        Err(err_resp) => return err_resp,
    };

    match output.response {
        InferenceTaskResponse::Grounding(grounding) => {
            gate.attach_headers(&state, Json(grounding).into_response())
        }
        _ => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type from vision grounding engine"),
                }),
            )
                .into_response(),
        ),
    }
}
