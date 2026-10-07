use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{
    ApiError, DetectObjectsRequest, ErrorResponse, ImageEmbeddingsRequest, RecognizeObjectsRequest,
};

#[axum::debug_handler]
pub async fn detect_objects_handler(
    State(state): State<Arc<AppState>>,
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

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
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

    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Detection(det) => Json(det).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_embeddings_handler(
    State(state): State<Arc<AppState>>,
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

    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::ImageEmbedding(emb) => Json(emb).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn recognize_objects_handler(
    State(state): State<Arc<AppState>>,
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

    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Recognition(rec) => Json(rec).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
