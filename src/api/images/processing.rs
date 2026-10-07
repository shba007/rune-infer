use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{
    ApiError, ErrorResponse, ImageCaptionRequest, ImageRestorationRequest,
    ImageStyleTransferRequest, ImageUpscaleRequest,
};

#[axum::debug_handler]
pub async fn image_caption_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageCaptionRequest>,
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

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.vision || m.modality == crate::config::Modality::VisionText)
        {
            Some(m) => m.id.clone(),
            None => "ornith-1.5-9b".to_string(),
        },
    };

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageCaption {
        model: Some(model_id.clone()),
        image_bytes,
        detail: request.detail.unwrap_or_else(|| "detailed".to_string()),
        max_tokens: request.max_tokens.unwrap_or(128),
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
            InferenceTaskResponse::Caption(cap) => Json(cap).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_upscale_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageUpscaleRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "real-esrgan-x4plus".to_string());

    let decoded_frames = match decode_media(&image_uri, (4096, 4096)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageUpscale {
        model: Some(model_id.clone()),
        image_bytes,
        scale: request.scale.unwrap_or(4),
        response_format: request.response_format,
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
            InferenceTaskResponse::Upscale(up) => Json(up).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_restoration_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageRestorationRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "codeformer-v0.1".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageRestoration {
        model: Some(model_id.clone()),
        image_bytes,
        fidelity_weight: request.fidelity_weight.unwrap_or(0.7),
        face_upsample: request.face_upsample.unwrap_or(true),
        response_format: request.response_format,
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
            InferenceTaskResponse::Restoration(rest) => Json(rest).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_style_transfer_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageStyleTransferRequest>,
) -> Response {
    let content_uri = match request.content_image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    let style_uri = match request.style_image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "neural-style-v2".to_string());

    let c_frames = match decode_media(&content_uri, (2048, 2048)) {
        Ok(f) if !f.is_empty() => f,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    let s_frames = match decode_media(&style_uri, (2048, 2048)) {
        Ok(f) if !f.is_empty() => f,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (content_bytes, _, _) = c_frames.into_iter().next().unwrap();
    let (style_bytes, _, _) = s_frames.into_iter().next().unwrap();

    let task = InferenceTaskRequest::ImageStyleTransfer {
        model: Some(model_id.clone()),
        content_bytes,
        style_bytes,
        strength: request.strength.unwrap_or(0.85),
        response_format: request.response_format,
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
            InferenceTaskResponse::StyleTransfer(st) => Json(st).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
