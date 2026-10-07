use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use std::sync::Arc;

use super::resolve_image_dimensions;
use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    ApiError, ErrorResponse, ImageData, ImageGenerationRequest, ImageGenerationResponse,
};

#[axum::debug_handler]
pub async fn image_generation_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageGenerationRequest>,
) -> Response {
    if request.prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'prompt' is a required string field for image generation.")
                    .with_type("invalid_request_error")
                    .with_param("prompt")
                    .with_code("missing_required_parameter"),
            }),
        )
            .into_response();
    }

    if let Some(ref s) = request.size {
        let allowed = ["256x256", "512x512", "1024x1024"];
        if !allowed.contains(&s.as_str()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!(
                        "'{}' is not one of ['256x256', '512x512', '1024x1024'] - 'size'",
                        s
                    ))
                    .with_type("invalid_request_error")
                    .with_param("size")
                    .with_code("invalid_parameter_value"),
                }),
            )
                .into_response();
        }
    }

    let model_id = match &request.model {
        Some(m) if !m.trim().is_empty() => m.clone(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.modality == crate::config::Modality::ImageGeneration)
        {
            Some(m) => m.id.clone(),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new(
                            "No image generation model specified and none found in config",
                        )
                        .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
        },
    };

    let model_config = match state.config.find(&model_id) {
        Some(c) => c.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found", model_id))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let model_id_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_id_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model failed to load: {err}"))
                            .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Thread join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let resolved_size = resolve_image_dimensions(
        request.size.as_deref(),
        request.aspect_ratio.as_deref(),
        model_config.max_resolution.as_deref(),
    );

    let task = InferenceTaskRequest::ImageGeneration {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        size: Some(resolved_size),
        response_format: request.response_format.clone(),
        steps: request.steps.or(model_config.default_steps),
        cfg_scale: request.cfg_scale.or(model_config.default_cfg_scale),
        seed: request.seed,
        sample_method: request
            .sample_method
            .clone()
            .or_else(|| model_config.default_sample_method.clone()),
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Image(img_resp) => Json(img_resp).into_response(),
            InferenceTaskResponse::Error(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type").with_type("server_error"),
                }),
            )
                .into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_edit_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut prompt: Option<String> = None;
    let mut image_bytes: Vec<u8> = Vec::new();
    let mut mask_bytes: Vec<u8> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "prompt" => {
                if let Ok(text) = field.text().await {
                    prompt = Some(text);
                }
            }
            "image" => {
                if let Ok(bytes) = field.bytes().await {
                    image_bytes = bytes.to_vec();
                }
            }
            "mask" => {
                if let Ok(bytes) = field.bytes().await {
                    mask_bytes = bytes.to_vec();
                }
            }
            _ => {}
        }
    }

    if prompt.as_ref().is_none_or(|p| p.trim().is_empty()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'prompt' is a required string field for image edits.")
                    .with_type("invalid_request_error")
                    .with_param("prompt")
                    .with_code("missing_required_parameter"),
            }),
        )
            .into_response();
    }

    let mask_mismatch = match (
        image::load_from_memory(&image_bytes),
        image::load_from_memory(&mask_bytes),
    ) {
        (Ok(img), Ok(mask)) => (img.width(), img.height()) != (mask.width(), mask.height()),
        _ => false,
    };

    if mask_mismatch {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Uploaded mask dimensions must match the source image dimensions exactly.",
                )
                .with_type("invalid_request_error")
                .with_param("mask")
                .with_code("mask_dimension_mismatch"),
            }),
        )
            .into_response();
    }

    let model_id = state
        .config
        .models
        .iter()
        .find(|m| m.modality == crate::config::Modality::ImageGeneration)
        .map(|m| m.id.clone())
        .unwrap_or_else(|| "z-image".to_string());

    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        _ => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let task = InferenceTaskRequest::ImageGeneration {
        prompt: prompt.unwrap_or_default(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        response_format: Some("b64_json".to_string()),
        steps: None,
        cfg_scale: None,
        seed: None,
        sample_method: None,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Image(img) => Json(img).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_variation_handler(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut image_bytes: Vec<u8> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name().unwrap_or("") == "image" {
            if let Ok(bytes) = field.bytes().await {
                image_bytes = bytes.to_vec();
            }
        }
    }

    if image_bytes.is_empty() || image_bytes.len() > 4 * 1024 * 1024 {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(ErrorResponse {
                error: ApiError::new("Uploaded image file must be a valid PNG, JPEG, or WEBP image format under 4 MB.")
                    .with_type("invalid_request_error")
                    .with_param("image")
                    .with_code("invalid_image_type"),
            }),
        )
            .into_response();
    }

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);
    Json(ImageGenerationResponse {
        created,
        data: vec![ImageData {
            b64_json: Some(b64),
            url: None,
        }],
    })
    .into_response()
}
