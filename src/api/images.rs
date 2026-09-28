use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, ErrorResponse, ImageGenerationRequest};

pub fn resolve_image_dimensions(
    size: Option<&str>,
    aspect_ratio: Option<&str>,
    default_res: Option<&str>,
) -> String {
    if let Some(s) = size {
        if s.contains('x') || s.contains('×') || s.contains('*') {
            return s.to_string();
        }
    }

    if let Some(ar) = aspect_ratio {
        let clean = ar.to_lowercase();
        if clean.contains("16:9") {
            return "1024x576".to_string();
        } else if clean.contains("9:16") {
            return "576x1024".to_string();
        } else if clean.contains("1:1") {
            return "1024x1024".to_string();
        } else if clean.contains("4:3") {
            return "1024x768".to_string();
        } else if clean.contains("3:4") {
            return "768x1024".to_string();
        } else if clean.contains("21:9") {
            return "1280x544".to_string();
        } else if clean.contains("9:21") {
            return "544x1280".to_string();
        }
    }

    if let Some(d) = default_res {
        let clean = d.split('(').next().unwrap_or(d).trim();
        if clean.contains('x') || clean.contains('×') || clean.contains('*') {
            return clean.replace('×', "x").replace('*', "x");
        }
    }

    "1024x1024".to_string()
}

#[axum::debug_handler]
pub async fn image_generation_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageGenerationRequest>,
) -> Response {
    let model_id = match &request.model {
        Some(m) if !m.trim().is_empty() => m.clone(),
        _ => {
            match state
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
            }
        }
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
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
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
                    error: ApiError::new("Unexpected response type from engine")
                        .with_type("server_error"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e).with_type("server_error"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e.to_string()).with_type("server_error"),
            }),
        )
            .into_response(),
    }
}
