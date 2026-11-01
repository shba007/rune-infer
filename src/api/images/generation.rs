use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::resolve_image_dimensions;
use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, ErrorResponse, ImageGenerationRequest};

#[axum::debug_handler]
pub async fn image_generation_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
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
        if super::parse_generation_dimensions(s).is_none() {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'size' must be WIDTHxHEIGHT with integer dimensions from 64 to 4096 pixels")
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
            .config()
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

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(g) => g,
        Err(resp) => return resp,
    };

    let resolved_size = resolve_image_dimensions(
        request.size.as_deref(),
        request.aspect_ratio.as_deref(),
        gate.model_config.max_resolution.as_deref(),
    );

    let task = InferenceTaskRequest::ImageGeneration {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        size: Some(resolved_size),
        response_format: request.response_format.clone(),
        steps: request.steps.or(gate.model_config.default_steps),
        cfg_scale: request.cfg_scale.or(gate.model_config.default_cfg_scale),
        seed: request.seed,
        sample_method: request
            .sample_method
            .clone()
            .or_else(|| gate.model_config.default_sample_method.clone()),
    };

    let engine = match gate.load_task_engine(&state, &task).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };

    let output = match gate.execute_task(&state, engine, task).await {
        Ok(out) => out,
        Err(resp) => return resp,
    };

    match output.response {
        InferenceTaskResponse::Image(img_resp) => {
            gate.attach_headers(&state, Json(img_resp).into_response())
        }
        InferenceTaskResponse::Error(e) => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response(),
        ),
        _ => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type").with_type("server_error"),
                }),
            )
                .into_response(),
        ),
    }
}

fn image_error(parameter: &str, message: impl Into<String>, variation: bool) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ApiError::new(message)
                .with_type("invalid_request_error")
                .with_param(parameter)
                .with_code(if variation {
                    "invalid_image_variation"
                } else {
                    "invalid_image_edit"
                }),
        }),
    )
        .into_response()
}

#[allow(clippy::result_large_err)]
fn edit_number<T: std::str::FromStr>(
    value: Option<String>,
    key: &str,
    variation: bool,
) -> Result<Option<T>, Response> {
    value
        .map(|text| {
            text.trim().parse().map_err(|_| {
                image_error(
                    key,
                    format!("'{key}' has an invalid numeric value"),
                    variation,
                )
            })
        })
        .transpose()
}

#[allow(clippy::result_large_err)]
async fn parse_image_request(
    mut multipart: Multipart,
    variation: bool,
) -> Result<(String, InferenceTaskRequest), Response> {
    let mut fields = std::collections::HashMap::new();
    let mut images = Vec::new();
    let mut mask = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| image_error("body", error.to_string(), variation))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" | "image[]" => images.push(
                field
                    .bytes()
                    .await
                    .map_err(|error| image_error("image", error.to_string(), variation))?
                    .to_vec(),
            ),
            "mask" => {
                if variation {
                    return Err(image_error(
                        "mask",
                        "Use image edits for masked changes",
                        true,
                    ));
                }
                if mask.is_some() {
                    return Err(image_error("mask", "Duplicate mask upload", false));
                }
                mask = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| image_error("mask", error.to_string(), false))?
                        .to_vec(),
                );
            }
            "model" | "prompt" | "size" | "n" | "response_format" | "steps" | "cfg_scale"
            | "seed" => {
                if variation && name == "prompt" {
                    return Err(image_error(
                        "prompt",
                        "Image variations use the source image; use image edits for a custom prompt",
                        true,
                    ));
                }
                let value = field
                    .text()
                    .await
                    .map_err(|error| image_error(&name, error.to_string(), variation))?;
                if fields.insert(name.clone(), value).is_some() {
                    return Err(image_error(&name, "Duplicate image parameter", variation));
                }
            }
            _ => {
                return Err(image_error(
                    &name,
                    format!("Unsupported image parameter '{name}'"),
                    variation,
                ));
            }
        }
    }
    let model = fields
        .remove("model")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| image_error("model", "Select an exact model ID", variation))?;
    if variation && images.len() != 1 {
        return Err(image_error(
            "image",
            "Image variations require exactly one source image",
            true,
        ));
    }
    let size = fields
        .remove("size")
        .map(|size| {
            super::parse_generation_dimensions(&size)
                .map(|(w, h)| format!("{w}x{h}"))
                .ok_or_else(|| {
                    image_error(
                        "size",
                        "Size must be WIDTHxHEIGHT from 64 to 4096 pixels",
                        variation,
                    )
                })
        })
        .transpose()?;
    let task = InferenceTaskRequest::ImageEdit {
        prompt: if variation {
            "Create a subtle variation of this image. Preserve the same main subject, its defining colors and materials, and the original visual style. Make small changes to composition and fine details while keeping the subject recognizable.".into()
        } else {
            fields.remove("prompt").unwrap_or_default()
        },
        images,
        mask,
        size,
        response_format: fields.remove("response_format"),
        n: edit_number(fields.remove("n"), "n", variation)?,
        steps: edit_number(fields.remove("steps"), "steps", variation)?,
        cfg_scale: edit_number(fields.remove("cfg_scale"), "cfg_scale", variation)?,
        seed: edit_number(fields.remove("seed"), "seed", variation)?,
    };
    crate::inference::engines::sd_server::edit::validate_task(&task)
        .map_err(|message| image_error("image", message, variation))?;
    Ok((model.trim().to_string(), task))
}

#[axum::debug_handler]
pub async fn image_edit_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    multipart: Multipart,
) -> Response {
    execute_image_request(state, headers, multipart, false).await
}

async fn execute_image_request(
    state: Arc<AppState>,
    headers: axum::http::HeaderMap,
    multipart: Multipart,
    variation: bool,
) -> Response {
    let (model_id, task) = match parse_image_request(multipart, variation).await {
        Ok(parsed) => parsed,
        Err(response) => return response,
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };

    let engine = match gate.load_task_engine(&state, &task).await {
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
            InferenceTaskResponse::Image(img) => Json(img).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
    )
}

#[axum::debug_handler]
pub async fn image_variation_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    multipart: Multipart,
) -> Response {
    execute_image_request(state, headers, multipart, true).await
}
