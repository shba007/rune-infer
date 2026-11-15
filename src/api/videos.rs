//! `POST /v1/videos/generations`: text-to-video, returned as a base64 container.
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::api::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, ErrorResponse, VideoGenerationRequest};

fn invalid(message: impl Into<String>, param: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ApiError::new(message)
                .with_type("invalid_request_error")
                .with_param(param)
                .with_code("invalid_parameter_value"),
        }),
    )
        .into_response()
}

#[axum::debug_handler]
pub async fn video_generation_handler(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(request): Json<VideoGenerationRequest>,
) -> Response {
    if request.prompt.trim().is_empty() {
        return invalid(
            "'prompt' is a required string field for video generation.",
            "prompt",
        );
    }
    if let Some(size) = &request.size
        && super::images::parse_generation_dimensions(size).is_none()
    {
        return invalid("'size' must be WIDTHxHEIGHT from 64 to 4096 pixels", "size");
    }
    if request
        .frames
        .is_some_and(|frames| !(1..=257).contains(&frames))
    {
        return invalid("'frames' must be between 1 and 257", "frames");
    }
    if request.fps.is_some_and(|fps| !(1..=60).contains(&fps)) {
        return invalid("'fps' must be between 1 and 60", "fps");
    }
    if request
        .output_format
        .as_deref()
        .is_some_and(|format| !matches!(format, "webm" | "webp" | "avi"))
    {
        return invalid("'output_format' must be webm, webp or avi", "output_format");
    }

    let model_id = match &request.model {
        Some(model) if !model.trim().is_empty() => model.clone(),
        _ => match state
            .config()
            .models
            .iter()
            .find(|model| model.modality == crate::config::Modality::VideoGeneration)
        {
            Some(model) => model.id.clone(),
            None => {
                return invalid(
                    "No video generation model specified and none found in config",
                    "model",
                );
            }
        },
    };

    let gate = match crate::api::ModelExecutionGate::enter(&state, &headers, &model_id) {
        Ok(gate) => gate,
        Err(response) => return response,
    };

    let task = InferenceTaskRequest::VideoGeneration {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        size: request.size.clone().or_else(|| {
            gate.model_config
                .max_resolution
                .as_deref()
                .map(|value| value.split('(').next().unwrap_or(value).trim().to_string())
        }),
        frames: request.frames,
        fps: request.fps,
        steps: request.steps.or(gate.model_config.default_steps),
        cfg_scale: request.cfg_scale.or(gate.model_config.default_cfg_scale),
        seed: request.seed,
        sample_method: request
            .sample_method
            .clone()
            .or_else(|| gate.model_config.default_sample_method.clone()),
        output_format: request.output_format.clone(),
    };

    let engine = match gate.load_task_engine(&state, &task).await {
        Ok(engine) => engine,
        Err(response) => return response,
    };
    let output = match gate.execute_task(&state, engine, task).await {
        Ok(output) => output,
        Err(response) => return response,
    };

    match output.response {
        InferenceTaskResponse::Video(video) => {
            gate.attach_headers(&state, Json(video).into_response())
        }
        InferenceTaskResponse::Error(message) => gate.attach_headers(
            &state,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(message).with_type("server_error"),
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
