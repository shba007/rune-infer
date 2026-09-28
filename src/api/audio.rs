use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, AudioSpeechRequest, ErrorResponse};

#[axum::debug_handler]
pub async fn audio_transcriptions_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut file_bytes = Vec::new();
    let mut filename = "audio.wav".to_string();
    let mut model_id: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut language: Option<String> = None;
    let mut temperature: Option<f32> = None;
    let mut response_format: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    filename = fname.to_string();
                }
                if let Ok(bytes) = field.bytes().await {
                    file_bytes = bytes.to_vec();
                }
            }
            "model" => {
                if let Ok(text) = field.text().await {
                    model_id = Some(text.trim().to_string());
                }
            }
            "prompt" => {
                if let Ok(text) = field.text().await {
                    prompt = Some(text);
                }
            }
            "language" => {
                if let Ok(text) = field.text().await {
                    language = Some(text);
                }
            }
            "temperature" => {
                if let Ok(text) = field.text().await {
                    temperature = text.parse().ok();
                }
            }
            "response_format" => {
                if let Ok(text) = field.text().await {
                    response_format = Some(text);
                }
            }
            _ => {}
        }
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing audio file in 'file' multipart field")
                    .with_type("invalid_request_error"),
            }),
        )
            .into_response();
    }

    let target_model = match model_id {
        Some(ref m) if !m.is_empty() => m.clone(),
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::SpeechToText
                || m.architecture.eq_ignore_ascii_case("crispasr")
                || m.architecture.eq_ignore_ascii_case("audio8")
        }) {
            Some(m) => m.id.clone(),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new(
                            "No speech-to-text model specified and none found in catalog",
                        )
                        .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
        },
    };

    let inference = state.inference.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&target_model)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model load failure: {err}"))
                            .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let task = InferenceTaskRequest::AudioTranscription {
        audio_bytes: file_bytes,
        filename,
        prompt,
        language,
        temperature,
        response_format,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Transcription(transcription) => {
                Json(transcription).into_response()
            }
            InferenceTaskResponse::Text(raw) => {
                Json(serde_json::json!({ "text": raw })).into_response()
            }
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
                    error: ApiError::new("Unexpected response type from ASR engine"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e.to_string()).with_type("server_error"),
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

#[axum::debug_handler]
pub async fn audio_speech_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response {
    let model_id = match &request.model {
        Some(m) if !m.is_empty() => m.clone(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.modality == crate::config::Modality::TextToSpeech)
        {
            Some(m) => m.id.clone(),
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new(
                            "No text-to-speech model specified and none found in config",
                        )
                        .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
        },
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

    let format_str = request
        .response_format
        .clone()
        .unwrap_or_else(|| "wav".to_string());
    let task = InferenceTaskRequest::AudioSpeech {
        input: request.input,
        voice: request.voice,
        response_format: Some(format_str.clone()),
        speed: request.speed,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Audio(bytes) => {
                let content_type = match format_str.as_str() {
                    "mp3" => "audio/mpeg",
                    "opus" => "audio/opus",
                    "flac" => "audio/flac",
                    _ => "audio/wav",
                };
                Response::builder()
                    .header("Content-Type", content_type)
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
            }
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
                    error: ApiError::new("Unexpected response type from TTS engine"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e.to_string()).with_type("server_error"),
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
