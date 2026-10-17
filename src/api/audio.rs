use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ApiError, AudioSpeechRequest, ErrorResponse};

const MAX_AUDIO_UPLOAD_BYTES: usize = 25 * 1024 * 1024; // 25 MB

async fn read_limited_field_bytes(
    field: &mut axum::extract::multipart::Field<'_>,
    limit: usize,
) -> Result<Vec<u8>, (StatusCode, Json<ErrorResponse>)> {
    let mut bytes = Vec::new();
    loop {
        match field.chunk().await {
            Ok(Some(chunk)) => {
                if bytes.len() + chunk.len() > limit {
                    return Err((
                        StatusCode::PAYLOAD_TOO_LARGE,
                        Json(ErrorResponse {
                            error: ApiError::new(
                                "Maximum audio file upload size is 25 MB. The uploaded file exceeds this limit.",
                            )
                            .with_type("invalid_request_error")
                            .with_param("file")
                            .with_code("file_too_large"),
                        }),
                    ));
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(err) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Failed reading audio upload stream: {err}"))
                            .with_type("invalid_request_error")
                            .with_param("file"),
                    }),
                ));
            }
        }
    }
    Ok(bytes)
}

#[axum::debug_handler]
pub async fn audio_transcriptions_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let start_time = std::time::Instant::now();
    let mut file_bytes = Vec::new();
    let mut filename = "audio.wav".to_string();
    let mut model_id: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut language: Option<String> = None;
    let mut temperature: Option<f32> = None;
    let mut response_format: Option<String> = None;
    let mut diarize = false;

    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    filename = fname.to_string();
                }
                match read_limited_field_bytes(&mut field, MAX_AUDIO_UPLOAD_BYTES).await {
                    Ok(bytes) => file_bytes = bytes,
                    Err(err_resp) => return err_resp.into_response(),
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
            "diarize" | "diarization" => {
                if let Ok(text) = field.text().await {
                    let val = text.trim().to_lowercase();
                    diarize = val == "true" || val == "1" || val == "yes";
                }
            }
            _ => {}
        }
    }

    if file_bytes.len() > MAX_AUDIO_UPLOAD_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Maximum audio file upload size is 25 MB. The uploaded file exceeds this limit.",
                )
                .with_type("invalid_request_error")
                .with_param("file")
                .with_code("file_too_large"),
            }),
        )
            .into_response();
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'file' is a required multipart/form-data field.")
                    .with_type("invalid_request_error")
                    .with_param("file")
                    .with_code("missing_required_file"),
            }),
        )
            .into_response();
    }

    let target_model = match model_id {
        Some(ref m) if !m.is_empty() => m.clone(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.modality == crate::config::Modality::SpeechToText)
        {
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
    let model_to_fetch = target_model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %err,
                    "Failed to load audio transcription engine"
                );
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
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %e,
                    "Join error when retrieving audio transcription engine"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let diar_audio_bytes = if diarize {
        Some(file_bytes.clone())
    } else {
        None
    };
    let diar_filename = filename.clone();

    let task = InferenceTaskRequest::AudioTranscription {
        audio_bytes: file_bytes,
        filename,
        prompt,
        language,
        temperature,
        response_format,
    };

    let model_for_stats = target_model.clone();
    let res = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let r = engine.execute(&task, None).map_err(|e| e.to_string());
        (r, gen_started.elapsed())
    })
    .await;

    let (exec_res, gen_elapsed) = match res {
        Ok((r, elapsed)) => (r, elapsed),
        Err(e) => (
            Err(format!("Thread join error: {e}")),
            std::time::Duration::ZERO,
        ),
    };

    match exec_res {
        Ok(mut output) => {
            if diarize {
                if let InferenceTaskResponse::Transcription(ref mut trans) = output.response {
                    let diar_model_id = state
                        .config
                        .models
                        .iter()
                        .find(|m| m.modality == crate::config::Modality::SpeechAnalysis)
                        .map(|m| m.id.clone())
                        .unwrap_or_else(|| "nemotron-3-diarization".to_string());

                    let inference_diar = state.inference.clone();
                    let diar_bytes = diar_audio_bytes.unwrap_or_default();
                    let diar_fname = diar_filename.clone();

                    let diar_res = tokio::task::spawn_blocking(move || {
                        let diar_engine = inference_diar.get_engine(&diar_model_id)?;
                        let diar_task = InferenceTaskRequest::AudioDiarization {
                            audio_bytes: diar_bytes,
                            filename: diar_fname,
                            speaker_threshold: None,
                            max_speakers: None,
                        };
                        diar_engine
                            .execute(&diar_task, None)
                            .map_err(|e| e.to_string())
                    })
                    .await;

                    if let Ok(Ok(diar_output)) = diar_res {
                        if let InferenceTaskResponse::Diarization(diar) = diar_output.response {
                            trans.speakers = Some(diar.speakers.clone());
                            trans.speaker_activity = Some(diar.speaker_activity);
                            if trans.duration.is_none() || trans.duration == Some(0.0) {
                                trans.duration = Some(diar.duration);
                            }

                            // Align segments with diarization turns
                            if let Some(serde_json::Value::Array(ref mut segs)) = trans.segments {
                                for seg in segs.iter_mut() {
                                    if let serde_json::Value::Object(obj) = seg {
                                        let seg_start = obj
                                            .get("start")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(0.0);
                                        let seg_end = obj
                                            .get("end")
                                            .and_then(|v| v.as_f64())
                                            .unwrap_or(seg_start);

                                        let mut best_speaker = None;
                                        let mut max_overlap = 0.0;
                                        for d in &diar.segments {
                                            let overlap = (seg_end.min(d.end)
                                                - seg_start.max(d.start))
                                            .max(0.0);
                                            if overlap > max_overlap {
                                                max_overlap = overlap;
                                                best_speaker = Some(d.speaker.clone());
                                            }
                                        }

                                        if let Some(spk) = best_speaker {
                                            obj.insert(
                                                "speaker".to_string(),
                                                serde_json::Value::String(spk),
                                            );
                                        } else if let Some(first_spk) = diar.speakers.first() {
                                            obj.insert(
                                                "speaker".to_string(),
                                                serde_json::Value::String(first_spk.clone()),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                gen_elapsed.as_millis(),
            );
            match output.response {
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
            }
        }
        Err(e) => {
            crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                &e,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response()
        }
    }
}

#[axum::debug_handler]
pub async fn audio_translations_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let start_time = std::time::Instant::now();
    let mut file_bytes = Vec::new();
    let mut filename = "audio.wav".to_string();
    let mut model_id: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut temperature: Option<f32> = None;
    let mut response_format: Option<String> = None;

    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    filename = fname.to_string();
                }
                match read_limited_field_bytes(&mut field, MAX_AUDIO_UPLOAD_BYTES).await {
                    Ok(bytes) => file_bytes = bytes,
                    Err(err_resp) => return err_resp.into_response(),
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

    if file_bytes.len() > MAX_AUDIO_UPLOAD_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Maximum audio file upload size is 25 MB. The uploaded file exceeds this limit.",
                )
                .with_type("invalid_request_error")
                .with_param("file")
                .with_code("file_too_large"),
            }),
        )
            .into_response();
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'file' is a required multipart/form-data field.")
                    .with_type("invalid_request_error")
                    .with_param("file")
                    .with_code("missing_required_file"),
            }),
        )
            .into_response();
    }

    let target_model = match model_id {
        Some(ref m) if !m.is_empty() => m.clone(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.modality == crate::config::Modality::SpeechToText)
        {
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
    let model_to_fetch = target_model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %err,
                    "Failed to load audio translation engine"
                );
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model load failure: {err}"))
                            .with_type("invalid_request_error")
                            .with_param("model")
                            .with_code("model_not_found"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %e,
                    "Join error when retrieving audio translation engine"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let task = InferenceTaskRequest::AudioTranslation {
        audio_bytes: file_bytes,
        filename,
        prompt,
        temperature,
        response_format,
    };

    let model_for_stats = target_model.clone();
    let res = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let r = engine.execute(&task, None).map_err(|e| e.to_string());
        (r, gen_started.elapsed())
    })
    .await;

    let (exec_res, gen_elapsed) = match res {
        Ok((r, elapsed)) => (r, elapsed),
        Err(e) => (
            Err(format!("Thread join error: {e}")),
            std::time::Duration::ZERO,
        ),
    };

    match exec_res {
        Ok(output) => {
            crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                gen_elapsed.as_millis(),
            );
            match output.response {
                InferenceTaskResponse::Translation(trans) => Json(trans).into_response(),
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
                        error: ApiError::new(
                            "Unexpected response type from ASR translation engine",
                        ),
                    }),
                )
                    .into_response(),
            }
        }
        Err(e) => {
            crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                &e,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response()
        }
    }
}

#[axum::debug_handler]
pub async fn audio_speech_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response {
    let start_time = std::time::Instant::now();
    if request.input.trim().is_empty() || request.input.len() > 4096 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(
                    "'input' field must be a non-empty string under 4096 characters.",
                )
                .with_type("invalid_request_error")
                .with_param("input")
                .with_code("invalid_payload"),
            }),
        )
            .into_response();
    }

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
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                crate::telemetry::record_failure(
                    &model_id,
                    404,
                    start_time.elapsed().as_millis(),
                    &err.to_string(),
                );
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
                crate::telemetry::record_failure(
                    &model_id,
                    500,
                    start_time.elapsed().as_millis(),
                    &e.to_string(),
                );
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

    let model_for_stats = model_id.clone();
    let res = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let r = engine.execute(&task, None).map_err(|e| e.to_string());
        (r, gen_started.elapsed())
    })
    .await;

    let (exec_res, gen_elapsed) = match res {
        Ok((r, elapsed)) => (r, elapsed),
        Err(e) => (
            Err(format!("Thread join error: {e}")),
            std::time::Duration::ZERO,
        ),
    };

    match exec_res {
        Ok(output) => {
            crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                gen_elapsed.as_millis(),
            );
            match output.response {
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
            }
        }
        Err(e) => {
            crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                &e,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response()
        }
    }
}

#[axum::debug_handler]
pub async fn speech_to_speech_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let start_time = std::time::Instant::now();
    let mut file_bytes = Vec::new();
    let mut model_id: Option<String> = None;
    let mut target_language = "fra".to_string();
    let mut source_language = Some("auto".to_string());
    let mut response_format = Some("wav".to_string());

    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => match read_limited_field_bytes(&mut field, MAX_AUDIO_UPLOAD_BYTES).await {
                Ok(bytes) => file_bytes = bytes,
                Err(err_resp) => return err_resp.into_response(),
            },
            "model" => {
                if let Ok(text) = field.text().await {
                    model_id = Some(text.trim().to_string());
                }
            }
            "target_language" => {
                if let Ok(text) = field.text().await {
                    target_language = text.trim().to_string();
                }
            }
            "source_language" => {
                if let Ok(text) = field.text().await {
                    source_language = Some(text.trim().to_string());
                }
            }
            "response_format" => {
                if let Ok(text) = field.text().await {
                    response_format = Some(text.trim().to_string());
                }
            }
            _ => {}
        }
    }

    if file_bytes.len() > MAX_AUDIO_UPLOAD_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Maximum audio file upload size is 25 MB. The uploaded file exceeds this limit.",
                )
                .with_type("invalid_request_error")
                .with_param("file")
                .with_code("file_too_large"),
            }),
        )
            .into_response();
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'file' is a required multipart/form-data field.")
                    .with_type("invalid_request_error")
                    .with_param("file")
                    .with_code("missing_required_file"),
            }),
        )
            .into_response();
    }

    let target_model = match model_id {
        Some(ref m) if !m.is_empty() => m.clone(),
        _ => match state.config.models.iter().find(|m| {
            m.modality == crate::config::Modality::SpeechToSpeech
                || m.id.contains("seamless")
                || m.id.contains("s2st")
        }) {
            Some(m) => m.id.clone(),
            None => "seamless-m4t-v2-large".to_string(),
        },
    };

    let inference = state.inference.clone();
    let model_to_fetch = target_model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
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

    let task = InferenceTaskRequest::SpeechToSpeech {
        audio_bytes: file_bytes,
        target_language,
        source_language,
        response_format: response_format.clone(),
    };

    let model_for_stats = target_model.clone();
    let res = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let r = engine.execute(&task, None).map_err(|e| e.to_string());
        (r, gen_started.elapsed())
    })
    .await;

    let (exec_res, gen_elapsed) = match res {
        Ok((r, elapsed)) => (r, elapsed),
        Err(e) => (
            Err(format!("Thread join error: {e}")),
            std::time::Duration::ZERO,
        ),
    };

    match exec_res {
        Ok(output) => {
            crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                gen_elapsed.as_millis(),
            );
            match output.response {
                InferenceTaskResponse::Audio(bytes) => Response::builder()
                    .header("Content-Type", "audio/wav")
                    .body(axum::body::Body::from(bytes))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
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
                        error: ApiError::new("Unexpected response type from S2ST engine"),
                    }),
                )
                    .into_response(),
            }
        }
        Err(e) => {
            crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                &e,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response()
        }
    }
}

fn attach_transcription_to_diarization(
    diar: &mut crate::types::AudioDiarizationResponse,
    asr_resp: &crate::types::AudioTranscriptionResponse,
) {
    if diar.text.is_none() && !asr_resp.text.is_empty() {
        diar.text = Some(asr_resp.text.clone());
    }

    struct WordItem {
        word: String,
        start: f64,
        end: f64,
    }

    let mut words: Vec<WordItem> = Vec::new();

    // 1. Check top-level asr_resp.words
    if let Some(serde_json::Value::Array(word_arr)) = &asr_resp.words {
        for w in word_arr {
            if let serde_json::Value::Object(obj) = w {
                let word_str = obj.get("word").and_then(|v| v.as_str()).unwrap_or("");
                let start = obj.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let end = obj.get("end").and_then(|v| v.as_f64()).unwrap_or(start);
                if !word_str.is_empty() {
                    words.push(WordItem {
                        word: word_str.to_string(),
                        start,
                        end,
                    });
                }
            }
        }
    }

    // 2. Check segment-level words in asr_resp.segments
    if words.is_empty() {
        if let Some(serde_json::Value::Array(segs)) = &asr_resp.segments {
            for seg in segs {
                if let serde_json::Value::Object(seg_obj) = seg {
                    if let Some(serde_json::Value::Array(word_arr)) = seg_obj.get("words") {
                        for w in word_arr {
                            if let serde_json::Value::Object(obj) = w {
                                let word_str =
                                    obj.get("word").and_then(|v| v.as_str()).unwrap_or("");
                                let start =
                                    obj.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                let end = obj.get("end").and_then(|v| v.as_f64()).unwrap_or(start);
                                if !word_str.is_empty() {
                                    words.push(WordItem {
                                        word: word_str.to_string(),
                                        start,
                                        end,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // If word-level timestamps exist, assign each word to the diarization turn whose interval it belongs to
    if !words.is_empty() && !diar.segments.is_empty() {
        let mut turn_words: Vec<Vec<String>> = vec![Vec::new(); diar.segments.len()];

        for w in &words {
            let mid = (w.start + w.end) / 2.0;
            let mut best_turn = None;
            let mut best_dist = f64::MAX;

            for (idx, seg) in diar.segments.iter().enumerate() {
                if mid >= seg.start && mid <= seg.end {
                    best_turn = Some(idx);
                    break;
                }
                let dist = if mid < seg.start {
                    seg.start - mid
                } else {
                    mid - seg.end
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_turn = Some(idx);
                }
            }

            if let Some(turn_idx) = best_turn {
                turn_words[turn_idx].push(w.word.clone());
            }
        }

        for (idx, seg) in diar.segments.iter_mut().enumerate() {
            if !turn_words[idx].is_empty() {
                let joined = turn_words[idx].join("");
                let trimmed = joined.trim();
                let clean = trimmed.trim_matches(|c: char| {
                    c.is_ascii_punctuation()
                        || c == '।'
                        || c == '?'
                        || c == '!'
                        || c.is_whitespace()
                });
                if !clean.is_empty() {
                    seg.text = Some(trimmed.to_string());
                }
            }
        }
        return;
    }

    // 3. Fallback: align ASR segments by maximum temporal overlap
    if let Some(serde_json::Value::Array(segs)) = &asr_resp.segments {
        let mut turn_texts: Vec<Vec<String>> = vec![Vec::new(); diar.segments.len()];

        for seg in segs {
            if let serde_json::Value::Object(seg_obj) = seg {
                let seg_text = seg_obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let seg_start = seg_obj.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let seg_end = seg_obj
                    .get("end")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(seg_start);

                if seg_text.trim().is_empty() {
                    continue;
                }

                let mut best_turn = None;
                let mut max_overlap = 0.0;

                for (idx, d) in diar.segments.iter().enumerate() {
                    let overlap = (seg_end.min(d.end) - seg_start.max(d.start)).max(0.0);
                    if overlap > max_overlap {
                        max_overlap = overlap;
                        best_turn = Some(idx);
                    }
                }

                if let Some(idx) = best_turn {
                    turn_texts[idx].push(seg_text.trim().to_string());
                }
            }
        }

        for (idx, seg) in diar.segments.iter_mut().enumerate() {
            if !turn_texts[idx].is_empty() {
                let joined = turn_texts[idx].join(" ");
                let trimmed = joined.trim();
                let clean = trimmed.trim_matches(|c: char| {
                    c.is_ascii_punctuation()
                        || c == '।'
                        || c == '?'
                        || c == '!'
                        || c.is_whitespace()
                });
                if !clean.is_empty() {
                    seg.text = Some(trimmed.to_string());
                }
            }
        }
    }
}

#[axum::debug_handler]
pub async fn audio_diarization_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let start_time = std::time::Instant::now();
    let mut file_bytes = Vec::new();
    let mut filename = "audio.wav".to_string();
    let mut model_id: Option<String> = None;
    let mut asr_model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut speaker_threshold: Option<f32> = None;
    let mut max_speakers: Option<usize> = None;

    while let Ok(Some(mut field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    filename = fname.to_string();
                }
                match read_limited_field_bytes(&mut field, MAX_AUDIO_UPLOAD_BYTES).await {
                    Ok(bytes) => file_bytes = bytes,
                    Err(err_resp) => return err_resp.into_response(),
                }
            }
            "model" => {
                if let Ok(text) = field.text().await {
                    model_id = Some(text.trim().to_string());
                }
            }
            "asr_model" => {
                if let Ok(text) = field.text().await {
                    let t = text.trim();
                    if !t.is_empty() {
                        asr_model = Some(t.to_string());
                    }
                }
            }
            "language" => {
                if let Ok(text) = field.text().await {
                    let t = text.trim();
                    if !t.is_empty() {
                        language = Some(t.to_string());
                    }
                }
            }
            "prompt" => {
                if let Ok(text) = field.text().await {
                    let t = text.trim();
                    if !t.is_empty() {
                        prompt = Some(t.to_string());
                    }
                }
            }
            "speaker_threshold" | "threshold" => {
                if let Ok(text) = field.text().await {
                    speaker_threshold = text.trim().parse().ok();
                }
            }
            "max_speakers" => {
                if let Ok(text) = field.text().await {
                    max_speakers = text.trim().parse().ok();
                }
            }
            _ => {}
        }
    }

    if file_bytes.len() > MAX_AUDIO_UPLOAD_BYTES {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Maximum audio file upload size is 25 MB. The uploaded file exceeds this limit.",
                )
                .with_type("invalid_request_error")
                .with_param("file")
                .with_code("file_too_large"),
            }),
        )
            .into_response();
    }

    if file_bytes.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'file' is a required multipart/form-data field.")
                    .with_type("invalid_request_error")
                    .with_param("file")
                    .with_code("missing_required_file"),
            }),
        )
            .into_response();
    }

    let target_model = match model_id {
        Some(ref m) if !m.is_empty() => m.clone(),
        _ => match state
            .config
            .models
            .iter()
            .find(|m| m.modality == crate::config::Modality::SpeechAnalysis)
        {
            Some(m) => m.id.clone(),
            None => "nemotron-3-diarization".to_string(),
        },
    };

    let inference = state.inference.clone();
    let model_to_fetch = target_model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %err,
                    "Failed to load speaker diarization engine"
                );
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
                tracing::error!(
                    target: "audit",
                    model = %target_model,
                    error = %e,
                    "Join error when retrieving speaker diarization engine"
                );
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let asr_audio_bytes = file_bytes.clone();
    let asr_filename = filename.clone();

    let task = InferenceTaskRequest::AudioDiarization {
        audio_bytes: file_bytes,
        filename,
        speaker_threshold,
        max_speakers,
    };

    let model_for_stats = target_model.clone();
    let res = tokio::task::spawn_blocking(move || {
        let gen_started = std::time::Instant::now();
        let r = engine.execute(&task, None).map_err(|e| e.to_string());
        (r, gen_started.elapsed())
    })
    .await;

    let (exec_res, gen_elapsed) = match res {
        Ok((r, elapsed)) => (r, elapsed),
        Err(e) => (
            Err(format!("Thread join error: {e}")),
            std::time::Duration::ZERO,
        ),
    };

    match exec_res {
        Ok(mut output) => {
            if let InferenceTaskResponse::Diarization(ref mut diarization) = output.response {
                let target_asr_model = asr_model.unwrap_or_else(|| {
                    state
                        .config
                        .models
                        .iter()
                        .find(|m| {
                            m.modality == crate::config::Modality::SpeechToText
                                && (m.id.contains("whisper")
                                    || m.name.to_lowercase().contains("whisper"))
                        })
                        .or_else(|| {
                            state
                                .config
                                .models
                                .iter()
                                .find(|m| m.modality == crate::config::Modality::SpeechToText)
                        })
                        .map(|m| m.id.clone())
                        .unwrap_or_else(|| "whisper-large-v3".to_string())
                });

                let inference_asr = state.inference.clone();
                let asr_task = InferenceTaskRequest::AudioTranscription {
                    audio_bytes: asr_audio_bytes,
                    filename: asr_filename,
                    prompt,
                    language,
                    temperature: None,
                    response_format: Some("verbose_json".to_string()),
                };

                let asr_res = tokio::task::spawn_blocking(move || {
                    let asr_engine = inference_asr.get_engine(&target_asr_model)?;
                    asr_engine
                        .execute(&asr_task, None)
                        .map_err(|e| e.to_string())
                })
                .await;

                if let Ok(Ok(asr_output)) = asr_res {
                    if let InferenceTaskResponse::Transcription(asr_resp) = asr_output.response {
                        attach_transcription_to_diarization(diarization, &asr_resp);
                    }
                }
            }

            crate::telemetry::record_completion(
                &model_for_stats,
                start_time.elapsed().as_millis(),
                output.usage.prompt_tokens,
                output.usage.completion_tokens,
                gen_elapsed.as_millis(),
            );
            match output.response {
                InferenceTaskResponse::Diarization(diarization) => {
                    Json(diarization).into_response()
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
                        error: ApiError::new("Unexpected response type from diarization engine"),
                    }),
                )
                    .into_response(),
            }
        }
        Err(e) => {
            crate::telemetry::record_failure(
                &model_for_stats,
                500,
                start_time.elapsed().as_millis(),
                &e,
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response()
        }
    }
}
