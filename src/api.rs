use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::{
    Json, Router,
    extract::{Query, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use serde::Deserialize;
use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::config::ModelRegistry;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceDelta, ErrorResponse,
    HealthResponse, ModelInfo, ModelsResponse, ResponseMessage, ToolCall, ToolCallChunk, Usage,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub config: ModelRegistry,
}

pub fn create_router(state: AppState) -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/inference", post(inference_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::new(state))
}

fn is_video_format(url_or_path: &str) -> bool {
    let lower = url_or_path.to_lowercase();
    lower.starts_with("data:video/")
        || lower.ends_with(".mp4")
        || lower.ends_with(".mkv")
        || lower.ends_with(".mov")
        || lower.ends_with(".webm")
        || lower.ends_with(".avi")
}

fn extract_video_frames(video_path: &Path) -> Result<Vec<Vec<u8>>, String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("rune_vid_{}_{}", std::process::id(), now));
    std::fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let out_pattern = temp_dir.join("frame_%04d.jpg");

    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-vf")
        .arg("fps=1,scale='min(768,iw)':-2")
        .arg("-vframes")
        .arg("16")
        .arg(&out_pattern)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => {
            let mut frames = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&temp_dir) {
                let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort();
                for p in paths {
                    if let Ok(bytes) = std::fs::read(&p) {
                        frames.push(bytes);
                    }
                }
            }
            let _ = std::fs::remove_dir_all(&temp_dir);
            if frames.is_empty() {
                Err("FFmpeg extracted 0 frames from video".to_string())
            } else {
                Ok(frames)
            }
        }
        Ok(s) => {
            let _ = std::fs::remove_dir_all(&temp_dir);
            Err(format!("FFmpeg failed with exit code: {s}"))
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&temp_dir);
            Err(format!("Failed to spawn ffmpeg: {e}. Is ffmpeg in PATH?"))
        }
    }
}

fn decode_media(url: &str) -> Result<Vec<Vec<u8>>, ErrorResponse> {
    if is_video_format(url) {
        if let Some(rest) = url.strip_prefix("data:") {
            let (_header, b64_data) = rest.split_once(',').ok_or_else(|| ErrorResponse {
                error: ApiError::new("Invalid video data URL: missing comma separator")
                    .with_type("invalid_request_error")
                    .with_code("invalid_video"),
            })?;

            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64_data.trim())
                .map_err(|e| ErrorResponse {
                    error: ApiError::new(&format!("Invalid base64 video payload: {e}"))
                        .with_type("invalid_request_error")
                        .with_code("invalid_video"),
                })?;

            let temp_video =
                std::env::temp_dir().join(format!("rune_tmp_{}.mp4", std::process::id()));
            std::fs::write(&temp_video, &bytes).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Failed to write temporary video: {e}")),
            })?;

            let frames = extract_video_frames(&temp_video).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Video processing error: {e}"))
                    .with_type("video_processing_error")
                    .with_code("ffmpeg_error"),
            });
            let _ = std::fs::remove_file(&temp_video);
            frames
        } else {
            let path = Path::new(url);
            if !path.exists() {
                return Err(ErrorResponse {
                    error: ApiError::new(&format!("Video file not found at '{}'", url))
                        .with_type("invalid_request_error")
                        .with_code("file_not_found"),
                });
            }
            extract_video_frames(path).map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Video processing error: {e}"))
                    .with_type("video_processing_error")
                    .with_code("ffmpeg_error"),
            })
        }
    } else if let Some(rest) = url.strip_prefix("data:") {
        let (header, b64_data) = rest.split_once(',').ok_or_else(|| ErrorResponse {
            error: ApiError::new("Invalid image data URL: missing comma separator")
                .with_type("invalid_request_error")
                .with_code("invalid_image"),
        })?;

        let header_lower = header.to_lowercase();
        if !header_lower.starts_with("image/") || !header_lower.contains(";base64") {
            return Err(ErrorResponse {
                error: ApiError::new("Invalid image data URL (expected `image/*;base64,<data>`)")
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            });
        }

        let clean_b64 = b64_data.trim();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(clean_b64)
            .map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Invalid base64 payload: {e}"))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            })?;

        Ok(vec![bytes])
    } else {
        match std::fs::read(url) {
            Ok(bytes) => Ok(vec![bytes]),
            Err(e) => Err(ErrorResponse {
                error: ApiError::new(&format!("Could not read file '{}': {e}", url))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            }),
        }
    }
}

#[axum::debug_handler]
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded_models = state.inference.loaded_models();
    Json(HealthResponse::new().with_loaded_models(loaded_models))
}

#[axum::debug_handler]
async fn models_handler(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let data = state
        .config
        .models
        .iter()
        .map(|m| ModelInfo::new(&m.id).with_ownership("Rune Infer".to_string()))
        .collect();
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

#[derive(Deserialize, Default)]
struct InferenceQuery {
    engine: Option<String>,
}

#[axum::debug_handler]
async fn inference_handler(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InferenceQuery>,
    Json(task): Json<InferenceTaskRequest>,
) -> Response {
    let engine_id = match q.engine.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(id) => id.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(InferenceTaskResponse::Error(
                    "Missing required query parameter: 'engine'".into(),
                )),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let res =
        tokio::task::spawn_blocking(move || inference.execute_task(&engine_id, &task, None)).await;

    match res {
        Ok(Ok(output)) => Json(output).into_response(),
        Ok(Err(e)) => {
            (StatusCode::NOT_FOUND, Json(InferenceTaskResponse::Error(e))).into_response()
        }
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(InferenceTaskResponse::Error(format!(
                "Task failed: {join_err}"
            ))),
        )
            .into_response(),
    }
}

fn parse_single_tool_call(val: &serde_json::Value, id: String) -> Option<ToolCall> {
    let obj = val.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let args = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id,
        r#type: "function".to_string(),
        function: crate::types::FunctionCall {
            name,
            arguments: args,
        },
    })
}

fn convert_to_tool_calls(value: &serde_json::Value, created: u64) -> Option<Vec<ToolCall>> {
    match value {
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            let mut calls = Vec::new();
            for (idx, item) in arr.iter().enumerate() {
                if let Some(call) =
                    parse_single_tool_call(item, format!("call_{}_{}", created, idx))
                {
                    calls.push(call);
                }
            }
            if calls.is_empty() { None } else { Some(calls) }
        }
        serde_json::Value::Object(_) => {
            parse_single_tool_call(value, format!("call_{}_0", created)).map(|c| vec![c])
        }
        _ => None,
    }
}

#[axum::debug_handler]
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.model.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing required field: 'model'")
                    .with_type("invalid_request_error")
                    .with_code("model_missing"),
            }),
        )
            .into_response();
    }

    let model_id = request.model.clone();
    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        Ok(Err(err)) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(&format!(
                        "Model '{}' not found or failed to load: {err}",
                        request.model
                    ))
                    .with_type("invalid_request_error")
                    .with_code("model_not_found"),
                }),
            )
                .into_response();
        }
        Err(join_err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(&format!("Thread execution error: {join_err}")),
                }),
            )
                .into_response();
        }
    };

    let tools_value = request.tools.clone().unwrap_or(serde_json::Value::Null);
    let has_tools = match &tools_value {
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
        _ => false,
    };
    let default_tool_system = if has_tools {
        let tools_json = serde_json::to_string_pretty(&tools_value).unwrap_or_default();
        format!(
            "You are a helpful assistant with access to tools.\n\
            When calling a tool, reply ONLY with a <tool_call> block containing a JSON array:\n\
            <tool_call>\n\
            [{{\"name\": \"tool_name\", \"arguments\": {{...}}}}]\n\
            </tool_call>\n\n\
            Available Tools:\n\
            {}",
            tools_json
        )
    } else {
        "You are a helpful assistant.".to_string()
    };

    let (prompt, images) = {
        let mut prompt = String::new();
        let mut images = Vec::new();

        let has_system = request.messages.iter().any(|m| m.role == "system");
        if !has_system {
            prompt.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n",
                default_tool_system
            ));
        }

        for msg in &request.messages {
            let (mut text, media_urls) = msg.split_text_and_images();
            let role = if msg.role.trim().is_empty() {
                "user"
            } else {
                msg.role.trim()
            };

            if role == "system" && has_tools {
                text = format!("{}\n\n{}", text, default_tool_system);
            }

            let mut decoded_in_msg = 0;
            for url in &media_urls {
                match decode_media(url) {
                    Ok(frames) => {
                        for frame in frames {
                            images.push(frame);
                            decoded_in_msg += 1;
                        }
                    }
                    Err(e) => return (StatusCode::BAD_REQUEST, Json(e)).into_response(),
                }
            }

            let existing_markers = text.matches("<__media__>").count();
            if existing_markers > 0 {
                let mut kept = 0;
                let mut cleaned = String::new();
                let parts: Vec<&str> = text.split("<__media__>").collect();
                for (i, part) in parts.iter().enumerate() {
                    cleaned.push_str(part);
                    if i < parts.len() - 1 {
                        if kept < decoded_in_msg {
                            cleaned.push_str("<__media__>");
                            kept += 1;
                        } else {
                            cleaned.push_str("[media]");
                        }
                    }
                }
                text = cleaned;
                while kept < decoded_in_msg {
                    text.push_str("\n<__media__>");
                    kept += 1;
                }
            } else {
                for _ in 0..decoded_in_msg {
                    text.push_str("\n<__media__>");
                }
            }

            prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text.trim()));
        }
        prompt.push_str("<|im_start|>assistant\n");
        (prompt, images)
    };

    let task = InferenceTaskRequest::ToolCall {
        prompt,
        schema: if has_tools {
            tools_value
        } else {
            serde_json::json!({})
        },
        images,
        messages: request.messages.clone(),
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let task_clone = task.clone();
        let engine_clone = engine.clone();
        let model_id = request.model.clone();

        tokio::task::spawn_blocking(move || {
            if has_tools {
                match engine_clone.execute(&task_clone, None) {
                    Ok(output) => match output.response {
                        InferenceTaskResponse::ToolCall(val) => {
                            if let Some(tool_calls) = convert_to_tool_calls(&val, created) {
                                let tool_chunks = tool_calls
                                    .into_iter()
                                    .enumerate()
                                    .map(|(idx, tc)| ToolCallChunk {
                                        index: idx,
                                        id: Some(tc.id),
                                        r#type: Some(tc.r#type),
                                        function: Some(crate::types::FunctionCallChunk {
                                            name: Some(tc.function.name),
                                            arguments: Some(tc.function.arguments),
                                        }),
                                    })
                                    .collect();

                                let chunk = ChatCompletionResponse {
                                    id: format!("chatcmpl-{created}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id.clone(),
                                    choices: vec![Choice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChoiceDelta {
                                            content: None,
                                            role: Some("assistant".to_string()),
                                            tool_calls: Some(tool_chunks),
                                        }),
                                        finish_reason: Some("tool_calls".to_string()),
                                    }],
                                    usage: output.usage,
                                };
                                let _ = tx.send(
                                    Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap_or_default()),
                                );
                            } else {
                                let chunk = ChatCompletionResponse {
                                    id: format!("chatcmpl-{created}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id.clone(),
                                    choices: vec![Choice {
                                        index: 0,
                                        message: None,
                                        delta: Some(ChoiceDelta {
                                            content: Some(val.to_string()),
                                            role: Some("assistant".to_string()),
                                            tool_calls: None,
                                        }),
                                        finish_reason: Some("stop".to_string()),
                                    }],
                                    usage: output.usage,
                                };
                                let _ = tx.send(
                                    Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap_or_default()),
                                );
                            }
                        }
                        InferenceTaskResponse::Text(text) => {
                            let chunk = ChatCompletionResponse {
                                id: format!("chatcmpl-{created}"),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id.clone(),
                                choices: vec![Choice {
                                    index: 0,
                                    message: None,
                                    delta: Some(ChoiceDelta {
                                        content: Some(text),
                                        role: Some("assistant".to_string()),
                                        tool_calls: None,
                                    }),
                                    finish_reason: Some("stop".to_string()),
                                }],
                                usage: output.usage,
                            };
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&chunk).unwrap_or_default()),
                            );
                        }
                        InferenceTaskResponse::Error(other) => {
                            eprintln!("[API] Unexpected error: {other}");
                        }
                    },
                    Err(e) => {
                        eprintln!("[API] Execution error: {e}");
                    }
                }
            } else {
                let tx_clone = tx.clone();
                let model_id_clone = model_id.clone();
                let mut completion_tokens = 0u32;
                let mut on_token = move |piece: &str| -> bool {
                    completion_tokens += 1;
                    let chunk = ChatCompletionResponse {
                        id: format!("chatcmpl-{created}"),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![Choice {
                            index: 0,
                            message: None,
                            delta: Some(ChoiceDelta {
                                content: Some(piece.to_string()),
                                role: Some("assistant".to_string()),
                                tool_calls: None,
                            }),
                            finish_reason: None,
                        }],
                        usage: Usage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                        },
                    };
                    tx_clone
                        .send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_ok()
                };

                match engine_clone.execute(&task_clone, Some(&mut on_token)) {
                    Ok(final_output) => {
                        let final_chunk = ChatCompletionResponse {
                            id: format!("chatcmpl-{created}"),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id.clone(),
                            choices: vec![Choice {
                                index: 0,
                                message: None,
                                delta: Some(ChoiceDelta {
                                    content: None,
                                    role: None,
                                    tool_calls: None,
                                }),
                                finish_reason: Some("stop".to_string()),
                            }],
                            usage: final_output.usage,
                        };
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                        );
                    }
                    Err(e) => {
                        eprintln!("[API] Engine execution error: {e}");
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));
        let sse_stream = stream.map(Ok::<_, Infallible>).chain(done_stream);
        return Sse::new(sse_stream).into_response();
    }

    let task_clone = task.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await;

    let response = match result {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::ToolCall(val) => {
                if let Some(tool_calls) = convert_to_tool_calls(&val, created) {
                    chat_completion_response(
                        &request.model,
                        None,
                        Some(tool_calls),
                        "tool_calls",
                        created,
                        output.usage,
                    )
                } else {
                    chat_completion_response(
                        &request.model,
                        Some(val.to_string()),
                        None,
                        "stop",
                        created,
                        output.usage,
                    )
                }
            }
            InferenceTaskResponse::Text(text) => chat_completion_response(
                &request.model,
                Some(text),
                None,
                "stop",
                created,
                output.usage,
            ),
            InferenceTaskResponse::Error(e) => chat_completion_response(
                &request.model,
                Some(e),
                None,
                "stop",
                created,
                output.usage,
            ),
        },
        Ok(Err(e)) => chat_completion_response(
            &request.model,
            Some(e),
            None,
            "stop",
            created,
            Usage::default(),
        ),
        Err(join_err) => chat_completion_response(
            &request.model,
            Some(format!("Thread execution error: {join_err}")),
            None,
            "stop",
            created,
            Usage::default(),
        ),
    };

    Json(response).into_response()
}

fn chat_completion_response(
    model: &str,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Some(ResponseMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            }),
            delta: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage,
    }
}
