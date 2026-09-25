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

fn decode_image(url: &str) -> Result<Vec<u8>, ErrorResponse> {
    if let Some(rest) = url.strip_prefix("data:") {
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

        Ok(bytes)
    } else {
        match std::fs::read(url) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(ErrorResponse {
                error: ApiError::new(&format!("Could not read image file '{}': {e}", url))
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
        Ok(Ok(response)) => Json(response).into_response(),
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
            let (mut text, img_urls) = msg.split_text_and_images();
            let role = if msg.role.trim().is_empty() {
                "user"
            } else {
                msg.role.trim()
            };

            if role == "system" && has_tools {
                text = format!("{}\n\n{}", text, default_tool_system);
            }

            let mut decoded_in_msg = 0;
            for url in &img_urls {
                match decode_image(url) {
                    Ok(bytes) => {
                        images.push(bytes);
                        decoded_in_msg += 1;
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
                    Ok(InferenceTaskResponse::ToolCall(val)) => {
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
                                usage: Usage {
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                    total_tokens: 0,
                                },
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
                                usage: Usage {
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                    total_tokens: 0,
                                },
                            };
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&chunk).unwrap_or_default()),
                            );
                        }
                    }
                    Ok(InferenceTaskResponse::Text(text)) => {
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
                            usage: Usage {
                                prompt_tokens: 0,
                                completion_tokens: 0,
                                total_tokens: 0,
                            },
                        };
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                    }
                    Ok(other) => {
                        eprintln!("[API] Unexpected response: {other:?}");
                    }
                    Err(e) => {
                        eprintln!("[API] Execution error: {e}");
                    }
                }
            } else {
                let tx_clone = tx.clone();
                let model_id_clone = model_id.clone();
                let mut on_token = move |piece: &str| -> bool {
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
                            completion_tokens: 0,
                            total_tokens: 0,
                        },
                    };
                    tx_clone
                        .send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_ok()
                };

                if let Err(e) = engine_clone.execute(&task_clone, Some(&mut on_token)) {
                    eprintln!("[API] Engine execution error: {e}");
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
        Ok(Ok(InferenceTaskResponse::ToolCall(val))) => {
            if let Some(tool_calls) = convert_to_tool_calls(&val, created) {
                chat_completion_response(
                    &request.model,
                    None,
                    Some(tool_calls),
                    "tool_calls",
                    created,
                )
            } else {
                chat_completion_response(
                    &request.model,
                    Some(val.to_string()),
                    None,
                    "stop",
                    created,
                )
            }
        }
        Ok(Ok(InferenceTaskResponse::Text(text))) => {
            chat_completion_response(&request.model, Some(text), None, "stop", created)
        }
        Ok(Ok(InferenceTaskResponse::Error(e))) => {
            chat_completion_response(&request.model, Some(e), None, "stop", created)
        }
        Ok(Err(e)) => chat_completion_response(&request.model, Some(e), None, "stop", created),
        Err(join_err) => chat_completion_response(
            &request.model,
            Some(format!("Thread execution error: {join_err}")),
            None,
            "stop",
            created,
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
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}
