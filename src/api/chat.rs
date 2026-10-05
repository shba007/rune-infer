use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceDelta, ErrorResponse,
    MediaItem, ResponseMessage, ToolCall, ToolCallChunk, Usage,
};

pub fn make_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: ChoiceDelta,
    finish_reason: Option<String>,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: Some("fp_rune_rust_v1".to_string()),
        choices: vec![Choice {
            index: 0,
            message: None,
            delta: Some(delta),
            logprobs: None,
            finish_reason,
        }],
        usage,
    }
}

pub fn chat_completion_response(
    model: &str,
    raw_content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionResponse {
    // Separate <think>...</think> from the actual message content
    let (reasoning_content, content) = match raw_content {
        Some(text) => {
            if let Some(start) = text.find("<think>") {
                if let Some(end) = text.find("</think>") {
                    let think_part = text[start + "<think>".len()..end].trim().to_string();
                    let rem = format!("{}{}", &text[..start], &text[end + "</think>".len()..])
                        .trim()
                        .to_string();
                    (
                        if think_part.is_empty() {
                            None
                        } else {
                            Some(think_part)
                        },
                        if rem.is_empty() { None } else { Some(rem) },
                    )
                } else {
                    (None, Some(text))
                }
            } else {
                (None, Some(text))
            }
        }
        None => (None, None),
    };

    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: Some("fp_rune_rust_v1".to_string()),
        choices: vec![Choice {
            index: 0,
            message: Some(ResponseMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content,
                tool_calls,
                refusal: None,
            }),
            delta: None,
            logprobs: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage,
    }
}

pub fn parse_single_tool_call(val: &serde_json::Value, fallback_id: String) -> Option<ToolCall> {
    let obj = val.as_object()?;

    if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
        let name = func.get("name")?.as_str()?.to_string();
        let args = match func.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
            None => "{}".to_string(),
        };
        let id = obj
            .get("id")
            .and_then(|i| i.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string())
            .unwrap_or(fallback_id);
        let r#type = obj
            .get("type")
            .and_then(|t| t.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("function")
            .to_string();

        return Some(ToolCall {
            id,
            r#type,
            function: crate::types::FunctionCall {
                name,
                arguments: args,
            },
        });
    }

    let name = obj.get("name")?.as_str()?.to_string();
    let args = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        None => "{}".to_string(),
    };
    let id = obj
        .get("id")
        .and_then(|i| i.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .unwrap_or(fallback_id);
    let r#type = obj
        .get("type")
        .and_then(|t| t.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("function")
        .to_string();

    Some(ToolCall {
        id,
        r#type,
        function: crate::types::FunctionCall {
            name,
            arguments: args,
        },
    })
}

pub fn convert_to_tool_calls(value: &serde_json::Value, created: u64) -> Option<Vec<ToolCall>> {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                convert_to_tool_calls(&parsed, created)
            } else {
                None
            }
        }
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
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    bytes: Bytes,
) -> Response {
    let start_time = std::time::Instant::now();

    let body_val: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Invalid JSON payload: {e}"))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

    match body_val.get("messages") {
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' field must be an array of message objects.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("invalid_type"),
                }),
            )
                .into_response();
        }
        Some(val) if !val.is_array() => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' field must be an array of message objects.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("invalid_type"),
                }),
            )
                .into_response();
        }
        Some(serde_json::Value::Array(arr)) if arr.is_empty() => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'messages' array cannot be empty.")
                        .with_type("invalid_request_error")
                        .with_param("messages")
                        .with_code("empty_array"),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    if let Some(serde_json::Value::Array(tools)) = body_val.get("tools") {
        for (idx, tool) in tools.iter().enumerate() {
            if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
                let name = tool
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str());
                if name.is_none() || name.unwrap().trim().is_empty() {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: ApiError::new(format!(
                                "'tools[{}].function.name' is a required string property.",
                                idx
                            ))
                            .with_type("invalid_request_error")
                            .with_param(format!("tools[{}].function.name", idx))
                            .with_code("missing_required_field"),
                        }),
                    )
                        .into_response();
                }
            }
        }
    }

    let request: ChatCompletionRequest = match serde_json::from_value(body_val) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Invalid request structure: {e}"))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

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

    let model_config = match state.config.find(&request.model) {
        Some(c) => c.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found", request.model))
                        .with_type("invalid_request_error")
                        .with_code("model_not_found"),
                }),
            )
                .into_response();
        }
    };

    // =========================================================================
    // Decoupled Dispatcher Branch: Remote Provider Proxy vs Local Execution
    // =========================================================================
    if !model_config.is_local() {
        return state.proxy.execute_chat(&model_config, &request).await;
    }

    // =========================================================================
    // Local Model Execution Hub (Unchanged)
    // =========================================================================
    let mut has_video = false;
    let mut has_media = false;

    for msg in &request.messages {
        let (_, media) = msg.split_text_and_media();
        if !media.is_empty() {
            has_media = true;
        }
        if media.iter().any(|m| matches!(m, MediaItem::Video(_))) {
            has_video = true;
        }
    }

    if has_media
        && !model_config.vision
        && model_config.modality != crate::config::Modality::VisionText
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support vision/media inputs.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("model_vision_unsupported"),
            }),
        )
            .into_response();
    }

    let is_structured_mode = match &request.response_format {
        Some(rf) => rf.r#type == "json_object" || rf.r#type == "json_schema",
        None => false,
    };

    if let Some(ref rf) = request.response_format {
        if rf.r#type == "json_schema" {
            if let Some(ref js) = rf.json_schema {
                if js.strict.unwrap_or(false) {
                    if let Some(ref schema) = js.schema {
                        let add_props = schema.get("additionalProperties");
                        if add_props != Some(&serde_json::Value::Bool(false)) {
                            return (
                                StatusCode::UNPROCESSABLE_ENTITY,
                                Json(ErrorResponse {
                                    error: ApiError::new(
                                        "When 'strict' is set to true in response_format, 'schema.additionalProperties' must be explicitly set to false.",
                                    )
                                    .with_type("invalid_request_error")
                                    .with_param("response_format.json_schema.schema.additionalProperties")
                                    .with_code("invalid_json_schema"),
                                }),
                            )
                                .into_response();
                        }
                    }
                }
            }
        }
    }

    if is_structured_mode && !model_config.supports_structured_output() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support structured JSON mode.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("unsupported_response_format"),
            }),
        )
            .into_response();
    }

    let model_id = request.model.clone();
    let inference = state.inference.clone();
    let model_target = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_target)).await {
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

    let structured_schema = if is_structured_mode {
        request
            .response_format
            .as_ref()
            .and_then(|rf| rf.json_schema.as_ref())
            .and_then(|js| js.schema.clone())
            .unwrap_or_else(|| serde_json::json!({ "type": "object" }))
    } else {
        serde_json::json!({})
    };

    let default_tool_system = if is_structured_mode {
        format!(
            "You are a helpful assistant.\nYou must respond ONLY with a valid JSON object matching this schema:\n{}",
            serde_json::to_string_pretty(&structured_schema).unwrap_or_default()
        )
    } else if has_tools {
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
    } else if has_video {
        "You are a helpful assistant with native video understanding capabilities. Analyze the video sequence directly, observing temporal motion, continuity, actions, and timestamps.".to_string()
    } else {
        "You are a helpful assistant.".to_string()
    };

    let max_dims = model_config.max_dimensions();
    let mut total_media_tokens = 0u32;
    let mut image_count = 0usize;
    let mut video_count = 0usize;

    let (prompt, images, augmented_messages) = {
        let mut prompt = String::new();
        let mut images = Vec::new();
        let mut augmented_messages = request.messages.clone();

        let has_system = request.messages.iter().any(|m| m.role == "system");
        if !has_system && (has_tools || is_structured_mode || has_video) {
            prompt.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n",
                default_tool_system
            ));
            augmented_messages.insert(
                0,
                crate::types::ChatMessage {
                    role: "system".to_string(),
                    content: Some(crate::types::MessageContent::Text(
                        default_tool_system.clone(),
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            );
        }

        for (m_idx, msg) in request.messages.iter().enumerate() {
            let (mut text, media_items) = msg.split_text_and_media();
            let role = if msg.role.trim().is_empty() {
                "user"
            } else {
                msg.role.trim()
            };

            if role == "system" && (has_tools || is_structured_mode) {
                if let Some(sys_msg) = augmented_messages.iter_mut().find(|m| m.role == "system") {
                    let current_text = sys_msg.text_content();
                    sys_msg.content = Some(crate::types::MessageContent::Text(format!(
                        "{}\n\n{}",
                        current_text, default_tool_system
                    )));
                }
            }

            if role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    if !calls.is_empty() {
                        let calls_json = serde_json::to_string(calls).unwrap_or_default();
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&format!("<tool_call>\n{}\n</tool_call>", calls_json));
                    }
                }
            } else if role == "tool" {
                let call_id = msg.tool_call_id.as_deref().unwrap_or("");
                text = format!(
                    "<tool_response id=\"{}\">\n{}\n</tool_response>",
                    call_id, text
                );
            }

            let mut media_blocks = String::new();
            for item in &media_items {
                match item {
                    MediaItem::Video(url) => {
                        video_count += 1;
                        match decode_media(url, max_dims) {
                            Ok(frames) => {
                                let (w, h) =
                                    frames.first().map(|f| (f.1, f.2)).unwrap_or((768, 768));
                                let patch_w = (w + 27) / 28;
                                let patch_h = (h + 27) / 28;
                                let tubelet_slices = ((frames.len() + 1) / 2) as u32;
                                total_media_tokens = total_media_tokens
                                    .saturating_add((tubelet_slices * patch_w * patch_h) + 32);

                                media_blocks
                                    .push_str(&format!("\nVideo {video_count}:\n<|video_start|>"));
                                for (frame_bytes, _, _) in frames {
                                    images.push(frame_bytes);
                                    media_blocks.push_str("<__media__>");
                                }
                                media_blocks.push_str("<|video_end|>\n");
                            }
                            Err(_) => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(ErrorResponse {
                                        error: ApiError::new(format!(
                                            "Invalid video format in 'messages[{}].content'.",
                                            m_idx
                                        ))
                                        .with_type("invalid_request_error")
                                        .with_param("video_url.url")
                                        .with_code("invalid_video_format"),
                                    }),
                                )
                                    .into_response();
                            }
                        }
                    }
                    MediaItem::Image(url) => {
                        image_count += 1;
                        match decode_media(url, max_dims) {
                            Ok(frames) => {
                                for (frame_bytes, w, h) in frames {
                                    let patch_w = (w + 27) / 28;
                                    let patch_h = (h + 27) / 28;
                                    total_media_tokens =
                                        total_media_tokens.saturating_add((patch_w * patch_h) + 32);

                                    images.push(frame_bytes);
                                    media_blocks.push_str(&format!(
                                        "\nPicture {image_count}:\n<__media__>\n"
                                    ));
                                }
                            }
                            Err(_) => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(ErrorResponse {
                                        error: ApiError::new(format!(
                                            "Invalid base64 image data URI format in 'messages[{}].content[1].image_url.url'.",
                                            m_idx
                                        ))
                                        .with_type("invalid_request_error")
                                        .with_param("image_url.url")
                                        .with_code("invalid_image_format"),
                                    }),
                                )
                                    .into_response();
                            }
                        }
                    }
                }
            }

            text.push_str(&media_blocks);
            prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text.trim()));
        }
        prompt.push_str("<|im_start|>assistant\n");
        (prompt, images, augmented_messages)
    };

    let est_text_tokens = ((prompt.len() / 4).max(1)) as u32;
    let reserved_tokens = request.max_tokens.unwrap_or(16384) as u32;
    let total_required = est_text_tokens
        .saturating_add(total_media_tokens)
        .saturating_add(reserved_tokens);
    let context_limit = model_config.effective_context_limit();

    if total_required > context_limit {
        let err_msg = format!(
            "Request token budget estimated at ~{} tokens exceeds model '{}' limit of {} tokens.",
            total_required, model_config.id, context_limit
        );
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(err_msg)
                    .with_type("invalid_request_error")
                    .with_code("context_length_exceeded"),
            }),
        )
            .into_response();
    }

    let task_schema = if is_structured_mode {
        structured_schema
    } else if has_tools {
        tools_value
    } else {
        serde_json::json!({})
    };

    let task = InferenceTaskRequest::ToolCall {
        prompt,
        schema: task_schema,
        images,
        messages: augmented_messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
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

        let initial_chunk = make_chunk(
            &format!("chatcmpl-{created}"),
            created,
            &model_id,
            ChoiceDelta {
                content: Some(String::new()),
                reasoning_content: None,
                role: Some("assistant".to_string()),
                tool_calls: None,
            },
            None,
            Usage::default(),
        );
        let _ = tx
            .send(Event::default().data(serde_json::to_string(&initial_chunk).unwrap_or_default()));

        tokio::task::spawn_blocking(move || {
            let tx_clone = tx.clone();
            let model_id_clone = model_id.clone();
            let mut completion_tokens = 0u32;
            let has_streamed_content =
                std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let has_streamed_clone = has_streamed_content.clone();
            let mut is_thinking = false;
            let mut tool_call_buffering = false;

            let mut on_token = move |piece: &str| -> bool {
                completion_tokens += 1;

                // 1. Detect start of thinking
                if piece.contains("<think>") {
                    is_thinking = true;
                    let clean = piece
                        .replace("<think>", "")
                        .trim_start_matches('\n')
                        .to_string();
                    if !clean.is_empty() {
                        let chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id_clone,
                            ChoiceDelta {
                                content: None,
                                reasoning_content: Some(clean),
                                role: None,
                                tool_calls: None,
                            },
                            None,
                            Usage {
                                prompt_tokens: 0,
                                completion_tokens,
                                total_tokens: completion_tokens,
                            },
                        );
                        let _ = tx_clone.send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                    }
                    return true;
                }

                // 2. Detect end of thinking
                if piece.contains("</think>") {
                    is_thinking = false;
                    let clean = piece.replace("</think>", "").trim_matches('\n').to_string();
                    if !clean.is_empty() {
                        let chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id_clone,
                            ChoiceDelta {
                                content: Some(clean),
                                reasoning_content: None,
                                role: None,
                                tool_calls: None,
                            },
                            None,
                            Usage {
                                prompt_tokens: 0,
                                completion_tokens,
                                total_tokens: completion_tokens,
                            },
                        );
                        let _ = tx_clone.send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                    }
                    return true;
                }

                // 3. During thinking: deliver strictly to reasoning_content
                if is_thinking {
                    let chunk = make_chunk(
                        &format!("chatcmpl-{created}"),
                        created,
                        &model_id_clone,
                        ChoiceDelta {
                            content: None,
                            reasoning_content: Some(piece.to_string()),
                            role: None,
                            tool_calls: None,
                        },
                        None,
                        Usage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                        },
                    );
                    return tx_clone
                        .send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_ok();
                }

                // 4. Outside thinking: ignore raw tool call syntax if tool buffering
                if piece.contains("<tool_call>") || tool_call_buffering {
                    tool_call_buffering = true;
                    return true;
                }

                // 5. Final answer generation: deliver strictly to content
                has_streamed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                let chunk = make_chunk(
                    &format!("chatcmpl-{created}"),
                    created,
                    &model_id_clone,
                    ChoiceDelta {
                        content: Some(piece.to_string()),
                        reasoning_content: None,
                        role: None,
                        tool_calls: None,
                    },
                    None,
                    Usage {
                        prompt_tokens: 0,
                        completion_tokens,
                        total_tokens: completion_tokens,
                    },
                );
                tx_clone
                    .send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
                    .is_ok()
            };

            match engine_clone.execute(&task_clone, Some(&mut on_token)) {
                Ok(output) => match output.response {
                    InferenceTaskResponse::ToolCall {
                        content,
                        tool_calls,
                    } => {
                        if let Some(parsed_calls) = convert_to_tool_calls(&tool_calls, created) {
                            // Only emit text_chunk if content was NOT already streamed chunk-by-chunk by on_token
                            if !has_streamed_content.load(std::sync::atomic::Ordering::Relaxed) {
                                if let Some(ref text) = content {
                                    if !text.is_empty() {
                                        let text_chunk = make_chunk(
                                            &format!("chatcmpl-{created}"),
                                            created,
                                            &model_id,
                                            ChoiceDelta {
                                                content: Some(text.clone()),
                                                reasoning_content: None,
                                                role: None,
                                                tool_calls: None,
                                            },
                                            None,
                                            output.usage.clone(),
                                        );
                                        let _ = tx.send(Event::default().data(
                                            serde_json::to_string(&text_chunk).unwrap_or_default(),
                                        ));
                                    }
                                }
                            }

                            let tool_chunks = parsed_calls
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

                            let tool_chunk = make_chunk(
                                &format!("chatcmpl-{created}"),
                                created,
                                &model_id,
                                ChoiceDelta {
                                    content: None,
                                    reasoning_content: None,
                                    role: None,
                                    tool_calls: Some(tool_chunks),
                                },
                                None,
                                output.usage.clone(),
                            );
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&tool_chunk).unwrap_or_default()),
                            );

                            let final_chunk = make_chunk(
                                &format!("chatcmpl-{created}"),
                                created,
                                &model_id,
                                ChoiceDelta {
                                    content: None,
                                    reasoning_content: None,
                                    role: None,
                                    tool_calls: None,
                                },
                                Some("tool_calls".to_string()),
                                output.usage,
                            );
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                            );
                        } else {
                            let chunk = make_chunk(
                                &format!("chatcmpl-{created}"),
                                created,
                                &model_id,
                                ChoiceDelta {
                                    content: Some(tool_calls.to_string()),
                                    reasoning_content: None,
                                    role: None,
                                    tool_calls: None,
                                },
                                Some("stop".to_string()),
                                output.usage,
                            );
                            let _ = tx.send(
                                Event::default()
                                    .data(serde_json::to_string(&chunk).unwrap_or_default()),
                            );
                        }
                    }
                    InferenceTaskResponse::Text(_text) => {
                        let final_chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id,
                            ChoiceDelta {
                                content: None,
                                reasoning_content: None,
                                role: None,
                                tool_calls: None,
                            },
                            Some("stop".to_string()),
                            output.usage,
                        );
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                        );
                    }
                    InferenceTaskResponse::Error(other) => {
                        let err_chunk = serde_json::json!({
                            "error": {
                                "message": other,
                                "type": "server_error",
                                "code": "inference_error"
                            }
                        });
                        let _ =
                            tx.send(Event::default().event("error").data(err_chunk.to_string()));
                    }
                    _ => {}
                },
                Err(e) => {
                    let err_chunk = serde_json::json!({
                        "error": {
                            "message": e.to_string(),
                            "type": "server_error",
                            "code": "engine_execution_failed"
                        }
                    });
                    let _ = tx.send(Event::default().event("error").data(err_chunk.to_string()));
                }
            }
        });

        tracing::info!(
            target: "audit",
            status = 200,
            latency_ms = start_time.elapsed().as_millis(),
            model = %request.model,
            media = %format!("images: {image_count}, videos: {video_count}"),
            "SSE stream started"
        );

        let stream = UnboundedReceiverStream::new(rx);
        let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));
        let sse_stream = stream.map(Ok::<_, Infallible>).chain(done_stream);
        let mut resp = Sse::new(sse_stream)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response();
        if let Some(heads) = model_config.mtp_heads() {
            if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                resp.headers_mut().insert("x-mtp-heads", val);
            }
        }
        return resp;
    }

    let task_clone = task.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::ToolCall {
                content,
                tool_calls,
            } => {
                let json_resp = if is_structured_mode {
                    let content_str = match &tool_calls {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string_pretty(other).unwrap_or_default(),
                    };
                    let resp = chat_completion_response(
                        &request.model,
                        Some(content_str),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else if let Some(parsed_calls) = convert_to_tool_calls(&tool_calls, created) {
                    let resp = chat_completion_response(
                        &request.model,
                        content,
                        Some(parsed_calls),
                        "tool_calls",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                } else {
                    let resp = chat_completion_response(
                        &request.model,
                        Some(tool_calls.to_string()),
                        None,
                        "stop",
                        created,
                        output.usage,
                    );
                    Json(resp).into_response()
                };

                let mut final_resp = json_resp;
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Text(text) => {
                let resp = chat_completion_response(
                    &request.model,
                    Some(text),
                    None,
                    "stop",
                    created,
                    output.usage,
                );
                let mut final_resp = Json(resp).into_response();
                if let Some(heads) = model_config.mtp_heads() {
                    if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
                        final_resp.headers_mut().insert("x-mtp-heads", val);
                    }
                }
                final_resp
            }
            InferenceTaskResponse::Error(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e)
                        .with_type("server_error")
                        .with_code("inference_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type from text inference")
                        .with_type("server_error"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => {
            let status = if e.contains("context") || e.contains("tokens") || e.contains("limit") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(ErrorResponse {
                    error: ApiError::new(e)
                        .with_type("invalid_request_error")
                        .with_code("inference_failure"),
                }),
            )
                .into_response()
        }
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(format!("Thread execution failure: {join_err}"))
                    .with_type("server_error")
                    .with_code("internal_error"),
            }),
        )
            .into_response(),
    }
}
