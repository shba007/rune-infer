use axum::Json;
use axum::http::StatusCode;

use super::tools::build_system_prompt;
use crate::config::ModelConfig;
use crate::inference::types::InferenceTaskRequest;
use crate::media::decode_media;
use crate::types::{ApiError, ChatCompletionRequest, ChatMessage, ErrorResponse, MediaItem};

pub struct PreparedPrompt {
    pub task: InferenceTaskRequest,
    pub is_structured_mode: bool,
    pub image_count: usize,
    pub video_count: usize,
}

pub fn prepare_prompt(
    request: &ChatCompletionRequest,
    model_config: &ModelConfig,
) -> Result<PreparedPrompt, (StatusCode, Json<ErrorResponse>)> {
    let mut has_video = false;
    let mut has_media = false;

    for msg in &request.messages {
        let (_, media) = msg.split_text_and_media();
        has_media |= !media.is_empty();
        has_video |= media.iter().any(|m| matches!(m, MediaItem::Video(_)));
    }

    let supports_vision =
        model_config.vision || model_config.modality == crate::config::Modality::VisionText;
    if has_media && !supports_vision {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support vision/media inputs.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("model_vision_unsupported"),
            }),
        ));
    }

    let is_structured_mode = match &request.response_format {
        Some(rf) => rf.r#type == "json_object" || rf.r#type == "json_schema",
        None => false,
    };

    if let Some(ref rf) = request.response_format {
        let invalid_strict_schema = rf.r#type == "json_schema"
            && rf.json_schema.as_ref().is_some_and(|js| {
                js.strict.unwrap_or(false)
                    && js.schema.as_ref().is_some_and(|schema| {
                        schema.get("additionalProperties") != Some(&serde_json::Value::Bool(false))
                    })
            });

        if invalid_strict_schema {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "When 'strict' is set to true in response_format, 'schema.additionalProperties' must be explicitly set to false.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("response_format.json_schema.schema.additionalProperties")
                    .with_code("invalid_json_schema"),
                }),
            ));
        }
    }

    if is_structured_mode && !model_config.supports_structured_output() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support structured JSON mode.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("unsupported_response_format"),
            }),
        ));
    }

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

    let default_tool_system = build_system_prompt(
        is_structured_mode,
        &structured_schema,
        has_tools,
        &tools_value,
        has_video,
    );

    let max_dims = model_config.max_dimensions();
    let mut total_media_tokens = 0u32;
    let mut image_count = 0usize;
    let mut video_count = 0usize;

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
            ChatMessage {
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
                            let (w, h) = frames.first().map(|f| (f.1, f.2)).unwrap_or((768, 768));
                            let patch_w = w.div_ceil(28);
                            let patch_h = h.div_ceil(28);
                            let tubelet_slices = frames.len().div_ceil(2) as u32;
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
                            return Err((
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
                            ));
                        }
                    }
                }
                MediaItem::Image(url) => {
                    image_count += 1;
                    match decode_media(url, max_dims) {
                        Ok(frames) => {
                            for (frame_bytes, w, h) in frames {
                                let patch_w = w.div_ceil(28);
                                let patch_h = h.div_ceil(28);
                                total_media_tokens =
                                    total_media_tokens.saturating_add((patch_w * patch_h) + 32);
                                images.push(frame_bytes);
                                media_blocks
                                    .push_str(&format!("\nPicture {image_count}:\n<__media__>\n"));
                            }
                        }
                        Err(_) => {
                            return Err((
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
                            ));
                        }
                    }
                }
            }
        }

        text.push_str(&media_blocks);
        prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text.trim()));
    }
    prompt.push_str("<|im_start|>assistant\n");

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
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: ApiError::new(err_msg)
                    .with_type("invalid_request_error")
                    .with_code("context_length_exceeded"),
            }),
        ));
    }

    let task_schema = if is_structured_mode {
        structured_schema
    } else if has_tools {
        tools_value
    } else {
        serde_json::json!({})
    };

    Ok(PreparedPrompt {
        task: InferenceTaskRequest::ToolCall {
            prompt,
            schema: task_schema,
            images,
            messages: augmented_messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        },
        is_structured_mode,
        image_count,
        video_count,
    })
}
