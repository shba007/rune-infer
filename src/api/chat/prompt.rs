use axum::Json;
use axum::http::StatusCode;
use base64::Engine;

use super::tools::build_system_prompt;
use crate::config::{ModelConfig, SamplingConfig};
use crate::inference::types::InferenceTaskRequest;
use crate::media::decode_media;
use crate::types::{ApiError, ChatCompletionRequest, ChatMessage, ErrorResponse, MediaItem};

pub struct PreparedPrompt {
    pub task: InferenceTaskRequest,
    pub is_structured_mode: bool,
    pub image_count: usize,
    pub video_count: usize,
    pub audio_count: usize,
}

pub fn prepare_prompt(
    request: &ChatCompletionRequest,
    model_config: &ModelConfig,
) -> Result<PreparedPrompt, (StatusCode, Json<ErrorResponse>)> {
    let mut has_video = false;
    let mut has_image = false;
    let mut has_audio = false;

    for msg in &request.messages {
        let (_, media) = msg.split_text_and_media();
        has_video |= media.iter().any(|m| matches!(m, MediaItem::Video(_)));
        has_image |= media.iter().any(|m| matches!(m, MediaItem::Image(_)));
        has_audio |= media.iter().any(|m| matches!(m, MediaItem::Audio { .. }));
    }

    let has_vision = has_image || has_video;
    let supports_vision = model_config.is_vision();
    if has_vision && !supports_vision {
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

    // A vision model is not an audio model: audio needs its own declared support.
    if has_audio && !model_config.supports_audio() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' does not support audio inputs.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("model_audio_unsupported"),
            }),
        ));
    }

    // llama.cpp refuses image/audio input without an mmproj file and answers with a bare 500.
    // Say what is actually wrong instead of forwarding the request to fail.
    if (has_vision || has_audio) && model_config.missing_projector() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "Model '{}' accepts image/audio input but has no usable vision projector. \
                     Set extra.mmproj_path in its catalog entry to an existing mmproj file.",
                    model_config.id
                ))
                .with_type("invalid_request_error")
                .with_code("model_projector_missing"),
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

    let sampling = model_config.effective_sampling(&SamplingConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        max_tokens: request.max_tokens,
        reasoning_effort: request.reasoning_effort.clone(),
    });

    let is_thinking_disabled = request.normalized_thinking()
        == crate::types::NormalizedThinking::Disabled
        || sampling.reasoning_effort.as_deref() == Some("none")
        || sampling.reasoning_effort.as_deref() == Some("disabled")
        || sampling.reasoning_effort.as_deref() == Some("off");

    let mut default_tool_system = build_system_prompt(
        is_structured_mode,
        &structured_schema,
        has_tools,
        &tools_value,
        has_video,
        request.tool_choice.as_ref(),
    );

    if is_thinking_disabled {
        let no_think = "Do not output any <think> tags or reasoning steps. Output only the final response directly.";
        if default_tool_system.is_empty() {
            default_tool_system = no_think.to_string();
        } else {
            default_tool_system.push_str("\n\n");
            default_tool_system.push_str(no_think);
        }
    }

    let max_dims = model_config.max_dimensions();
    let mut total_media_tokens = 0u32;
    let mut image_count = 0usize;
    let mut video_count = 0usize;
    let mut audio_count = 0usize;

    let mut prompt = String::new();
    let mut images = Vec::new();
    let mut audios = Vec::new();
    let mut augmented_messages = request.messages.clone();

    let has_system = request.messages.iter().any(|m| m.role == "system");
    if !has_system && (has_tools || is_structured_mode || has_video || is_thinking_disabled) {
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
                MediaItem::Audio { data, format } => {
                    audio_count += 1;
                    let fmt_lower = format.trim().to_ascii_lowercase();
                    if fmt_lower != "wav" && fmt_lower != "mp3" {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: ApiError::new(format!(
                                    "Unsupported audio format '{format}' in 'messages[{}].content'. Supported formats are 'wav' and 'mp3'.",
                                    m_idx
                                ))
                                .with_type("invalid_request_error")
                                .with_param("input_audio.format")
                                .with_code("unsupported_audio_format"),
                            }),
                        ));
                    }

                    let audio_bytes =
                        match base64::engine::general_purpose::STANDARD.decode(data.trim()) {
                            Ok(b) => b,
                            Err(_) => {
                                return Err((
                                    StatusCode::BAD_REQUEST,
                                    Json(ErrorResponse {
                                        error: ApiError::new(format!(
                                            "Invalid base64 audio data in 'messages[{}].content'.",
                                            m_idx
                                        ))
                                        .with_type("invalid_request_error")
                                        .with_param("input_audio.data")
                                        .with_code("invalid_audio_format"),
                                    }),
                                ));
                            }
                        };

                    if audio_bytes.len() > 50 * 1024 * 1024 {
                        return Err((
                            StatusCode::PAYLOAD_TOO_LARGE,
                            Json(ErrorResponse {
                                error: ApiError::new(
                                    "Audio payload exceeds maximum supported size of 50MB.",
                                )
                                .with_type("invalid_request_error")
                                .with_code("audio_payload_too_large"),
                            }),
                        ));
                    }

                    let est_audio_tokens = ((audio_bytes.len() / 640).clamp(64, 4096)) as u32;
                    total_media_tokens = total_media_tokens.saturating_add(est_audio_tokens);

                    audios.push(audio_bytes);
                    media_blocks.push_str(&format!(
                        "\nAudio {audio_count}:\n<|audio_start|><__media__><|audio_end|>\n"
                    ));
                }
            }
        }

        text.push_str(&media_blocks);
        prompt.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text.trim()));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let est_text_tokens = ((prompt.len() / 4).max(1)) as u32;
    let context_limit = model_config.effective_context_limit();
    // Without an explicit max_tokens, reserve room for the reply — but never
    // more than a quarter of the window, or small-context models would reject
    // every request from clients that omit max_tokens.
    let reserved_tokens = sampling
        .max_tokens
        .map(|t| t as u32)
        .unwrap_or_else(|| 16384.min(context_limit / 4));
    let total_required = est_text_tokens
        .saturating_add(total_media_tokens)
        .saturating_add(reserved_tokens);

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
            tool_choice: request.tool_choice.clone(),
            images,
            audios,
            messages: augmented_messages,
            max_tokens: sampling.max_tokens,
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            min_p: sampling.min_p,
            reasoning_effort: sampling.reasoning_effort,
            thinking: request.thinking.clone(),
            chat_template_kwargs: request.chat_template_kwargs.clone(),
        },
        is_structured_mode,
        image_count,
        video_count,
        audio_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ModelEntry;

    /// A local llama-server model. `mmproj` is spliced into `extra`.
    fn model(modality: &str, caps: &str, mmproj: Option<&str>) -> ModelConfig {
        let extra = mmproj
            .map(|p| format!(r#""mmproj_path": {p:?}"#))
            .unwrap_or_default();
        let entry: ModelEntry = serde_json::from_str(&format!(
            r#"{{"id":"m","name":"M","modality":"{modality}","capabilities":{caps},
                "engine":{{"supported":["llama-server"]}},
                "local":{{"path":"m.gguf"}},"extra":{{{extra}}}}}"#
        ))
        .unwrap();
        ModelConfig::try_from(entry).unwrap()
    }

    fn request(parts: &str) -> ChatCompletionRequest {
        let comma = if parts.is_empty() { "" } else { "," };
        serde_json::from_str(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":[
                {{"type":"text","text":"hi"}}{comma}{parts}]}}]}}"#
        ))
        .unwrap()
    }

    const IMAGE: &str = r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#;
    const AUDIO: &str = r#"{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}"#;
    /// Any file that exists, standing in for a projector.
    const EXISTING_FILE: &str = env!("CARGO_MANIFEST_DIR");

    fn rejected_with(
        r: Result<PreparedPrompt, (StatusCode, Json<ErrorResponse>)>,
    ) -> Option<String> {
        r.err().and_then(|(_, Json(e))| e.error.code)
    }

    #[test]
    fn a_vision_model_does_not_accept_audio() {
        // The bug this guards: `is_vision()` used to waive the audio check, so audio reached
        // llama-server, which answered "audio input is not supported" with a 500.
        let vision = model("VisionText", r#"["Chat","Vision"]"#, Some(EXISTING_FILE));
        assert_eq!(
            rejected_with(prepare_prompt(&request(AUDIO), &vision)).as_deref(),
            Some("model_audio_unsupported")
        );
    }

    #[test]
    fn omni_and_audio_capable_models_pass_the_audio_gate() {
        let omni = model("Omni", r#"["Chat","Vision","Audio"]"#, Some(EXISTING_FILE));
        let code = rejected_with(prepare_prompt(&request(AUDIO), &omni));
        assert!(
            !matches!(
                code.as_deref(),
                Some("model_audio_unsupported" | "model_projector_missing")
            ),
            "{code:?}"
        );
        let audio_chat = model("Text", r#"["Chat","Audio"]"#, Some(EXISTING_FILE));
        let code = rejected_with(prepare_prompt(&request(AUDIO), &audio_chat));
        assert!(
            !matches!(code.as_deref(), Some("model_audio_unsupported")),
            "{code:?}"
        );
    }

    #[test]
    fn text_models_reject_images_and_audio() {
        let text = model("Text", r#"["Chat"]"#, None);
        assert_eq!(
            rejected_with(prepare_prompt(&request(IMAGE), &text)).as_deref(),
            Some("model_vision_unsupported")
        );
        assert_eq!(
            rejected_with(prepare_prompt(&request(AUDIO), &text)).as_deref(),
            Some("model_audio_unsupported")
        );
    }

    #[test]
    fn a_missing_projector_is_reported_instead_of_forwarded() {
        for (label, mmproj) in [
            ("not configured", None),
            (
                "file does not exist",
                Some("Z:/definitely/missing-mmproj.gguf"),
            ),
        ] {
            let vision = model("VisionText", r#"["Chat","Vision"]"#, mmproj);
            let err = prepare_prompt(&request(IMAGE), &vision).err().expect(label);
            assert_eq!(err.0, StatusCode::BAD_REQUEST, "{label}");
            let Json(body) = err.1;
            assert_eq!(
                body.error.code.as_deref(),
                Some("model_projector_missing"),
                "{label}"
            );
            assert!(body.error.message.contains("extra.mmproj_path"), "{label}");
        }
        // With a projector present the request moves past this gate.
        let ok = model("VisionText", r#"["Chat","Vision"]"#, Some(EXISTING_FILE));
        let code = rejected_with(prepare_prompt(&request(IMAGE), &ok));
        assert!(
            !matches!(
                code.as_deref(),
                Some("model_projector_missing" | "model_vision_unsupported")
            ),
            "{code:?}"
        );
    }

    #[test]
    fn text_only_requests_never_need_a_projector() {
        let vision = model("VisionText", r#"["Chat","Vision"]"#, None);
        let code = rejected_with(prepare_prompt(&request(""), &vision));
        assert!(
            code.is_none() || code.as_deref() != Some("model_projector_missing"),
            "{code:?}"
        );
    }
}
