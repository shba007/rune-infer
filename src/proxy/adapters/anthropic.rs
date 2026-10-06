use axum::Json;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt;

use crate::api::chat::{chat_completion_response, make_chunk};
use crate::config::ModelConfig;
use crate::proxy::adapters::{BoxEventStream, parse_data_uri, parse_sse_stream};
use crate::types::{
    ApiError, ChatCompletionRequest, ChoiceDelta, FunctionCall, ToolCall, ToolCallChunk, Usage,
};

pub async fn execute_chat(
    client: &reqwest::Client,
    config: &ModelConfig,
    request: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let base_url = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.anthropic.com/v1");
    let endpoint = format!("{}/messages", base_url.trim_end_matches('/'));

    let api_key = config.api_key.as_deref().unwrap_or("");
    let upstream_model = config.upstream_model.as_deref().unwrap_or(&request.model);

    let mut system_text = String::new();
    let mut anthropic_messages = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            let text = msg.text_content();
            if !text.is_empty() {
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&text);
            }
            continue;
        }

        let role = match msg.role.as_str() {
            "assistant" => "assistant",
            _ => "user",
        };

        let mut content_parts = Vec::new();

        if msg.role == "tool" {
            let tool_id = msg.tool_call_id.clone().unwrap_or_default();
            content_parts.push(serde_json::json!({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": msg.text_content()
            }));
            anthropic_messages.push(serde_json::json!({
                "role": "user",
                "content": content_parts
            }));
            continue;
        }

        if role == "assistant" {
            let text = msg.text_content();
            if !text.is_empty() {
                content_parts.push(serde_json::json!({
                    "type": "text",
                    "text": text
                }));
            }
            if let Some(ref calls) = msg.tool_calls {
                for call in calls {
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| serde_json::json!({}));
                    content_parts.push(serde_json::json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": parsed_args
                    }));
                }
            }
            anthropic_messages.push(serde_json::json!({
                "role": "assistant",
                "content": content_parts
            }));
            continue;
        }

        let (text, media_items) = msg.split_text_and_media();
        if !text.is_empty() {
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": text
            }));
        }

        for item in media_items {
            let url = match item {
                crate::types::MediaItem::Image(u) => u,
                crate::types::MediaItem::Video(u) => u,
            };

            if let Some((mime, b64)) = parse_data_uri(&url) {
                content_parts.push(serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64
                    }
                }));
            } else if url.starts_with("http://") || url.starts_with("https://") {
                content_parts.push(serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url
                    }
                }));
            }
        }

        anthropic_messages.push(serde_json::json!({
            "role": "user",
            "content": content_parts
        }));
    }

    let mut anthropic_tools = Vec::new();
    if let Some(serde_json::Value::Array(tools)) = &request.tools {
        for t in tools {
            if t.get("type").and_then(|v| v.as_str()) == Some("function") {
                if let Some(func) = t.get("function") {
                    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let desc = func
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let schema = func
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

                    anthropic_tools.push(serde_json::json!({
                        "name": name,
                        "description": desc,
                        "input_schema": schema
                    }));
                }
            }
        }
    }

    let mut body = serde_json::json!({
        "model": upstream_model,
        "messages": anthropic_messages,
        "max_tokens": request.max_tokens.unwrap_or(4096),
        "stream": request.stream
    });

    // Map Structured Output JSON Schema into Anthropic system prompt
    if let Some(ref rf) = request.response_format {
        if rf.r#type == "json_object" {
            let instruction = "\nYou must output ONLY a valid JSON object without any markdown formatting, backticks, or commentary.";
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str(instruction);
        } else if rf.r#type == "json_schema" {
            if let Some(ref js) = rf.json_schema {
                if let Some(ref schema) = js.schema {
                    let instruction = format!(
                        "You must output ONLY a valid JSON object strictly matching this schema without any markdown formatting, backticks, or commentary:\n{}",
                        serde_json::to_string_pretty(schema).unwrap_or_default()
                    );
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(&instruction);
                }
            }
        }
    }

    if !system_text.is_empty() {
        body["system"] = serde_json::Value::String(system_text);
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(p) = request.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if !anthropic_tools.is_empty() {
        body["tools"] = serde_json::Value::Array(anthropic_tools);
    }

    let resp = client
        .post(&endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ApiError::new(format!("Failed to connect to Anthropic endpoint: {e}"))
                .with_type("api_error")
        })?;

    let status = resp.status();
    if !status.is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(
            ApiError::new(format!("Anthropic returned HTTP {status}: {err_text}"))
                .with_type("upstream_error")
                .with_code(status.as_u16().to_string()),
        );
    }

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let client_model = request.model.clone();

    if request.stream {
        let byte_stream = resp.bytes_stream();
        let sse_stream: BoxEventStream = Box::pin(async_stream::stream! {
            let data_stream = parse_sse_stream(Box::pin(byte_stream));
            tokio::pin!(data_stream);

            let chunk_id = format!("chatcmpl-{created}");

            let initial_chunk = make_chunk(
                &chunk_id,
                created,
                &client_model,
                ChoiceDelta {
                    content: Some(String::new()),
                    reasoning_content: None,
                    role: Some("assistant".to_string()),
                    tool_calls: None,
                },
                None,
                Usage::default(),
            );
            yield Ok(Event::default().data(serde_json::to_string(&initial_chunk).unwrap_or_default()));

            let mut tool_idx = 0usize;

            while let Some(line) = data_stream.next().await {
                if let Ok(event_val) = serde_json::from_str::<serde_json::Value>(&line) {
                    let event_type = event_val.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    match event_type {
                        "content_block_start" => {
                            if let Some(cb) = event_val.get("content_block") {
                                if cb.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                                    let id = cb.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let name = cb.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();

                                    let chunk = make_chunk(
                                        &chunk_id,
                                        created,
                                        &client_model,
                                        ChoiceDelta {
                                            content: None,
                                            reasoning_content: None,
                                            role: None,
                                            tool_calls: Some(vec![ToolCallChunk {
                                                index: tool_idx,
                                                id: Some(id),
                                                r#type: Some("function".to_string()),
                                                function: Some(crate::types::FunctionCallChunk {
                                                    name: Some(name),
                                                    arguments: Some(String::new()),
                                                }),
                                            }]),
                                        },
                                        None,
                                        Usage::default(),
                                    );
                                    tool_idx += 1;
                                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                                }
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = event_val.get("delta") {
                                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                match delta_type {
                                    "text_delta" => {
                                        let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                        let chunk = make_chunk(
                                            &chunk_id,
                                            created,
                                            &client_model,
                                            ChoiceDelta {
                                                content: Some(text.to_string()),
                                                reasoning_content: None,
                                                role: None,
                                                tool_calls: None,
                                            },
                                            None,
                                            Usage::default(),
                                        );
                                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                                    }
                                    "thinking_delta" => {
                                        let thinking = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                        let chunk = make_chunk(
                                            &chunk_id,
                                            created,
                                            &client_model,
                                            ChoiceDelta {
                                                content: None,
                                                reasoning_content: Some(thinking.to_string()),
                                                role: None,
                                                tool_calls: None,
                                            },
                                            None,
                                            Usage::default(),
                                        );
                                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                                    }
                                    "input_json_delta" => {
                                        let partial_json = delta.get("partial_json").and_then(|v| v.as_str()).unwrap_or("");
                                        let current_idx = tool_idx.saturating_sub(1);
                                        let chunk = make_chunk(
                                            &chunk_id,
                                            created,
                                            &client_model,
                                            ChoiceDelta {
                                                content: None,
                                                reasoning_content: None,
                                                role: None,
                                                tool_calls: Some(vec![ToolCallChunk {
                                                    index: current_idx,
                                                    id: None,
                                                    r#type: None,
                                                    function: Some(crate::types::FunctionCallChunk {
                                                        name: None,
                                                        arguments: Some(partial_json.to_string()),
                                                    }),
                                                }]),
                                            },
                                            None,
                                            Usage::default(),
                                        );
                                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "message_delta" => {
                            let stop_reason = event_val
                                .get("delta")
                                .and_then(|d| d.get("stop_reason"))
                                .and_then(|r| r.as_str())
                                .unwrap_or("stop");

                            let finish_reason = match stop_reason {
                                "tool_use" => "tool_calls",
                                "max_tokens" => "length",
                                _ => "stop",
                            };

                            let out_tokens = event_val
                                .get("usage")
                                .and_then(|u| u.get("output_tokens"))
                                .and_then(|t| t.as_u64())
                                .unwrap_or(0) as u32;

                            let chunk = make_chunk(
                                &chunk_id,
                                created,
                                &client_model,
                                ChoiceDelta {
                                    content: None,
                                    reasoning_content: None,
                                    role: None,
                                    tool_calls: None,
                                },
                                Some(finish_reason.to_string()),
                                Usage::new(0, out_tokens),
                            );
                            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                        }
                        "message_stop" => {
                            yield Ok(Event::default().data("[DONE]"));
                        }
                        _ => {}
                    }
                }
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response())
    } else {
        let val: serde_json::Value = resp.json().await.map_err(|e| {
            ApiError::new(format!("Failed to parse Anthropic JSON response: {e}"))
                .with_type("api_error")
        })?;

        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        if let Some(contents) = val.get("content").and_then(|c| c.as_array()) {
            for block in contents {
                let b_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match b_type {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            content_text.push_str(t);
                        }
                    }
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        tool_calls.push(ToolCall {
                            id,
                            r#type: "function".to_string(),
                            function: FunctionCall {
                                name,
                                arguments: serde_json::to_string(&input).unwrap_or_default(),
                            },
                        });
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = match val.get("stop_reason").and_then(|r| r.as_str()) {
            Some("tool_use") => "tool_calls",
            Some("max_tokens") => "length",
            _ => "stop",
        };

        let in_tokens = val
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;
        let out_tokens = val
            .get("usage")
            .and_then(|u| u.get("output_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;

        let response = chat_completion_response(
            &client_model,
            if content_text.is_empty() {
                None
            } else {
                Some(content_text)
            },
            if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            finish_reason,
            created,
            Usage::new(in_tokens, out_tokens),
        );

        Ok(Json(response).into_response())
    }
}
