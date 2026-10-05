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

/// Recursively removes schema keywords unsupported by Google's Schema protobuf
/// (e.g. `additionalProperties`, `$schema`, `strict`, `title`).
fn sanitize_gemini_schema(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut cleaned = serde_json::Map::new();
            for (k, v) in map {
                // Strip fields unsupported by Google Gemini Schema protobuf
                if k == "additionalProperties"
                    || k == "$schema"
                    || k == "strict"
                    || k == "$defs"
                    || k == "definitions"
                {
                    continue;
                }
                cleaned.insert(k.clone(), sanitize_gemini_schema(v));
            }
            serde_json::Value::Object(cleaned)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(sanitize_gemini_schema).collect())
        }
        other => other.clone(),
    }
}

pub async fn execute_chat(
    client: &reqwest::Client,
    config: &ModelConfig,
    request: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let api_key = config.api_key.as_deref().unwrap_or("");
    let upstream_model = config.upstream_model.as_deref().unwrap_or(&request.model);

    let base_url = config
        .base_url
        .as_deref()
        .unwrap_or("https://generativelanguage.googleapis.com/v1beta");

    let (action, query_sep) = if request.stream {
        ("streamGenerateContent?alt=sse", "&")
    } else {
        ("generateContent", "?")
    };

    let endpoint = format!(
        "{}/models/{}:{}{}key={}",
        base_url.trim_end_matches('/'),
        upstream_model,
        action,
        query_sep,
        api_key
    );

    let mut system_instruction_parts = Vec::new();
    let mut contents = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            let text = msg.text_content();
            if !text.is_empty() {
                system_instruction_parts.push(serde_json::json!({ "text": text }));
            }
            continue;
        }

        let role = match msg.role.as_str() {
            "assistant" => "model",
            _ => "user",
        };

        let mut parts = Vec::new();

        if msg.role == "tool" {
            let func_name = msg
                .name
                .clone()
                .unwrap_or_else(|| "function_result".to_string());
            parts.push(serde_json::json!({
                "functionResponse": {
                    "name": func_name,
                    "response": {
                        "name": func_name,
                        "content": msg.text_content()
                    }
                }
            }));
            contents.push(serde_json::json!({
                "role": "user",
                "parts": parts
            }));
            continue;
        }

        let (text, media_items) = msg.split_text_and_media();
        if !text.is_empty() {
            parts.push(serde_json::json!({ "text": text }));
        }

        for item in media_items {
            let url = match item {
                crate::types::MediaItem::Image(u) => u,
                crate::types::MediaItem::Video(u) => u,
            };
            if let Some((mime, b64)) = parse_data_uri(&url) {
                parts.push(serde_json::json!({
                    "inlineData": {
                        "mimeType": mime,
                        "data": b64
                    }
                }));
            }
        }

        contents.push(serde_json::json!({
            "role": role,
            "parts": parts
        }));
    }

    let mut func_decls = Vec::new();
    if let Some(serde_json::Value::Array(tools)) = &request.tools {
        for t in tools {
            if t.get("type").and_then(|v| v.as_str()) == Some("function") {
                if let Some(func) = t.get("function") {
                    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let desc = func
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let params = func
                        .get("parameters")
                        .map(sanitize_gemini_schema)
                        .unwrap_or_else(|| serde_json::json!({}));
                    func_decls.push(serde_json::json!({
                        "name": name,
                        "description": desc,
                        "parameters": params
                    }));
                }
            }
        }
    }

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": {
            "temperature": request.temperature.unwrap_or(0.7),
            "topP": request.top_p.unwrap_or(0.95),
            "maxOutputTokens": request.max_tokens.unwrap_or(8192)
        }
    });

    // =========================================================================
    // Map & Sanitize Structured Output JSON Schema for Gemini
    // =========================================================================
    if let Some(ref rf) = request.response_format {
        if rf.r#type == "json_object" {
            body["generationConfig"]["responseMimeType"] = serde_json::json!("application/json");
        } else if rf.r#type == "json_schema" {
            body["generationConfig"]["responseMimeType"] = serde_json::json!("application/json");
            if let Some(ref js) = rf.json_schema {
                if let Some(ref schema) = js.schema {
                    // Sanitize away `additionalProperties: false` and other non-proto fields
                    let sanitized = sanitize_gemini_schema(schema);
                    body["generationConfig"]["responseSchema"] = sanitized;
                }
            }
        }
    }

    if !system_instruction_parts.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": system_instruction_parts
        });
    }

    if !func_decls.is_empty() {
        body["tools"] = serde_json::json!([{
            "functionDeclarations": func_decls
        }]);
    }

    let resp = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            ApiError::new(format!("Failed to connect to Gemini endpoint: {e}"))
                .with_type("api_error")
        })?;

    let status = resp.status();
    if !status.is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(
            ApiError::new(format!("Gemini returned HTTP {status}: {err_text}"))
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

            // Initial assistant role handshake chunk
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
            let mut is_done = false;

            while let Some(line) = data_stream.next().await {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(err) = val.get("error") {
                        let err_msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("Gemini streaming error");
                        let err_chunk = serde_json::json!({
                            "error": {
                                "message": err_msg,
                                "type": "upstream_error",
                                "code": "gemini_error"
                            }
                        });
                        yield Ok(Event::default().event("error").data(err_chunk.to_string()));
                        is_done = true;
                        break;
                    }

                    if let Some(candidate) = val.get("candidates").and_then(|c| c.get(0)) {
                        if let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                            for part in parts {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
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
                                }

                                if let Some(fc) = part.get("functionCall") {
                                    let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                    let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
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
                                                id: Some(format!("call_gemini_{created}_{tool_idx}")),
                                                r#type: Some("function".to_string()),
                                                function: Some(crate::types::FunctionCallChunk {
                                                    name: Some(name.to_string()),
                                                    arguments: Some(serde_json::to_string(&args).unwrap_or_default()),
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

                        // Only complete if finishReason is present and non-empty
                        if let Some(finish_str) = candidate
                            .get("finishReason")
                            .and_then(|r| r.as_str())
                            .filter(|s| !s.trim().is_empty())
                        {
                            let finish_reason = if finish_str == "MAX_TOKENS" { "length" } else { "stop" };

                            let in_tokens = val.get("usageMetadata").and_then(|u| u.get("promptTokenCount")).and_then(|c| c.as_u64()).unwrap_or(0) as u32;
                            let out_tokens = val.get("usageMetadata").and_then(|u| u.get("candidatesTokenCount")).and_then(|c| c.as_u64()).unwrap_or(0) as u32;

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
                                Usage::new(in_tokens, out_tokens),
                            );
                            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                            yield Ok(Event::default().data("[DONE]"));
                            is_done = true;
                            break;
                        }
                    }
                }
            }

            if !is_done {
                yield Ok(Event::default().data("[DONE]"));
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response())
    } else {
        let val: serde_json::Value = resp.json().await.map_err(|e| {
            ApiError::new(format!("Failed to parse Gemini JSON response: {e}"))
                .with_type("api_error")
        })?;

        let candidate = val.get("candidates").and_then(|c| c.get(0));
        let mut content_text = String::new();
        let mut tool_calls = Vec::new();

        if let Some(cand) = candidate {
            if let Some(parts) = cand
                .get("content")
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
            {
                for (idx, part) in parts.iter().enumerate() {
                    if let Some(t) = part.get("text").and_then(|s| s.as_str()) {
                        content_text.push_str(t);
                    }
                    if let Some(fc) = part.get("functionCall") {
                        let name = fc
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        let args = fc
                            .get("args")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        tool_calls.push(ToolCall {
                            id: format!("call_gemini_{created}_{idx}"),
                            r#type: "function".to_string(),
                            function: FunctionCall {
                                name,
                                arguments: serde_json::to_string(&args).unwrap_or_default(),
                            },
                        });
                    }
                }
            }
        }

        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };

        let in_tokens = val
            .get("usageMetadata")
            .and_then(|u| u.get("promptTokenCount"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as u32;
        let out_tokens = val
            .get("usageMetadata")
            .and_then(|u| u.get("candidatesTokenCount"))
            .and_then(|c| c.as_u64())
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
