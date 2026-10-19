use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::AppState;
use crate::api::chat::prepare_prompt;
use crate::inference::types::InferenceTaskResponse;
use crate::types::{
    ApiError, ChatCompletionRequest, ChatMessage, ContentPart, ErrorResponse, ImageUrlContent,
    MessageContent,
};

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub async fn anthropic_messages_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AnthropicMessagesRequest>,
) -> Response {
    if payload.model.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing required field: 'model'")
                    .with_type("invalid_request_error"),
            }),
        )
            .into_response();
    }

    let model_config = match state.config().find(&payload.model) {
        Some(c) => c.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found in catalog", payload.model))
                        .with_type("not_found_error"),
                }),
            )
                .into_response();
        }
    };

    let mut chat_messages: Vec<ChatMessage> = Vec::new();

    if let Some(ref sys_val) = payload.system {
        let sys_text = match sys_val {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n"),
            _ => String::new(),
        };
        if !sys_text.trim().is_empty() {
            chat_messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(sys_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    for msg in &payload.messages {
        match &msg.content {
            serde_json::Value::String(s) => {
                chat_messages.push(ChatMessage {
                    role: msg.role.clone(),
                    content: Some(MessageContent::Text(s.clone())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
            serde_json::Value::Array(blocks) => {
                let mut parts = Vec::new();
                for b in blocks {
                    let b_type = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match b_type {
                        "text" => {
                            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                parts.push(ContentPart {
                                    r#type: Some("text".to_string()),
                                    text: Some(t.to_string()),
                                    image_url: None,
                                    video_url: None,
                                    input_audio: None,
                                });
                            }
                        }
                        "image" => {
                            if let Some(source) = b.get("source") {
                                let s_type =
                                    source.get("type").and_then(|s| s.as_str()).unwrap_or("");
                                if s_type == "base64" {
                                    let media_type = source
                                        .get("media_type")
                                        .and_then(|m| m.as_str())
                                        .unwrap_or("image/jpeg");
                                    let data =
                                        source.get("data").and_then(|d| d.as_str()).unwrap_or("");
                                    let data_uri = format!("data:{media_type};base64,{data}");
                                    parts.push(ContentPart {
                                        r#type: Some("image_url".to_string()),
                                        text: None,
                                        image_url: Some(ImageUrlContent { url: data_uri }),
                                        video_url: None,
                                        input_audio: None,
                                    });
                                }
                            }
                        }
                        "tool_result" => {
                            let tool_use_id = b
                                .get("tool_use_id")
                                .and_then(|id| id.as_str())
                                .map(|s| s.to_string());
                            let tool_content = b
                                .get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            chat_messages.push(ChatMessage {
                                role: "tool".to_string(),
                                content: Some(MessageContent::Text(tool_content)),
                                tool_calls: None,
                                tool_call_id: tool_use_id,
                                name: None,
                            });
                        }
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    chat_messages.push(ChatMessage {
                        role: msg.role.clone(),
                        content: Some(MessageContent::Parts(parts)),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
            _ => {}
        }
    }

    let openai_tools = payload.tools.as_ref().map(|tools| {
        let list: Vec<serde_json::Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema
                    }
                })
            })
            .collect();
        serde_json::Value::Array(list)
    });

    let chat_req = ChatCompletionRequest {
        model: payload.model.clone(),
        stream: payload.stream,
        messages: chat_messages,
        temperature: payload.temperature,
        top_p: payload.top_p,
        max_tokens: payload.max_tokens,
        top_k: None,
        min_p: None,
        tools: openai_tools,
        tool_choice: None,
        response_format: None,
    };

    if !model_config.is_local() {
        return state.proxy.execute_chat(&model_config, &chat_req).await;
    }

    let prepared = match prepare_prompt(&chat_req, &model_config) {
        Ok(p) => p,
        Err(err_resp) => return err_resp.into_response(),
    };

    let inference = state.inference.clone();
    let model_target = payload.model.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_target)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!(
                            "Failed to load engine for '{}': {err}",
                            payload.model
                        )),
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

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let message_id = format!("msg_{:016x}", created);
    let requested_model = payload.model.clone();

    if payload.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let task_clone = prepared.task.clone();
        let msg_id_clone = message_id.clone();
        let model_id_clone = requested_model.clone();

        tokio::task::spawn_blocking(move || {
            let start_event = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": msg_id_clone,
                    "type": "message",
                    "role": "assistant",
                    "model": model_id_clone,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            });
            let _ = tx.send(
                Event::default()
                    .event("message_start")
                    .data(start_event.to_string()),
            );

            let block_start = serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            });
            let _ = tx.send(
                Event::default()
                    .event("content_block_start")
                    .data(block_start.to_string()),
            );

            let tx_clone = tx.clone();
            let mut on_token = move |piece: &str| -> bool {
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": piece }
                });
                tx_clone
                    .send(
                        Event::default()
                            .event("content_block_delta")
                            .data(delta.to_string()),
                    )
                    .is_ok()
            };

            let res = engine.execute(&task_clone, Some(&mut on_token));

            let block_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": 0
            });
            let _ = tx.send(
                Event::default()
                    .event("content_block_stop")
                    .data(block_stop.to_string()),
            );

            let final_usage = res.map(|o| o.usage).unwrap_or_default();
            let msg_delta = serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": null
                },
                "usage": { "output_tokens": final_usage.completion_tokens }
            });
            let _ = tx.send(
                Event::default()
                    .event("message_delta")
                    .data(msg_delta.to_string()),
            );

            let msg_stop = serde_json::json!({ "type": "message_stop" });
            let _ = tx.send(
                Event::default()
                    .event("message_stop")
                    .data(msg_stop.to_string()),
            );
        });

        let stream = UnboundedReceiverStream::new(rx);
        return Sse::new(stream.map(Ok::<_, Infallible>))
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response();
    }

    let task_clone = prepared.task.clone();
    let result = tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await;

    match result {
        Ok(Ok(output)) => {
            let (content_blocks, stop_reason) = match output.response {
                InferenceTaskResponse::Text(t) => (
                    vec![AnthropicContentBlock::Text { text: t }],
                    Some("end_turn".to_string()),
                ),
                InferenceTaskResponse::ToolCall {
                    content,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();
                    if let Some(text) = content {
                        if !text.trim().is_empty() {
                            blocks.push(AnthropicContentBlock::Text { text });
                        }
                    }
                    if let Some(arr) = tool_calls.as_array() {
                        for (i, tc) in arr.iter().enumerate() {
                            let name = tc
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("function")
                                .to_string();
                            let input = tc
                                .get("arguments")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!({}));
                            blocks.push(AnthropicContentBlock::ToolUse {
                                id: format!("toolu_{created}_{i}"),
                                name,
                                input,
                            });
                        }
                    }
                    (blocks, Some("tool_use".to_string()))
                }
                _ => (
                    vec![AnthropicContentBlock::Text {
                        text: String::new(),
                    }],
                    Some("end_turn".to_string()),
                ),
            };

            let response = AnthropicMessageResponse {
                id: message_id,
                r#type: "message".to_string(),
                role: "assistant".to_string(),
                model: requested_model,
                content: content_blocks,
                stop_reason,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: output.usage.prompt_tokens,
                    output_tokens: output.usage.completion_tokens,
                },
            };

            Json(response).into_response()
        }
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err).with_type("api_error"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(format!("Execution failed: {e}")),
            }),
        )
            .into_response(),
    }
}
