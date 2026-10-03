use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    ApiError, ChatMessage, CreateResponseRequest, CreateResponseResponse, ErrorResponse,
    MessageContent, ResponseContentPart, ResponseOutputItem, ResponseUsage,
};

#[axum::debug_handler]
pub async fn responses_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateResponseRequest>,
) -> Response {
    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'model' is a required parameter when creating a response.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("model")
                    .with_code("missing_required_parameter"),
                }),
            )
                .into_response();
        }
    };

    let input_text = match request.input {
        Some(serde_json::Value::String(ref s)) if !s.trim().is_empty() => s.trim().to_string(),
        Some(serde_json::Value::Array(ref arr)) if !arr.is_empty() => {
            let mut parts = Vec::new();
            for item in arr {
                if let Some(s) = item.as_str() {
                    parts.push(s.to_string());
                } else {
                    parts.push(item.to_string());
                }
            }
            parts.join("\n")
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'input' is a required non-empty string or array parameter.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("input")
                    .with_code("missing_required_parameter"),
                }),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let target_model = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&target_model)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model failed to load: {err}"))
                            .with_type("invalid_request_error")
                            .with_param("model")
                            .with_code("model_not_found"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Execution thread error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let mut messages = Vec::new();
    if let Some(ref inst) = request.instructions {
        if !inst.trim().is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(inst.trim().to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text(input_text.clone())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });

    let tools_val = request.tools.unwrap_or(serde_json::Value::Null);

    let task = InferenceTaskRequest::ToolCall {
        prompt: input_text.clone(),
        schema: tools_val,
        images: Vec::new(),
        messages,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let resp_id = format!(
        "resp_{}",
        hex::encode(&Sha256::digest(format!("resp_{created_at}_{input_text}").as_bytes())[..8])
    );
    let msg_id = format!(
        "msg_{}",
        hex::encode(&Sha256::digest(format!("msg_{created_at}_{resp_id}").as_bytes())[..8])
    );

    match res {
        Ok(Ok(output)) => {
            let (answer_text, prompt_tokens, comp_tokens) = match output.response {
                InferenceTaskResponse::Text(t) => (
                    t,
                    output.usage.prompt_tokens,
                    output.usage.completion_tokens,
                ),
                InferenceTaskResponse::ToolCall {
                    content,
                    tool_calls,
                } => {
                    let text = content.unwrap_or_else(|| tool_calls.to_string());
                    (
                        text,
                        output.usage.prompt_tokens,
                        output.usage.completion_tokens,
                    )
                }
                InferenceTaskResponse::Error(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: ApiError::new(e).with_type("server_error"),
                        }),
                    )
                        .into_response();
                }
                _ => (
                    "Execution completed.".to_string(),
                    output.usage.prompt_tokens,
                    output.usage.completion_tokens,
                ),
            };

            let response_data = CreateResponseResponse {
                id: resp_id,
                object: "response".to_string(),
                created_at,
                model: request.model.unwrap_or_default(),
                status: "completed".to_string(),
                output: vec![ResponseOutputItem {
                    r#type: "message".to_string(),
                    id: msg_id,
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart {
                        r#type: "text".to_string(),
                        text: answer_text,
                    }],
                }],
                usage: ResponseUsage {
                    input_tokens: prompt_tokens,
                    output_tokens: comp_tokens,
                    total_tokens: prompt_tokens.saturating_add(comp_tokens),
                },
            };

            Json(response_data).into_response()
        }
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err).with_type("server_error"),
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err.to_string()).with_type("server_error"),
            }),
        )
            .into_response(),
    }
}
