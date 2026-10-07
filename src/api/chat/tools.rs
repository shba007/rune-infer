use axum::Json;
use axum::http::StatusCode;

use crate::types::{ApiError, ErrorResponse, FunctionCall, ToolCall};

/// Validates that tools passed in the request body have valid function names.
pub fn validate_tools(
    body_val: &serde_json::Value,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if let Some(serde_json::Value::Array(tools)) = body_val.get("tools") {
        for (idx, tool) in tools.iter().enumerate() {
            if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
                let name = tool
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str());
                if name.is_none() || name.unwrap().trim().is_empty() {
                    return Err((
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
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Builds the system prompt injection for tool calling, structured JSON, or video modes.
pub fn build_system_prompt(
    is_structured_mode: bool,
    structured_schema: &serde_json::Value,
    has_tools: bool,
    tools_value: &serde_json::Value,
    has_video: bool,
) -> String {
    if is_structured_mode {
        format!(
            "You are a helpful assistant.\nYou must respond ONLY with a valid JSON object matching this schema:\n{}",
            serde_json::to_string_pretty(structured_schema).unwrap_or_default()
        )
    } else if has_tools {
        let tools_json = serde_json::to_string_pretty(tools_value).unwrap_or_default();
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
    }
}

pub fn parse_single_tool_call(val: &serde_json::Value, fallback_id: String) -> Option<ToolCall> {
    let obj = val.as_object()?;

    if let Some(func) = obj.get("function").and_then(|f| f.as_object()) {
        let name = func.get("name")?.as_str()?.to_string();
        let args = match func.get("arguments") {
            Some(serde_json::Value::String(s)) => {
                if s.trim().is_empty() {
                    "{}".to_string()
                } else {
                    s.clone()
                }
            }
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
            function: FunctionCall {
                name,
                arguments: args,
            },
        });
    }

    let name = obj.get("name")?.as_str()?.to_string();
    let args = match obj.get("arguments") {
        Some(serde_json::Value::String(s)) => {
            if s.trim().is_empty() {
                "{}".to_string()
            } else {
                s.clone()
            }
        }
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
        function: FunctionCall {
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
