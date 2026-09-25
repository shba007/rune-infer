use crate::types::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceTaskRequest {
    ToolCall {
        prompt: String,
        schema: serde_json::Value,
        #[serde(default)]
        images: Vec<Vec<u8>>,
        #[serde(default)]
        messages: Vec<ChatMessage>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InferenceTaskResponse {
    ToolCall(serde_json::Value),
    Text(String),
    Error(String),
}
