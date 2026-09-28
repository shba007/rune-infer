use crate::types::{ChatMessage, ImageGenerationResponse, Usage};
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
    ImageGeneration {
        prompt: String,
        negative_prompt: Option<String>,
        size: Option<String>,
        response_format: Option<String>,
        steps: Option<u32>,
        cfg_scale: Option<f32>,
        seed: Option<i64>,
        sample_method: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InferenceTaskResponse {
    ToolCall(serde_json::Value),
    Text(String),
    Image(ImageGenerationResponse),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub response: InferenceTaskResponse,
    pub usage: Usage,
}
