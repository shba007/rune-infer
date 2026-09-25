use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum InferenceTaskRequest {
    ToolCall {
        prompt: String,
        schema: serde_json::Value,
        images: Vec<Vec<u8>>,
    },

    Chat {
        messages: Vec<ChatMessage>,
    },

    GenerateImage {
        prompt: String,
        width: Option<u32>,
        height: Option<u32>,
    },

    GenerateAudio {
        text: String,
        voice_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
pub enum InferenceTaskResponse {
    ToolCall(serde_json::Value),
    Text(String),
    ImageBase64(String),
    AudioBuffer(Vec<u8>),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
