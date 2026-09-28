use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default, alias = "functions")]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaItem {
    Image(String),
    Video(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn text_content(&self) -> String {
        match &self.content {
            None => String::new(),
            Some(MessageContent::Text(text)) => text.clone(),
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn split_text_and_media(&self) -> (String, Vec<MediaItem>) {
        match &self.content {
            None => (String::new(), Vec::new()),
            Some(MessageContent::Text(text)) => (text.clone(), Vec::new()),
            Some(MessageContent::Parts(parts)) => {
                let mut text = String::new();
                let mut media_items = Vec::new();
                for p in parts {
                    if let Some(t) = &p.text {
                        if !t.is_empty() {
                            text.push_str(t);
                            text.push('\n');
                        }
                    }
                    if let Some(img) = &p.image_url {
                        media_items.push(MediaItem::Image(img.url.clone()));
                    }
                    if let Some(vid) = &p.video_url {
                        media_items.push(MediaItem::Video(vid.url.clone()));
                    }
                }
                (text.trim().to_string(), media_items)
            }
        }
    }

    pub fn split_text_and_images(&self) -> (String, Vec<String>) {
        let (text, items) = self.split_text_and_media();
        let urls = items
            .into_iter()
            .map(|item| match item {
                MediaItem::Image(url) => url,
                MediaItem::Video(url) => url,
            })
            .collect();
        (text, urls)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_url: Option<VideoUrlContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlContent {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrlContent {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChunk {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChoiceDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default, alias = "guidance")]
    pub cfg_scale: Option<f32>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    #[serde(default)]
    pub sample_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SseResponse {
    pub choices: Vec<Choice>,
}

impl SseResponse {
    pub fn new() -> Self {
        Self {
            choices: Vec::new(),
        }
    }
}

impl IntoResponse for SseResponse {
    fn into_response(self) -> axum::response::Response {
        let body = format!("data: {}\n\n", serde_json::to_string(&self).unwrap());
        axum::response::Response::builder()
            .status(axum::http::StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(body.into())
            .unwrap()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            type_: None,
            code: None,
        }
    }

    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.type_ = Some(t.into());
        self
    }

    pub fn with_code(mut self, c: impl Into<String>) -> Self {
        self.code = Some(c.into());
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_heads: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

impl ModelInfo {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "model".to_string(),
            owned_by: None,
            mtp_heads: None,
            capabilities: None,
        }
    }

    pub fn with_ownership(mut self, owner: impl Into<String>) -> Self {
        self.owned_by = Some(owner.into());
        self
    }

    pub fn with_mtp_heads(mut self, heads: Option<u32>) -> Self {
        self.mtp_heads = heads;
        self
    }

    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.capabilities = Some(caps);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded_models: Option<Vec<String>>,
}

impl HealthResponse {
    pub fn new() -> Self {
        Self {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            loaded_models: None,
        }
    }

    pub fn with_loaded_models(mut self, models: Vec<String>) -> Self {
        self.loaded_models = Some(models);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSpeechRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}
