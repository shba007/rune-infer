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
    #[serde(default, alias = "max_completion_tokens")]
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
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<ResponseMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<ChoiceDelta>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

fn default_chat_object() -> String {
    "chat.completion".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(default = "default_chat_object")]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Usage,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            type_: None,
            param: None,
            code: None,
        }
    }

    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.type_ = Some(t.into());
        self
    }

    pub fn with_param(mut self, p: impl Into<String>) -> Self {
        self.param = Some(p.into());
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    pub permission: Vec<ModelPermission>,
    pub root: String,
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_heads: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

impl ModelInfo {
    pub fn new(id: &str) -> Self {
        let clean_id = id.trim().to_string();
        let perm_id = format!(
            "modelperm-{}",
            clean_id.split('-').next().unwrap_or(&clean_id)
        );
        Self {
            id: clean_id.clone(),
            object: "model".to_string(),
            created: 1727827200,
            owned_by: Some("rune-infer".to_string()),
            permission: vec![ModelPermission {
                id: perm_id,
                object: "model_permission".to_string(),
                created: 1727827200,
                allow_create_engine: false,
                allow_sampling: true,
                allow_logprobs: true,
                allow_search_indices: false,
                allow_view: true,
                allow_fine_tuning: false,
                organization: "*".to_string(),
                group: None,
                is_blocking: false,
            }],
            root: clean_id,
            parent: None,
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
    pub service: String,
    pub version: String,
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded_models: Option<Vec<String>>,
}

impl HealthResponse {
    pub fn new() -> Self {
        let backend = if cfg!(feature = "cuda") {
            "candle-cuda"
        } else if cfg!(feature = "metal") {
            "candle-metal"
        } else if cfg!(feature = "vulkan") {
            "candle-vulkan"
        } else {
            "candle-cuda"
        };
        Self {
            status: "ok".to_string(),
            service: "rune-infer".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: backend.to_string(),
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
pub struct CreateResponseRequest {
    pub model: Option<String>,
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseContentPart {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseOutputItem {
    pub r#type: String,
    pub id: String,
    pub role: String,
    pub content: Vec<ResponseContentPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponseResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponseOutputItem>,
    pub usage: ResponseUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    pub r#type: Option<String>,
    pub image_url: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentOcrRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub document: Option<DocumentSource>,
    #[serde(default)]
    pub features: Option<Vec<String>>,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    pub bbox: [u32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrSelectionMark {
    pub id: String,
    pub question_index: u32,
    pub value: u32,
    pub state: String,
    pub confidence: f64,
    pub bbox: [u32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPage {
    pub page: u32,
    pub dimensions: OcrDimensions,
    pub lines: Vec<OcrLine>,
    pub selection_marks: Vec<OcrSelectionMark>,
    pub tables: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrUsage {
    pub pages_processed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentOcrResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub text: String,
    pub pages: Vec<OcrPage>,
    pub usage: OcrUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCaptionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCaptionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub caption: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranslationResponse {
    pub text: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEmbeddingRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    pub sexual: bool,
    pub hate: bool,
    pub harassment: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
    pub violence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategoryScores {
    pub sexual: f64,
    pub hate: f64,
    pub harassment: f64,
    #[serde(rename = "self-harm")]
    pub self_harm: f64,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
    pub violence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: ModerationCategories,
    pub category_scores: ModerationCategoryScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluIntentRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub text: Option<String>,
    #[serde(default)]
    pub candidate_intents: Option<Vec<String>>,
    #[serde(default)]
    pub candidate_entities: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluIntent {
    pub name: String,
    pub confidence: f64,
}

impl NluIntent {
    pub fn new(name: impl Into<String>, confidence: f64) -> Self {
        Self {
            name: name.into(),
            confidence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluEntity {
    pub r#type: String,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NluIntentResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<NluIntent>,
    pub entities: Vec<NluEntity>,
}

// --- Additional Postman Schemas ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechToSpeechRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub target_language: String,
    #[serde(default)]
    pub source_language: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCrop {
    #[serde(rename = "box")]
    pub box_: [f64; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub regions: Option<Vec<RegionCrop>>,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRegionEmbeddingObject {
    pub object: String,
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingsResponse {
    pub object: String,
    pub data: Vec<ImageRegionEmbeddingObject>,
    pub model: String,
    pub usage: ImageEmbeddingsUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingsUsage {
    pub total_regions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectObjectsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub categories: Option<Vec<String>>,
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    #[serde(default)]
    pub iou_threshold: Option<f32>,
    #[serde(default)]
    pub features: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionBboxPixels {
    pub x_min: u32,
    pub y_min: u32,
    pub x_max: u32,
    pub y_max: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPoseKeypoint {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionAttributes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glasses: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liveness: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionItem {
    pub object: String,
    pub index: usize,
    pub label: String,
    pub confidence: f64,
    #[serde(rename = "box")]
    pub box_: DetectionBox,
    pub bbox_pixels: DetectionBboxPixels,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<Vec<DetectionPoseKeypoint>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<DetectionAttributes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectObjectsResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub data: Vec<DetectionItem>,
    pub usage: DetectObjectsUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectObjectsUsage {
    pub image_width: u32,
    pub image_height: u32,
    pub total_objects: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeTarget {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeObjectsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    pub targets: Vec<RecognizeTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionMatch {
    pub object: String,
    pub index: usize,
    pub label: String,
    pub similarity: f64,
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeObjectsResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<RecognitionMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub scale: Option<u32>,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleUsage {
    pub input_width: u32,
    pub input_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub scale: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
    pub usage: ImageUpscaleUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRestorationRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub image: Option<String>,
    #[serde(default)]
    pub fidelity_weight: Option<f32>,
    #[serde(default)]
    pub face_upsample: Option<bool>,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRestorationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageStyleTransferRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub content_image: Option<String>,
    pub style_image: Option<String>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageStyleTransferResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}
