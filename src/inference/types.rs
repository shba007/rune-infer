use crate::types::{
    AudioTranscriptionResponse, AudioTranslationResponse, ChatMessage, DetectObjectsResponse,
    DocumentOcrResponse, EmbeddingResponse, ImageCaptionResponse, ImageEmbeddingsResponse,
    ImageGenerationResponse, ImageRestorationResponse, ImageStyleTransferResponse,
    ImageUpscaleResponse, ModerationResponse, NluIntentResponse, RecognizeObjectsResponse,
    RecognizeTarget, RegionCrop, Usage,
};
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
    AudioTranscription {
        audio_bytes: Vec<u8>,
        filename: String,
        prompt: Option<String>,
        language: Option<String>,
        temperature: Option<f32>,
        response_format: Option<String>,
    },
    AudioTranslation {
        audio_bytes: Vec<u8>,
        filename: String,
        prompt: Option<String>,
        temperature: Option<f32>,
        response_format: Option<String>,
    },
    AudioSpeech {
        input: String,
        voice: Option<String>,
        response_format: Option<String>,
        speed: Option<f32>,
    },
    SpeechToSpeech {
        audio_bytes: Vec<u8>,
        target_language: String,
        source_language: Option<String>,
        response_format: Option<String>,
    },
    Embedding {
        model: Option<String>,
        input: Vec<String>,
        dimensions: Option<usize>,
    },
    Moderation {
        model: Option<String>,
        input: Vec<String>,
    },
    NluIntent {
        model: Option<String>,
        text: String,
        candidate_intents: Option<Vec<String>>,
        candidate_entities: Option<Vec<String>>,
    },
    DocumentOcr {
        model: Option<String>,
        image_bytes: Vec<u8>,
        width: u32,
        height: u32,
        features: Vec<String>,
        response_format: Option<String>,
    },
    ImageCaption {
        model: Option<String>,
        image_bytes: Vec<u8>,
        detail: String,
        max_tokens: usize,
    },
    DetectObjects {
        model: Option<String>,
        image_bytes: Vec<u8>,
        prompt: Option<String>,
        categories: Option<Vec<String>>,
        confidence_threshold: Option<f32>,
        iou_threshold: Option<f32>,
        features: Vec<String>,
    },
    ImageEmbedding {
        model: Option<String>,
        image_bytes: Vec<u8>,
        regions: Option<Vec<RegionCrop>>,
        encoding_format: Option<String>,
    },
    RecognizeObjects {
        model: Option<String>,
        image_bytes: Vec<u8>,
        top_k: usize,
        targets: Vec<RecognizeTarget>,
    },
    ImageUpscale {
        model: Option<String>,
        image_bytes: Vec<u8>,
        scale: u32,
        response_format: Option<String>,
    },
    ImageRestoration {
        model: Option<String>,
        image_bytes: Vec<u8>,
        fidelity_weight: f32,
        face_upsample: bool,
        response_format: Option<String>,
    },
    ImageStyleTransfer {
        model: Option<String>,
        content_bytes: Vec<u8>,
        style_bytes: Vec<u8>,
        strength: f32,
        response_format: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InferenceTaskResponse {
    ToolCall {
        content: Option<String>,
        tool_calls: serde_json::Value,
    },
    Text(String),
    Image(ImageGenerationResponse),
    Audio(Vec<u8>),
    Transcription(AudioTranscriptionResponse),
    Translation(AudioTranslationResponse),
    Embedding(EmbeddingResponse),
    Moderation(ModerationResponse),
    NluIntent(NluIntentResponse),
    Ocr(DocumentOcrResponse),
    Caption(ImageCaptionResponse),
    Detection(DetectObjectsResponse),
    ImageEmbedding(ImageEmbeddingsResponse),
    Recognition(RecognizeObjectsResponse),
    Upscale(ImageUpscaleResponse),
    Restoration(ImageRestorationResponse),
    StyleTransfer(ImageStyleTransferResponse),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub response: InferenceTaskResponse,
    pub usage: Usage,
}
