use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    #[default]
    Local,
    #[serde(alias = "openai")]
    OpenAi,
    #[serde(alias = "anthropic")]
    Anthropic,
    #[serde(alias = "google", alias = "gemini")]
    Google,
    #[serde(alias = "openrouter")]
    OpenRouter,
    #[serde(alias = "custom")]
    Custom,
}

impl ProviderType {
    pub fn is_local(&self) -> bool {
        matches!(self, ProviderType::Local)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderType::Local => "local",
            ProviderType::OpenAi => "openai",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Google => "google",
            ProviderType::OpenRouter => "openrouter",
            ProviderType::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,
    #[serde(default)]
    pub tokens_per_minute: Option<u32>,
    #[serde(default)]
    pub requests_per_day: Option<u32>,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_queue_timeout")]
    pub queue_timeout_seconds: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub retry_delay_seconds: Option<u64>,
    #[serde(default = "default_retry_delay_ms")]
    pub retry_delay_ms: u64,
    #[serde(default = "default_retry_timeout")]
    pub retry_timeout_seconds: u64,
}

fn default_rpm() -> u32 {
    60
}

fn default_max_concurrent() -> usize {
    5
}

fn default_queue_timeout() -> u64 {
    60
}

fn default_max_retries() -> u32 {
    5
}

fn default_retry_delay_ms() -> u64 {
    1000
}

fn default_retry_timeout() -> u64 {
    30
}

impl RateLimitConfig {
    pub fn effective_retry_delay_ms(&self) -> u64 {
        if let Some(sec) = self.retry_delay_seconds {
            sec * 1000
        } else {
            self.retry_delay_ms
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: default_rpm(),
            tokens_per_minute: None,
            requests_per_day: None,
            max_concurrent: default_max_concurrent(),
            queue_timeout_seconds: default_queue_timeout(),
            max_retries: default_max_retries(),
            retry_delay_seconds: None,
            retry_delay_ms: default_retry_delay_ms(),
            retry_timeout_seconds: default_retry_timeout(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Modality {
    Text,
    VisionText,
    ImageGeneration,
    #[serde(alias = "AudioToText", alias = "AudioTranscription")]
    SpeechToText,
    #[serde(alias = "TextToAudio", alias = "AudioSpeech")]
    TextToSpeech,
    #[serde(alias = "SpeechToSpeechTranslation", alias = "S2ST")]
    SpeechToSpeech,
    #[serde(alias = "embedding", alias = "embeddings")]
    Embedding,
    #[serde(alias = "moderation", alias = "moderations", alias = "guardrails")]
    Moderation,
    #[serde(alias = "nlu", alias = "intent", alias = "classifier")]
    Nlu,
    #[serde(alias = "ocr", alias = "omr")]
    Ocr,
    #[serde(alias = "detection", alias = "object-detection")]
    ObjectDetection,
    #[serde(alias = "image_embedding", alias = "image-embeddings")]
    ImageEmbedding,
    #[serde(alias = "restoration", alias = "face-restoration")]
    ImageRestoration,
    #[serde(alias = "upscale", alias = "super-resolution")]
    ImageUpscale,
    #[serde(alias = "style_transfer", alias = "style")]
    ImageStyleTransfer,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::VisionText => "vision-text",
            Modality::ImageGeneration => "image-generation",
            Modality::SpeechToText => "speech-to-text",
            Modality::TextToSpeech => "text-to-speech",
            Modality::SpeechToSpeech => "speech-to-speech",
            Modality::Embedding => "embedding",
            Modality::Moderation => "moderation",
            Modality::Nlu => "nlu",
            Modality::Ocr => "ocr",
            Modality::ObjectDetection => "object-detection",
            Modality::ImageEmbedding => "image-embedding",
            Modality::ImageRestoration => "image-restoration",
            Modality::ImageUpscale => "image-upscale",
            Modality::ImageStyleTransfer => "image-style-transfer",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub max_loaded_models: usize,
    pub idle_unload_seconds: u64,
    #[serde(default = "default_vram_budget_ratio")]
    pub vram_budget_ratio: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            api_key: None,
            max_loaded_models: 1,
            idle_unload_seconds: 300,
            vram_budget_ratio: 0.97,
        }
    }
}

fn default_vram_budget_ratio() -> f64 {
    0.97
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub schema_version: u32,
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            schema_version: 1,
            server: ServerConfig::default(),
            models: Vec::new(),
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw = std::fs::read_to_string(path).context("Failed to read config file")?;
        let interpolated = interpolate_env_vars(&raw)?;
        let registry: ModelRegistry =
            serde_json::from_str(&interpolated).context("Failed to parse config as JSON")?;
        registry.validate()?;
        Ok(registry)
    }

    pub fn validate(&self) -> Result<()> {
        let mut seen_ids = std::collections::HashMap::new();
        for model in &self.models {
            if seen_ids.contains_key(&model.id) {
                anyhow::bail!("Duplicate model id '{}'", model.id);
            }
            seen_ids.insert(model.id.clone(), model.name.clone());
            model.validate()?;
        }
        Ok(())
    }

    pub fn find(&self, id: &str) -> Option<&ModelConfig> {
        self.models
            .iter()
            .find(|m| {
                m.id == id
                    || m.id.eq_ignore_ascii_case(id)
                    || m.name.eq_ignore_ascii_case(id)
                    || (m.id == "ternary-Bonsai-2-27b" && id == "ternary-Bonsai-2-27b")
            })
            .or_else(|| {
                let lower = id.to_lowercase();
                if lower.contains("claude") || lower.contains("anthropic") {
                    self.models
                        .iter()
                        .find(|m| m.provider == ProviderType::Anthropic)
                } else if lower.contains("gemini") || lower.contains("google") {
                    self.models
                        .iter()
                        .find(|m| m.provider == ProviderType::Google)
                } else if lower.contains("space-bunny")
                    || lower.contains("bunny")
                    || lower.contains("stealth")
                {
                    self.models.iter().find(|m| {
                        m.id.contains("space-bunny")
                            || m.name.to_lowercase().contains("space bunny")
                    })
                } else if lower.contains("dots") {
                    self.models
                        .iter()
                        .find(|m| m.id.contains("dots") || m.name.to_lowercase().contains("dots"))
                } else if lower.contains("apodex") || lower.contains("openrouter") {
                    self.models.iter().find(|m| {
                        m.provider == ProviderType::OpenRouter
                            || m.id.to_lowercase().contains("apodex")
                    })
                } else if lower.contains("gpt-") || lower.contains("openai") {
                    self.models
                        .iter()
                        .find(|m| m.provider == ProviderType::OpenAi)
                } else if lower.contains("embed")
                    || lower.contains("bge")
                    || lower.contains("granite")
                {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::Embedding)
                } else if lower.contains("moderation") || lower.contains("guard") {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::Moderation)
                } else if lower.contains("intent")
                    || lower.contains("nlu")
                    || lower.contains("gliner")
                    || lower.contains("mmbert")
                {
                    self.models.iter().find(|m| m.modality == Modality::Nlu)
                } else if lower.contains("got-ocr") || lower.contains("ocr") {
                    self.models.iter().find(|m| m.modality == Modality::Ocr)
                } else if lower.contains("whisper")
                    || lower.contains("transcription")
                    || lower.contains("asr")
                {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::SpeechToText)
                } else if lower.contains("tts") || lower.contains("speech") {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::TextToSpeech)
                } else if lower.contains("s2st") || lower.contains("seamless") {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::SpeechToSpeech)
                } else if lower.contains("yolo") || lower.contains("detection") {
                    self.models
                        .iter()
                        .find(|m| m.modality == Modality::ObjectDetection)
                } else {
                    None
                }
            })
    }

    pub fn find_owned(&self, id: &str) -> Option<ModelConfig> {
        self.find(id).cloned()
    }
}

pub fn interpolate_env_vars(raw: &str) -> Result<String> {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'{') {
            chars.next();
            let mut var_name = String::new();
            let mut closed = false;

            while let Some(inner) = chars.next() {
                if inner == '}' && chars.peek() == Some(&'}') {
                    chars.next();
                    closed = true;
                    break;
                }
                var_name.push(inner);
            }

            if !closed {
                anyhow::bail!("Unclosed template parameter '{{{{{var_name}' in config");
            }

            let var_trimmed = var_name.trim();
            match std::env::var(var_trimmed) {
                Ok(val) => {
                    let escaped = val.replace('\\', "\\\\").replace('"', "\\\"");
                    result.push_str(&escaped);
                }
                Err(_) => {
                    anyhow::bail!(
                        "Configuration references environment variable '{}' via '{{{{{}}}}}', but it is not set in .env or the system environment",
                        var_trimmed,
                        var_trimmed
                    );
                }
            }
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    #[serde(default = "default_context_length")]
    pub context_length: i64,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    #[serde(default)]
    pub mtp_heads: Option<u32>,
    #[serde(default)]
    pub extra_args: Option<String>,
}

fn default_context_length() -> i64 {
    8192
}

fn default_gpu_layers() -> u32 {
    99
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub provider: ProviderType,
    pub modality: Modality,
    #[serde(default)]
    pub architecture: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub upstream_model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
    pub mmproj_path: Option<String>,
    #[serde(default)]
    pub mtp_path: Option<String>,
    #[serde(default)]
    pub vae_path: Option<String>,
    #[serde(default)]
    pub text_encoder_path: Option<String>,
    #[serde(default)]
    pub llm_vision_path: Option<String>,
    #[serde(default)]
    pub default_steps: Option<u32>,
    #[serde(default)]
    pub default_cfg_scale: Option<f32>,
    #[serde(default)]
    pub default_sample_method: Option<String>,
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub bits_per_weight: Option<f64>,
    #[serde(default)]
    pub total_params: Option<u64>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub kv_bytes_per_token: Option<u64>,
    #[serde(default)]
    pub max_resolution: Option<String>,
    #[serde(default)]
    pub mtp_heads: Option<u32>,
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub runtime: Option<RuntimeConfig>,
}

impl ModelConfig {
    pub fn is_local(&self) -> bool {
        self.provider == ProviderType::Local
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            anyhow::bail!("Model '{}' has empty id", self.name);
        }

        if self.is_local() {
            if self.model_path.trim().is_empty() {
                anyhow::bail!(
                    "Local model '{}' must specify a valid 'model_path'",
                    self.id
                );
            }
        } else {
            if self
                .upstream_model
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                anyhow::bail!(
                    "Remote model '{}' (provider: {:?}) must specify 'upstream_model'",
                    self.id,
                    self.provider
                );
            }
            if self.api_key.as_deref().unwrap_or("").trim().is_empty() {
                anyhow::bail!(
                    "Remote model '{}' (provider: {:?}) must provide an 'api_key'",
                    self.id,
                    self.provider
                );
            }
        }
        Ok(())
    }

    pub fn mtp_heads(&self) -> Option<u32> {
        self.mtp_heads
            .or_else(|| self.runtime.as_ref().and_then(|r| r.mtp_heads))
    }

    pub fn has_mtp(&self) -> bool {
        if self.mtp_path.as_ref().is_some_and(|p| !p.trim().is_empty()) {
            return true;
        }
        self.mtp_heads().is_some_and(|h| h > 0)
    }

    pub fn resolved_capabilities(&self) -> Vec<String> {
        if let Some(ref caps) = self.capabilities {
            if !caps.is_empty() {
                return caps.clone();
            }
        }
        let mut caps = vec!["Chat".to_string(), "Tool Call".to_string()];
        if self.vision || self.modality == Modality::VisionText {
            caps.push("Vision".to_string());
        }
        if self.has_mtp() && !caps.iter().any(|c| c.eq_ignore_ascii_case("mtp")) {
            caps.push("MTP".to_string());
        }
        if self.modality == Modality::ImageGeneration
            && !caps.iter().any(|c| c.eq_ignore_ascii_case("text-to-image"))
        {
            caps.push("Text-to-Image".to_string());
        }
        if (self.modality == Modality::SpeechToText || self.architecture == "crispasr")
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("speech-to-text") || c.eq_ignore_ascii_case("asr"))
        {
            caps.push("Speech-to-Text".to_string());
        }
        if self.modality == Modality::TextToSpeech
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("text-to-speech") || c.eq_ignore_ascii_case("tts"))
        {
            caps.push("Text-to-Speech".to_string());
        }
        if self.modality == Modality::Embedding
            && !caps.iter().any(|c| c.eq_ignore_ascii_case("embeddings"))
        {
            caps.push("Embeddings".to_string());
        }
        if self.modality == Modality::Moderation
            && !caps.iter().any(|c| c.eq_ignore_ascii_case("moderations"))
        {
            caps.push("Moderations".to_string());
        }
        if self.modality == Modality::Nlu && !caps.iter().any(|c| c.eq_ignore_ascii_case("nlu")) {
            caps.push("NLU".to_string());
        }
        if self.modality == Modality::Ocr && !caps.iter().any(|c| c.eq_ignore_ascii_case("ocr")) {
            caps.push("OCR".to_string());
        }
        if self.modality == Modality::SpeechToSpeech
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("speech-to-speech"))
        {
            caps.push("Speech-to-Speech".to_string());
        }
        if self.modality == Modality::ObjectDetection
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("object-detection"))
        {
            caps.push("Object Detection".to_string());
        }
        if self.modality == Modality::ImageEmbedding
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("image-embeddings"))
        {
            caps.push("Image Embeddings".to_string());
        }
        if self.modality == Modality::ImageRestoration
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("image-restoration"))
        {
            caps.push("Image Restoration".to_string());
        }
        if self.modality == Modality::ImageUpscale
            && !caps.iter().any(|c| c.eq_ignore_ascii_case("image-upscale"))
        {
            caps.push("Image Upscale".to_string());
        }
        if self.modality == Modality::ImageStyleTransfer
            && !caps
                .iter()
                .any(|c| c.eq_ignore_ascii_case("style-transfer"))
        {
            caps.push("Style Transfer".to_string());
        }
        caps
    }

    pub fn supports_structured_output(&self) -> bool {
        self.resolved_capabilities().iter().any(|c| {
            c.eq_ignore_ascii_case("Structured Output")
                || c.eq_ignore_ascii_case("Structured JSON")
                || c.eq_ignore_ascii_case("Structured Output Mode")
                || c.eq_ignore_ascii_case("Structured Extraction")
        })
    }

    pub fn max_dimensions(&self) -> (u32, u32) {
        if let Some(ref res_str) = self.max_resolution {
            let clean = res_str.split('(').next().unwrap_or(res_str).trim();
            let parts: Vec<&str> = if clean.contains('×') {
                clean.split('×').collect()
            } else if clean.contains('x') {
                clean.split('x').collect()
            } else if clean.contains('*') {
                clean.split('*').collect()
            } else {
                Vec::new()
            };
            if parts.len() == 2 {
                if let (Ok(w), Ok(h)) = (
                    parts[0].trim().parse::<u32>(),
                    parts[1].trim().parse::<u32>(),
                ) {
                    return (w, h);
                }
            }
        }
        (2048, 2048)
    }

    pub fn effective_context_limit(&self) -> u32 {
        if let Some(ref r) = self.runtime {
            if r.context_length > 0 {
                return (r.context_length as u32).min(self.max_context_length.unwrap_or(131072));
            }
        }
        self.max_context_length.unwrap_or(8192)
    }
}
