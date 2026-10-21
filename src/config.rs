use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    pub fn from_name(name: &str) -> Option<Self> {
        [
            ProviderType::Local,
            ProviderType::OpenAi,
            ProviderType::Anthropic,
            ProviderType::Google,
            ProviderType::OpenRouter,
            ProviderType::Custom,
        ]
        .into_iter()
        .find(|p| p.as_str() == name)
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
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_day: Option<u32>,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
    #[serde(default = "default_queue_timeout")]
    pub queue_timeout_seconds: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
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
        self.retry_delay_ms
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
    #[serde(
        alias = "omni",
        alias = "Omni",
        alias = "multimodal",
        alias = "Multimodal"
    )]
    Omni,
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
    #[serde(
        alias = "SpeechAnalysis",
        alias = "diarization",
        alias = "speaker_diarization"
    )]
    SpeechAnalysis,
    #[serde(
        alias = "decision",
        alias = "decisions",
        alias = "system1",
        alias = "system_one"
    )]
    Decision,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::VisionText => "vision-text",
            Modality::Omni => "omni",
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
            Modality::SpeechAnalysis => "speech-analysis",
            Modality::Decision => "decision",
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

#[derive(Debug, Clone)]
pub struct ModelRegistry {
    pub schema_version: u32,
    pub server: ServerConfig,
    pub models: Vec<ModelConfig>,
    pub roles: HashMap<String, String>,
    pub aliases: HashMap<String, String>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: ServerConfig::default(),
            models: Vec::new(),
            roles: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    pub fn from_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        server_path: P1,
        catalog_path: P2,
    ) -> Result<Self> {
        Self::from_files_with_env(server_path, catalog_path, &[])
    }

    pub fn from_files_with_env<P1: AsRef<Path>, P2: AsRef<Path>>(
        server_path: P1,
        catalog_path: P2,
        overrides: &[(String, String)],
    ) -> Result<Self> {
        let server_path = server_path.as_ref();
        let catalog_path = catalog_path.as_ref();

        let raw_server = std::fs::read_to_string(server_path)
            .with_context(|| format!("Failed to read server config: {}", server_path.display()))?;
        let raw_catalog = std::fs::read_to_string(catalog_path).with_context(|| {
            format!("Failed to read catalog config: {}", catalog_path.display())
        })?;

        let lookup = |name: &str| {
            overrides
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .or_else(|| std::env::var(name).ok())
        };

        let interp_server = interpolate_with(&raw_server, &lookup)?;
        let interp_catalog = interpolate_with(&raw_catalog, &lookup)?;

        let server_file: crate::schema::ServerFile = serde_json::from_str(&interp_server)
            .with_context(|| format!("Invalid server config at {}", server_path.display()))?;
        if server_file.schema_version != crate::schema::SCHEMA_VERSION {
            anyhow::bail!(
                "{} uses schema_version {} but rune-infer requires schema_version {}. Run `rune-infer migrate`.",
                server_path.display(),
                server_file.schema_version,
                crate::schema::SCHEMA_VERSION
            );
        }

        let catalog_file: crate::schema::CatalogFile = serde_json::from_str(&interp_catalog)
            .with_context(|| format!("Invalid catalog config at {}", catalog_path.display()))?;
        if catalog_file.schema_version != crate::schema::SCHEMA_VERSION {
            anyhow::bail!(
                "{} uses schema_version {} but rune-infer requires schema_version {}. Run `rune-infer migrate`.",
                catalog_path.display(),
                catalog_file.schema_version,
                crate::schema::SCHEMA_VERSION
            );
        }

        let models = catalog_file
            .models
            .into_iter()
            .map(ModelConfig::try_from)
            .collect::<Result<Vec<_>>>()?;

        let registry = Self {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: server_file.server,
            models,
            roles: catalog_file.roles,
            aliases: catalog_file.aliases,
        };
        registry.validate()?;
        Ok(registry)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file_with_env(path, &[])
    }

    /// Load a config, resolving `{{VAR}}` from `overrides` first and the
    /// process environment second (used to validate before `.env` is loaded).
    pub fn from_file_with_env<P: AsRef<Path>>(
        path: P,
        overrides: &[(String, String)],
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.file_name().is_some_and(|n| n == "server.json") {
            let cat = path.parent().unwrap_or(Path::new(".")).join("catalog.json");
            if cat.exists() {
                return Self::from_files_with_env(path, &cat, overrides);
            }
        } else if path.file_name().is_some_and(|n| n == "catalog.json") {
            let srv = path.parent().unwrap_or(Path::new(".")).join("server.json");
            if srv.exists() {
                return Self::from_files_with_env(&srv, path, overrides);
            }
        }

        let path_display = Some(path.display().to_string());
        let raw = std::fs::read_to_string(path).context("Failed to read config file")?;
        let interpolated = interpolate_with(&raw, &|name| {
            overrides
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
                .or_else(|| std::env::var(name).ok())
        })?;
        let registry = Self::from_json(&interpolated, path_display.as_deref())?;
        registry.validate()?;
        Ok(registry)
    }

    /// Parse a (placeholder-free) config document of the current schema.
    pub fn from_json(json: &str, origin: Option<&str>) -> Result<Self> {
        #[derive(Deserialize)]
        struct Version {
            #[serde(default)]
            schema_version: u32,
        }
        let version: Version =
            serde_json::from_str(json).context("Failed to parse config as JSON")?;
        if version.schema_version != crate::schema::SCHEMA_VERSION && version.schema_version != 2 {
            anyhow::bail!(
                "{} uses config schema_version {} but this rune-infer needs {}. Run `rune-infer migrate{}` to upgrade it (a backup is kept).",
                origin.unwrap_or("The config"),
                version.schema_version,
                crate::schema::SCHEMA_VERSION,
                origin
                    .map(|o| format!(" --config \"{o}\""))
                    .unwrap_or_default()
            );
        }
        let file: crate::schema::RegistryFile =
            serde_json::from_str(json).context("Invalid config")?;
        let models = file
            .models
            .into_iter()
            .map(ModelConfig::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: file.server,
            models,
            roles: file.roles,
            aliases: file.aliases,
        })
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

    fn clean_model_slug(s: &str) -> &str {
        let after_slash = s.split_once('/').map(|(_, r)| r).unwrap_or(s);
        after_slash
            .strip_suffix("-GGUF")
            .or_else(|| after_slash.strip_suffix("-gguf"))
            .or_else(|| after_slash.strip_suffix(".gguf"))
            .unwrap_or(after_slash)
    }

    /// Resolve a requested model name: exact id, then case-insensitive id or
    /// display name. If nothing matches, generic OpenAI-SDK default names
    /// ("text-embedding-3-small", "whisper-1", "tts-1", ...) fall back to the
    /// first configured model of the matching modality. There are no
    /// model- or provider-specific shortcuts: give each model its own id.
    pub fn find(&self, id: &str) -> Option<&ModelConfig> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .or_else(|| {
                // Check server-level roles and aliases
                if let Some(target) = self.roles.get(id).or_else(|| self.aliases.get(id)) {
                    self.models.iter().find(|m| m.id == *target)
                } else {
                    None
                }
            })
            .or_else(|| {
                // Check model-level roles and aliases
                self.models.iter().find(|m| {
                    m.roles.iter().any(|r| r.eq_ignore_ascii_case(id))
                        || m.aliases.iter().any(|a| a.eq_ignore_ascii_case(id))
                })
            })
            .or_else(|| {
                self.models
                    .iter()
                    .find(|m| m.id.eq_ignore_ascii_case(id) || m.name.eq_ignore_ascii_case(id))
            })
            .or_else(|| {
                let req_clean = Self::clean_model_slug(id);
                self.models.iter().find(|m| {
                    let m_clean = Self::clean_model_slug(&m.id);
                    m_clean.eq_ignore_ascii_case(req_clean)
                        || m.name.eq_ignore_ascii_case(req_clean)
                        || id.split_once('/').is_some_and(|(_, short)| {
                            m.id.eq_ignore_ascii_case(short) || m.name.eq_ignore_ascii_case(short)
                        })
                        || m.id
                            .split_once('/')
                            .is_some_and(|(_, short)| short.eq_ignore_ascii_case(id))
                })
            })
            .or_else(|| {
                let lower = id.to_lowercase();
                let has = |words: &[&str]| words.iter().any(|w| lower.contains(w));
                let modality = if has(&["embed"]) {
                    Modality::Embedding
                } else if has(&["moderation"]) {
                    Modality::Moderation
                } else if has(&["intent", "nlu"]) {
                    Modality::Nlu
                } else if has(&["ocr"]) {
                    Modality::Ocr
                } else if has(&["whisper", "transcription", "asr"]) {
                    Modality::SpeechToText
                } else if has(&["tts", "speech"]) {
                    Modality::TextToSpeech
                } else if has(&["s2st"]) {
                    Modality::SpeechToSpeech
                } else if has(&["detection"]) {
                    Modality::ObjectDetection
                } else if has(&["diar", "diariz", "nemotron", "speaker"]) {
                    Modality::SpeechAnalysis
                } else if has(&["decision", "system1", "d1-3b"]) {
                    Modality::Decision
                } else if has(&["omni", "multimodal"]) {
                    Modality::Omni
                } else {
                    return None;
                };
                self.models.iter().find(|m| m.modality == modality)
            })
    }

    pub fn find_owned(&self, id: &str) -> Option<ModelConfig> {
        self.find(id).cloned()
    }
}

pub fn interpolate_env_vars(raw: &str) -> Result<String> {
    interpolate_with(raw, &|name| std::env::var(name).ok())
}

/// Like [`interpolate_env_vars`] but resolving names through `lookup`.
pub fn interpolate_with(raw: &str, lookup: &dyn Fn(&str) -> Option<String>) -> Result<String> {
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
            match lookup(var_trimmed).ok_or(()) {
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

/// Inference backends a local model can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    /// llama.cpp `llama-server` (upstream build): GGUF text and vision models.
    LlamaServer,
    /// llama.cpp `llama-server` (Prism build): ternary Bonsai models.
    LlamaServerPrism,
    /// stable-diffusion.cpp `sd-server`: image generation.
    SdServer,
    /// audio.cpp: Audio8 speech models.
    AudioCpp,
    /// NeMo-Speech.cpp: Parakeet / FastConformer speech models.
    NemoSpeech,
    /// CrispASR: Whisper-family and other speech models.
    Crispasr,
    /// Native Rust encoders (embeddings, moderation, NLU, vision encoders).
    Encoder,
    /// Cactus Needle (`.cact`) tool-calling engine.
    Needle,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::LlamaServer => "llama-server",
            EngineKind::LlamaServerPrism => "llama-server-prism",
            EngineKind::SdServer => "sd-server",
            EngineKind::AudioCpp => "audio-cpp",
            EngineKind::NemoSpeech => "nemo-speech",
            EngineKind::Crispasr => "crispasr",
            EngineKind::Encoder => "encoder",
            EngineKind::Needle => "needle",
        }
    }
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which engines may serve a model and which one to try first.
///
/// ```json
/// "engine": { "supported": ["nemo-speech", "crispasr"], "preferred": "nemo-speech" }
/// ```
///
/// The preferred engine is tried first; if it fails to start, the remaining
/// `supported` engines are tried in the order listed. `preferred` defaults to
/// the first supported engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub supported: Vec<EngineKind>,
    #[serde(default)]
    pub preferred: Option<EngineKind>,
}

impl EngineConfig {
    pub fn validate(&self, model_id: &str) -> Result<()> {
        if self.supported.is_empty() {
            anyhow::bail!("Model '{model_id}': engine.supported must list at least one engine");
        }
        for (i, kind) in self.supported.iter().enumerate() {
            if self.supported[..i].contains(kind) {
                anyhow::bail!("Model '{model_id}': engine.supported lists '{kind}' twice");
            }
        }
        if let Some(pref) = self.preferred {
            if !self.supported.contains(&pref) {
                anyhow::bail!(
                    "Model '{model_id}': engine.preferred '{pref}' is not in engine.supported"
                );
            }
        }
        Ok(())
    }

    /// Engines in the order they should be tried.
    pub fn plan(&self) -> Vec<EngineKind> {
        let first = self.preferred.unwrap_or(self.supported[0]);
        std::iter::once(first)
            .chain(self.supported.iter().copied().filter(|k| *k != first))
            .collect()
    }
}

/// Default sampling parameters for a model. Every field is optional; a value
/// supplied in the request wins, then this config, then the engine's own
/// built-in default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

impl SamplingConfig {
    /// `self` (per-request values) layered over `defaults` (model config).
    pub fn over(&self, defaults: &SamplingConfig) -> SamplingConfig {
        SamplingConfig {
            temperature: self.temperature.or(defaults.temperature),
            top_p: self.top_p.or(defaults.top_p),
            top_k: self.top_k.or(defaults.top_k),
            min_p: self.min_p.or(defaults.min_p),
            max_tokens: self.max_tokens.or(defaults.max_tokens),
            reasoning_effort: self
                .reasoning_effort
                .clone()
                .or_else(|| defaults.reasoning_effort.clone()),
        }
    }

    pub fn validate(&self, model_id: &str) -> Result<()> {
        let bad = |what: &str| anyhow::bail!("Model '{model_id}': invalid sampling.{what}");
        if self.temperature.is_some_and(|t| !(0.0..=5.0).contains(&t)) {
            return bad("temperature (expected 0.0 to 5.0)");
        }
        if self.top_p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
            return bad("top_p (expected greater than 0 and at most 1)");
        }
        if self.min_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return bad("min_p (expected 0.0 to 1.0)");
        }
        if self.max_tokens == Some(0) {
            return bad("max_tokens (must be at least 1)");
        }
        if let Some(ref effort) = self.reasoning_effort {
            let valid = ["none", "low", "medium", "high", "disabled", "off"];
            if !valid.contains(&effort.to_lowercase().as_str()) {
                return bad("reasoning_effort (expected one of: none, low, medium, high)");
            }
        }
        Ok(())
    }
}

/// Engine process settings for a local model (the file's `local` block).
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Context window in tokens; `-1` = automatic.
    pub context_length: i64,
    pub gpu_layers: u32,
    pub extra_args: Option<String>,
}

/// A LoRA adapter listed under `extra.lora`. Parsed and validated, but not
/// applied by any engine yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraConfig {
    pub name: String,
    pub path: String,
    #[serde(default = "default_lora_scale")]
    pub scale: f32,
}

fn default_lora_scale() -> f32 {
    1.0
}

/// A model as the rest of the server sees it, flattened from the file's
/// [`ModelEntry`](crate::schema::ModelEntry).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    pub provider: ProviderType,
    pub modality: Modality,
    pub model_path: String,
    pub upstream_model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub rate_limit: Option<RateLimitConfig>,
    pub mmproj_path: Option<String>,
    pub mtp_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub llm_vision_path: Option<String>,
    pub default_steps: Option<u32>,
    pub default_cfg_scale: Option<f32>,
    pub default_sample_method: Option<String>,
    pub description: String,
    pub bits_per_weight: Option<f64>,
    pub total_params: Option<u64>,
    pub max_context_length: Option<u32>,
    pub max_resolution: Option<String>,
    pub mtp_heads: Option<u32>,
    pub capabilities: Option<Vec<String>>,
    pub sampling: Option<SamplingConfig>,
    /// `Some` for local models, `None` for cloud models.
    pub engine: Option<EngineConfig>,
    pub lora: Vec<LoraConfig>,
    pub runtime: Option<RuntimeConfig>,
    pub aliases: Vec<String>,
    pub roles: Vec<String>,
}

impl ModelConfig {
    pub fn is_local(&self) -> bool {
        self.provider == ProviderType::Local
    }

    pub fn locality(&self) -> &'static str {
        if self.is_local() { "local" } else { "cloud" }
    }

    pub fn modalities_in(&self) -> Vec<String> {
        let mut list = Vec::new();
        match self.modality {
            Modality::Omni => {
                list.push("text".to_string());
                list.push("image".to_string());
                list.push("audio".to_string());
                list.push("video".to_string());
            }
            Modality::VisionText => {
                list.push("text".to_string());
                list.push("image".to_string());
            }
            Modality::SpeechToText | Modality::SpeechAnalysis => {
                list.push("audio".to_string());
            }
            Modality::SpeechToSpeech => {
                list.push("audio".to_string());
            }
            Modality::ImageGeneration => {
                list.push("text".to_string());
                list.push("image".to_string());
            }
            Modality::Text
            | Modality::TextToSpeech
            | Modality::Embedding
            | Modality::Moderation
            | Modality::Nlu
            | Modality::Decision => {
                list.push("text".to_string());
                if self.is_vision() {
                    list.push("image".to_string());
                }
            }
            Modality::Ocr | Modality::ObjectDetection => {
                list.push("image".to_string());
            }
            _ => {
                list.push("image".to_string());
            }
        }
        list
    }

    pub fn modalities_out(&self) -> Vec<String> {
        let mut list = Vec::new();
        match self.modality {
            Modality::Omni => {
                list.push("text".to_string());
                list.push("audio".to_string());
            }
            Modality::TextToSpeech | Modality::SpeechToSpeech => {
                list.push("audio".to_string());
            }
            Modality::ImageGeneration => {
                list.push("image".to_string());
            }
            Modality::SpeechToText
            | Modality::Text
            | Modality::VisionText
            | Modality::Ocr
            | Modality::Nlu
            | Modality::Decision
            | Modality::Moderation
            | Modality::SpeechAnalysis => {
                list.push("text".to_string());
            }
            Modality::Embedding => {
                list.push("embedding".to_string());
            }
            Modality::ObjectDetection => {
                list.push("box_2d".to_string());
            }
            _ => {
                list.push("image".to_string());
            }
        }
        list
    }

    pub fn supports_tool_calls(&self) -> bool {
        self.capabilities
            .as_ref()
            .map(|caps| {
                caps.iter().any(|c| {
                    c.eq_ignore_ascii_case("tool call")
                        || c.eq_ignore_ascii_case("tool_calls")
                        || c.eq_ignore_ascii_case("tools")
                })
            })
            .unwrap_or(false)
    }

    pub fn thinking_control(&self) -> &'static str {
        let name_lower = self.name.to_lowercase();
        let id_lower = self.id.to_lowercase();
        if id_lower.contains("qwen")
            || name_lower.contains("qwen")
            || id_lower.contains("deepseek")
            || name_lower.contains("deepseek")
            || id_lower.contains("gemma")
            || name_lower.contains("gemma")
            || id_lower.contains("thinking")
            || name_lower.contains("thinking")
            || id_lower.contains("reasoning")
            || name_lower.contains("reasoning")
        {
            "thinking"
        } else if self.is_local() {
            "chat_template_kwargs"
        } else {
            "reasoning_effort"
        }
    }

    pub fn quantization(&self) -> Option<String> {
        let path_or_id = format!("{} {}", self.model_path, self.id);
        let path_upper = path_or_id.to_uppercase();
        for q in &[
            "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q8_0", "Q8_K",
            "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ2_XS", "IQ2_XXS", "BF16", "FP16", "F16",
            "FP32", "F32",
        ] {
            if path_upper.contains(q) {
                return Some(q.to_string());
            }
        }
        if let Some(bpw) = self.bits_per_weight {
            return Some(format!("{:.1}bpw", bpw));
        }
        None
    }

    pub fn vram_mb_estimate(&self) -> Option<u64> {
        if !self.is_local() {
            return None;
        }
        let plan = crate::inference::budget::calculate_budget(self, &ServerConfig::default());
        Some(plan.total_estimated_vram_bytes() / (1024 * 1024))
    }

    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            anyhow::bail!("Model '{}' has empty id", self.name);
        }

        if let Some(sampling) = &self.sampling {
            sampling.validate(&self.id)?;
        }
        if let Some(engine) = &self.engine {
            engine.validate(&self.id)?;
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

    /// Engines to try, in order: the configured preference first, then the other
    /// supported engines. Empty for cloud models.
    pub fn engine_plan(&self) -> Vec<EngineKind> {
        self.engine
            .as_ref()
            .map(EngineConfig::plan)
            .unwrap_or_default()
    }

    /// Whether the model accepts image input.
    pub fn is_vision(&self) -> bool {
        self.modality == Modality::VisionText
            || self.modality == Modality::Omni
            || (self.modality == Modality::Decision
                && self
                    .capabilities
                    .as_ref()
                    .map(|caps| caps.iter().any(|c| c.eq_ignore_ascii_case("vision")))
                    .unwrap_or(false))
    }

    /// A local llama.cpp model can only take image or audio input with an `mmproj`
    /// file. True when the model is served by llama-server but none is configured
    /// or the configured file does not exist.
    pub fn missing_projector(&self) -> bool {
        let llama = matches!(
            self.engine_plan().first(),
            Some(EngineKind::LlamaServer | EngineKind::LlamaServerPrism)
        );
        llama
            && !self
                .mmproj_path
                .as_deref()
                .is_some_and(|p| !p.trim().is_empty() && Path::new(p).exists())
    }

    /// Whether the model accepts audio input.
    pub fn supports_audio(&self) -> bool {
        self.modality == Modality::Omni
            || self.modality == Modality::SpeechToText
            || self.modality == Modality::SpeechAnalysis
            || self.modality == Modality::SpeechToSpeech
            || self
                .capabilities
                .as_ref()
                .map(|caps| {
                    caps.iter().any(|c| {
                        let lc = c.to_ascii_lowercase();
                        lc == "audio"
                            || lc == "speech"
                            || lc == "speech-to-text"
                            || lc == "audio-input"
                            || lc == "omni"
                            || lc == "multimodal"
                    })
                })
                .unwrap_or(false)
    }

    /// Whether the model is an omni multimodal model (text, image, audio, video).
    pub fn is_omni(&self) -> bool {
        self.modality == Modality::Omni
            || self
                .capabilities
                .as_ref()
                .map(|caps| {
                    caps.iter().any(|c| {
                        let lc = c.to_ascii_lowercase();
                        lc == "omni" || lc == "multimodal"
                    })
                })
                .unwrap_or(false)
    }

    /// Whether the model is a fast System 1 decision model.
    pub fn is_decision(&self) -> bool {
        self.modality == Modality::Decision
    }

    /// Per-request sampling values layered over this model's configured defaults.
    pub fn effective_sampling(&self, request: &SamplingConfig) -> SamplingConfig {
        match &self.sampling {
            Some(defaults) => request.over(defaults),
            None => request.clone(),
        }
    }

    pub fn mtp_heads(&self) -> Option<u32> {
        self.mtp_heads
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
        if self.is_vision() {
            caps.push("Vision".to_string());
        }
        if self.is_omni() {
            if !caps.iter().any(|c| c.eq_ignore_ascii_case("audio")) {
                caps.push("Audio".to_string());
            }
            if !caps.iter().any(|c| c.eq_ignore_ascii_case("video")) {
                caps.push("Video".to_string());
            }
            if !caps.iter().any(|c| c.eq_ignore_ascii_case("omni")) {
                caps.push("Omni".to_string());
            }
        }
        if self.has_mtp() && !caps.iter().any(|c| c.eq_ignore_ascii_case("mtp")) {
            caps.push("MTP".to_string());
        }
        if self.modality == Modality::ImageGeneration
            && !caps.iter().any(|c| c.eq_ignore_ascii_case("text-to-image"))
        {
            caps.push("Text-to-Image".to_string());
        }
        if self.modality == Modality::SpeechToText
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ModelEntry;

    fn try_model(json: &str) -> Result<ModelConfig> {
        ModelConfig::try_from(serde_json::from_str::<ModelEntry>(json)?)
    }

    /// A valid local entry; `more` is spliced in as extra top-level keys.
    fn local(more: &str) -> Result<ModelConfig> {
        try_model(&format!(
            r#"{{"id":"m","name":"M","modality":"Text",
                "engine":{{"supported":["llama-server"]}},
                "local":{{"path":"m.gguf"}}{more}}}"#
        ))
    }

    fn err(r: Result<ModelConfig>) -> String {
        format!("{:#}", r.unwrap_err())
    }

    // ---- sampling

    #[test]
    fn request_overrides_config_which_overrides_nothing() {
        let m =
            local(r#","sampling":{"temperature":0.2,"top_p":0.9,"top_k":20,"max_tokens":2048}"#)
                .unwrap();
        let eff = m.effective_sampling(&SamplingConfig {
            temperature: Some(1.1),
            ..Default::default()
        });
        assert_eq!(eff.temperature, Some(1.1), "request wins");
        assert_eq!(eff.top_p, Some(0.9), "config fills the gap");
        assert_eq!(
            (eff.top_k, eff.max_tokens, eff.min_p),
            (Some(20), Some(2048), None)
        );

        let bare = local("").unwrap();
        assert!(
            bare.sampling.is_none(),
            "an empty sampling block means no defaults"
        );
        assert_eq!(
            bare.effective_sampling(&SamplingConfig::default()),
            SamplingConfig::default()
        );
    }

    #[test]
    fn nonsense_sampling_values_are_rejected_at_load() {
        for bad in [
            r#"{"temperature":-1.0}"#,
            r#"{"top_p":0.0}"#,
            r#"{"top_p":1.5}"#,
            r#"{"min_p":2.0}"#,
            r#"{"max_tokens":0}"#,
        ] {
            assert!(local(&format!(r#","sampling":{bad}"#)).is_err(), "{bad}");
        }
        assert!(local(r#","sampling":{"temperature":0.0,"top_p":1.0}"#).is_ok());
    }

    // ---- engine

    #[test]
    fn engine_object_orders_the_fallbacks() {
        let m = try_model(
            r#"{"id":"p","name":"n","modality":"SpeechToText","local":{"path":"m"},
                "engine":{"supported":["crispasr","nemo-speech","audio-cpp"],"preferred":"nemo-speech"}}"#,
        )
        .unwrap();
        assert_eq!(
            m.engine_plan(),
            vec![
                EngineKind::NemoSpeech,
                EngineKind::Crispasr,
                EngineKind::AudioCpp
            ],
            "preferred first, then the rest in listed order"
        );
        let no_pref = try_model(
            r#"{"id":"m","name":"n","modality":"Text","local":{"path":"m"},
                "engine":{"supported":["llama-server-prism","llama-server"]}}"#,
        )
        .unwrap();
        assert_eq!(
            no_pref.engine_plan()[0],
            EngineKind::LlamaServerPrism,
            "defaults to first supported"
        );
    }

    #[test]
    fn invalid_engine_objects_are_rejected_at_load() {
        let bad = |engine: &str| {
            err(try_model(&format!(
                r#"{{"id":"m","name":"n","modality":"Text","local":{{"path":"m"}},"engine":{engine}}}"#
            )))
        };
        assert!(bad(r#"{"supported":[]}"#).contains("at least one"));
        assert!(bad(r#"{"supported":["crispasr","crispasr"]}"#).contains("twice"));
        assert!(
            bad(r#"{"supported":["crispasr"],"preferred":"nemo-speech"}"#)
                .contains("not in engine.supported")
        );
        let unknown = bad(r#"{"supported":["fast-engine"]}"#);
        assert!(
            unknown.contains("fast-engine") && unknown.contains("llama-server"),
            "{unknown}"
        );
    }

    #[test]
    fn local_models_must_name_their_engine() {
        let e = err(try_model(
            r#"{"id":"m","name":"n","modality":"Text","local":{"path":"m"}}"#,
        ));
        assert!(e.contains("'engine' is required"), "{e}");
    }

    // ---- structure

    const REMOTE: &str = r#""remote":{"provider":"google","upstream_model":"x","api_key":"k"}"#;

    #[test]
    fn exactly_one_of_local_or_remote() {
        let both = err(try_model(&format!(
            r#"{{"id":"m","name":"n","modality":"Text","engine":{{"supported":["llama-server"]}},"local":{{"path":"p"}},{REMOTE}}}"#
        )));
        assert!(both.contains("not both"), "{both}");
        let neither = err(try_model(r#"{"id":"m","name":"n","modality":"Text"}"#));
        assert!(neither.contains("needs a 'local' block"), "{neither}");
    }

    #[test]
    fn cloud_models_take_no_engine_and_no_extra() {
        let ok = try_model(&format!(
            r#"{{"id":"g","name":"n","modality":"Text",{REMOTE}}}"#
        ))
        .unwrap();
        assert!(!ok.is_local() && ok.engine.is_none() && ok.engine_plan().is_empty());
        let with_engine = err(try_model(&format!(
            r#"{{"id":"g","name":"n","modality":"Text","engine":{{"supported":["encoder"]}},{REMOTE}}}"#
        )));
        assert!(
            with_engine.contains("only applies to local"),
            "{with_engine}"
        );
        let with_extra = err(try_model(&format!(
            r#"{{"id":"g","name":"n","modality":"Text",{REMOTE},"extra":{{"mmproj_path":"x"}}}}"#
        )));
        assert!(
            with_extra.contains("not supported on cloud"),
            "{with_extra}"
        );
    }

    #[test]
    fn remote_blocks_reject_local_provider_and_blank_keys() {
        let e = err(try_model(
            r#"{"id":"g","name":"n","modality":"Text","remote":{"provider":"local","upstream_model":"x","api_key":"k"}}"#,
        ));
        assert!(e.contains("cannot be 'local'"), "{e}");
        let blank = err(try_model(
            r#"{"id":"g","name":"n","modality":"Text","remote":{"provider":"google","upstream_model":"x","api_key":" "}}"#,
        ));
        assert!(blank.contains("api_key"), "{blank}");
    }

    #[test]
    fn unknown_keys_are_errors_not_silently_ignored() {
        let top = err(local(r#","vison":true"#));
        assert!(top.contains("vison"), "{top}");
        let block = err(try_model(
            r#"{"id":"m","name":"n","modality":"Text","engine":{"supported":["encoder"]},"local":{"path":"p","gpu_layer":1}}"#,
        ));
        assert!(block.contains("gpu_layer"), "{block}");
    }

    // ---- extra

    #[test]
    fn extra_is_validated_against_the_models_engines() {
        let m = local(r#","extra":{"mmproj_path":"mm.gguf","mtp_heads":2}"#).unwrap();
        assert_eq!(
            (m.mmproj_path.as_deref(), m.mtp_heads()),
            (Some("mm.gguf"), Some(2))
        );

        let wrong_engine = err(local(r#","extra":{"vae_path":"v.safetensors"}"#));
        assert!(
            wrong_engine.contains("extra.vae_path") && wrong_engine.contains("mmproj_path"),
            "{wrong_engine}"
        );

        let sd = try_model(
            r#"{"id":"z","name":"n","modality":"ImageGeneration","engine":{"supported":["sd-server"]},
                "local":{"path":"z"},"extra":{"vae_path":"v","default_steps":8,
                "lora":[{"name":"a","path":"a.safetensors","scale":0.5}]}}"#,
        )
        .unwrap();
        assert_eq!(
            (sd.default_steps, sd.lora.len(), sd.lora[0].scale),
            (Some(8), 1, 0.5)
        );

        let none_allowed = err(try_model(
            r#"{"id":"w","name":"n","modality":"SpeechToText","engine":{"supported":["crispasr"]},
                "local":{"path":"w"},"extra":{"mmproj_path":"x"}}"#,
        ));
        assert!(
            none_allowed.contains("Allowed keys: none"),
            "{none_allowed}"
        );

        let bad_type = err(local(r#","extra":{"mtp_heads":"two"}"#));
        assert!(bad_type.contains("invalid extra"), "{bad_type}");
    }

    #[test]
    fn local_block_defaults_are_explicit_and_sane() {
        let m = local("").unwrap();
        let rt = m.runtime.unwrap();
        assert_eq!((rt.context_length, rt.gpu_layers), (-1, 99));
        assert!(
            rt.extra_args.is_none(),
            "empty args are not passed to the engine"
        );
        let with_args = try_model(
            r#"{"id":"m","name":"n","modality":"Text","engine":{"supported":["llama-server"]},
                "local":{"path":"p","args":"-fa on","context_length":4096,"gpu_layers":0}}"#,
        )
        .unwrap();
        let rt = with_args.runtime.unwrap();
        assert_eq!(
            (rt.context_length, rt.gpu_layers, rt.extra_args.as_deref()),
            (4096, 0, Some("-fa on"))
        );
    }

    #[test]
    fn vision_comes_from_the_modality_only() {
        let v = try_model(
            r#"{"id":"v","name":"n","modality":"VisionText","engine":{"supported":["llama-server"]},"local":{"path":"p"}}"#,
        )
        .unwrap();
        assert!(v.is_vision() && v.resolved_capabilities().contains(&"Vision".to_string()));
        assert!(!local("").unwrap().is_vision());
    }

    // ---- model lookup

    fn registry(models: &[(&str, &str, &str)]) -> ModelRegistry {
        let mut r = ModelRegistry::new();
        for (id, name, modality) in models {
            r.models.push(
                try_model(&format!(
                    r#"{{"id":"{id}","name":"{name}","modality":"{modality}",
                        "engine":{{"supported":["encoder"]}},"local":{{"path":"p"}}}}"#
                ))
                .unwrap(),
            );
        }
        r
    }

    #[test]
    fn find_uses_ids_and_names_and_no_model_specific_shortcuts() {
        let r = registry(&[
            ("qwen-9b", "Qwen 9B", "Text"),
            ("bge-small", "BGE Small", "Embedding"),
            ("gemini-flash", "Gemini Flash", "Text"),
        ]);
        assert_eq!(r.find("qwen-9b").unwrap().id, "qwen-9b");
        assert_eq!(
            r.find("QWEN-9B").unwrap().id,
            "qwen-9b",
            "case-insensitive id"
        );
        assert_eq!(r.find("Qwen 9B").unwrap().id, "qwen-9b", "display name");
        assert!(r.find("claude-3-opus").is_none(), "no vendor shortcuts");
        assert!(r.find("gemini").is_none(), "partial names do not match");
        assert!(r.find("gpt-4o").is_none());
    }

    #[test]
    fn find_keeps_only_generic_modality_fallbacks_for_sdk_default_names() {
        let r = registry(&[
            ("a", "A", "Text"),
            ("emb", "E", "Embedding"),
            ("stt", "S", "SpeechToText"),
            ("tts", "T", "TextToSpeech"),
        ]);
        assert_eq!(r.find("text-embedding-3-small").unwrap().id, "emb");
        assert_eq!(r.find("whisper-1").unwrap().id, "stt");
        assert_eq!(r.find("tts-1").unwrap().id, "tts");
        assert!(
            r.find("moderation-latest").is_none(),
            "no model of that modality configured"
        );
    }

    #[test]
    fn omni_modality_capabilities_and_find_resolution() {
        let r = registry(&[("gemma-4-12b-it", "Gemma 4 12B IT", "Omni")]);
        let m = r.find("gemma-4-12b-it").expect("finds by exact id");
        assert!(m.is_omni());
        assert!(m.is_vision());
        assert!(m.supports_audio());
        let caps = m.resolved_capabilities();
        assert!(caps.contains(&"Vision".to_string()));
        assert!(caps.contains(&"Audio".to_string()));
        assert!(caps.contains(&"Video".to_string()));
        assert!(caps.contains(&"Omni".to_string()));

        // Slug / repo resolution tests
        assert_eq!(
            r.find("unsloth/gemma-4-12b-it-GGUF").unwrap().id,
            "gemma-4-12b-it"
        );
        assert_eq!(
            r.find("unsloth/gemma-4-12b-it").unwrap().id,
            "gemma-4-12b-it"
        );
        assert_eq!(r.find("gemma-4-12b-it-GGUF").unwrap().id, "gemma-4-12b-it");
    }

    // ---- loader

    #[test]
    fn rate_limit_blocks_reject_the_removed_seconds_key() {
        let e = err(try_model(
            r#"{"id":"g","name":"n","modality":"Text","remote":{"provider":"google","upstream_model":"x",
                "api_key":"k","rate_limit":{"retry_delay_seconds":2}}}"#,
        ));
        assert!(e.contains("retry_delay_seconds"), "{e}");
    }

    #[test]
    fn the_repo_config_loads() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config");
        let server_path = dir.join("server.json");
        let catalog_path = dir.join("catalog.json");
        let reg = ModelRegistry::from_files_with_env(
            &server_path,
            &catalog_path,
            &[
                ("GEMINI_API_KEY".into(), "x".into()),
                ("OPENROUTER_API_KEY".into(), "x".into()),
            ],
        )
        .unwrap_or_else(|e| panic!("{e:#}"));
        assert!(
            reg.models
                .iter()
                .filter(|m| m.is_local())
                .all(|m| !m.engine_plan().is_empty())
        );
    }
}
