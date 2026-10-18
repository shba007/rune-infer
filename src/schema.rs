//! The on-disk model catalog format (`schema_version: 2`).
//!
//! Every model has the same core keys. Local models carry a `local` block,
//! cloud models a `remote` block, and anything specific to one engine lives in
//! the free-form `extra` object, which is validated against the engines the
//! model supports so typos are caught when the config loads.
//!
//! ```json
//! {
//!   "id": "qwen3.8-27b", "name": "Qwen3.8 27B", "description": "",
//!   "modality": "VisionText",
//!   "capabilities": ["Chat", "Vision"],
//!   "engine": { "supported": ["llama-server"], "preferred": "llama-server" },
//!   "max_context_length": 262144,
//!   "sampling": { "temperature": 0.6 },
//!   "local": { "path": "D:/Models/q.gguf", "params": null, "bits_per_weight": null,
//!              "context_length": -1, "gpu_layers": 99, "args": "" },
//!   "extra": { "mmproj_path": "D:/Models/mmproj.gguf" }
//! }
//! ```
//!
//! The loader converts these entries into the flat [`ModelConfig`] the rest of
//! the server uses; `rune-infer migrate` upgrades a v1 file.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config::{
    EngineConfig, EngineKind, LoraConfig, Modality, ModelConfig, ProviderType, RateLimitConfig,
    RuntimeConfig, SamplingConfig, ServerConfig,
};

pub const SCHEMA_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFile {
    pub schema_version: u32,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogFile {
    pub schema_version: u32,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryFile {
    pub schema_version: u32,
    pub server: ServerConfig,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub modality: Modality,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Required for local models, forbidden for cloud models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineConfig>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub sampling: SamplingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteBlock>,
    /// Engine-specific settings; see [`allowed_extra_keys`].
    #[serde(default)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalBlock {
    pub path: String,
    #[serde(default)]
    pub params: Option<u64>,
    #[serde(default)]
    pub bits_per_weight: Option<f64>,
    /// Context window in tokens, `-1` = chosen automatically from free VRAM.
    #[serde(default = "default_context_length")]
    pub context_length: i64,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    /// Extra command-line arguments passed verbatim to the engine process.
    #[serde(default)]
    pub args: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteBlock {
    pub provider: ProviderType,
    pub upstream_model: String,
    pub api_key: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
}

fn default_context_length() -> i64 {
    -1
}

fn default_gpu_layers() -> u32 {
    99
}

/// The `extra` keys each engine understands.
pub fn allowed_extra_keys(kind: EngineKind) -> &'static [&'static str] {
    match kind {
        EngineKind::LlamaServer | EngineKind::LlamaServerPrism => {
            &["mmproj_path", "mtp_path", "mtp_heads", "max_resolution"]
        }
        EngineKind::SdServer => &[
            "vae_path",
            "text_encoder_path",
            "llm_vision_path",
            "default_steps",
            "default_cfg_scale",
            "default_sample_method",
            "max_resolution",
            "lora",
        ],
        EngineKind::AudioCpp
        | EngineKind::NemoSpeech
        | EngineKind::Crispasr
        | EngineKind::Encoder
        | EngineKind::Needle => &[],
    }
}

/// Typed view of `extra` (every key optional).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtraFields {
    mmproj_path: Option<String>,
    mtp_path: Option<String>,
    mtp_heads: Option<u32>,
    vae_path: Option<String>,
    text_encoder_path: Option<String>,
    llm_vision_path: Option<String>,
    default_steps: Option<u32>,
    default_cfg_scale: Option<f32>,
    default_sample_method: Option<String>,
    max_resolution: Option<String>,
    #[serde(default)]
    lora: Vec<LoraConfig>,
}

fn check_extra_keys(id: &str, supported: &[EngineKind], extra: &Map<String, Value>) -> Result<()> {
    for key in extra.keys() {
        if !supported
            .iter()
            .any(|k| allowed_extra_keys(*k).contains(&key.as_str()))
        {
            let mut allowed: Vec<&str> = supported
                .iter()
                .flat_map(|k| allowed_extra_keys(*k).iter().copied())
                .collect();
            allowed.sort_unstable();
            allowed.dedup();
            let engines: Vec<String> = supported.iter().map(|k| k.to_string()).collect();
            anyhow::bail!(
                "Model '{id}': extra.{key} is not used by its engine(s) [{}]. Allowed keys: {}",
                engines.join(", "),
                if allowed.is_empty() {
                    "none".to_string()
                } else {
                    allowed.join(", ")
                }
            );
        }
    }
    Ok(())
}

impl TryFrom<ModelEntry> for ModelConfig {
    type Error = anyhow::Error;

    fn try_from(e: ModelEntry) -> Result<Self> {
        let id = e.id.clone();
        let (
            provider,
            model_path,
            upstream_model,
            api_key,
            base_url,
            rate_limit,
            runtime,
            params,
            bpw,
        ) = match (e.local, e.remote) {
            (Some(l), None) => (
                ProviderType::Local,
                l.path,
                None,
                None,
                None,
                None,
                Some(RuntimeConfig {
                    context_length: l.context_length,
                    gpu_layers: l.gpu_layers,
                    extra_args: (!l.args.trim().is_empty()).then_some(l.args),
                }),
                l.params,
                l.bits_per_weight,
            ),
            (None, Some(r)) => {
                if r.provider == ProviderType::Local {
                    anyhow::bail!("Model '{id}': remote.provider cannot be 'local'");
                }
                (
                    r.provider,
                    String::new(),
                    Some(r.upstream_model),
                    Some(r.api_key),
                    r.base_url,
                    r.rate_limit,
                    None,
                    None,
                    None,
                )
            }
            (Some(_), Some(_)) => {
                anyhow::bail!("Model '{id}': give either a 'local' or a 'remote' block, not both")
            }
            (None, None) => anyhow::bail!(
                "Model '{id}': needs a 'local' block (own weights) or a 'remote' block (cloud provider)"
            ),
        };

        let engine = match (provider == ProviderType::Local, e.engine) {
            (true, Some(cfg)) => Some(cfg),
            (true, None) => anyhow::bail!(
                "Model '{id}': 'engine' is required for local models, e.g. \
                 \"engine\": {{ \"supported\": [\"llama-server\"] }}"
            ),
            (false, Some(_)) => anyhow::bail!(
                "Model '{id}': 'engine' only applies to local models (the provider adapter handles cloud models)"
            ),
            (false, None) => None,
        };

        match &engine {
            Some(cfg) => check_extra_keys(&id, &cfg.supported, &e.extra)?,
            None if !e.extra.is_empty() => {
                anyhow::bail!("Model '{id}': 'extra' is not supported on cloud models")
            }
            None => {}
        }
        let extra: ExtraFields = serde_json::from_value(Value::Object(e.extra))
            .map_err(|err| anyhow::anyhow!("Model '{id}': invalid extra: {err}"))?;

        let sampling = e.sampling;
        let model = ModelConfig {
            id: e.id,
            name: e.name,
            provider,
            modality: e.modality,
            model_path,
            upstream_model,
            api_key,
            base_url,
            rate_limit,
            mmproj_path: extra.mmproj_path,
            mtp_path: extra.mtp_path,
            vae_path: extra.vae_path,
            text_encoder_path: extra.text_encoder_path,
            llm_vision_path: extra.llm_vision_path,
            default_steps: extra.default_steps,
            default_cfg_scale: extra.default_cfg_scale,
            default_sample_method: extra.default_sample_method,
            description: e.description,
            bits_per_weight: bpw,
            total_params: params,
            max_context_length: e.max_context_length,
            max_resolution: extra.max_resolution,
            mtp_heads: extra.mtp_heads,
            capabilities: (!e.capabilities.is_empty()).then_some(e.capabilities),
            sampling: (sampling != SamplingConfig::default()).then_some(sampling),
            engine,
            lora: extra.lora,
            runtime,
        };
        model.validate()?;
        Ok(model)
    }
}
