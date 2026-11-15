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
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub roles: std::collections::HashMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub aliases: std::collections::HashMap<String, String>,
}

const CATALOG_ROOT_KEYS: &[&str] = &["schema_version", "models", "roles", "aliases"];
const CATALOG_MODEL_KEYS: &[&str] = &[
    "id",
    "name",
    "description",
    "capabilities",
    "modality",
    "tasks",
    "aliases",
    "roles",
    "engine",
    "max_context_length",
    "sampling",
    "local",
    "remote",
    "extra",
    "audio_operations",
    "image_reference_checks",
    "structured_output_support",
    "tool_call_support",
    "benchmarks",
];
const CATALOG_LOCAL_KEYS: &[&str] = &[
    "path",
    "params",
    "bits_per_weight",
    "context_length",
    "gpu_layers",
    "args",
];
const CATALOG_SAMPLING_KEYS: &[&str] = &["temperature", "top_p", "top_k", "min_p", "max_tokens"];
const CATALOG_REMOTE_KEYS: &[&str] = &[
    "provider",
    "upstream_model",
    "api_key",
    "base_url",
    "rate_limit",
];
const CATALOG_RATE_LIMIT_KEYS: &[&str] = &[
    "requests_per_minute",
    "tokens_per_minute",
    "requests_per_day",
    "tokens_per_day",
    "max_concurrent",
    "retry_delay_ms",
    "max_retries",
    "queue_timeout_seconds",
    "retry_timeout_seconds",
];
const CATALOG_BENCHMARK_KEYS: &[&str] = &[
    "index_name",
    "score",
    "unit",
    "mode",
    "reasoning_effort",
    "domains",
    "cost_per_task_usd",
    "time_per_task_minutes",
    "output_tokens_per_task",
];
const CATALOG_DOMAIN_KEYS: &[&str] = &[
    "finance_and_accounting",
    "strategy_and_ops",
    "legal",
    "healthcare_and_medical",
    "engineering",
    "economics",
];

fn catalog_key_order(path: &[&str]) -> &'static [&'static str] {
    match path {
        [] => CATALOG_ROOT_KEYS,
        ["models", _] => CATALOG_MODEL_KEYS,
        ["models", _, "local"] => CATALOG_LOCAL_KEYS,
        ["models", _, "sampling"] => CATALOG_SAMPLING_KEYS,
        ["models", _, "remote"] => CATALOG_REMOTE_KEYS,
        ["models", _, "remote", "rate_limit"] => CATALOG_RATE_LIMIT_KEYS,
        ["models", _, "benchmarks"] => CATALOG_BENCHMARK_KEYS,
        ["models", _, "benchmarks", "domains"] => CATALOG_DOMAIN_KEYS,
        _ => &[],
    }
}

fn write_catalog_value(out: &mut String, value: &Value, path: &mut Vec<String>, depth: usize) {
    let pad = |out: &mut String, level: usize| out.push_str(&"  ".repeat(level));
    match value {
        Value::Object(map) if !map.is_empty() => {
            let route: Vec<&str> = path.iter().map(String::as_str).collect();
            let order = catalog_key_order(&route);
            let mut keys: Vec<&String> = map.keys().collect();
            // Listed keys first in their canonical order; the rest stay alphabetical.
            keys.sort_by_key(|key| {
                order
                    .iter()
                    .position(|known| known == key)
                    .unwrap_or(usize::MAX)
            });
            out.push_str("{\n");
            for (index, key) in keys.iter().enumerate() {
                pad(out, depth + 1);
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push_str(": ");
                path.push((*key).clone());
                write_catalog_value(out, &map[*key], path, depth + 1);
                path.pop();
                out.push_str(if index + 1 < keys.len() { ",\n" } else { "\n" });
            }
            pad(out, depth);
            out.push('}');
        }
        Value::Array(items) if !items.is_empty() => {
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                pad(out, depth + 1);
                // Array positions are unnamed in the key-order lookup, except models.
                path.push(index.to_string());
                write_catalog_value(out, item, path, depth + 1);
                path.pop();
                out.push_str(if index + 1 < items.len() { ",\n" } else { "\n" });
            }
            pad(out, depth);
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// Render a catalog document in the canonical, hand-maintained layout.
///
/// `serde_json::Value` objects are key-sorted, so writers that edit the raw JSON
/// used to emit models with alphabetical keys. This prints the same data with
/// each model's fields in schema order (`id`, `name`, ..., `tasks`, `engine`,
/// `local`, `extra`, ...). Values are never altered, only their order.
pub fn render_catalog(value: &Value) -> Result<String> {
    let mut out = String::new();
    write_catalog_value(&mut out, value, &mut Vec::new(), 0);
    out.push('\n');
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryFile {
    pub schema_version: u32,
    pub server: ServerConfig,
    pub models: Vec<ModelEntry>,
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub roles: std::collections::HashMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub aliases: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "legacy_modality")]
    pub modality: Modality,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub tasks: Vec<String>,
    #[serde(default)]
    pub audio_operations: crate::audio_operations::AudioOperations,
    #[serde(default)]
    pub tool_call_support: Option<crate::audio_operations::OperationSupport>,
    #[serde(default)]
    pub structured_output_support: Option<crate::audio_operations::OperationSupport>,
    #[serde(default)]
    pub image_reference_checks: Vec<crate::config::ImageReferenceVerification>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub roles: Vec<String>,
    /// Required for local models, forbidden for cloud models.
    #[serde(default)]
    pub engine: Option<EngineConfig>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub sampling: SamplingConfig,
    #[serde(default)]
    pub local: Option<LocalBlock>,
    #[serde(default)]
    pub remote: Option<RemoteBlock>,
    /// Engine-specific settings; see [`allowed_extra_keys`].
    #[serde(default)]
    pub extra: Map<String, Value>,
    /// Benchmark and evaluation metrics (e.g. Artificial Analysis Intelligence Index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks: Option<BenchmarkMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_name: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "intelligence_index"
    )]
    pub score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "domain_indexes"
    )]
    pub domains: Option<DomainScores>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_per_task_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_per_task_minutes: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_task: Option<u64>,
}

impl Eq for BenchmarkMetrics {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DomainScores {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finance_and_accounting: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy_and_ops: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legal: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub healthcare_and_medical: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engineering: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub economics: Option<f64>,
}

impl Eq for DomainScores {}

impl Serialize for ModelEntry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("id", &self.id)?;
        map.serialize_entry("name", &self.name)?;
        map.serialize_entry("description", &self.description)?;
        map.serialize_entry("capabilities", &self.capabilities)?;
        if !self.audio_operations.is_empty() {
            map.serialize_entry("audio_operations", &self.audio_operations)?;
        }
        if let Some(support) = &self.tool_call_support {
            map.serialize_entry("tool_call_support", support)?;
        }
        if let Some(support) = &self.structured_output_support {
            map.serialize_entry("structured_output_support", support)?;
        }
        if !self.image_reference_checks.is_empty() {
            map.serialize_entry("image_reference_checks", &self.image_reference_checks)?;
        }
        map.serialize_entry("max_context_length", &self.max_context_length)?;
        if self.tasks.is_empty() {
            // Preserve legacy routing facts without inventing verified tasks.
            map.serialize_entry("modality", &self.modality)?;
        } else {
            map.serialize_entry("tasks", &self.tasks)?;
        }
        if !self.aliases.is_empty() {
            map.serialize_entry("aliases", &self.aliases)?;
        }
        if !self.roles.is_empty() {
            map.serialize_entry("roles", &self.roles)?;
        }
        if let Some(engine) = &self.engine {
            map.serialize_entry("engine", engine)?;
        }
        if !sampling_is_default(&self.sampling) {
            map.serialize_entry("sampling", &self.sampling)?;
        }
        if let Some(local) = &self.local {
            map.serialize_entry("local", local)?;
        }
        if let Some(remote) = &self.remote {
            map.serialize_entry("remote", remote)?;
        }
        if !self.extra.is_empty() {
            map.serialize_entry("extra", &self.extra)?;
        }
        if let Some(benchmarks) = &self.benchmarks {
            map.serialize_entry("benchmarks", benchmarks)?;
        }
        map.end()
    }
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

fn sampling_is_default(value: &SamplingConfig) -> bool {
    *value == SamplingConfig::default()
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
            "tokenizer_path",
            "llm_vision_path",
            "embeddings_connectors_path",
            "audio_vae_path",
            "default_steps",
            "default_cfg_scale",
            "default_sample_method",
            "max_resolution",
            "lora",
        ],
        EngineKind::Onnx
        | EngineKind::AudioCpp
        | EngineKind::NemoSpeech
        | EngineKind::Crispasr
        | EngineKind::WhisperCpp
        | EngineKind::Encoder
        | EngineKind::Litert
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
    tokenizer_path: Option<String>,
    llm_vision_path: Option<String>,
    embeddings_connectors_path: Option<String>,
    audio_vae_path: Option<String>,
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

pub(crate) fn valid_task(task: &str) -> bool {
    let Some((inputs, outputs)) = task.split_once("-to-") else {
        return false;
    };
    [inputs, outputs].into_iter().all(|side| {
        let mut seen = std::collections::HashSet::new();
        side.split('-').all(|kind| {
            matches!(
                kind,
                "text"
                    | "image"
                    | "audio"
                    | "video"
                    | "embedding"
                    | "decision"
                    | "diarization"
                    | "detection"
                    | "moderation"
                    | "depth"
                    | "segmentation"
                    | "matting"
            ) && seen.insert(kind)
        })
    })
}

fn legacy_modality() -> Modality {
    Modality::Text
}

fn derived_modality(tasks: &[String]) -> Modality {
    let outputs = |kind: &str| {
        tasks.iter().any(|task| {
            task.split_once("-to-")
                .is_some_and(|(_, out)| out.split('-').any(|v| v == kind))
        })
    };
    let inputs = |kind: &str| {
        tasks.iter().any(|task| {
            task.split_once("-to-")
                .is_some_and(|(input, _)| input.split('-').any(|v| v == kind))
        })
    };
    if outputs("depth") {
        Modality::DepthEstimation
    } else if outputs("decision") {
        Modality::Decision
    } else if outputs("embedding") {
        if inputs("image") {
            Modality::ImageEmbedding
        } else {
            Modality::Embedding
        }
    } else if outputs("diarization") {
        Modality::SpeechAnalysis
    } else if outputs("detection") {
        Modality::ObjectDetection
    } else if outputs("moderation") {
        Modality::Moderation
    } else if outputs("video") {
        Modality::VideoGeneration
    } else if outputs("image") {
        Modality::ImageGeneration
    } else if outputs("audio") {
        if tasks.iter().any(|task| {
            task.split_once("-to-").is_some_and(|(input, output)| {
                input.split('-').any(|kind| kind == "audio")
                    && output.split('-').any(|kind| kind == "audio")
            })
        }) {
            Modality::SpeechToSpeech
        } else {
            Modality::TextToSpeech
        }
    } else if inputs("image") {
        Modality::VisionText
    } else if inputs("audio") {
        Modality::SpeechToText
    } else {
        Modality::Text
    }
}

impl TryFrom<ModelEntry> for ModelConfig {
    type Error = anyhow::Error;

    fn try_from(e: ModelEntry) -> Result<Self> {
        let id = e.id.clone();
        let mut seen = std::collections::HashSet::new();
        for task in &e.tasks {
            anyhow::ensure!(
                valid_task(task),
                "Model '{id}' has invalid input/output task '{task}'"
            );
            anyhow::ensure!(seen.insert(task), "Model '{id}' repeats task '{task}'");
        }

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
            modality: if e.tasks.is_empty() {
                e.modality
            } else {
                derived_modality(&e.tasks)
            },
            model_path,
            upstream_model,
            api_key,
            base_url,
            rate_limit,
            mmproj_path: extra.mmproj_path,
            mtp_path: extra.mtp_path,
            vae_path: extra.vae_path,
            text_encoder_path: extra.text_encoder_path,
            tokenizer_path: extra.tokenizer_path,
            llm_vision_path: extra.llm_vision_path,
            embeddings_connectors_path: extra.embeddings_connectors_path,
            audio_vae_path: extra.audio_vae_path,
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
            tasks: e.tasks,
            audio_operations: e.audio_operations,
            tool_call_support: e.tool_call_support,
            structured_output_support: e.structured_output_support,
            image_reference_checks: e.image_reference_checks,
            sampling: (sampling != SamplingConfig::default()).then_some(sampling),
            engine,
            lora: extra.lora,
            runtime,
            aliases: e.aliases,
            roles: e.roles,
            benchmarks: e.benchmarks,
        };
        model.validate()?;
        model.validate_tool_call_support()?;
        model.validate_structured_output_support()?;
        model.validate_image_reference_checks()?;
        model
            .audio_operations
            .validate(&model.engine_plan())
            .map_err(|error| anyhow::anyhow!("Model '{}': {error}", model.id))?;
        Ok(model)
    }
}

#[cfg(test)]
mod audio_modality_tests {
    use super::*;

    #[test]
    fn separate_asr_and_tts_tasks_do_not_claim_speech_to_speech() {
        assert_eq!(
            derived_modality(&["audio-to-text".into(), "text-to-audio".into()]),
            Modality::TextToSpeech
        );
        assert_eq!(
            derived_modality(&["audio-to-audio".into()]),
            Modality::SpeechToSpeech
        );
        assert_eq!(
            derived_modality(&["text-audio-to-audio".into()]),
            Modality::SpeechToSpeech
        );
        assert_eq!(
            derived_modality(&["audio-to-text-audio".into()]),
            Modality::SpeechToSpeech
        );
    }
}

#[cfg(test)]
mod render_catalog_tests {
    use super::*;

    #[test]
    fn rendering_keeps_the_canonical_key_order() {
        let value = serde_json::json!({
            "models": [{
                "tasks": ["text-to-text"], "name": "A", "id": "a", "description": "",
                "capabilities": ["Chat"], "max_context_length": null,
                "engine": ["llama-server"], "extra": {"z": 1, "a": []},
                "local": {"path": "a.gguf", "args": "", "params": 1}
            }],
            "schema_version": SCHEMA_VERSION
        });
        let text = render_catalog(&value).unwrap();
        let position = |key: &str| text.find(&format!("\"{key}\"")).unwrap();
        assert!(position("schema_version") < position("models"));
        assert!(position("id") < position("name"));
        assert!(position("capabilities") < position("tasks"));
        assert!(position("tasks") < position("engine"));
        assert!(position("engine") < position("local"));
        assert!(position("path") < position("params"));
        assert!(position("params") < position("args"));
        assert!(text.contains("\"a\": []"));
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    }
}
