//! `rune-infer migrate`: upgrade a `schema_version: 1` models.json to the
//! current schema (see [`crate::schema`]).
//!
//! The v1 format let the engine be *inferred* from a model's architecture,
//! modality, file type and id substrings. That inference now lives only here,
//! so it runs once, at migration time, and the result is written into the
//! file as an explicit `engine` object.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::config::{
    EngineConfig, EngineKind, LoraConfig, Modality, ProviderType, RateLimitConfig, SamplingConfig,
    ServerConfig,
};
use crate::schema::{
    LocalBlock, ModelEntry, RegistryFile, RemoteBlock, SCHEMA_VERSION, allowed_extra_keys,
};

#[derive(Debug, Deserialize)]
struct V1File {
    #[serde(default)]
    schema_version: u32,
    server: ServerConfig,
    models: Vec<V1Model>,
}

#[derive(Debug, Deserialize)]
struct V1Runtime {
    context_length: Option<i64>,
    gpu_layers: Option<u32>,
    mtp_heads: Option<u32>,
    extra_args: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct V1RateLimit {
    requests_per_minute: Option<u32>,
    tokens_per_minute: Option<u32>,
    requests_per_day: Option<u32>,
    max_concurrent: Option<usize>,
    queue_timeout_seconds: Option<u64>,
    max_retries: Option<u32>,
    retry_delay_seconds: Option<u64>,
    retry_delay_ms: Option<u64>,
    retry_timeout_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct V1Model {
    id: String,
    name: String,
    #[serde(default)]
    provider: ProviderType,
    modality: Modality,
    #[serde(default)]
    architecture: String,
    #[serde(default)]
    model_path: String,
    upstream_model: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    rate_limit: Option<V1RateLimit>,
    mmproj_path: Option<String>,
    mtp_path: Option<String>,
    vae_path: Option<String>,
    text_encoder_path: Option<String>,
    llm_vision_path: Option<String>,
    default_steps: Option<u32>,
    default_cfg_scale: Option<f32>,
    default_sample_method: Option<String>,
    vision: Option<bool>,
    #[serde(default)]
    description: String,
    bits_per_weight: Option<f64>,
    total_params: Option<u64>,
    max_context_length: Option<u32>,
    kv_bytes_per_token: Option<u64>,
    max_resolution: Option<String>,
    mtp_heads: Option<u32>,
    capabilities: Option<Vec<String>>,
    sampling: Option<SamplingConfig>,
    engine: Option<EngineConfig>,
    runtime: Option<V1Runtime>,
    /// Accepted so it is not reported as an unknown key; engines detect the format themselves.
    #[allow(dead_code)]
    format: Option<String>,
    #[serde(default)]
    lora: Vec<Value>,
    #[serde(flatten)]
    unknown: Map<String, Value>,
}

/// The engine v1 inferred for a model that had no explicit `engine`.
/// Rules are checked top to bottom; the first match wins.
fn legacy_engine(m: &V1Model, id: &str) -> EngineKind {
    let arch = |name: &str| m.architecture.eq_ignore_ascii_case(name);
    let id_lower = id.to_lowercase();

    if m.modality == Modality::ImageGeneration {
        return EngineKind::SdServer;
    }
    if arch("audio8") || arch("audiocpp") || id_lower.contains("audio8") {
        return EngineKind::AudioCpp;
    }
    if arch("parakeet")
        || arch("nemo-speech")
        || arch("fastconformer")
        || arch("fastconformer-tdt")
        || id_lower.contains("parakeet")
        || id_lower.contains("nemotron-speech")
    {
        return EngineKind::NemoSpeech;
    }
    if m.modality == Modality::SpeechToText
        || m.modality == Modality::TextToSpeech
        || arch("crispasr")
        || arch("voxtral4b")
    {
        return EngineKind::Crispasr;
    }
    let is_encoder_path = m.model_path.ends_with(".safetensors")
        || m.model_path.ends_with(".bin")
        || m.model_path.ends_with(".onnx");
    let is_encoder_task = m.modality == Modality::Embedding
        || m.modality == Modality::Moderation
        || m.modality == Modality::Nlu
        || arch("modernbert")
        || arch("encoder")
        || id.contains("embedding")
        || id.contains("granite")
        || id.contains("guard")
        || id.contains("intent")
        || id.contains("mmbert");
    if is_encoder_task || is_encoder_path {
        return EngineKind::Encoder;
    }
    match m.architecture.as_str() {
        "cactus-needle-3" => EngineKind::Needle,
        _ => llama_engine_for_architecture(&m.architecture),
    }
}

/// Which llama.cpp build serves a GGUF architecture.
pub fn llama_engine_for_architecture(architecture: &str) -> EngineKind {
    match architecture {
        "ternary-bonsai" | "bonsai" | "bonsai-vl" | "ternary-bonsai-vl" => {
            EngineKind::LlamaServerPrism
        }
        _ => EngineKind::LlamaServer,
    }
}

/// Result of converting a v1 document.
#[derive(Debug)]
pub struct Migration {
    pub file: RegistryFile,
    /// Human-readable notes about inferred engines, dropped keys and warnings.
    pub notes: Vec<String>,
}

fn rate_limit(v1: V1RateLimit) -> RateLimitConfig {
    let d = RateLimitConfig::default();
    RateLimitConfig {
        requests_per_minute: v1.requests_per_minute.unwrap_or(d.requests_per_minute),
        tokens_per_minute: v1.tokens_per_minute,
        requests_per_day: v1.requests_per_day,
        max_concurrent: v1.max_concurrent.unwrap_or(d.max_concurrent),
        queue_timeout_seconds: v1.queue_timeout_seconds.unwrap_or(d.queue_timeout_seconds),
        max_retries: v1.max_retries.unwrap_or(d.max_retries),
        retry_delay_ms: v1
            .retry_delay_seconds
            .map(|s| s * 1000)
            .or(v1.retry_delay_ms)
            .unwrap_or(d.retry_delay_ms),
        retry_timeout_seconds: v1.retry_timeout_seconds.unwrap_or(d.retry_timeout_seconds),
    }
}

fn convert(m: V1Model, notes: &mut Vec<String>) -> Result<ModelEntry> {
    let id = m.id.clone();
    let mut note = |msg: String| notes.push(format!("{id}: {msg}"));

    for key in m.unknown.keys() {
        note(format!("unknown key '{key}' dropped"));
    }
    if m.vision == Some(true) && m.modality != Modality::VisionText {
        note(format!(
            "vision=true dropped (modality is {:?}, so it no longer counts as a vision model)",
            m.modality
        ));
    }
    if let Some(bytes) = m.kv_bytes_per_token {
        note(format!(
            "kv_bytes_per_token={bytes} dropped (estimated from params instead)"
        ));
    }

    let mut entry = ModelEntry {
        id: m.id.clone(),
        name: m.name.clone(),
        description: m.description.clone(),
        modality: m.modality.clone(),
        capabilities: m.capabilities.clone().unwrap_or_default(),
        tasks: Vec::new(),
        roles: Vec::new(),
        aliases: Vec::new(),
        engine: None,
        max_context_length: m.max_context_length,
        sampling: m.sampling.clone().unwrap_or_default(),
        local: None,
        remote: None,
        extra: Map::new(),
    };

    if m.provider != ProviderType::Local {
        let missing = |what: &str| anyhow::anyhow!("Model '{}': cloud model has no {what}", m.id);
        entry.remote = Some(RemoteBlock {
            provider: m.provider,
            upstream_model: m
                .upstream_model
                .clone()
                .ok_or_else(|| missing("upstream_model"))?,
            api_key: m.api_key.clone().ok_or_else(|| missing("api_key"))?,
            base_url: m.base_url.clone(),
            rate_limit: m.rate_limit.map(rate_limit),
        });
        return Ok(entry);
    }

    let engine = match m.engine.clone() {
        Some(cfg) => cfg,
        None => {
            let kind = legacy_engine(&m, &m.id);
            if legacy_engine(&m, "") == kind {
                note(format!(
                    "engine: {kind} (inferred from modality/architecture)"
                ));
            } else {
                note(format!(
                    "engine: {kind} (inferred from the model id — verify it is what you want)"
                ));
            }
            EngineConfig {
                supported: vec![kind],
                preferred: Some(kind),
            }
        }
    };
    let engine = EngineConfig {
        preferred: engine
            .preferred
            .or_else(|| engine.supported.first().copied()),
        ..engine
    };
    let llama = engine
        .supported
        .iter()
        .any(|k| matches!(k, EngineKind::LlamaServer | EngineKind::LlamaServerPrism));

    let runtime = m.runtime.as_ref();
    // v1 silently defaulted an omitted context_length to 8192; that only ever
    // mattered to llama.cpp engines, so keep it there and use "auto" elsewhere.
    let context_length = match (
        runtime.and_then(|r| r.context_length),
        runtime.is_some() && llama,
    ) {
        (Some(n), _) => n,
        (None, true) => 8192,
        (None, false) => -1,
    };
    entry.local = Some(LocalBlock {
        path: m.model_path.clone(),
        params: m.total_params,
        bits_per_weight: m.bits_per_weight,
        context_length,
        gpu_layers: runtime.and_then(|r| r.gpu_layers).unwrap_or(99),
        args: runtime
            .and_then(|r| r.extra_args.clone())
            .unwrap_or_default(),
    });

    let mut extra = Map::new();
    let mut put = |key: &str, value: Option<Value>| {
        if let Some(v) = value {
            extra.insert(key.to_string(), v);
        }
    };
    put("mmproj_path", m.mmproj_path.clone().map(Value::from));
    put("mtp_path", m.mtp_path.clone().map(Value::from));
    put(
        "mtp_heads",
        m.mtp_heads
            .or(runtime.and_then(|r| r.mtp_heads))
            .map(Value::from),
    );
    put("vae_path", m.vae_path.clone().map(Value::from));
    put(
        "text_encoder_path",
        m.text_encoder_path.clone().map(Value::from),
    );
    put(
        "llm_vision_path",
        m.llm_vision_path.clone().map(Value::from),
    );
    put("default_steps", m.default_steps.map(Value::from));
    put("default_cfg_scale", m.default_cfg_scale.map(Value::from));
    put(
        "default_sample_method",
        m.default_sample_method.clone().map(Value::from),
    );
    put("max_resolution", m.max_resolution.clone().map(Value::from));
    if !m.lora.is_empty() {
        let adapters: Vec<LoraConfig> = serde_json::from_value(Value::Array(m.lora.clone()))
            .with_context(|| format!("Model '{}': lora entries need name, path and scale", m.id))?;
        put("lora", Some(serde_json::to_value(adapters)?));
    }

    for (key, _) in extra.clone() {
        if !engine
            .supported
            .iter()
            .any(|k| allowed_extra_keys(*k).contains(&key.as_str()))
        {
            extra.remove(&key);
            note(format!(
                "'{key}' dropped (none of its engines [{}] use it)",
                engine
                    .supported
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    entry.engine = Some(engine);
    entry.extra = extra;
    Ok(entry)
}

/// Convert a v1 document (placeholders like `{{VAR}}` are kept as written).
pub fn migrate_json(json: &str) -> Result<Migration> {
    let v1: V1File = serde_json::from_str(json).context("Cannot read the config as schema 1")?;
    if v1.schema_version != 1 {
        anyhow::bail!(
            "Expected schema_version 1, found {}. Nothing to migrate.",
            v1.schema_version
        );
    }
    let mut notes = Vec::new();
    let models = v1
        .models
        .into_iter()
        .map(|m| convert(m, &mut notes))
        .collect::<Result<Vec<_>>>()?;
    Ok(Migration {
        file: RegistryFile {
            schema_version: SCHEMA_VERSION,
            server: v1.server,
            models,
            roles: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
        },
        notes,
    })
}

fn backup_path(config: &Path, ver: u64) -> PathBuf {
    let base = format!("{}.v{ver}.bak", config.display());
    let mut candidate = PathBuf::from(&base);
    let mut n = 1;
    while candidate.exists() {
        candidate = PathBuf::from(format!("{base}.{n}"));
        n += 1;
    }
    candidate
}

/// Entry point for `rune-infer migrate`. Prints a report; splits and writes the upgraded
/// schema 3 files (config/server.json and config/catalog.json) unless `dry_run`.
pub fn run(config: &Path, dry_run: bool) -> Result<()> {
    let cfg_dir = config.parent().unwrap_or(Path::new("."));
    let server_path = cfg_dir.join("server.json");
    let catalog_path = cfg_dir.join("catalog.json");

    if server_path.exists() && catalog_path.exists() && !config.exists() {
        println!(
            "{} and {} are already schema {SCHEMA_VERSION}; nothing to do.",
            server_path.display(),
            catalog_path.display()
        );
        return Ok(());
    }

    let raw = std::fs::read_to_string(config)
        .with_context(|| format!("Cannot read {}", config.display()))?;
    let version = serde_json::from_str::<Value>(&raw)
        .with_context(|| format!("{} is not valid JSON", config.display()))?
        .get("schema_version")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let (registry_file, notes) = if version <= 1 {
        let migration = migrate_json(&raw)?;
        (migration.file, migration.notes)
    } else if version == 2 {
        let file: RegistryFile = serde_json::from_str(&raw)
            .with_context(|| format!("Invalid schema 2 file at {}", config.display()))?;
        (
            file,
            vec!["Decoupled unified models.json into server.json and catalog.json".to_string()],
        )
    } else if version == u64::from(SCHEMA_VERSION) {
        println!(
            "{} is already schema {SCHEMA_VERSION}; nothing to do.",
            config.display()
        );
        return Ok(());
    } else {
        anyhow::bail!(
            "Unsupported schema version {version} in {}",
            config.display()
        );
    };

    let local = registry_file
        .models
        .iter()
        .filter(|m| m.local.is_some())
        .count();
    let remote = registry_file.models.len() - local;
    println!(
        "Migrating {} ({} local, {} cloud models) to schema {SCHEMA_VERSION} (decoupled server.json and catalog.json)",
        config.display(),
        local,
        remote
    );
    for note in &notes {
        println!("  - {note}");
    }

    let server_file = crate::schema::ServerFile {
        schema_version: SCHEMA_VERSION,
        server: registry_file.server,
    };
    let catalog_file = crate::schema::CatalogFile {
        schema_version: SCHEMA_VERSION,
        models: registry_file.models,
        roles: registry_file.roles,
        aliases: registry_file.aliases,
    };

    let server_output = serde_json::to_string_pretty(&server_file)? + "\n";
    let catalog_output = serde_json::to_string_pretty(&catalog_file)? + "\n";

    // Validate that the decoupled outputs load properly before touching anything on disk
    let check_server =
        crate::config::interpolate_with(&server_output, &|_| Some("placeholder".to_string()))?;
    let check_catalog =
        crate::config::interpolate_with(&catalog_output, &|_| Some("placeholder".to_string()))?;
    let test_server: crate::schema::ServerFile = serde_json::from_str(&check_server)?;
    let test_catalog: crate::schema::CatalogFile = serde_json::from_str(&check_catalog)?;
    let models = test_catalog
        .models
        .into_iter()
        .map(crate::config::ModelConfig::try_from)
        .collect::<Result<Vec<_>>>()?;
    let registry = crate::config::ModelRegistry {
        schema_version: SCHEMA_VERSION,
        server: test_server.server,
        models,
        roles: test_catalog.roles,
        aliases: test_catalog.aliases,
    };
    registry
        .validate()
        .context("The migrated config does not validate; the original was left untouched")?;

    if dry_run {
        println!(
            "Dry run: would write {} and {}",
            server_path.display(),
            catalog_path.display()
        );
        return Ok(());
    }

    let backup = backup_path(config, version);
    std::fs::copy(config, &backup)
        .with_context(|| format!("Cannot write backup {}", backup.display()))?;
    std::fs::write(&server_path, server_output)
        .with_context(|| format!("Cannot write {}", server_path.display()))?;
    std::fs::write(&catalog_path, catalog_output)
        .with_context(|| format!("Cannot write {}", catalog_path.display()))?;
    let _ = std::fs::remove_file(config);

    println!("Done. Server settings written to {}", server_path.display());
    println!("Done. Model catalog written to {}", catalog_path.display());
    println!("Original saved as {}", backup.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelRegistry;

    fn v1_model(extra: &str) -> V1Model {
        serde_json::from_str(&format!(
            r#"{{"id":"m","name":"n","modality":"Text","model_path":"m.gguf"{extra}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn legacy_engine_rules_are_pinned() {
        use EngineKind::*;
        let cases = [
            ("z-image", "ImageGeneration", "z-image", "m.gguf", SdServer),
            ("audio8-asr", "SpeechToText", "arkasr", "m", AudioCpp),
            ("x", "TextToSpeech", "audiocpp", "m", AudioCpp),
            (
                "parakeet-tdt",
                "SpeechToText",
                "crispasr",
                "m.gguf",
                NemoSpeech,
            ),
            ("x", "SpeechToText", "fastconformer-tdt", "m", NemoSpeech),
            ("whisper", "SpeechToText", "whisper", "m", Crispasr),
            ("tts", "TextToSpeech", "qwen3-tts", "m", Crispasr),
            ("v", "Text", "voxtral4b", "m.gguf", Crispasr),
            ("emb", "Embedding", "modernbert", "m.safetensors", Encoder),
            ("my-guard", "Text", "x", "m.gguf", Encoder),
            ("w", "Text", "x", "m.safetensors", Encoder),
            ("needle", "Text", "cactus-needle-3", "m.cact", Needle),
            (
                "b",
                "VisionText",
                "ternary-bonsai-vl",
                "m.gguf",
                LlamaServerPrism,
            ),
            ("q", "Text", "qwen35", "m.gguf", LlamaServer),
            (
                "dir-model",
                "VisionText",
                "sam2",
                "/hub/snapshots/main",
                LlamaServer,
            ),
        ];
        for (id, modality, arch, path, want) in cases {
            let m: V1Model = serde_json::from_str(&format!(
                r#"{{"id":"{id}","name":"n","modality":"{modality}","architecture":"{arch}","model_path":"{path}"}}"#
            ))
            .unwrap();
            assert_eq!(legacy_engine(&m, id), want, "{id}");
        }
    }

    const V1: &str = r#"{
      "schema_version": 1,
      "server": { "host": "127.0.0.1", "port": 9, "api_key": null, "max_loaded_models": 1, "idle_unload_seconds": 300 },
      "models": [
        { "id": "vision-llm", "name": "V", "modality": "VisionText", "architecture": "qwen35", "format": "gguf",
          "model_path": "D:/m/v.gguf", "mmproj_path": "D:/m/mm.gguf", "mtp_path": null, "mtp_heads": null,
          "vision": true, "total_params": 27000000000, "bits_per_weight": 3.4, "max_context_length": 262144,
          "sampling": { "temperature": 0.6, "top_k": 20 }, "lora": [],
          "runtime": { "context_length": -1, "gpu_layers": 99, "extra_args": "-fa on" } },
        { "id": "z-image", "name": "Z", "modality": "ImageGeneration", "architecture": "z-image", "format": "gguf",
          "model_path": "D:/m/z.gguf", "vae_path": "D:/m/vae.safetensors", "default_steps": 8, "vision": false,
          "max_resolution": "1024x1024", "mmproj_path": "D:/m/never-used.gguf",
          "lora": [ { "name": "px", "path": "D:/m/px.safetensors", "scale": 0.8 } ],
          "runtime": { "gpu_layers": 99, "extra_args": "--diffusion-fa" } },
        { "id": "parakeet-tdt-0.6b-v3", "name": "P", "modality": "SpeechToText", "architecture": "crispasr",
          "model_path": "D:/m/p.gguf", "runtime": { "gpu_layers": 99 } },
        { "id": "edit", "name": "E", "modality": "ImageGeneration", "architecture": "qwen-image-edit",
          "vision": true, "model_path": "D:/m/e.gguf", "whatever": 1,
          "engine": { "supported": ["sd-server"] } },
        { "id": "gemini", "name": "G", "provider": "google", "modality": "VisionText",
          "upstream_model": "gemini-x", "api_key": "{{GEMINI_API_KEY}}", "base_url": "https://g/v1beta", "vision": true,
          "rate_limit": { "requests_per_minute": 5, "retry_delay_seconds": 2, "max_concurrent": 2 } }
      ]
    }"#;

    #[test]
    fn migrates_every_kind_of_model_and_the_result_loads() {
        let mig = migrate_json(V1).unwrap();
        let by_id = |id: &str| mig.file.models.iter().find(|m| m.id == id).unwrap();

        let v = by_id("vision-llm");
        assert_eq!(
            v.engine.as_ref().unwrap().plan(),
            vec![EngineKind::LlamaServer]
        );
        assert_eq!(v.extra["mmproj_path"], "D:/m/mm.gguf");
        assert!(!v.extra.contains_key("mtp_path"), "null values are omitted");
        let l = v.local.as_ref().unwrap();
        assert_eq!(
            (l.context_length, l.gpu_layers, l.args.as_str()),
            (-1, 99, "-fa on")
        );

        let z = by_id("z-image");
        assert_eq!(
            z.engine.as_ref().unwrap().plan(),
            vec![EngineKind::SdServer]
        );
        assert_eq!(z.extra["vae_path"], "D:/m/vae.safetensors");
        assert!((z.extra["lora"][0]["scale"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert!(
            !z.extra.contains_key("mmproj_path"),
            "keys the engine ignores are dropped"
        );
        assert_eq!(z.local.as_ref().unwrap().context_length, -1);

        let p = by_id("parakeet-tdt-0.6b-v3");
        assert_eq!(
            p.engine.as_ref().unwrap().plan(),
            vec![EngineKind::NemoSpeech]
        );

        let g = by_id("gemini");
        let r = g.remote.as_ref().unwrap();
        assert_eq!(
            r.api_key, "{{GEMINI_API_KEY}}",
            "placeholders are preserved"
        );
        assert_eq!(
            r.rate_limit.as_ref().unwrap().retry_delay_ms,
            2000,
            "seconds become ms"
        );
        assert!(g.engine.is_none() && g.local.is_none());

        let joined = mig.notes.join("\n");
        assert!(
            joined
                .contains("parakeet-tdt-0.6b-v3: engine: nemo-speech (inferred from the model id")
        );
        assert!(
            joined
                .contains("vision-llm: engine: llama-server (inferred from modality/architecture)")
        );
        assert!(joined.contains("edit: vision=true dropped"));
        assert!(joined.contains("edit: unknown key 'whatever' dropped"));
        assert!(
            !joined.contains("edit: engine:"),
            "explicit engines are kept"
        );

        let json = serde_json::to_string_pretty(&mig.file).unwrap();
        let loaded =
            ModelRegistry::from_json(&json.replace("{{GEMINI_API_KEY}}", "k"), None).unwrap();
        assert_eq!(loaded.models.len(), 5);
        assert_eq!(loaded.find("z-image").unwrap().lora.len(), 1);
        assert!(loaded.find("gemini").unwrap().engine.is_none());
    }

    #[test]
    fn omitted_context_length_keeps_the_old_default_only_for_llama_engines() {
        let llama = convert(v1_model(r#","runtime":{"gpu_layers":1}"#), &mut vec![]).unwrap();
        assert_eq!(llama.local.unwrap().context_length, 8192);

        let sd: V1Model = serde_json::from_str(
            r#"{"id":"z","name":"n","modality":"ImageGeneration","architecture":"z-image","model_path":"z.gguf","runtime":{"gpu_layers":1}}"#,
        )
        .unwrap();
        assert_eq!(
            convert(sd, &mut vec![])
                .unwrap()
                .local
                .unwrap()
                .context_length,
            -1
        );
    }

    #[test]
    fn already_current_files_are_refused_by_migrate_json() {
        let err = migrate_json(r#"{"schema_version":2,"server":{"host":"h","port":1,"max_loaded_models":1,"idle_unload_seconds":1},"models":[]}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected schema_version 1"), "{err}");
    }

    #[test]
    fn loader_points_v1_files_at_the_migrate_command() {
        let err = ModelRegistry::from_json(V1, Some("/x/models.json"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("schema_version 1") && err.contains("rune-infer migrate"),
            "{err}"
        );
    }

    #[test]
    fn run_writes_backup_and_upgrades_in_place() {
        let dir = std::env::temp_dir().join(format!("rune-mig-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("models.json");
        std::fs::write(&cfg, V1).unwrap();

        run(&cfg, true).unwrap();
        assert_eq!(
            std::fs::read_to_string(&cfg).unwrap(),
            V1,
            "dry run changes nothing"
        );

        run(&cfg, false).unwrap();
        assert!(dir.join("models.json.v1.bak").exists());
        let server_cfg = std::fs::read_to_string(dir.join("server.json")).unwrap();
        let catalog_cfg = std::fs::read_to_string(dir.join("catalog.json")).unwrap();
        assert!(server_cfg.contains("\"schema_version\": 3"));
        assert!(catalog_cfg.contains("\"schema_version\": 3"));
        run(&cfg, false).unwrap(); // second run is a no-op
        assert_eq!(
            std::fs::read_to_string(dir.join("server.json")).unwrap(),
            server_cfg
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
