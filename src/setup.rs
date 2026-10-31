//! First-run setup logic: prerequisite detection, local model discovery,
//! cloud provider presets and safe config/`.env` generation.
//!
//! This module is UI-agnostic (no terminal code) so it can be unit tested and
//! reused by other front-ends; the interactive wizard lives in `tui::wizard`.

mod registry;
pub use registry::{ModelMetadata, RegistryStatus, registered_model};
mod cloud;
pub use cloud::{CloudCapabilities, CloudModel, list_provider_models};
mod local;
#[cfg(test)]
pub(crate) use local::prepare_local_models_with;
pub use local::{LocalPreparationEvent, PreparedLocal, prepare_local_models, validate_local_files};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::{
    EngineConfig, EngineKind, Modality, ModelRegistry, ProviderType, SamplingConfig, ServerConfig,
};
use crate::schema::{CatalogFile, LocalBlock, ModelEntry, RemoteBlock, SCHEMA_VERSION, ServerFile};

// ---------------------------------------------------------------------------
// Prerequisites
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Prerequisites {
    pub os: &'static str,
    pub arch: &'static str,
    pub home: PathBuf,
    pub gpu: Option<GpuInfo>,
    pub ffmpeg: Option<String>,
    pub llama_server_installed: bool,
}

/// Probe the host. Spawns `nvidia-smi` / `ffmpeg`, so call off the UI thread.
pub fn detect_prerequisites(home: &Path) -> Prerequisites {
    Prerequisites {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        home: home.to_path_buf(),
        gpu: detect_gpu(),
        ffmpeg: detect_ffmpeg(),
        llama_server_installed: llama_server_installed(home),
    }
}

fn detect_gpu() -> Option<GpuInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?;
    let (name, mib) = line.rsplit_once(',')?;
    Some(GpuInfo {
        name: name.trim().to_string(),
        total_bytes: mib.trim().parse::<u64>().ok()? * 1024 * 1024,
    })
}

fn detect_ffmpeg() -> Option<String> {
    let out = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .ok()?;
    out.status.success().then(|| {
        let text = String::from_utf8_lossy(&out.stdout);
        let version = text
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(2))
            .unwrap_or("");
        let short: String = version.chars().take(24).collect();
        format!("FFmpeg {short}").trim_end().to_string()
    })
}

fn llama_server_installed(home: &Path) -> bool {
    let exe = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    if std::env::var_os("LLAMA_SERVER_PATH").is_some_and(|p| Path::new(&p).exists()) {
        return true;
    }
    for base in [home.join("bin"), home.join("bin").join("llama-upstream")] {
        if base.join(exe).exists() {
            return true;
        }
    }
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join(exe).exists()))
        .unwrap_or(false)
}

/// Download (or locate) the upstream `llama-server` engine.
pub fn install_llama_server() -> Result<PathBuf, String> {
    install_engine(EngineKind::LlamaServer)
}

/// Install or reuse a selected external engine without starting a model process.
/// Uses the same installation locations and resolvers as request-time loading.
pub fn install_engine(engine: EngineKind) -> Result<PathBuf, String> {
    use crate::inference::engines::{
        audio::AudioEngine,
        crispasr::CrispAsrEngine,
        llama_server::{RuntimeFlavor, install},
        needle::NeedleEngine,
        nemo_speech::NemoSpeechEngine,
        sd_server::SdServerEngine,
        whisper::WhisperCppEngine,
    };
    let result = match engine {
        EngineKind::LlamaServer => install::ensure_binary_installed(RuntimeFlavor::Upstream),
        EngineKind::LlamaServerPrism => install::ensure_binary_installed(RuntimeFlavor::Prism),
        EngineKind::SdServer => SdServerEngine::ensure_binary_installed(),
        EngineKind::AudioCpp => AudioEngine::ensure_binary_installed(),
        EngineKind::NemoSpeech => NemoSpeechEngine::ensure_binary_installed(),
        EngineKind::Crispasr => CrispAsrEngine::ensure_binary_installed(),
        EngineKind::WhisperCpp => WhisperCppEngine::ensure_binary_installed(),
        EngineKind::Needle => NeedleEngine::ensure_binary_installed(),
        EngineKind::Litert => crate::inference::engines::litert::ensure_runtime_installed(),
        EngineKind::Encoder => return Err(
            "encoder is built in; external installation does not validate its inference support"
                .into(),
        ),
    };
    let path = result.map_err(|error| format!("{engine}: {error}"))?;
    if !path.is_file() {
        return Err(format!(
            "{engine}: installer did not provide an executable file at {}",
            path.display()
        ));
    }
    Ok(path)
}

/// Keep compatible selections in preference order; the first engine is the default.
pub fn select_local_engines(
    models: &[LocalCandidate],
    choices: &[Vec<EngineKind>],
) -> Result<Vec<LocalCandidate>, String> {
    if models.len() != choices.len() {
        return Err("Choose engines for every local model.".into());
    }
    models
        .iter()
        .zip(choices)
        .map(|(model, engines)| {
            if engines.is_empty() {
                return Err(format!(
                    "Choose at least one engine for {}.",
                    model.display_name
                ));
            }
            let mut selected = model.clone();
            selected.engines.clear();
            for engine in engines {
                if *engine == EngineKind::Encoder || !model.engines.contains(engine) {
                    return Err(format!(
                        "{} is not available for {}",
                        engine, model.display_name
                    ));
                }
                if !selected.engines.contains(engine) {
                    selected.engines.push(*engine);
                }
            }
            Ok(selected)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Secrets
// ---------------------------------------------------------------------------

/// Generate a `rune-` prefixed random API key (256 bits).
pub fn generate_api_key() -> String {
    use sha2::{Digest, Sha256};
    use std::hash::{BuildHasher, Hasher};

    let mut hasher = Sha256::new();
    // `RandomState` is seeded from the OS CSPRNG; mix several independent
    // instances with the clock and pid so keys never repeat.
    for _ in 0..8 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        hasher.update(h.finish().to_le_bytes());
    }
    hasher.update(std::process::id().to_le_bytes());
    format!("rune-{}", hex::encode(hasher.finalize()))
}

// ---------------------------------------------------------------------------
// Cloud providers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ProviderPreset {
    pub label: &'static str,
    /// Value for the config `provider` field.
    pub provider: &'static str,
    pub env_var: &'static str,
    pub base_url: &'static str,
    pub id_prefix: &'static str,
}

pub const PROVIDERS: &[ProviderPreset] = &[
    ProviderPreset {
        label: "Google Gemini",
        provider: "google",
        env_var: "GEMINI_API_KEY",
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        id_prefix: "gemini",
    },
    ProviderPreset {
        label: "OpenAI",
        provider: "openai",
        env_var: "OPENAI_API_KEY",
        base_url: "https://api.openai.com/v1",
        id_prefix: "openai",
    },
    ProviderPreset {
        label: "Anthropic",
        provider: "anthropic",
        env_var: "ANTHROPIC_API_KEY",
        base_url: "https://api.anthropic.com/v1",
        id_prefix: "anthropic",
    },
    ProviderPreset {
        label: "OpenRouter",
        provider: "openrouter",
        env_var: "OPENROUTER_API_KEY",
        base_url: "https://openrouter.ai/api/v1",
        id_prefix: "openrouter",
    },
];

/// Validate credentials using the same authenticated discovery path as setup.
pub fn validate_provider_key(preset: &ProviderPreset, key: &str) -> Result<(), String> {
    list_provider_models(preset, key).map(|_| ())
}

// ---------------------------------------------------------------------------
// Local model discovery lives in `crate::detect`
// ---------------------------------------------------------------------------

pub use crate::detect::{Discovery, LocalCandidate, Skipped, discover as discover_models};

// ---------------------------------------------------------------------------
// Config generation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exposure {
    /// 127.0.0.1 — only this computer.
    Local,
    /// 0.0.0.0 — reachable from the network.
    Network,
}

impl Exposure {
    pub fn host(self) -> &'static str {
        match self {
            Exposure::Local => "127.0.0.1",
            Exposure::Network => "0.0.0.0",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CloudChoice {
    pub preset: ProviderPreset,
    pub api_key: String,
    pub model: ModelMetadata,
}

#[derive(Debug, Clone)]
pub struct SetupPlan {
    pub exposure: Exposure,
    pub port: u16,
    /// Plain-text server API key (stored in `.env`, referenced from config).
    pub server_api_key: Option<String>,
    pub cloud: Vec<CloudChoice>,
    pub local: Vec<LocalCandidate>,
    pub gpu_available: bool,
}

pub fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '.' {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn local_entry(c: &LocalCandidate, gpu: bool) -> ModelEntry {
    let id = slugify(&c.path.file_stem().unwrap_or_default().to_string_lossy());
    let is_llm = matches!(
        c.modality,
        Modality::Text | Modality::VisionText | Modality::Omni
    );
    let mut description = format!(
        "Imported by setup wizard ({} · {} architecture)",
        c.kind_label, c.architecture
    );
    if !c.needs.is_empty() {
        description.push_str(&format!(
            ". Still needs extra.{} before it will run",
            c.needs.join(", extra.")
        ));
    }
    if let Some(gap) = c.omni_gap() {
        description.push_str(&format!(". {gap}"));
    }
    ModelEntry {
        id,
        name: c.display_name.clone(),
        description,
        modality: c.modality.clone(),
        capabilities: c.capabilities.clone(),
        tasks: Vec::new(),
        roles: Vec::new(),
        aliases: Vec::new(),
        engine: Some(EngineConfig {
            supported: c.engines.clone(),
            preferred: Some(c.engine()),
        }),
        max_context_length: is_llm.then_some(c.context_length).flatten(),
        sampling: if is_llm {
            SamplingConfig {
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: Some(40),
                min_p: Some(0.05),
                max_tokens: Some(4096),
                reasoning_effort: None,
            }
        } else {
            SamplingConfig::default()
        },
        local: Some(LocalBlock {
            path: c.path.to_string_lossy().into_owned(),
            params: {
                let s = format!("{} {}", c.display_name, c.architecture).to_lowercase();
                if s.contains("70b") {
                    Some(70_000_000_000)
                } else if s.contains("32b") {
                    Some(32_000_000_000)
                } else if s.contains("27b") {
                    Some(27_000_000_000)
                } else if s.contains("14b") {
                    Some(14_000_000_000)
                } else if s.contains("9b") || s.contains("8b") {
                    Some(8_000_000_000)
                } else if s.contains("7b") {
                    Some(7_000_000_000)
                } else if s.contains("3b") || s.contains("4b") {
                    Some(3_500_000_000)
                } else {
                    None
                }
            },
            bits_per_weight: (c.engine() == EngineKind::Litert).then_some(32.0),
            context_length: -1,
            gpu_layers: if gpu || c.engine() == EngineKind::Litert {
                99
            } else {
                0
            },
            // Enable flash attention and quantized KV cache for performance and VRAM safety
            args: if c.engine() == EngineKind::SdServer {
                "--diffusion-fa".to_string()
            } else if gpu
                && matches!(
                    c.engine(),
                    EngineKind::LlamaServer | EngineKind::LlamaServerPrism
                )
            {
                "-fa on --cache-type-k q8_0 --cache-type-v q8_0".to_string()
            } else {
                String::new()
            },
        }),
        remote: None,
        extra: c.extra.clone(),
    }
}

fn cloud_entry(c: &CloudChoice) -> ModelEntry {
    ModelEntry {
        id: format!("{}-{}", c.preset.id_prefix, slugify(&c.model.id)),
        name: format!("{} · {}", c.preset.label, c.model.name),
        description: format!("{} model added by setup wizard", c.preset.label),
        modality: if c.model.capabilities.vision == Some(true) {
            Modality::VisionText
        } else {
            Modality::Text
        },
        capabilities: c.model.catalog_capabilities(),
        tasks: Vec::new(),
        roles: Vec::new(),
        aliases: Vec::new(),
        engine: None,
        max_context_length: c.model.context_tokens,
        sampling: SamplingConfig::default(),
        local: None,
        remote: Some(RemoteBlock {
            provider: ProviderType::from_name(c.preset.provider).unwrap_or(ProviderType::Custom),
            upstream_model: c.model.id.clone(),
            api_key: format!("{{{{{}}}}}", c.preset.env_var),
            base_url: Some(c.preset.base_url.to_string()),
            rate_limit: None,
        }),
        extra: serde_json::Map::new(),
    }
}

/// Merge `plan` into `server_path` and `catalog_path` (creating them when absent) and
/// the secrets into `env_path`. Existing models with the same id are kept
/// untouched. Returns the number of models added.
pub fn write_setup(
    plan: &SetupPlan,
    server_path: &Path,
    catalog_path: &Path,
    env_path: &Path,
) -> Result<usize> {
    write_setup_inner(plan, None, server_path, catalog_path, env_path)
}

/// Save verified local task metadata; cloud entries remain unverified.
pub fn write_verified_setup(
    plan: &SetupPlan,
    prepared: Option<&PreparedLocal>,
    server_path: &Path,
    catalog_path: &Path,
    env_path: &Path,
) -> Result<usize> {
    anyhow::ensure!(
        plan.local.is_empty() || prepared.is_some(),
        "Local models require successful inference checks before saving."
    );
    write_setup_inner(plan, prepared, server_path, catalog_path, env_path)
}

fn write_setup_inner(
    plan: &SetupPlan,
    prepared: Option<&PreparedLocal>,
    server_path: &Path,
    catalog_path: &Path,
    env_path: &Path,
) -> Result<usize> {
    let local_entries = match prepared {
        Some(proof) => proof
            .entries(&plan.local, plan.gpu_available)
            .map_err(anyhow::Error::msg)?,
        None => plan
            .local
            .iter()
            .map(|model| local_entry(model, plan.gpu_available))
            .collect(),
    };
    for choice in &plan.cloud {
        let reviewed = registered_model(choice.preset.provider, &choice.model.id)
            .map_err(anyhow::Error::msg)?
            .ok_or_else(|| {
                anyhow::anyhow!("Selected cloud model is missing from registry/models.json.")
            })?;
        if reviewed != choice.model {
            anyhow::bail!("Selected cloud metadata differs from its reviewed registry entry.");
        }
        if choice.api_key.trim().is_empty() {
            anyhow::bail!("Selected cloud provider requires an API key.");
        }
        if choice.model.provider != choice.preset.provider {
            anyhow::bail!("Cloud model/provider mismatch.");
        }
        if let Some(error) = choice.model.selection_error() {
            anyhow::bail!("{}: {error}", choice.model.id);
        }
    }
    let mut server_file: ServerFile = if server_path.exists() {
        let text = std::fs::read_to_string(server_path)?;
        let version = serde_json::from_str::<Value>(&text)
            .context("existing server config is not valid JSON; refusing to overwrite it")?
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if version != u64::from(SCHEMA_VERSION) {
            anyhow::bail!(
                "{} uses an older config schema (version {version}). Run `rune-infer migrate` first, then run setup again.",
                server_path.display()
            );
        }
        serde_json::from_str(&text)
            .context("existing server config is not valid; refusing to overwrite it")?
    } else {
        ServerFile {
            schema_version: SCHEMA_VERSION,
            server: ServerConfig::default(),
        }
    };

    server_file.server.host = plan.exposure.host().to_string();
    server_file.server.port = plan.port;
    server_file.server.api_key = plan
        .server_api_key
        .is_some()
        .then(|| "{{RUNE_API_KEY}}".to_string());

    let mut catalog_file: CatalogFile = if catalog_path.exists() {
        let text = std::fs::read_to_string(catalog_path)?;
        let version = serde_json::from_str::<Value>(&text)
            .context("existing catalog config is not valid JSON; refusing to overwrite it")?
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if version != u64::from(SCHEMA_VERSION) {
            anyhow::bail!(
                "{} uses an older config schema (version {version}). Run `rune-infer migrate` first, then run setup again.",
                catalog_path.display()
            );
        }
        serde_json::from_str(&text)
            .context("existing catalog config is not valid; refusing to overwrite it")?
    } else {
        CatalogFile {
            schema_version: SCHEMA_VERSION,
            models: Vec::new(),
            roles: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
        }
    };

    let mut added = 0;
    let new_entries = local_entries
        .into_iter()
        .chain(plan.cloud.iter().map(cloud_entry));
    for mut entry in new_entries {
        // Re-running setup must not duplicate a model it already added, but two
        // different files that slug to the same id both need to be kept.
        let same_source = |m: &ModelEntry| {
            m.id == entry.id
                && match (&m.local, &entry.local) {
                    (Some(a), Some(b)) => a.path == b.path,
                    _ => m.remote.is_some() == entry.remote.is_some(),
                }
        };
        if let Some(existing) = catalog_file
            .models
            .iter_mut()
            .find(|model| same_source(model))
        {
            if existing.tasks.is_empty() && !entry.tasks.is_empty() {
                let execution = |model: &ModelEntry| -> Result<Value> {
                    let value = serde_json::to_value(model)?;
                    Ok(
                        serde_json::json!({"engine":value["engine"], "local":value["local"],
                        "sampling":value["sampling"], "extra":value["extra"], "max_context_length":value["max_context_length"]}),
                    )
                };
                anyhow::ensure!(
                    execution(existing)? == execution(&entry)?,
                    "Model '{}' has settings that differ from the checked setup selection. Existing configuration is unchanged; review the conflict before replacing it.",
                    existing.id
                );
                existing.tasks = entry.tasks;
                added += 1;
            }
            continue;
        }
        let base = entry.id.clone();
        let mut n = 2;
        while catalog_file.models.iter().any(|m| m.id == entry.id) {
            entry.id = format!("{base}-{n}");
            n += 1;
        }
        catalog_file.models.push(entry);
        added += 1;
    }

    let mut secrets: Vec<(String, String)> = plan
        .cloud
        .iter()
        .map(|c| (c.preset.env_var.to_string(), c.api_key.clone()))
        .collect();
    if let Some(key) = &plan.server_api_key {
        secrets.push(("RUNE_API_KEY".into(), key.clone()));
    }
    let combined = serde_json::json!({"schema_version": SCHEMA_VERSION, "server": server_file.server,
        "models": catalog_file.models, "roles": catalog_file.roles, "aliases": catalog_file.aliases});
    let interpolated = crate::config::interpolate_with(&combined.to_string(), &|name| {
        secrets
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
            .or_else(|| std::env::var(name).ok())
    })?;
    ModelRegistry::from_json(&interpolated, None)?
        .validate()
        .context("generated config failed validation; existing files are unchanged")?;
    if let Some(proof) = prepared {
        proof
            .validate(&plan.local, plan.gpu_available)
            .map_err(anyhow::Error::msg)?;
    }

    if let Some(dir) = server_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        server_path,
        serde_json::to_string_pretty(&server_file)? + "\n",
    )?;

    if let Some(dir) = catalog_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        catalog_path,
        serde_json::to_string_pretty(&catalog_file)? + "\n",
    )?;

    merge_env_file(env_path, &secrets)?;

    // Fail now (not at next start-up) if what we wrote doesn't load.
    ModelRegistry::from_files_with_env(server_path, catalog_path, &secrets)
        .context("generated config failed validation")?;
    Ok(added)
}

fn env_quote(v: &str) -> String {
    if v.chars()
        .any(|c| c.is_whitespace() || "#'\"\\$".contains(c))
    {
        format!("'{}'", v.replace('\'', ""))
    } else {
        v.to_string()
    }
}

/// Upsert `KEY=value` lines, preserving all other content, and restrict file
/// permissions to the owner on unix.
pub fn merge_env_file(path: &Path, secrets: &[(String, String)]) -> Result<()> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut remaining: BTreeMap<&str, &str> = secrets
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let mut lines: Vec<String> = existing
        .lines()
        .map(|line| {
            let key = line.trim_start().split('=').next().unwrap_or("").trim();
            match remaining.remove(key) {
                Some(v) if !line.trim_start().starts_with('#') => format!("{key}={}", env_quote(v)),
                _ => line.to_string(),
            }
        })
        .collect();
    for (k, v) in remaining {
        lines.push(format!("{k}={}", env_quote(v)));
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, lines.join("\n") + "\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs() {
        assert_eq!(slugify("OxCoder-9B.i1-Q6_K"), "oxcoder-9b.i1-q6-k");
        assert_eq!(slugify("  Hello World! "), "hello-world");
    }

    #[test]
    fn api_keys_are_unique_and_prefixed() {
        let (a, b) = (generate_api_key(), generate_api_key());
        assert_ne!(a, b);
        assert!(a.starts_with("rune-") && a.len() == 5 + 64);
    }

    #[test]
    fn env_merge_preserves_and_updates() {
        let dir = std::env::temp_dir().join(format!("rune-env-{}", generate_api_key()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(".env");
        std::fs::write(&p, "# keep me\nOTHER=1\nGEMINI_API_KEY=old\n").unwrap();
        merge_env_file(
            &p,
            &[
                ("GEMINI_API_KEY".into(), "new".into()),
                ("RUNE_API_KEY".into(), "k".into()),
            ],
        )
        .unwrap();
        let out = std::fs::read_to_string(&p).unwrap();
        assert!(out.contains("# keep me") && out.contains("OTHER=1"));
        assert!(out.contains("GEMINI_API_KEY=new") && !out.contains("old"));
        assert!(out.contains("RUNE_API_KEY=k"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn written_config_matches_what_each_file_actually_is() {
        use crate::detect::testing::Gguf;
        let dir = std::env::temp_dir().join(format!("rune-cfg-{}", generate_api_key()));
        let models = dir.join("models");
        Gguf::llm("qwen35", "Chat 9B").write(&models.join("chat/chat-Q6_K.gguf"));
        Gguf::new("clip").write(&models.join("chat/mmproj-F16.gguf"));
        Gguf::new("audiocpp")
            .string("audiocpp.model_spec.family", "voxcpm2")
            .write(&models.join("voice/voxcpm2-q8_0.gguf"));
        Gguf::new("lumina2").write(&models.join("img/z-image-Q8_0.gguf"));

        let found = discover_models(&models).unwrap();
        assert_eq!(found.models.len(), 3);
        let plan = SetupPlan {
            exposure: Exposure::Local,
            port: 8080,
            server_api_key: Some("k".into()),
            cloud: vec![CloudChoice {
                preset: PROVIDERS[0],
                api_key: "secret".into(),
                model: registered_model("google", "gemini-3.5-flash-lite")
                    .unwrap()
                    .unwrap(),
            }],
            local: found.models,
            gpu_available: true,
        };
        let srv = dir.join("config/server.json");
        let cat = dir.join("config/catalog.json");
        assert_eq!(
            write_setup(&plan, &srv, &cat, &dir.join(".env")).unwrap(),
            4
        );
        assert_eq!(
            write_setup(&plan, &srv, &cat, &dir.join(".env")).unwrap(),
            0,
            "idempotent"
        );

        let reg = ModelRegistry::from_files_with_env(
            &srv,
            &cat,
            &[
                ("GEMINI_API_KEY".into(), "k".into()),
                ("RUNE_API_KEY".into(), "k".into()),
            ],
        )
        .unwrap();
        let get = |id: &str| {
            reg.models
                .iter()
                .find(|m| m.id == id)
                .unwrap_or_else(|| panic!("{id}"))
        };
        let chat = get("chat-q6-k");
        assert_eq!(
            (chat.modality.clone(), chat.engine_plan()),
            (Modality::VisionText, vec![EngineKind::LlamaServer])
        );
        assert!(
            chat.mmproj_path
                .as_deref()
                .unwrap()
                .ends_with("mmproj-F16.gguf")
        );
        assert!(chat.sampling.is_some());
        let voice = get("voxcpm2-q8-0");
        assert_eq!(
            (voice.modality.clone(), voice.engine_plan()),
            (Modality::TextToSpeech, vec![EngineKind::AudioCpp])
        );
        assert!(
            voice.sampling.is_none(),
            "chat sampling defaults are not written for non-chat models"
        );
        let img = get("z-image-q8-0");
        assert_eq!(
            (img.modality.clone(), img.engine_plan()),
            (Modality::ImageGeneration, vec![EngineKind::SdServer])
        );
        assert!(
            img.description.contains("extra.vae_path"),
            "{}",
            img.description
        );
        assert!(
            reg.models.iter().all(|model| model.tasks.is_empty()),
            "Discovery alone must not establish verified tasks"
        );
        let raw = std::fs::read_to_string(&cat).unwrap();
        assert!(raw.contains("{{GEMINI_API_KEY}}") && !raw.contains("secret"));
        std::fs::remove_dir_all(dir).ok();
    }

    /// `RUNE_PROBE_DIR=D:/Models RUNE_PROBE_OUT=/tmp/out cargo test write_probe_config -- --ignored`
    #[test]
    #[ignore]
    fn write_probe_config() {
        let (Ok(dir), Ok(out)) = (
            std::env::var("RUNE_PROBE_DIR"),
            std::env::var("RUNE_PROBE_OUT"),
        ) else {
            return;
        };
        let found = discover_models(std::path::Path::new(&dir)).unwrap();
        let plan = SetupPlan {
            exposure: Exposure::Local,
            port: 8080,
            server_api_key: None,
            cloud: vec![],
            local: found.models.into_iter().filter(|c| c.is_ready()).collect(),
            gpu_available: true,
        };
        let out = std::path::PathBuf::from(out);
        write_setup(
            &plan,
            &out.join("config/server.json"),
            &out.join("config/catalog.json"),
            &out.join(".env"),
        )
        .unwrap();
    }

    #[test]
    fn different_files_with_the_same_slug_both_survive() {
        use crate::detect::testing::Gguf;
        let dir = std::env::temp_dir().join(format!("rune-dup-{}", generate_api_key()));
        Gguf::llm("llama", "Model A").write(&dir.join("a/model.gguf"));
        Gguf::llm("llama", "Model B").write(&dir.join("b/model.gguf"));
        let found = discover_models(&dir).unwrap();
        let plan = SetupPlan {
            exposure: Exposure::Local,
            port: 8080,
            server_api_key: None,
            cloud: vec![],
            local: found.models,
            gpu_available: false,
        };
        let srv = dir.join("config/server.json");
        let cat = dir.join("config/catalog.json");
        assert_eq!(
            write_setup(&plan, &srv, &cat, &dir.join(".env")).unwrap(),
            2
        );
        let raw = std::fs::read_to_string(&cat).unwrap();
        assert!(raw.contains("\"id\": \"model\"") && raw.contains("\"id\": \"model-2\""));
        std::fs::remove_dir_all(dir).ok();
    }
}
