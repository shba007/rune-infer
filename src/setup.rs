//! First-run setup logic: prerequisite detection, local model discovery,
//! cloud provider presets and safe config/`.env` generation.
//!
//! This module is UI-agnostic (no terminal code) so it can be unit tested and
//! reused by other front-ends; the interactive wizard lives in `tui::wizard`.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::config::{
    EngineConfig, Modality, ModelRegistry, ProviderType, SamplingConfig, ServerConfig,
};
use crate::schema::{LocalBlock, ModelEntry, RegistryFile, RemoteBlock, SCHEMA_VERSION};

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
    use crate::inference::engines::llama_server::{RuntimeFlavor, install};
    install::ensure_binary_installed(RuntimeFlavor::Upstream).map_err(|e| e.to_string())
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
    pub default_model: &'static str,
    pub id_prefix: &'static str,
}

pub const PROVIDERS: &[ProviderPreset] = &[
    ProviderPreset {
        label: "Google Gemini",
        provider: "google",
        env_var: "GEMINI_API_KEY",
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        default_model: "gemini-flash-latest",
        id_prefix: "gemini",
    },
    ProviderPreset {
        label: "OpenAI",
        provider: "openai",
        env_var: "OPENAI_API_KEY",
        base_url: "https://api.openai.com/v1",
        default_model: "gpt-4o-mini",
        id_prefix: "openai",
    },
    ProviderPreset {
        label: "Anthropic",
        provider: "anthropic",
        env_var: "ANTHROPIC_API_KEY",
        base_url: "https://api.anthropic.com/v1",
        default_model: "claude-haiku-5-5",
        id_prefix: "anthropic",
    },
    ProviderPreset {
        label: "OpenRouter",
        provider: "openrouter",
        env_var: "OPENROUTER_API_KEY",
        base_url: "https://openrouter.ai/api/v1",
        default_model: "openrouter/auto",
        id_prefix: "openrouter",
    },
];

/// Check an API key by listing models. Blocking; run on a worker thread.
pub fn validate_provider_key(preset: &ProviderPreset, key: &str) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/models", preset.base_url.trim_end_matches('/'));
    let req = client.get(url);
    let req = match preset.provider {
        "google" => req.header("x-goog-api-key", key),
        "anthropic" => req
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01"),
        _ => req.bearer_auth(key),
    };
    let resp = req.send().map_err(|e| format!("network error: {e}"))?;
    match resp.status().as_u16() {
        200..=299 => Ok(()),
        401 | 403 => Err("the provider rejected this key (unauthorized)".into()),
        code => Err(format!("provider returned HTTP {code}")),
    }
}

// ---------------------------------------------------------------------------
// Local model discovery
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LocalCandidate {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub architecture: String,
    pub display_name: String,
    pub context_length: Option<u32>,
    pub mmproj: Option<PathBuf>,
}

/// Find GGUF models at `input` (a file, or a folder searched up to 5 levels).
pub fn discover_gguf(input: &Path) -> Result<Vec<LocalCandidate>> {
    if !input.exists() {
        anyhow::bail!("'{}' does not exist", input.display());
    }
    let mut files = Vec::new();
    if input.is_file() {
        files.push(input.to_path_buf());
    } else {
        walk_gguf(input, 0, &mut files);
    }

    let mut out = Vec::new();
    for path in files {
        let fname = path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !fname.ends_with(".gguf") || fname.contains("mmproj") {
            continue;
        }
        let Ok(meta) = read_gguf_metadata(&path) else {
            continue;
        };
        out.push(LocalCandidate {
            size_bytes: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            mmproj: find_mmproj(&path),
            display_name: meta.name.unwrap_or_else(|| {
                path.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into()
            }),
            architecture: meta.architecture,
            context_length: meta.context_length,
            path,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn walk_gguf(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 5 || out.len() >= 300 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_gguf(&path, depth + 1, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

fn find_mmproj(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .is_some_and(|n| n.contains("mmproj") && n.ends_with(".gguf"))
        })
        .collect();
    found.sort_by_key(|p| {
        // Prefer f16 projectors over quantized/bf16 ones.
        let n = p.to_string_lossy().to_lowercase();
        (!n.contains("f16") || n.contains("bf16"), n)
    });
    found.into_iter().next()
}

#[derive(Debug, Default)]
pub struct GgufMetadata {
    pub architecture: String,
    pub name: Option<String>,
    pub context_length: Option<u32>,
}

/// Read the handful of GGUF header keys the wizard needs, stopping before the
/// (huge) tokenizer section.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let file = std::fs::File::open(path)?;
    parse_gguf_metadata(&mut std::io::BufReader::new(file))
}

fn rd<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn rd_u32(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(rd(r)?))
}
fn rd_u64(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(rd(r)?))
}
fn rd_string(r: &mut impl Read) -> Result<String> {
    let len = rd_u64(r)? as usize;
    anyhow::ensure!(len <= 1 << 20, "implausible string length in GGUF header");
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_value(r: &mut impl Read, ty: u32) -> Result<()> {
    let fixed = |r: &mut dyn Read, n: u64| -> Result<()> {
        std::io::copy(&mut r.take(n), &mut std::io::sink())?;
        Ok(())
    };
    match ty {
        0 | 1 | 7 => fixed(r, 1),
        2 | 3 => fixed(r, 2),
        4..=6 => fixed(r, 4),
        10..=12 => fixed(r, 8),
        8 => rd_string(r).map(|_| ()),
        9 => {
            let elem = rd_u32(r)?;
            let n = rd_u64(r)?;
            for _ in 0..n {
                skip_value(r, elem)?;
            }
            Ok(())
        }
        _ => anyhow::bail!("unknown GGUF value type {ty}"),
    }
}

pub fn parse_gguf_metadata(r: &mut impl Read) -> Result<GgufMetadata> {
    anyhow::ensure!(&rd::<4>(r)? == b"GGUF", "not a GGUF file");
    let version = rd_u32(r)?;
    anyhow::ensure!(
        (2..=3).contains(&version),
        "unsupported GGUF version {version}"
    );
    let _tensors = rd_u64(r)?;
    let kv_count = rd_u64(r)?;

    let mut meta = GgufMetadata::default();
    let mut ctx_key = String::new();
    for _ in 0..kv_count.min(512) {
        let key = rd_string(r)?;
        let ty = rd_u32(r)?;
        if key.starts_with("tokenizer.") {
            break;
        }
        let wants_u32 = ty == 4 && key == ctx_key;
        match (key.as_str(), ty) {
            ("general.architecture", 8) => {
                meta.architecture = rd_string(r)?;
                ctx_key = format!("{}.context_length", meta.architecture);
            }
            ("general.name", 8) => meta.name = Some(rd_string(r)?),
            _ if wants_u32 => meta.context_length = Some(rd_u32(r)?),
            _ => skip_value(r, ty)?,
        }
        if !meta.architecture.is_empty() && meta.name.is_some() && meta.context_length.is_some() {
            break;
        }
    }
    anyhow::ensure!(
        !meta.architecture.is_empty(),
        "GGUF has no architecture key"
    );
    Ok(meta)
}

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
    pub upstream_model: String,
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
    let vision = c.mmproj.is_some();
    let mut caps = vec!["Chat", "Tool Call", "Structured Output"];
    if vision {
        caps.push("Vision");
    }
    let kind = crate::migrate::llama_engine_for_architecture(&c.architecture);
    let mut extra = serde_json::Map::new();
    if let Some(mm) = &c.mmproj {
        extra.insert("mmproj_path".into(), json!(mm.to_string_lossy()));
    }
    ModelEntry {
        id,
        name: c.display_name.clone(),
        description: format!("Imported by setup wizard ({} architecture)", c.architecture),
        modality: if vision {
            Modality::VisionText
        } else {
            Modality::Text
        },
        capabilities: caps.into_iter().map(String::from).collect(),
        engine: Some(EngineConfig {
            supported: vec![kind],
            preferred: Some(kind),
        }),
        max_context_length: c.context_length,
        sampling: SamplingConfig {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            max_tokens: Some(4096),
        },
        local: Some(LocalBlock {
            path: c.path.to_string_lossy().into_owned(),
            params: None,
            bits_per_weight: None,
            context_length: -1,
            gpu_layers: if gpu { 99 } else { 0 },
            args: String::new(),
        }),
        remote: None,
        extra,
    }
}

fn cloud_entry(c: &CloudChoice) -> ModelEntry {
    ModelEntry {
        id: format!("{}-{}", c.preset.id_prefix, slugify(&c.upstream_model)),
        name: format!("{} · {}", c.preset.label, c.upstream_model),
        description: format!("{} model added by setup wizard", c.preset.label),
        modality: Modality::VisionText,
        capabilities: ["Chat", "Tool Call", "Vision", "Structured Output"]
            .into_iter()
            .map(String::from)
            .collect(),
        engine: None,
        max_context_length: None,
        sampling: SamplingConfig::default(),
        local: None,
        remote: Some(RemoteBlock {
            provider: ProviderType::from_name(c.preset.provider).unwrap_or(ProviderType::Custom),
            upstream_model: c.upstream_model.clone(),
            api_key: format!("{{{{{}}}}}", c.preset.env_var),
            base_url: Some(c.preset.base_url.to_string()),
            rate_limit: None,
        }),
        extra: serde_json::Map::new(),
    }
}

/// Merge `plan` into the config at `config_path` (creating it when absent) and
/// the secrets into `env_path`. Existing models with the same id are kept
/// untouched. Returns the number of models added.
pub fn write_setup(plan: &SetupPlan, config_path: &Path, env_path: &Path) -> Result<usize> {
    let mut file: RegistryFile = if config_path.exists() {
        let text = std::fs::read_to_string(config_path)?;
        let version = serde_json::from_str::<Value>(&text)
            .context("existing config is not valid JSON; refusing to overwrite it")?
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if version != u64::from(SCHEMA_VERSION) {
            anyhow::bail!(
                "{} uses an older config schema (version {version}). Run `rune-infer migrate` first, then run setup again.",
                config_path.display()
            );
        }
        serde_json::from_str(&text)
            .context("existing config is not valid; refusing to overwrite it")?
    } else {
        RegistryFile {
            schema_version: SCHEMA_VERSION,
            server: ServerConfig::default(),
            models: Vec::new(),
        }
    };

    file.server.host = plan.exposure.host().to_string();
    file.server.port = plan.port;
    file.server.api_key = plan
        .server_api_key
        .is_some()
        .then(|| "{{RUNE_API_KEY}}".to_string());

    let mut added = 0;
    let new_entries = plan
        .local
        .iter()
        .map(|c| local_entry(c, plan.gpu_available))
        .chain(plan.cloud.iter().map(cloud_entry));
    for entry in new_entries {
        if file.models.iter().any(|m| m.id == entry.id) {
            continue;
        }
        file.models.push(entry);
        added += 1;
    }

    if let Some(dir) = config_path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        config_path,
        serde_json::to_string_pretty(&file)?
            + "
",
    )?;

    let mut secrets: Vec<(String, String)> = plan
        .cloud
        .iter()
        .map(|c| (c.preset.env_var.to_string(), c.api_key.clone()))
        .collect();
    if let Some(k) = &plan.server_api_key {
        secrets.push(("RUNE_API_KEY".into(), k.clone()));
    }
    merge_env_file(env_path, &secrets)?;

    // Fail now (not at next start-up) if what we wrote doesn't load.
    ModelRegistry::from_file_with_env(config_path, &secrets)
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

    fn gguf_bytes(arch: &str, name: &str, ctx: u32) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(4u64.to_le_bytes());
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, arch);
        // a value we must skip: array of 2 strings
        s(&mut b, "general.tags");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        s(&mut b, "a");
        s(&mut b, "bb");
        s(&mut b, "general.name");
        b.extend(8u32.to_le_bytes());
        s(&mut b, name);
        s(&mut b, &format!("{arch}.context_length"));
        b.extend(4u32.to_le_bytes());
        b.extend(ctx.to_le_bytes());
        b
    }

    #[test]
    fn parses_gguf_header_and_skips_arrays() {
        let bytes = gguf_bytes("qwen35", "My Model", 32768);
        let m = parse_gguf_metadata(&mut bytes.as_slice()).unwrap();
        assert_eq!(m.architecture, "qwen35");
        assert_eq!(m.name.as_deref(), Some("My Model"));
        assert_eq!(m.context_length, Some(32768));
    }

    #[test]
    fn rejects_non_gguf() {
        assert!(parse_gguf_metadata(&mut b"NOPE....".as_slice()).is_err());
    }

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
    fn written_config_loads_and_validates() {
        let dir = std::env::temp_dir().join(format!("rune-cfg-{}", generate_api_key()));
        let model = dir.join("m.gguf");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&model, gguf_bytes("llama", "Tiny", 4096)).unwrap();

        let local = discover_gguf(&model).unwrap();
        assert_eq!(local.len(), 1);
        let plan = SetupPlan {
            exposure: Exposure::Local,
            port: 8080,
            server_api_key: Some("k".into()),
            cloud: vec![CloudChoice {
                preset: PROVIDERS[0],
                api_key: "secret".into(),
                upstream_model: "gemini-flash-latest".into(),
            }],
            local,
            gpu_available: false,
        };
        let cfg = dir.join("config/models.json");
        assert_eq!(write_setup(&plan, &cfg, &dir.join(".env")).unwrap(), 2);
        // Re-running is idempotent.
        assert_eq!(write_setup(&plan, &cfg, &dir.join(".env")).unwrap(), 0);
        let raw = std::fs::read_to_string(&cfg).unwrap();
        assert!(raw.contains("{{GEMINI_API_KEY}}") && !raw.contains("secret"));
        std::fs::remove_dir_all(dir).ok();
    }
}
