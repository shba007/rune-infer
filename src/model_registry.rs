//! Local model entries in the remotely hosted, reviewed model registry.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/shba007/rune-infer/main/registry/models.json";
const LIMIT: u64 = 4 * 1024 * 1024;

#[derive(Deserialize)]
struct Registry {
    schema_version: u32,
    #[serde(default)]
    local_models: Vec<LocalModel>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalModel {
    pub id: String,
    #[serde(default)]
    pub install_dir: Option<String>,
    pub name: String,
    pub description: String,
    pub license: String,
    pub source_url: String,
    pub engine: String,
    pub format: String,
    pub capabilities: Vec<String>,
    pub latest: String,
    pub versions: BTreeMap<String, Version>,
    #[serde(default)]
    pub catalog: Option<CatalogDefaults>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogDefaults {
    pub tasks: Vec<String>,
    pub capabilities: Vec<String>,
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub sampling: Option<serde_json::Value>,
    pub local: LocalDefaults,
    #[serde(default)]
    pub engines: Option<Vec<String>>,
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub audio_operations: Option<crate::audio_operations::AudioOperations>,
    #[serde(default)]
    pub tool_call_support: Option<crate::audio_operations::OperationSupport>,
    #[serde(default)]
    pub structured_output_support: Option<crate::audio_operations::OperationSupport>,
    #[serde(default)]
    pub image_reference_checks: Vec<crate::config::ImageReferenceVerification>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalDefaults {
    #[serde(default)]
    pub path: Option<String>,
    pub params: Option<u64>,
    pub bits_per_weight: Option<f64>,
    pub context_length: i64,
    pub gpu_layers: u32,
    pub args: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Version {
    pub release_number: u64,
    pub files: Vec<File>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    pub id: String,
    pub filename: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

pub fn fetch() -> Result<Vec<LocalModel>> {
    parse(&fetch_document()?)
}

/// The registry reviewed and shipped with this Infer build. Hosts can use this
/// when model selection must match the exact Infer version they have bundled.
pub fn bundled() -> Result<Vec<LocalModel>> {
    parse(include_bytes!("../registry/models.json")).context("Invalid bundled model registry")
}

/// Load model metadata from an explicitly selected local registry file.
pub fn load_from_path(path: &Path) -> Result<Vec<LocalModel>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open model registry {}", path.display()))?;
    let mut body = Vec::new();
    file.take(LIMIT + 1)
        .read_to_end(&mut body)
        .with_context(|| format!("Cannot read model registry {}", path.display()))?;
    ensure!(
        body.len() as u64 <= LIMIT,
        "Model registry exceeds size limit"
    );
    parse(&body).with_context(|| format!("Invalid model registry {}", path.display()))
}

pub(crate) fn fetch_document() -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(15))
        .build()?;
    let mut body = Vec::new();
    client
        .get(REGISTRY_URL)
        .send()?
        .error_for_status()?
        .take(LIMIT + 1)
        .read_to_end(&mut body)?;
    ensure!(
        body.len() as u64 <= LIMIT,
        "Model registry exceeds size limit"
    );
    Ok(body)
}

fn parse(bytes: &[u8]) -> Result<Vec<LocalModel>> {
    let registry: Registry = serde_json::from_slice(bytes).context("Invalid model registry")?;
    ensure!(
        registry.schema_version == 1,
        "Unsupported model registry schema"
    );
    let mut ids = HashSet::new();
    for model in &registry.local_models {
        ensure!(
            valid_model_id(&model.id) && ids.insert(&model.id),
            "Invalid or duplicate local model ID"
        );
        ensure!(
            !model.name.trim().is_empty() && !model.license.trim().is_empty(),
            "Missing model name or license"
        );
        ensure!(
            reqwest::Url::parse(&model.source_url)?.scheme() == "https",
            "Model source must use HTTPS"
        );
        ensure!(
            matches!(
                model.engine.as_str(),
                "llama-server" | "llama-server-prism" | "nemo-speech" | "audio-cpp" | "sd-server"
            ) && model.format == "gguf"
                || (model.engine == "onnx"
                    && model.format == "onnx"
                    && model.capabilities == ["Portrait Matting"]),
            "Unsupported model format or engine"
        );
        let latest = model
            .versions
            .get(&model.latest)
            .context("Missing selected model version")?;
        if let Some(dir) = &model.install_dir {
            validate_install_path(&format!("{dir}/model.gguf"), "model.gguf")?;
            ensure!(
                dir.split('/').count() == 2,
                "Model install directory must contain creator and family"
            );
        }
        if let Some(path) = model
            .catalog
            .as_ref()
            .and_then(|catalog| catalog.local.path.as_deref())
        {
            let main = latest
                .files
                .iter()
                .find(|file| file.id == "model")
                .context("Model file missing")?;
            validate_install_path(path, &main.filename)?;
        }
        let mut order = HashSet::new();
        for version in model.versions.values() {
            ensure!(
                version.release_number > 0
                    && order.insert(version.release_number)
                    && version.release_number <= latest.release_number,
                "Invalid model release order"
            );
            validate_bundle(model, version)?;
            for file in &version.files {
                ensure!(
                    !file.filename.is_empty()
                        && (file.filename.ends_with(".gguf")
                            || file.filename.ends_with(".safetensors")
                            || file.filename.ends_with(".json")
                            || file.filename.ends_with(".onnx"))
                        && file
                            .filename
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
                    "Invalid model filename"
                );
                let url = reqwest::Url::parse(&file.url)?;
                ensure!(
                    url.scheme() == "https"
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.fragment().is_none(),
                    "Invalid model download URL"
                );
                ensure!(
                    file.size_bytes > 0
                        && file.sha256.len() == 64
                        && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                    "Invalid model integrity metadata"
                );
            }
        }
    }
    Ok(registry.local_models)
}

fn valid_model_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && !id.ends_with('.')
        && !id.contains("..")
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'.'))
}

fn validate_install_path<'a>(path: &'a str, filename: &str) -> Result<Vec<&'a str>> {
    let parts: Vec<_> = path.split('/').collect();
    ensure!(
        !parts.is_empty() && parts.len() <= 32 && parts.last() == Some(&filename),
        "Registry model path must end in the model filename"
    );
    for part in &parts {
        ensure!(
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && !part.ends_with('.')
                && part.trim() == *part
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._- ".contains(&b)),
            "Registry model path must stay under the models directory"
        );
        let stem = part.split('.').next().unwrap_or("").to_ascii_uppercase();
        ensure!(
            !matches!(
                stem.as_str(),
                "CON"
                    | "PRN"
                    | "AUX"
                    | "NUL"
                    | "COM1"
                    | "COM2"
                    | "COM3"
                    | "COM4"
                    | "COM5"
                    | "COM6"
                    | "COM7"
                    | "COM8"
                    | "COM9"
                    | "LPT1"
                    | "LPT2"
                    | "LPT3"
                    | "LPT4"
                    | "LPT5"
                    | "LPT6"
                    | "LPT7"
                    | "LPT8"
                    | "LPT9"
            ),
            "Registry model path contains a reserved filename"
        );
    }
    Ok(parts)
}

fn encoder_directory_file(file: &File) -> bool {
    matches!(
        file.id.as_str(),
        "encoder_config" | "encoder_generation_config" | "encoder_index"
    ) || file
        .id
        .strip_prefix("encoder_shard_")
        .is_some_and(|n| n.len() == 3 && n.bytes().all(|b| b.is_ascii_digit()) && n != "000")
}

fn bundle_path(model_dir: &Path, file: &File) -> PathBuf {
    if encoder_directory_file(file) {
        model_dir.join("text_encoder").join(&file.filename)
    } else {
        model_dir.join(&file.filename)
    }
}

fn validate_encoder_index(version: &Version, index_path: &Path) -> Result<()> {
    let index: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
    let weights = index["weight_map"]
        .as_object()
        .context("Encoder index has no weight map")?;
    ensure!(!weights.is_empty(), "Encoder index is empty");
    let shards: HashSet<_> = version
        .files
        .iter()
        .filter(|file| file.id.starts_with("encoder_shard_"))
        .map(|file| file.filename.as_str())
        .collect();
    let used: HashSet<_> = weights
        .values()
        .map(|value| value.as_str().context("Invalid encoder shard reference"))
        .collect::<Result<_>>()?;
    ensure!(
        used == shards,
        "Encoder index and registry shards do not match"
    );
    Ok(())
}

fn validate_bundle(model: &LocalModel, version: &Version) -> Result<()> {
    let vision = model.capabilities == ["Chat", "Vision"]
        || model.capabilities == ["Chat", "Vision", "Audio", "Video"];
    let image = model.capabilities == ["Image Generation"];
    let matting = model.capabilities == ["Portrait Matting"];
    ensure!(
        matting
            || vision
            || image
            || model.capabilities == ["Chat"]
            || model.capabilities == ["Decision"]
            || model.capabilities == ["Transcription"]
            || model.capabilities == ["Speech Generation"],
        "Unsupported model capabilities"
    );
    if matting
        || image
        || model.capabilities == ["Decision"]
        || model.capabilities == ["Transcription"]
        || model.capabilities == ["Speech Generation"]
        || model.capabilities == ["Chat", "Vision", "Audio", "Video"]
    {
        ensure!(
            model.catalog.is_some(),
            "Specialized model requires reviewed catalog metadata"
        );
    }
    ensure!(
        if image {
            (3..=32).contains(&version.files.len())
        } else {
            version.files.len() == 1 || version.files.len() == 2
        },
        "Model has missing or unsupported companion files"
    );
    if matting {
        ensure!(
            model.engine == "onnx" && model.format == "onnx" && version.files.len() == 1,
            "Unsupported portrait matting bundle"
        );
    } else if image {
        ensure!(model.engine == "sd-server", "Unsupported image engine");
    } else if model.capabilities == ["Transcription"] || model.capabilities == ["Speech Generation"]
    {
        ensure!(
            version.files.len() == 1,
            "Speech model requires a single reviewed GGUF"
        );
        ensure!(
            (model.capabilities == ["Transcription"] && model.engine == "nemo-speech")
                || (model.capabilities == ["Speech Generation"] && model.engine == "audio-cpp"),
            "Unsupported speech engine combination"
        );
    } else {
        ensure!(
            matches!(model.engine.as_str(), "llama-server" | "llama-server-prism"),
            "Unsupported model engine combination"
        );
    }
    let mut roles = HashSet::new();
    let mut names = HashSet::new();
    for file in &version.files {
        ensure!(
            (if image {
                matches!(
                    file.id.as_str(),
                    "model" | "text_encoder" | "vae" | "tokenizer"
                ) || encoder_directory_file(file)
            } else {
                matches!(file.id.as_str(), "model" | "projector")
            }) && roles.insert(&file.id),
            "Duplicate or unsupported model file role"
        );
        ensure!(
            !file.filename.is_empty()
                && (match file.id.as_str() {
                    "tokenizer" => file.filename.ends_with(".json"),
                    "vae" => file.filename.ends_with(".safetensors"),
                    "encoder_config" => file.filename == "config.json",
                    "encoder_generation_config" => file.filename == "generation_config.json",
                    "encoder_index" => file.filename == "model.safetensors.index.json",
                    id if id.starts_with("encoder_shard_") =>
                        file.filename.ends_with(".safetensors"),
                    "text_encoder" =>
                        file.filename.ends_with(".safetensors") || file.filename.ends_with(".gguf"),
                    _ if matting => file.filename.ends_with(".onnx"),
                    _ => file.filename.ends_with(".gguf"),
                })
                && file
                    .filename
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                && names.insert(file.filename.to_ascii_lowercase()),
            "Invalid or duplicate model filename"
        );
    }
    ensure!(
        roles.contains(&"model".to_string())
            && (!vision || roles.contains(&"projector".to_string()))
            && (!image
                || (roles.contains(&"vae".to_string())
                    && (roles.contains(&"text_encoder".to_string())
                        != roles.contains(&"encoder_index".to_string())))),
        "Model capabilities do not match companion files"
    );
    if roles.contains(&"encoder_index".to_string()) {
        ensure!(
            roles.contains(&"encoder_config".to_string())
                && version
                    .files
                    .iter()
                    .any(|file| file.id.starts_with("encoder_shard_")),
            "Sharded encoder is incomplete"
        );
    } else {
        ensure!(
            !version.files.iter().any(encoder_directory_file),
            "Unexpected sharded encoder file"
        );
    }
    Ok(())
}

/// Download a reviewed model and register it in the user's catalog.
pub fn install(model: &LocalModel, root: &Path, server: &Path, catalog: &Path) -> Result<PathBuf> {
    let client = crate::inference::engines::download::create_download_client()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    install_with(model, root, server, catalog, |file, dest| {
        crate::inference::engines::download::with_download_label(&file.filename, || {
            crate::inference::engines::download::download_resumable_model(
                &client,
                &file.url,
                dest,
                "[Model]",
                &file.sha256,
                file.size_bytes,
            )
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))
    })
}

/// Install through the same verified, resumable path used by the dashboard,
/// while exposing progress and cancellation to an embedding host.
pub fn install_observed(
    model: &LocalModel,
    root: &Path,
    server: &Path,
    catalog: &Path,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    progress: impl FnMut(crate::inference::engines::download::DownloadProgress) + 'static,
) -> Result<PathBuf> {
    crate::inference::engines::download::with_cancellation(cancel, || {
        crate::inference::engines::download::with_progress(progress, || {
            install(model, root, server, catalog)
        })
    })
}

fn regular_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    let meta = std::fs::symlink_metadata(path)?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "{} is not a regular directory",
        path.display()
    );
    Ok(())
}

fn read_regular(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            ensure!(
                meta.is_file() && !meta.file_type().is_symlink(),
                "{} is not a regular file",
                path.display()
            );
            Ok(Some(std::fs::read(path)?))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn unique_path(dir: &Path, label: &str) -> PathBuf {
    dir.join(format!(
        ".rune-{label}-{}-{}",
        std::process::id(),
        crate::setup::generate_api_key()
    ))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn verify_file(path: &Path, file: &File) -> Result<()> {
    let meta = std::fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink() && meta.len() == file.size_bytes,
        "Model size or file type does not match registry"
    );
    let mut input = std::fs::File::open(path)?;
    if file.filename.ends_with(".json") {
        ensure!(
            meta.len() <= 64 * 1024 * 1024,
            "Tokenizer exceeds size limit"
        );
        let json: serde_json::Value = serde_json::from_reader(input)?;
        ensure!(json.is_object(), "Invalid tokenizer JSON");
    } else if file.filename.ends_with(".safetensors") {
        let mut length = [0; 8];
        input.read_exact(&mut length)?;
        let header_size = u64::from_le_bytes(length);
        ensure!(
            (2..=100_000_000).contains(&header_size) && header_size <= meta.len().saturating_sub(8),
            "Invalid safetensors header length"
        );
        let mut first = [0; 1];
        input.read_exact(&mut first)?;
        ensure!(first[0] == b'{', "Invalid safetensors header");
    } else if !file.filename.ends_with(".onnx") {
        let mut magic = [0; 4];
        input.read_exact(&mut magic)?;
        ensure!(&magic == b"GGUF", "Downloaded file is not GGUF");
    }
    let hash = crate::inference::engines::download::file_sha256(path)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    ensure!(
        hash.eq_ignore_ascii_case(&file.sha256),
        "Model SHA-256 mismatch"
    );
    Ok(())
}

// Publish without replacing a file created by another process.
fn publish_new(source: &Path, target: &Path) -> Result<()> {
    move_no_replace(source, target)
        .with_context(|| format!("Cannot publish {} without overwriting", target.display()))
}

fn install_with(
    model: &LocalModel,
    root: &Path,
    server_path: &Path,
    catalog_path: &Path,
    mut download: impl FnMut(&File, &Path) -> Result<()>,
) -> Result<PathBuf> {
    // Revalidate even if a caller constructed the model instead of using fetch().
    let version = model
        .versions
        .get(&model.latest)
        .context("Missing selected model version")?;
    ensure!(
        matches!(
            model.engine.as_str(),
            "llama-server" | "llama-server-prism" | "nemo-speech" | "audio-cpp" | "sd-server"
        ) && model.format == "gguf"
            || (model.engine == "onnx"
                && model.format == "onnx"
                && model.capabilities == ["Portrait Matting"]),
        "Unsupported model entry"
    );
    validate_bundle(model, version)?;
    let file = version
        .files
        .iter()
        .find(|file| file.id == "model")
        .context("Model file missing")?;
    let projector = version.files.iter().find(|file| file.id == "projector");
    ensure!(valid_model_id(&model.id), "Invalid model ID");
    regular_dir(root)?;
    let root = std::fs::canonicalize(root)?;
    let models = root.join("models");
    regular_dir(&models)?;
    let mut model_dir = models.clone();
    let catalog_before = read_regular(catalog_path)?;
    let legacy_dir = models.join(&model.id);
    let legacy_dest = legacy_dir.join(&file.filename);
    let registered = catalog_before
        .as_ref()
        .and_then(|raw| serde_json::from_slice::<crate::schema::CatalogFile>(raw).ok())
        .and_then(|catalog| {
            catalog
                .models
                .into_iter()
                .find(|entry| entry.id == model.id)
        });
    let registered_legacy = registered
        .as_ref()
        .and_then(|entry| entry.local.as_ref())
        .is_some_and(|local| Path::new(&local.path) == legacy_dest);
    let interrupted_legacy = registered.is_none()
        && (legacy_dest.exists()
            || version.files.iter().any(|file| {
                let target = bundle_path(&legacy_dir, file);
                target.parent().is_some_and(|parent| {
                    parent
                        .join(format!(
                            ".rune-model-download-{}-{}.part",
                            file.id,
                            file.sha256.to_ascii_lowercase()
                        ))
                        .exists()
                })
            }));
    let use_legacy = registered_legacy || interrupted_legacy;
    if use_legacy {
        model_dir = legacy_dir;
        regular_dir(&model_dir)?;
    } else if let Some(path) = model
        .catalog
        .as_ref()
        .and_then(|catalog| catalog.local.path.as_deref())
    {
        let parts = validate_install_path(path, &file.filename)?;
        for part in &parts[..parts.len() - 1] {
            model_dir.push(part);
            regular_dir(&model_dir)?;
        }
    } else {
        let dir = model
            .install_dir
            .as_deref()
            .context("Registry model has no creator and family install directory")?;
        let path = format!("{dir}/{}", file.filename);
        let parts = validate_install_path(&path, &file.filename)?;
        ensure!(
            parts.len() == 3,
            "Model install directory must contain creator and family"
        );
        for part in &parts[..2] {
            model_dir.push(part);
            regular_dir(&model_dir)?;
        }
    }
    let encoder_dir = model_dir.join("text_encoder");
    if version.files.iter().any(encoder_directory_file) {
        regular_dir(&encoder_dir)?;
    }
    let lock_path = model_dir.join(".rune-model-install.lock");
    if let Ok(meta) = std::fs::symlink_metadata(&lock_path) {
        ensure!(
            meta.is_file() && !meta.file_type().is_symlink(),
            "Model installation lock is not a regular file"
        );
    }
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .context("Cannot open model installation lock")?;
    lock.try_lock().map_err(|_| {
        anyhow::anyhow!("Another installation is already using this model directory")
    })?;
    let dest = model_dir.join(&file.filename);
    let server_before = read_regular(server_path)?;
    if server_before.is_none() && catalog_before.is_none() {
        ensure!(
            !root.join("config/models.json").exists(),
            "Legacy model configuration exists; run migrate before installing models"
        );
    }
    ensure!(
        server_before.is_some() == catalog_before.is_some(),
        "Server and catalog configuration are incomplete; run setup or migrate first"
    );
    let server_value: serde_json::Value = match &server_before {
        Some(raw) => serde_json::from_slice(raw).context("Invalid server configuration")?,
        None => serde_json::to_value(crate::schema::ServerFile {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: crate::config::ServerConfig {
                host: "127.0.0.1".into(),
                ..crate::config::ServerConfig::default()
            },
        })?,
    };
    let mut catalog_value: serde_json::Value = match &catalog_before {
        Some(raw) => serde_json::from_slice(raw).context("Invalid model catalog")?,
        None => {
            serde_json::json!({"schema_version": crate::schema::SCHEMA_VERSION, "models": [], "roles": {}, "aliases": {}})
        }
    };
    let existing: crate::schema::CatalogFile =
        serde_json::from_value(catalog_value.clone()).context("Invalid model catalog")?;
    ensure!(
        existing.schema_version == crate::schema::SCHEMA_VERSION,
        "Catalog requires migration"
    );
    if let Some(entry) = existing.models.iter().find(|entry| entry.id == model.id) {
        ensure!(
            entry
                .local
                .as_ref()
                .is_some_and(|local| Path::new(&local.path) == dest),
            "Model ID already exists with different configuration"
        );
        if let Some(projector) = projector {
            let projector_path = model_dir.join(&projector.filename);
            ensure!(
                entry
                    .extra
                    .get("mmproj_path")
                    .and_then(|value| value.as_str())
                    .is_some_and(|path| Path::new(path) == projector_path),
                "Model ID already exists with a different projector"
            );
        }
        for (role, key) in [
            ("text_encoder", "text_encoder_path"),
            ("vae", "vae_path"),
            ("tokenizer", "tokenizer_path"),
        ] {
            if let Some(companion) = version.files.iter().find(|file| file.id == role) {
                let expected = model_dir.join(&companion.filename);
                ensure!(
                    entry
                        .extra
                        .get(key)
                        .and_then(|value| value.as_str())
                        .is_some_and(|path| Path::new(path) == expected),
                    "Model ID already exists with different companion paths"
                );
            }
        }
        if version.files.iter().any(|file| file.id == "encoder_index") {
            ensure!(
                entry
                    .extra
                    .get("text_encoder_path")
                    .and_then(|value| value.as_str())
                    .is_some_and(|path| Path::new(path) == encoder_dir),
                "Model ID already exists with different encoder directory"
            );
        }
        for file in &version.files {
            verify_file(&bundle_path(&model_dir, file), file)
                .context("Existing model does not match registry")?;
        }
        if let Some(index) = version.files.iter().find(|file| file.id == "encoder_index") {
            validate_encoder_index(version, &bundle_path(&model_dir, index))?;
        }
        return Ok(dest);
    }
    let path = dest.to_str().context("Model path is not valid UTF-8")?;
    let tasks = if projector.is_some() {
        vec!["text-to-text", "text-image-to-text"]
    } else {
        vec!["text-to-text"]
    };
    let mut entry = serde_json::json!({"id": model.id, "name": model.name, "description": model.description,
        "tasks": tasks, "capabilities": model.capabilities, "engine": [model.engine], "local": {"path": path}});
    if let Some(defaults) = &model.catalog {
        let allowed = match model.capabilities.as_slice() {
            [a] if a == "Image Generation" => {
                defaults.tasks == ["text-to-image"]
                    || defaults.tasks == ["text-image-to-image", "image-to-image"]
                    || defaults.tasks == ["text-to-image", "text-image-to-image", "image-to-image"]
            }
            [a] if a == "Decision" => defaults.tasks == ["text-to-decision"],
            [a] if a == "Transcription" => defaults.tasks == ["audio-to-text"],
            [a] if a == "Speech Generation" => defaults.tasks == ["text-to-audio"],
            [a] if a == "Portrait Matting" => {
                defaults.tasks == ["image-to-matting"]
                    && defaults.capabilities == ["Portrait Matting"]
            }
            [a, b, c, d] if a == "Chat" && b == "Vision" && c == "Audio" && d == "Video" => {
                projector.is_some()
                    && defaults.tasks
                        == [
                            "text-to-text",
                            "text-image-to-text",
                            "text-audio-to-text",
                            "text-video-to-text",
                            "text-to-audio",
                        ]
            }
            _ => {
                defaults.tasks == ["text-to-text"]
                    || (projector.is_some()
                        && defaults.tasks == ["text-to-text", "text-image-to-text"])
            }
        };
        ensure!(
            allowed,
            "Catalog tasks do not match the verified model files"
        );
        if version.files.iter().any(|file| file.id == "tokenizer") {
            ensure!(
                !defaults
                    .local
                    .args
                    .split_whitespace()
                    .any(|arg| arg == "--tokenizer" || arg.starts_with("--tokenizer=")),
                "Tokenizer must be provided by a verified file"
            );
        }
        ensure!(
            !defaults.extra.contains_key("mmproj_path")
                && !defaults.extra.contains_key("mtp_path")
                && !defaults.extra.contains_key("text_encoder_path")
                && !defaults.extra.contains_key("vae_path")
                && !defaults.extra.contains_key("tokenizer_path"),
            "Catalog paths must be provided by verified files"
        );
        entry["tasks"] = serde_json::to_value(&defaults.tasks)?;
        entry["capabilities"] = serde_json::to_value(&defaults.capabilities)?;
        entry["max_context_length"] = serde_json::to_value(defaults.max_context_length)?;
        if let Some(sampling) = &defaults.sampling {
            entry["sampling"] = sampling.clone();
        }
        if let Some(support) = &defaults.audio_operations {
            entry["audio_operations"] = serde_json::to_value(support)?;
        }
        if let Some(support) = &defaults.tool_call_support {
            entry["tool_call_support"] = serde_json::to_value(support)?;
        }
        if let Some(support) = &defaults.structured_output_support {
            entry["structured_output_support"] = serde_json::to_value(support)?;
        }
        if !defaults.image_reference_checks.is_empty() {
            entry["image_reference_checks"] =
                serde_json::to_value(&defaults.image_reference_checks)?;
        }
        entry["local"]["params"] = serde_json::to_value(defaults.local.params)?;
        entry["local"]["bits_per_weight"] = serde_json::to_value(defaults.local.bits_per_weight)?;
        entry["local"]["context_length"] = serde_json::to_value(defaults.local.context_length)?;
        entry["local"]["gpu_layers"] = serde_json::to_value(defaults.local.gpu_layers)?;
        entry["local"]["args"] = serde_json::to_value(&defaults.local.args)?;
        if let Some(engines) = &defaults.engines {
            ensure!(
                engines.first() == Some(&model.engine),
                "Catalog engine does not match the reviewed model engine"
            );
            entry["engine"] = serde_json::to_value(engines)?;
        }
        if !defaults.extra.is_empty() {
            entry["extra"] = serde_json::Value::Object(defaults.extra.clone());
        }
    }
    if let Some(projector) = projector {
        let projector_path = model_dir.join(&projector.filename);
        entry["extra"]["mmproj_path"] = serde_json::json!(
            projector_path
                .to_str()
                .context("Projector path is not valid UTF-8")?
        );
    }
    for (role, key) in [
        ("text_encoder", "text_encoder_path"),
        ("vae", "vae_path"),
        ("tokenizer", "tokenizer_path"),
    ] {
        if let Some(companion) = version.files.iter().find(|file| file.id == role) {
            let path = model_dir.join(&companion.filename);
            entry["extra"][key] =
                serde_json::json!(path.to_str().context("Companion path is not valid UTF-8")?);
        }
    }
    if version.files.iter().any(|file| file.id == "encoder_index") {
        entry["extra"]["text_encoder_path"] = serde_json::json!(
            encoder_dir
                .to_str()
                .context("Encoder path is not valid UTF-8")?
        );
    }
    catalog_value["models"]
        .as_array_mut()
        .context("Invalid catalog models")?
        .push(entry);
    let mut combined = catalog_value.clone();
    combined["server"] = server_value
        .get("server")
        .context("Invalid server configuration")?
        .clone();
    let env_file = root.join(".env");
    let file_env: BTreeMap<String, String> = if env_file.exists() {
        dotenvy::from_path_iter(&env_file)?.collect::<Result<BTreeMap<_, _>, _>>()?
    } else {
        BTreeMap::new()
    };
    let interpolated = crate::config::interpolate_with(&combined.to_string(), &|key| {
        std::env::var(key)
            .ok()
            .or_else(|| file_env.get(key).cloned())
    })?;
    crate::config::ModelRegistry::from_json(&interpolated, None)?.validate()?;
    for file in &version.files {
        let path = bundle_path(&model_dir, file);
        if path.exists() {
            verify_file(&path, file).context("Conflicting model file; existing file preserved")?;
        }
    }
    let mut staged_files = Vec::new();
    let staged_result = (|| -> Result<()> {
        for file in &version.files {
            crate::inference::engines::download::check_cancelled()?;
            let target = bundle_path(&model_dir, file);
            if target.exists() {
                continue;
            }
            ensure!(
                file.sha256.len() == 64 && file.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "Invalid model checksum"
            );
            let staged = target
                .parent()
                .context("Missing model file directory")?
                .join(format!(
                    ".rune-model-download-{}-{}.part",
                    file.id,
                    file.sha256.to_ascii_lowercase()
                ));
            staged_files.push((staged, target));
            let (staged, _) = staged_files.last().expect("just added");
            download(file, staged)?;
            verify_file(staged, file)?;
            crate::inference::engines::download::check_cancelled()?;
        }
        if let Some(index) = version.files.iter().find(|file| file.id == "encoder_index") {
            let target = bundle_path(&model_dir, index);
            let index_path = staged_files
                .iter()
                .find(|(_, path)| *path == target)
                .map(|(staged, _)| staged)
                .unwrap_or(&target);
            validate_encoder_index(version, index_path)?;
        }
        for (staged, target) in &staged_files {
            crate::inference::engines::download::check_cancelled()?;
            publish_new(staged, target)?;
        }
        Ok(())
    })();
    staged_result?;
    let catalog_dir = catalog_path
        .parent()
        .context("Catalog has no parent directory")?;
    let server_dir = server_path
        .parent()
        .context("Server configuration has no parent directory")?;
    regular_dir(catalog_dir)?;
    regular_dir(server_dir)?;
    let staged_catalog = unique_path(catalog_dir, "catalog");
    let catalog_bytes = serde_json::to_vec_pretty(&catalog_value)?;
    write_new(&staged_catalog, &catalog_bytes)?;
    let result = (|| -> Result<()> {
        crate::inference::engines::download::check_cancelled()?;
        ensure!(
            read_regular(server_path)? == server_before,
            "Server configuration changed during model download; retry"
        );
        ensure!(
            read_regular(catalog_path)? == catalog_before,
            "Catalog changed during model download; retry"
        );
        if let Some(original) = &catalog_before {
            let backup = unique_path(catalog_dir, "catalog-backup");
            move_no_replace(catalog_path, &backup).context("Cannot preserve existing catalog")?;
            let publish = (|| -> Result<()> {
                ensure!(
                    read_regular(&backup)?.as_deref() == Some(original),
                    "Catalog changed during installation"
                );
                move_no_replace(&staged_catalog, catalog_path)
                    .context("Catalog appeared during installation")?;
                Ok(())
            })();
            if let Err(error) = publish {
                return match move_no_replace(&backup, catalog_path) {
                    Ok(()) => Err(error),
                    Err(_) => Err(error.context(format!(
                        "Previous catalog preserved at {}",
                        backup.display()
                    ))),
                };
            }
        } else {
            let server_bytes = serde_json::to_vec_pretty(&server_value)?;
            write_new(server_path, &server_bytes)
                .context("Server configuration appeared during installation")?;
            publish_new(&staged_catalog, catalog_path)
                .context("Catalog appeared during installation; server configuration preserved")?;
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&staged_catalog);
    result?;
    Ok(dest)
}

fn move_no_replace(source: &Path, target: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn MoveFileW(existing: *const u16, new: *const u16) -> i32;
        }
        let target: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let ok = unsafe { MoveFileW(source.as_ptr(), target.as_ptr()) };
        if ok == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())?;
        let target = std::ffi::CString::new(target.as_os_str().as_bytes())?;
        if unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())?;
        let target = std::ffi::CString::new(target.as_os_str().as_bytes())?;
        if unsafe { libc::renamex_np(source.as_ptr(), target.as_ptr(), libc::RENAME_EXCL) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(not(any(
        windows,
        target_os = "linux",
        target_os = "android",
        target_os = "macos"
    )))]
    {
        let _ = (source, target);
        anyhow::bail!("Safe catalog installation is unsupported on this platform");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn interrupted_install_reuses_staged_partial_before_registering() {
        let (root, model, bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let first = install_with(&model, &root, &server, &catalog, |_, path| {
            std::fs::write(path, &bytes[..5])?;
            anyhow::bail!("interrupted");
        });
        assert!(first.is_err());
        assert!(!catalog.exists());
        let installed = install_with(&model, &root, &server, &catalog, |_, path| {
            assert_eq!(std::fs::read(path)?, &bytes[..5]);
            let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
            file.write_all(&bytes[5..])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(installed).unwrap(), bytes);
        assert!(catalog.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_legacy_download_resumes_in_place() {
        let (root, model, bytes) = fixture();
        let file = &model.versions[&model.latest].files[0];
        let legacy = root.join("models").join(&model.id);
        std::fs::create_dir_all(&legacy).unwrap();
        let partial = legacy.join(format!(
            ".rune-model-download-{}-{}.part",
            file.id, file.sha256
        ));
        std::fs::write(&partial, &bytes[..5]).unwrap();
        let installed = install_with(
            &model,
            &root,
            &root.join("config/server.json"),
            &root.join("config/catalog.json"),
            |_, path| {
                assert_eq!(std::fs::read(path)?, &bytes[..5]);
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(path)?
                    .write_all(&bytes[5..])?;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            installed,
            std::fs::canonicalize(&legacy).unwrap().join(&file.filename)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_model_install_is_refused_without_touching_partial() {
        let (root, model, bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let worker_root = root.clone();
        let worker_model = model.clone();
        let worker_server = server.clone();
        let worker_catalog = catalog.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            install_with(
                &worker_model,
                &worker_root,
                &worker_server,
                &worker_catalog,
                |_, path| {
                    std::fs::write(path, &bytes[..5])?;
                    ready_tx.send(()).unwrap();
                    release_rx
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                    std::fs::write(path, bytes)?;
                    Ok(())
                },
            )
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let error = install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("second installer must not write")
        })
        .unwrap_err();
        assert!(error.to_string().contains("Another installation"));
        release_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_local_registry_is_loaded_and_invalid_files_fail() {
        let (root, _, _) = fixture();
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("local-models.json");
        std::fs::write(&path, include_bytes!("../registry/models.json")).unwrap();
        assert!(
            load_from_path(&path)
                .unwrap()
                .iter()
                .any(|model| model.id == "lfm3-1.7b-q4-k-m")
        );
        std::fs::write(&path, b"not JSON").unwrap();
        assert!(load_from_path(&path).is_err());
        std::fs::write(&path, vec![b' '; LIMIT as usize + 1]).unwrap();
        assert!(
            load_from_path(&path)
                .unwrap_err()
                .to_string()
                .contains("size limit")
        );
        std::fs::remove_file(&path).unwrap();
        assert!(
            load_from_path(&path)
                .unwrap_err()
                .to_string()
                .contains("Cannot open")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    fn fixture_bytes(file: &File) -> &'static [u8] {
        if file.id == "encoder_index" && file.url.contains("Qwen-Image-Edit-2511") {
            br#"{"weight_map":{"a":"model-00001-of-00004.safetensors","b":"model-00002-of-00004.safetensors","c":"model-00003-of-00004.safetensors","d":"model-00004-of-00004.safetensors"}}"#
        } else if file.id == "encoder_index" {
            br#"{"weight_map":{"a":"model-00001-of-00003.safetensors","b":"model-00002-of-00003.safetensors","c":"model-00003-of-00003.safetensors"}}"#
        } else if file.filename.ends_with(".json") {
            b"{\"model\":{}}"
        } else if file.filename.ends_with(".safetensors") {
            b"\x02\0\0\0\0\0\0\0{}"
        } else {
            b"GGUF fixture model bytes"
        }
    }

    #[test]
    fn lfm3_installs_at_registry_path_with_runtime_and_sampling_defaults() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "lfm3-1.7b-q4-k-m")
            .unwrap();
        let file = &mut model.versions.get_mut(&model.latest).unwrap().files[0];
        let bytes = fixture_bytes(file);
        file.sha256 = hex::encode(Sha256::digest(bytes));
        file.size_bytes = bytes.len() as u64;
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let path = install_with(&model, &root, &server, &catalog, |file, destination| {
            std::fs::write(destination, fixture_bytes(file))?;
            Ok(())
        })
        .unwrap();
        let expected = std::fs::canonicalize(&root)
            .unwrap()
            .join("models/liquidai/lfm3-1.7b/LFM3-1.7B-Q4_K_M.gguf");
        assert_eq!(path, expected);
        let loaded = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        let installed = loaded
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        assert_eq!(Path::new(&installed.model_path), expected);
        assert_eq!(
            installed.runtime.as_ref().unwrap().extra_args.as_deref(),
            Some("--repeat-penalty 1.1")
        );
        let defaults = installed.effective_sampling(&crate::config::SamplingConfig::default());
        assert_eq!(defaults.temperature, Some(0.1));
        assert_eq!(defaults.top_k, Some(50));
        let overridden = installed.effective_sampling(&crate::config::SamplingConfig {
            temperature: Some(0.7),
            ..Default::default()
        });
        assert_eq!(overridden.temperature, Some(0.7));
        install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("verified file should be reused")
        })
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn omni_installs_in_liquidai_family_directory_with_projector() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "lfm3-omni-2b")
            .unwrap();
        for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
            let bytes = fixture_bytes(file);
            file.sha256 = hex::encode(Sha256::digest(bytes));
            file.size_bytes = bytes.len() as u64;
        }
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let path = install_with(&model, &root, &server, &catalog, |file, dest| {
            std::fs::write(dest, fixture_bytes(file))?;
            Ok(())
        })
        .unwrap();
        let dir = std::fs::canonicalize(&root)
            .unwrap()
            .join("models/liquidai/lfm3-omni-2b");
        assert_eq!(path, dir.join("LFM3-Omni-2B-Q8_0.gguf"));
        assert!(dir.join("mmproj-LFM3-Omni-2B-F16.gguf").exists());
        let old_dir = root.join("models").join(&model.id);
        std::fs::create_dir_all(&old_dir).unwrap();
        let file = &model.versions[&model.latest].files[0];
        std::fs::write(
            old_dir.join(format!(
                ".rune-model-download-{}-{}.part",
                file.id, file.sha256
            )),
            b"stale",
        )
        .unwrap();
        assert_eq!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "installed model should be reused"
            ))
            .unwrap(),
            path
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reviewed_matting_entry_installs_and_registers_its_capability() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "birefnet-matting")
            .unwrap();
        let onnx_fixture = b"fixture ONNX bytes";
        let file = &mut model.versions.get_mut(&model.latest).unwrap().files[0];
        file.sha256 = hex::encode(Sha256::digest(onnx_fixture));
        file.size_bytes = onnx_fixture.len() as u64;
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let installed = install_with(&model, &root, &server, &catalog, |_, dest| {
            std::fs::write(dest, onnx_fixture)?;
            Ok(())
        })
        .unwrap();
        assert!(
            installed
                .ends_with("models/zhengpeng7/birefnet-matting/BiRefNet-matting-epoch_100.onnx")
        );
        let config = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        let entry = config
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        assert_eq!(entry.tasks, ["image-to-matting"]);
        assert!(
            entry
                .resolved_capabilities()
                .contains(&"Portrait Matting".to_string())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registered_legacy_directory_remains_in_use() {
        let (root, model) = vision_fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        install_with(&model, &root, &server, &catalog, |file, dest| {
            let bytes = if file.id == "model" {
                b"GGUF model".as_slice()
            } else {
                b"GGUF projector".as_slice()
            };
            std::fs::write(dest, bytes)?;
            Ok(())
        })
        .unwrap();
        let new_dir = root
            .join("models")
            .join(model.install_dir.as_deref().unwrap());
        let legacy = root.join("models").join(&model.id);
        std::fs::rename(&new_dir, &legacy).unwrap();
        let legacy = std::fs::canonicalize(legacy).unwrap();
        let mut data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&catalog).unwrap()).unwrap();
        let entry = data["models"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == model.id)
            .unwrap();
        for (role, key) in [("model", "local"), ("projector", "mmproj_path")] {
            let file = model.versions[&model.latest]
                .files
                .iter()
                .find(|file| file.id == role)
                .unwrap();
            let path = legacy.join(&file.filename).to_str().unwrap().to_owned();
            if key == "local" {
                entry["local"]["path"] = path.into();
            } else {
                entry["extra"][key] = path.into();
            }
        }
        std::fs::write(&catalog, serde_json::to_vec(&data).unwrap()).unwrap();
        let installed = install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("legacy files should be reused")
        })
        .unwrap();
        assert!(installed.starts_with(&legacy));
        assert!(!new_dir.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_install_path_rejects_escape_and_conflicting_filenames() {
        let value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../registry/models.json")).unwrap();
        for dir in [
            "../escape",
            "creator/../escape",
            "creator",
            "creator/family/extra",
            "/creator/family",
        ] {
            let mut candidate = value.clone();
            candidate["local_models"][0]["install_dir"] = dir.into();
            assert!(
                parse(&serde_json::to_vec(&candidate).unwrap()).is_err(),
                "{dir}"
            );
        }
        for invalid in [
            "../LFM3-1.7B-Q4_K_M.gguf",
            "/LFM3-1.7B-Q4_K_M.gguf",
            "C:/LFM3-1.7B-Q4_K_M.gguf",
            "dir\\LFM3-1.7B-Q4_K_M.gguf",
            "NUL/LFM3-1.7B-Q4_K_M.gguf",
            "dir/other.gguf",
            "dir//LFM3-1.7B-Q4_K_M.gguf",
        ] {
            let mut candidate = value.clone();
            let model = candidate["local_models"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|model| model["id"] == "lfm3-1.7b-q4-k-m")
                .unwrap();
            model["catalog"]["local"]["path"] = invalid.into();
            assert!(
                parse(&serde_json::to_vec(&candidate).unwrap()).is_err(),
                "{invalid}"
            );
        }
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_ok());
        assert!(
            validate_install_path(
                "Liquid Models/LFM3-1.7B-Q4_K_M.gguf",
                "LFM3-1.7B-Q4_K_M.gguf"
            )
            .is_ok()
        );
    }

    fn fixture() -> (PathBuf, LocalModel, &'static [u8]) {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "model-fixture-{}",
                crate::setup::generate_api_key()
            ));
        let bytes: &'static [u8] = b"GGUF fixture model bytes";
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .remove(0);
        let file = &mut model.versions.get_mut(&model.latest).unwrap().files[0];
        file.sha256 = hex::encode(Sha256::digest(bytes));
        file.size_bytes = bytes.len() as u64;
        (root, model, bytes)
    }

    fn vision_fixture() -> (PathBuf, LocalModel) {
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "vision-fixture-{}",
                crate::setup::generate_api_key()
            ));
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "smolvlm-256m-instruct-q8-0")
            .unwrap();
        for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
            let bytes = if file.id == "model" {
                b"GGUF model".as_slice()
            } else {
                b"GGUF projector".as_slice()
            };
            file.sha256 = hex::encode(Sha256::digest(bytes));
            file.size_bytes = bytes.len() as u64;
        }
        (root, model)
    }

    #[test]
    fn vision_pair_installs_and_registers_its_projector() {
        let (root, model) = vision_fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let installed = install_with(&model, &root, &server, &catalog, |file, path| {
            let bytes = if file.id == "model" {
                b"GGUF model".as_slice()
            } else {
                b"GGUF projector".as_slice()
            };
            std::fs::write(path, bytes)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(installed).unwrap(), b"GGUF model");
        let registry = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        assert_eq!(registry.server.host, "127.0.0.1");
        let entry = registry
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        assert!(entry.tasks.iter().any(|task| task == "text-image-to-text"));
        assert_eq!(
            std::fs::read(entry.mmproj_path.as_ref().unwrap()).unwrap(),
            b"GGUF projector"
        );
        install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("verified pair must be reused")
        })
        .unwrap();
        let projector = root
            .join("models")
            .join(model.install_dir.as_deref().unwrap())
            .join(
                &model.versions[&model.latest]
                    .files
                    .iter()
                    .find(|file| file.id == "projector")
                    .unwrap()
                    .filename,
            );
        std::fs::write(&projector, b"conflicting data").unwrap();
        let saved_catalog = std::fs::read(&catalog).unwrap();
        assert!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "must not overwrite a conflicting projector"
            ))
            .is_err()
        );
        assert_eq!(std::fs::read(&projector).unwrap(), b"conflicting data");
        assert_eq!(std::fs::read(&catalog).unwrap(), saved_catalog);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_projector_does_not_publish_or_register_pair() {
        let (root, model) = vision_fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        assert!(
            install_with(&model, &root, &server, &catalog, |file, path| {
                let bytes = if file.id == "model" {
                    b"GGUF model".as_slice()
                } else {
                    b"GGUF bad".as_slice()
                };
                std::fs::write(path, bytes)?;
                Ok(())
            })
            .is_err()
        );
        assert!(!server.exists() && !catalog.exists());
        for file in &model.versions[&model.latest].files {
            assert!(
                !root
                    .join("models")
                    .join(model.install_dir.as_deref().unwrap())
                    .join(&file.filename)
                    .exists()
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn vision_registry_requires_unique_model_and_projector() {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../registry/models.json")).unwrap();
        let model = &mut value["local_models"][1];
        let latest = model["latest"].as_str().unwrap().to_owned();
        model["versions"][&latest]["files"][1]["id"] = "model".into();
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn image_bundle_installs_all_companions_and_rejects_bad_or_conflicting_files() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "qwen-image-2.1")
            .unwrap();
        for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
            let contents = fixture_bytes(file);
            file.sha256 = hex::encode(Sha256::digest(contents));
            file.size_bytes = contents.len() as u64;
        }
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        assert!(
            install_with(&model, &root, &server, &catalog, |file, path| {
                std::fs::write(
                    path,
                    if file.id == "vae" {
                        b"invalid".as_slice()
                    } else {
                        fixture_bytes(file)
                    },
                )?;
                Ok(())
            })
            .is_err()
        );
        assert!(!catalog.exists());
        for file in &model.versions[&model.latest].files {
            assert!(
                !root
                    .join("models")
                    .join(model.install_dir.as_deref().unwrap())
                    .join(&file.filename)
                    .exists()
            );
        }
        install_with(&model, &root, &server, &catalog, |file, path| {
            std::fs::write(path, fixture_bytes(file))?;
            Ok(())
        })
        .unwrap();
        let installed: crate::schema::CatalogFile =
            serde_json::from_slice(&std::fs::read(&catalog).unwrap()).unwrap();
        let entry = installed
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        for (role, key) in [("text_encoder", "text_encoder_path"), ("vae", "vae_path")] {
            let file = model.versions[&model.latest]
                .files
                .iter()
                .find(|file| file.id == role)
                .unwrap();
            assert_eq!(
                entry.extra[key].as_str().unwrap(),
                std::fs::canonicalize(&root)
                    .unwrap()
                    .join("models")
                    .join(model.install_dir.as_deref().unwrap())
                    .join(&file.filename)
                    .to_str()
                    .unwrap()
            );
        }
        install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("existing verified bundle should be reused")
        })
        .unwrap();
        let vae = model.versions[&model.latest]
            .files
            .iter()
            .find(|file| file.id == "vae")
            .unwrap();
        let path = root
            .join("models")
            .join(model.install_dir.as_deref().unwrap())
            .join(&vae.filename);
        std::fs::write(&path, b"conflicting file").unwrap();
        let original = std::fs::read(&catalog).unwrap();
        assert!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "must not overwrite"
            ))
            .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"conflicting file");
        assert_eq!(std::fs::read(&catalog).unwrap(), original);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn image_registry_requires_all_distinct_companions() {
        let original: serde_json::Value =
            serde_json::from_slice(include_bytes!("../registry/models.json")).unwrap();
        for change in ["missing", "duplicate", "wrong-format"] {
            let mut registry = original.clone();
            let model = registry["local_models"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|model| model["id"] == "qwen-image-2.1")
                .unwrap();
            let latest = model["latest"].as_str().unwrap().to_owned();
            let files = model["versions"][&latest]["files"].as_array_mut().unwrap();
            match change {
                "missing" => {
                    files.pop();
                }
                "duplicate" => files[2]["id"] = "text_encoder".into(),
                _ => files[2]["filename"] = "vae.gguf".into(),
            }
            assert!(
                parse(&serde_json::to_vec(&registry).unwrap()).is_err(),
                "{change}"
            );
        }
    }

    #[test]
    fn moss_bundle_installs_verified_tokenizer_and_reuses_it() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "moss-image-sft")
            .unwrap();
        for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
            let bytes = fixture_bytes(file);
            file.sha256 = hex::encode(Sha256::digest(bytes));
            file.size_bytes = bytes.len() as u64;
        }
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        assert!(
            install_with(&model, &root, &server, &catalog, |file, path| {
                std::fs::write(
                    path,
                    if file.id == "tokenizer" {
                        b"bad".as_slice()
                    } else {
                        fixture_bytes(file)
                    },
                )?;
                Ok(())
            })
            .is_err()
        );
        assert!(!catalog.exists());
        install_with(&model, &root, &server, &catalog, |file, path| {
            std::fs::write(path, fixture_bytes(file))?;
            Ok(())
        })
        .unwrap();
        let loaded = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        let installed = loaded
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        assert!(
            installed
                .tokenizer_path
                .as_ref()
                .unwrap()
                .ends_with("tokenizer.json")
        );
        assert_eq!(
            std::fs::read(installed.tokenizer_path.as_ref().unwrap()).unwrap(),
            b"{\"model\":{}}"
        );
        install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("verified bundle should be reused")
        })
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sharded_encoder_requires_matching_index_before_publishing() {
        let mut model = parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .find(|model| model.id == "z-image-turbo-q8-0")
            .unwrap();
        for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
            let bytes = fixture_bytes(file);
            file.sha256 = hex::encode(Sha256::digest(bytes));
            file.size_bytes = bytes.len() as u64;
        }
        let (root, _, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let mut invalid = model.clone();
        let bad_index = br#"{"weight_map":{"a":"missing.safetensors"}}"#;
        let file = invalid
            .versions
            .get_mut(&invalid.latest)
            .unwrap()
            .files
            .iter_mut()
            .find(|file| file.id == "encoder_index")
            .unwrap();
        file.sha256 = hex::encode(Sha256::digest(bad_index));
        file.size_bytes = bad_index.len() as u64;
        assert!(
            install_with(&invalid, &root, &server, &catalog, |file, path| {
                std::fs::write(
                    path,
                    if file.id == "encoder_index" {
                        bad_index.as_slice()
                    } else {
                        fixture_bytes(file)
                    },
                )?;
                Ok(())
            })
            .is_err()
        );
        assert!(!catalog.exists());
        for file in &model.versions[&model.latest].files {
            assert!(
                !bundle_path(
                    &root
                        .join("models")
                        .join(model.install_dir.as_deref().unwrap()),
                    file
                )
                .exists()
            );
        }
        install_with(&model, &root, &server, &catalog, |file, path| {
            std::fs::write(path, fixture_bytes(file))?;
            Ok(())
        })
        .unwrap();
        let loaded = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        let installed = loaded
            .models
            .iter()
            .find(|entry| entry.id == model.id)
            .unwrap();
        let encoder = Path::new(installed.text_encoder_path.as_ref().unwrap());
        assert!(encoder.join("model.safetensors.index.json").is_file());
        assert!(encoder.join("model-00003-of-00003.safetensors").is_file());
        install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("verified bundle should be reused")
        })
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn migrated_model_keeps_catalog_settings_when_installed() {
        let configured: crate::schema::CatalogFile =
            serde_json::from_slice(include_bytes!("../config/catalog.json")).unwrap();
        for mut model in parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .filter(|entry| {
                entry.catalog.is_some()
                    && configured.models.iter().any(|model| model.id == entry.id)
            })
        {
            let (root, _, bytes) = fixture();
            for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
                let contents = fixture_bytes(file);
                file.sha256 = hex::encode(Sha256::digest(contents));
                file.size_bytes = contents.len() as u64;
            }
            let server = root.join("config/server.json");
            let catalog = root.join("config/catalog.json");
            let _ = bytes;
            install_with(&model, &root, &server, &catalog, |file, path| {
                std::fs::write(path, fixture_bytes(file))?;
                Ok(())
            })
            .unwrap();
            let installed: crate::schema::CatalogFile =
                serde_json::from_slice(&std::fs::read(&catalog).unwrap()).unwrap();
            let actual = serde_json::to_value(
                installed
                    .models
                    .iter()
                    .find(|entry| entry.id == model.id)
                    .unwrap(),
            )
            .unwrap();
            let original = serde_json::to_value(
                configured
                    .models
                    .iter()
                    .find(|entry| entry.id == model.id)
                    .unwrap(),
            )
            .unwrap();
            for field in [
                "id",
                "name",
                "description",
                "tasks",
                "capabilities",
                "engine",
                "sampling",
                "max_context_length",
                "audio_operations",
                "tool_call_support",
                "structured_output_support",
                "image_reference_checks",
            ] {
                assert_eq!(actual[field], original[field], "{}: {field}", model.id);
            }
            for field in ["params", "bits_per_weight", "context_length", "gpu_layers"] {
                assert_eq!(
                    actual["local"][field], original["local"][field],
                    "{}: local.{field}",
                    model.id
                );
            }
            if model.id == "moss-image-sft" {
                assert_eq!(
                    actual["local"]["args"].as_str().unwrap(),
                    original["local"]["args"]
                        .as_str()
                        .unwrap()
                        .split(" --tokenizer ")
                        .next()
                        .unwrap()
                );
                assert!(
                    actual["extra"]["tokenizer_path"]
                        .as_str()
                        .unwrap()
                        .ends_with("tokenizer.json")
                );
            } else {
                assert_eq!(
                    actual["local"]["args"], original["local"]["args"],
                    "{}: local.args",
                    model.id
                );
            }
            if !original["extra"].is_null() {
                assert_eq!(
                    actual["extra"]["max_resolution"],
                    original["extra"]["max_resolution"]
                );
            }
            if model.id == "z-image" {
                assert!(actual["extra"]["lora"].is_null());
                assert!(!original["extra"]["lora"].as_array().unwrap().is_empty());
            }
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn installs_and_registers_a_verified_model_and_retries_idempotently() {
        let (root, model, bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let path = install_with(&model, &root, &server, &catalog, |_, path| {
            std::fs::write(path, bytes)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let registry = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        assert!(registry.models.iter().any(|entry| entry.id == model.id));
        let again = install_with(&model, &root, &server, &catalog, |_, _| {
            panic!("must reuse verified model")
        })
        .unwrap();
        assert_eq!(again, path);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn observed_install_honors_cancellation_before_downloading() {
        let (root, model, _) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let error = install_observed(&model, &root, &server, &catalog, cancel, |_| {}).unwrap_err();
        assert!(format!("{error:#}").to_lowercase().contains("cancel"));
        assert!(!server.exists());
        assert!(!catalog.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_bad_download_and_preserves_conflicting_file() {
        let (root, model, _bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        assert!(
            install_with(&model, &root, &server, &catalog, |_, path| {
                std::fs::write(path, b"GGUF bad")?;
                Ok(())
            })
            .is_err()
        );
        assert!(!server.exists() && !catalog.exists());
        let target = root
            .join("models")
            .join(model.install_dir.as_deref().unwrap())
            .join(&model.versions[&model.latest].files[0].filename);
        std::fs::write(&target, b"keep this").unwrap();
        assert!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "must not download"
            ))
            .is_err()
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"keep this");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_changes_during_download_are_preserved() {
        let (root, model, bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        let server_data = crate::schema::ServerFile {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: crate::config::ServerConfig::default(),
        };
        std::fs::write(&server, serde_json::to_vec(&server_data).unwrap()).unwrap();
        std::fs::write(&catalog, br#"{"schema_version":3,"models":[]}"#).unwrap();
        let changed = br#"{"schema_version":3,"models":[],"roles":{}}"#;
        assert!(
            install_with(&model, &root, &server, &catalog, |_, path| {
                std::fs::write(path, bytes)?;
                std::fs::write(&catalog, changed)?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(std::fs::read(&catalog).unwrap(), changed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn existing_catalog_is_extended_and_original_is_preserved() {
        let (root, model, bytes) = fixture();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        std::fs::create_dir_all(server.parent().unwrap()).unwrap();
        let server_data = crate::schema::ServerFile {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: crate::config::ServerConfig::default(),
        };
        std::fs::write(&server, serde_json::to_vec(&server_data).unwrap()).unwrap();
        let original = br#"{"schema_version":3,"models":[],"roles":{},"aliases":{}}"#;
        std::fs::write(&catalog, original).unwrap();
        install_with(&model, &root, &server, &catalog, |_, path| {
            std::fs::write(path, bytes)?;
            Ok(())
        })
        .unwrap();
        let registry = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        assert_eq!(registry.models.len(), 1);
        assert!(
            std::fs::read_dir(catalog.parent().unwrap())
                .unwrap()
                .filter_map(Result::ok)
                .any(|e| e
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".rune-catalog-backup-")
                    && std::fs::read(e.path()).unwrap() == original)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_config_and_conflicting_model_ids_are_not_replaced() {
        let (root, model, _bytes) = fixture();
        let config = root.join("config");
        let server = config.join("server.json");
        let catalog = config.join("catalog.json");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(config.join("models.json"), b"legacy").unwrap();
        assert!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "must not download"
            ))
            .is_err()
        );
        assert_eq!(
            std::fs::read(config.join("models.json")).unwrap(),
            b"legacy"
        );
        std::fs::remove_file(config.join("models.json")).unwrap();
        let server_data = crate::schema::ServerFile {
            schema_version: crate::schema::SCHEMA_VERSION,
            server: crate::config::ServerConfig::default(),
        };
        std::fs::write(&server, serde_json::to_vec(&server_data).unwrap()).unwrap();
        let contents = serde_json::json!({"schema_version":3,"models":[{"id":model.id,"name":"Existing", "capabilities":["Chat"], "engine":["llama-server"], "local":{"path":"existing.gguf"}}]});
        let original = serde_json::to_vec(&contents).unwrap();
        std::fs::write(&catalog, &original).unwrap();
        assert!(
            install_with(&model, &root, &server, &catalog, |_, _| panic!(
                "must not download"
            ))
            .is_err()
        );
        assert_eq!(std::fs::read(&catalog).unwrap(), original);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn bundled_local_entries_are_valid() {
        let models = parse(include_bytes!("../registry/models.json")).unwrap();
        assert_eq!(models[0].id, "smollm2-135m-instruct-q4-k-m");
    }
    #[test]
    fn bad_integrity_and_release_order_are_rejected() {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../registry/models.json")).unwrap();
        let model = &mut value["local_models"][0];
        let latest = model["latest"].as_str().unwrap().to_owned();
        model["versions"][&latest]["files"][0]["sha256"] = "bad".into();
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    #[test]
    fn model_identifiers_allow_catalog_dots_but_not_path_components() {
        assert!(valid_model_id("qwen3.5-0.8b"));
        for invalid in [
            "../model",
            "..",
            ".hidden",
            "foo..bar",
            "foo.",
            "model/name",
            "MODEL",
        ] {
            assert!(!valid_model_id(invalid), "{invalid}");
        }
    }
}
