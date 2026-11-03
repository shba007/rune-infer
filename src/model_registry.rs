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
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalDefaults {
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
                "llama-server" | "nemo-speech" | "audio-cpp"
            ) && model.format == "gguf",
            "Unsupported model format or engine"
        );
        let latest = model
            .versions
            .get(&model.latest)
            .context("Missing selected model version")?;
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
                        && file.filename.ends_with(".gguf")
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

fn validate_bundle(model: &LocalModel, version: &Version) -> Result<()> {
    let vision = model.capabilities == ["Chat", "Vision"]
        || model.capabilities == ["Chat", "Vision", "Audio", "Video"];
    ensure!(
        vision
            || model.capabilities == ["Chat"]
            || model.capabilities == ["Decision"]
            || model.capabilities == ["Transcription"]
            || model.capabilities == ["Speech Generation"],
        "Unsupported model capabilities"
    );
    if model.capabilities == ["Decision"]
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
        version.files.len() == 1 || version.files.len() == 2,
        "Model has missing or unsupported companion files"
    );
    if model.capabilities == ["Transcription"] || model.capabilities == ["Speech Generation"] {
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
            model.engine == "llama-server",
            "Unsupported model engine combination"
        );
    }
    let mut roles = HashSet::new();
    let mut names = HashSet::new();
    for file in &version.files {
        ensure!(
            matches!(file.id.as_str(), "model" | "projector") && roles.insert(&file.id),
            "Duplicate or unsupported model file role"
        );
        ensure!(
            !file.filename.is_empty()
                && file.filename.ends_with(".gguf")
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
            && (!vision || roles.contains(&"projector".to_string())),
        "Model capabilities do not match companion files"
    );
    Ok(())
}

/// Download a reviewed model and register it in the user's catalog.
pub fn install(model: &LocalModel, root: &Path, server: &Path, catalog: &Path) -> Result<PathBuf> {
    let client = crate::inference::engines::download::create_download_client()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    install_with(model, root, server, catalog, |file, dest| {
        crate::inference::engines::download::download_verified(
            &client,
            &file.url,
            dest,
            "[Model]",
            &file.sha256,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))
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
    let mut magic = [0; 4];
    std::fs::File::open(path)?.read_exact(&mut magic)?;
    ensure!(&magic == b"GGUF", "Downloaded file is not GGUF");
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
            "llama-server" | "nemo-speech" | "audio-cpp"
        ) && model.format == "gguf",
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
    let model_dir = models.join(&model.id);
    regular_dir(&model_dir)?;
    let dest = model_dir.join(&file.filename);
    let server_before = read_regular(server_path)?;
    let catalog_before = read_regular(catalog_path)?;
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
            server: crate::config::ServerConfig::default(),
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
        for file in &version.files {
            verify_file(&model_dir.join(&file.filename), file)
                .context("Existing model does not match registry")?;
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
            [a] if a == "Decision" => defaults.tasks == ["text-to-decision"],
            [a] if a == "Transcription" => defaults.tasks == ["audio-to-text"],
            [a] if a == "Speech Generation" => defaults.tasks == ["text-to-audio"],
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
        ensure!(
            !defaults.extra.contains_key("mmproj_path") && !defaults.extra.contains_key("mtp_path"),
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
        let path = model_dir.join(&file.filename);
        if path.exists() {
            verify_file(&path, file).context("Conflicting model file; existing file preserved")?;
        }
    }
    let mut staged_files = Vec::new();
    let staged_result = (|| -> Result<()> {
        for file in &version.files {
            let target = model_dir.join(&file.filename);
            if target.exists() {
                continue;
            }
            let staged = unique_path(&model_dir, "model-download");
            staged_files.push((staged, target));
            let (staged, _) = staged_files.last().expect("just added");
            download(file, staged)?;
            verify_file(staged, file)?;
        }
        for (staged, target) in &staged_files {
            publish_new(staged, target)?;
        }
        Ok(())
    })();
    for (staged, _) in staged_files {
        let _ = std::fs::remove_file(staged);
    }
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
        let projector = root.join("models").join(&model.id).join(
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
                    .join(&model.id)
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
    fn migrated_model_keeps_catalog_settings_when_installed() {
        let configured: crate::schema::CatalogFile =
            serde_json::from_slice(include_bytes!("../config/catalog.json")).unwrap();
        for mut model in parse(include_bytes!("../registry/models.json"))
            .unwrap()
            .into_iter()
            .filter(|entry| entry.catalog.is_some())
        {
            let (root, _, bytes) = fixture();
            for file in &mut model.versions.get_mut(&model.latest).unwrap().files {
                file.sha256 = hex::encode(Sha256::digest(bytes));
                file.size_bytes = bytes.len() as u64;
            }
            let server = root.join("config/server.json");
            let catalog = root.join("config/catalog.json");
            install_with(&model, &root, &server, &catalog, |_, path| {
                std::fs::write(path, bytes)?;
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
            ] {
                assert_eq!(actual[field], original[field], "{}: {field}", model.id);
            }
            for field in [
                "params",
                "bits_per_weight",
                "context_length",
                "gpu_layers",
                "args",
            ] {
                assert_eq!(
                    actual["local"][field], original["local"][field],
                    "{}: local.{field}",
                    model.id
                );
            }
            if !original["extra"].is_null() {
                assert_eq!(
                    actual["extra"]["max_resolution"],
                    original["extra"]["max_resolution"]
                );
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
            .join(&model.id)
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
