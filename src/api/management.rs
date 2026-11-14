//! Authenticated catalog management for servers with explicit config paths.
use super::AppState;
use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub(crate) struct CatalogStore {
    server: PathBuf,
    catalog: PathBuf,
    lock: Mutex<()>,
}
impl CatalogStore {
    pub(crate) fn new(server: PathBuf, catalog: PathBuf) -> Self {
        Self {
            server,
            catalog,
            lock: Mutex::new(()),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct CatalogDocument {
    revision: String,
    catalog: String,
}
type Reply<T> = Result<T, (StatusCode, &'static str)>;

#[derive(Deserialize)]
pub(super) struct CandidateQuery {
    #[serde(default)]
    purpose: Option<String>,
}

pub(super) async fn model_candidates(
    State(state): State<Arc<AppState>>,
    Query(query): Query<CandidateQuery>,
) -> Reply<Json<Vec<crate::model_management::Candidate>>> {
    store(&state)?;
    let computer_use = match query.purpose.as_deref() {
        None | Some("all") => false,
        Some("computer-use") => true,
        _ => return Err((StatusCode::BAD_REQUEST, "Unknown candidate purpose")),
    };
    let root = crate::engine_inventory::managed_root().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot resolve Infer model storage",
        )
    })?;
    let candidates = tokio::task::spawn_blocking(move || {
        let (memory, gpu) = crate::model_management::detected_memory();
        let disk = crate::model_management::available_disk(&root);
        crate::model_management::managed_candidates(computer_use, memory, gpu, disk)
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model candidates unavailable",
        )
    })?
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Model candidates unavailable",
        )
    })?;
    Ok(Json(candidates))
}

#[derive(Serialize)]
pub(super) struct EngineInventory {
    engines: Vec<EngineEntry>,
    installation: Option<EngineInstallation>,
}

#[derive(Serialize)]
struct EngineEntry {
    name: &'static str,
    installed: bool,
    managed: bool,
    version: String,
    release_check_available: bool,
    installation_available: bool,
    update_available: bool,
}

pub(super) async fn engines(State(state): State<Arc<AppState>>) -> Reply<Json<EngineInventory>> {
    store(&state)?;
    let root = crate::engine_inventory::managed_root().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot resolve Infer engine storage",
        )
    })?;
    let engines = tokio::task::spawn_blocking(move || {
        crate::engine_inventory::discover_engines(&root)
            .into_iter()
            .map(|row| EngineEntry {
                installation_available: crate::engine_inventory::installation_available(&row),
                update_available: can_update_engine(&row),
                name: row.name,
                installed: row.installed,
                managed: row.managed,
                version: row.version,
                release_check_available: can_check_engine_release(row.name)
                    && (!row.installed || row.managed),
            })
            .collect()
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine inventory unavailable",
        )
    })?;
    let installation = state
        .engine_installation
        .lock()
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Engine installation status unavailable",
            )
        })?
        .clone();
    Ok(Json(EngineInventory {
        engines,
        installation,
    }))
}

#[derive(Clone, Serialize)]
pub(crate) struct EngineInstallation {
    name: String,
    operation: &'static str,
    status: &'static str,
    message: String,
}

pub(super) async fn engine_installation(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Reply<Json<EngineInstallation>> {
    store(&state)?;
    let current = state.engine_installation.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine installation status unavailable",
        )
    })?;
    current
        .as_ref()
        .filter(|job| job.name == name && job.operation == "install")
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "No engine installation found"))
}

pub(super) async fn start_engine_installation(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Reply<(StatusCode, Json<EngineInstallation>)> {
    store(&state)?;
    let kind = crate::engine_inventory::install_engine_kind(&name)
        .ok_or((StatusCode::BAD_REQUEST, "Engine installation unavailable"))?;
    let root = crate::engine_inventory::managed_root().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot resolve Infer engine storage",
        )
    })?;
    let checked_name = name.clone();
    let available = tokio::task::spawn_blocking(move || {
        crate::engine_inventory::discover_engines(&root)
            .iter()
            .any(|row| {
                row.name == checked_name && crate::engine_inventory::installation_available(row)
            })
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine inventory unavailable",
        )
    })?;
    if !available {
        return Err((
            StatusCode::CONFLICT,
            "Engine installation unavailable or already installed",
        ));
    }
    let job = EngineInstallation {
        name: name.clone(),
        operation: "install",
        status: "running",
        message: "Installing engine".into(),
    };
    {
        let mut current = state.engine_installation.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Engine installation status unavailable",
            )
        })?;
        if current.as_ref().is_some_and(|job| job.status == "running") {
            return Err((
                StatusCode::CONFLICT,
                "An engine installation is already running",
            ));
        }
        *current = Some(job.clone());
    }
    let status = state.engine_installation.clone();
    let inference = state.inference.clone();
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (events, receiver) = std::sync::mpsc::channel();
            drop(receiver);
            crate::inference::engines::download::with_cancellation(stop.clone(), || {
                crate::inference::engines::update::install_engine(inference, stop, kind, events)
            })
        })
        .await;
        if let Ok(mut current) = status.lock() {
            if let Some(job) = current.as_mut().filter(|job| job.name == name) {
                match result {
                    Ok(Ok(message)) => {
                        job.status = "completed";
                        job.message = message;
                    }
                    Err(error) => {
                        tracing::error!(engine = %name, %error, "Engine installation task failed");
                        job.status = "failed";
                        job.message = "Engine installation failed; check Infer logs".into();
                    }
                    Ok(Err(error)) => {
                        tracing::warn!(engine = %name, %error, "Engine installation failed");
                        job.status = "failed";
                        job.message = "Engine installation failed; check Infer logs".into();
                    }
                }
            }
        }
    });
    Ok((StatusCode::ACCEPTED, Json(job)))
}

fn can_update_engine(row: &crate::engine_inventory::EngineRow) -> bool {
    cfg!(all(windows, target_arch = "x86_64"))
        && row.installed
        && row.managed
        && matches!(
            row.name,
            "llama-server"
                | "llama-server-prism"
                | "nemo-speech"
                | "audio-cpp"
                | "crispasr"
                | "sd-server"
                | "needle"
                | "whisper-cpp"
        )
}

pub(super) async fn engine_update(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Reply<Json<EngineInstallation>> {
    store(&state)?;
    let current = state.engine_installation.lock().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine update status unavailable",
        )
    })?;
    current
        .as_ref()
        .filter(|job| job.name == name && job.operation == "update")
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "No engine update found"))
}

pub(super) async fn start_engine_update(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Reply<(StatusCode, Json<EngineInstallation>)> {
    store(&state)?;
    let root = crate::engine_inventory::managed_root().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot resolve Infer engine storage",
        )
    })?;
    let checked_name = name.clone();
    let discovery_root = root.clone();
    let row = tokio::task::spawn_blocking(move || {
        crate::engine_inventory::discover_engines(&discovery_root)
            .into_iter()
            .find(|row| row.name == checked_name && can_update_engine(row))
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine inventory unavailable",
        )
    })?
    .ok_or((StatusCode::CONFLICT, "Engine update unavailable"))?;
    let job = EngineInstallation {
        name: name.clone(),
        operation: "update",
        status: "running",
        message: "Checking and updating engine".into(),
    };
    {
        let mut current = state.engine_installation.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Engine update status unavailable",
            )
        })?;
        if current.as_ref().is_some_and(|job| job.status == "running") {
            return Err((
                StatusCode::CONFLICT,
                "An engine operation is already running",
            ));
        }
        *current = Some(job.clone());
    }
    let status = state.engine_installation.clone();
    let inference = state.inference.clone();
    tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            let release = crate::engine_releases::check_latest_release(row.name)?;
            let mut engine = row;
            crate::engine_releases::apply_release(&mut engine, release, &root);
            match crate::engine_releases::engine_action(&engine) {
                crate::engine_releases::EngineAction::Update => {}
                crate::engine_releases::EngineAction::Unavailable(reason)
                    if matches!(
                        crate::engine_releases::release_status(&engine),
                        "Up to date" | "Ahead"
                    ) =>
                {
                    return Ok(reason);
                }
                crate::engine_releases::EngineAction::Unavailable(reason) => return Err(reason),
                crate::engine_releases::EngineAction::Install => {
                    return Err("Engine installation is required before updating".into());
                }
            }
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (events, receiver) = std::sync::mpsc::channel();
            drop(receiver);
            crate::inference::engines::download::with_cancellation(stop.clone(), || {
                crate::engine_releases::perform_update(inference, stop, engine, events)
            })
        })
        .await;
        if let Ok(mut current) = status.lock() {
            if let Some(job) = current
                .as_mut()
                .filter(|job| job.name == name && job.operation == "update")
            {
                match result {
                    Ok(Ok(message)) => {
                        job.status = "completed";
                        job.message = message;
                    }
                    Ok(Err(error)) => {
                        tracing::warn!(engine = %name, %error, "Engine update failed");
                        job.status = "failed";
                        job.message = "Engine update failed; check Infer logs".into();
                    }
                    Err(error) => {
                        tracing::error!(engine = %name, %error, "Engine update task failed");
                        job.status = "failed";
                        job.message = "Engine update failed; check Infer logs".into();
                    }
                }
            }
        }
    });
    Ok((StatusCode::ACCEPTED, Json(job)))
}

#[derive(Serialize)]
pub(super) struct EngineRelease {
    latest: String,
    detail: String,
}

fn can_check_engine_release(name: &str) -> bool {
    matches!(
        name,
        "llama-server"
            | "llama-server-prism"
            | "nemo-speech"
            | "audio-cpp"
            | "crispasr"
            | "sd-server"
            | "needle"
            | "whisper-cpp"
    )
}

pub(super) async fn check_engine_release(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Reply<Json<EngineRelease>> {
    store(&state)?;
    if !can_check_engine_release(&name) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Release checks are unavailable for this engine",
        ));
    }
    static CHECKS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
    let permit = CHECKS.try_acquire().map_err(|_| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "An engine release check is already running",
        )
    })?;
    let release = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        crate::engine_releases::check_latest_release(&name)
    })
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Engine release check failed",
        )
    })?
    .map_err(|_| {
        (
            StatusCode::BAD_GATEWAY,
            "Engine release check failed; retry later",
        )
    })?;
    Ok(Json(EngineRelease {
        latest: release.tag,
        detail: release.detail,
    }))
}

#[derive(Serialize)]
pub(super) struct EffectiveRoles {
    roles: std::collections::BTreeMap<String, String>,
}

pub(super) async fn roles(State(state): State<Arc<AppState>>) -> Reply<Json<EffectiveRoles>> {
    store(&state)?;
    let config = state.config();
    let mut roles = std::collections::BTreeMap::new();
    for model in &config.models {
        for role in &model.roles {
            roles
                .entry(role.to_ascii_lowercase())
                .or_insert_with(|| model.id.clone());
        }
    }
    for (role, id) in &config.roles {
        if config.models.iter().any(|model| model.id == *id) {
            roles.insert(role.clone(), id.clone());
        }
    }
    Ok(Json(EffectiveRoles { roles }))
}

fn store(state: &AppState) -> Reply<&CatalogStore> {
    let configured = state
        .config()
        .server
        .api_key
        .clone()
        .or_else(|| std::env::var("RUNE_API_KEY").ok())
        .or_else(|| std::env::var("API_KEY").ok());
    if configured.as_deref().is_none_or(|s| s.trim().is_empty()) {
        return Err((
            StatusCode::FORBIDDEN,
            "Catalog management requires an API key",
        ));
    }
    state.management.as_deref().ok_or((
        StatusCode::NOT_IMPLEMENTED,
        "Catalog management is unavailable",
    ))
}

fn references_only(raw: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    value
        .get("models")
        .and_then(|v| v.as_array())
        .is_some_and(|models| {
            models.iter().all(|model| {
                let Some(remote) = model.get("remote").filter(|v| !v.is_null()) else {
                    return true;
                };
                remote
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .is_some_and(|key| {
                        key.starts_with("{{")
                            && key.ends_with("}}")
                            && key.len() > 4
                            && !key[2..key.len() - 2].contains(['{', '}'])
                    })
            })
        })
}

pub(super) async fn read(State(state): State<Arc<AppState>>) -> Reply<Json<CatalogDocument>> {
    let store = store(&state)?;
    let _guard = store
        .lock
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Catalog unavailable"))?;
    let bytes = crate::model_registry::read_regular(&store.catalog)
        .ok()
        .flatten()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Cannot read catalog"))?;
    if bytes.len() > 1_048_576 {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Catalog is too large to edit",
        ));
    }
    let raw = std::str::from_utf8(&bytes)
        .map_err(|_| (StatusCode::UNPROCESSABLE_ENTITY, "Catalog is not UTF-8"))?;
    if !references_only(raw) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Catalog contains inline credentials or invalid JSON",
        ));
    }
    Ok(Json(CatalogDocument {
        revision: crate::catalog_management::revision(&bytes),
        catalog: raw.to_owned(),
    }))
}

pub(super) async fn write(
    State(state): State<Arc<AppState>>,
    Json(document): Json<CatalogDocument>,
) -> Reply<Json<CatalogDocument>> {
    let store = store(&state)?;
    if document.catalog.len() > 1_048_576 || !references_only(&document.catalog) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid catalog or inline credentials",
        ));
    }
    let _guard = store
        .lock
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Catalog unavailable"))?;
    let current = crate::model_registry::read_regular(&store.catalog)
        .ok()
        .flatten()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Cannot read catalog"))?;
    commit(store, &current, &document.revision, &document.catalog)?;
    Ok(Json(CatalogDocument {
        revision: crate::catalog_management::revision(document.catalog.as_bytes()),
        catalog: document.catalog,
    }))
}

fn commit(store: &CatalogStore, current: &[u8], revision: &str, catalog: &str) -> Reply<()> {
    use std::io::Write;
    if crate::catalog_management::revision(current) != revision {
        return Err((
            StatusCode::CONFLICT,
            "Catalog changed; reload before saving",
        ));
    }
    let server = std::fs::read_to_string(&store.server).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot read server configuration",
        )
    })?;
    let env_path = store
        .server
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(".env"));
    let mut variables = std::collections::HashMap::new();
    if let Some(path) = env_path.filter(|p| p.exists()) {
        for entry in dotenvy::from_path_iter(path).map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cannot read Infer environment",
            )
        })? {
            let (key, value) = entry.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Invalid Infer environment",
                )
            })?;
            variables.insert(key, value);
        }
    }
    crate::catalog_management::validate_update(current, revision, &server, catalog, &|key| {
        std::env::var(key)
            .ok()
            .or_else(|| variables.get(key).cloned())
    })
    .map_err(|_| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            "Catalog validation failed",
        )
    })?;
    let parent = store
        .catalog
        .parent()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Invalid catalog path"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Clock error"))?
        .as_nanos();
    let temp = parent.join(format!(".catalog-{}-{stamp}.tmp", std::process::id()));
    let backup = parent.join(format!(".catalog-{}-{stamp}.backup", std::process::id()));
    let result = (|| -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(catalog.as_bytes())?;
        file.sync_all()?;
        drop(file);
        if crate::model_registry::read_regular(&store.catalog)?.as_deref() != Some(current) {
            anyhow::bail!("catalog changed");
        }
        crate::model_registry::move_no_replace(&store.catalog, &backup)?;
        let publish = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                crate::model_registry::read_regular(&backup)?.as_deref() == Some(current),
                "catalog changed during save"
            );
            crate::model_registry::move_no_replace(&temp, &store.catalog)?;
            Ok(())
        })();
        if let Err(error) = publish {
            if crate::model_registry::move_no_replace(&backup, &store.catalog).is_err() {
                return Err(error.context(format!(
                    "Previous catalog preserved at {}",
                    backup.display()
                )));
            }
            return Err(error);
        }
        Ok(())
    })();
    if let Err(error) = result {
        tracing::warn!(%error, "Catalog replacement failed");
        let _ = std::fs::remove_file(&temp);
        return Err((
            StatusCode::CONFLICT,
            "Catalog could not be saved; reload and retry",
        ));
    }
    Ok(())
}

#[derive(Serialize)]
pub(super) struct Removal {
    revision: String,
    removed: Vec<String>,
    failed: Vec<String>,
}

fn strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) => out.push(text.clone()),
        serde_json::Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
        serde_json::Value::Object(map) => map.values().for_each(|item| strings(item, out)),
        _ => {}
    }
}

pub(super) async fn remove_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Reply<Json<Removal>> {
    let store = store(&state)?;
    if !crate::model_registry::valid_model_id(&id) {
        return Err((StatusCode::BAD_REQUEST, "Invalid model ID"));
    }
    let registered = crate::model_registry::bundled()
        .ok()
        .and_then(|models| models.into_iter().find(|model| model.id == id))
        .ok_or((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Only reviewed registry models can be removed",
        ))?;
    remove_with(store, &id, &registered)
}

fn remove_with(
    store: &CatalogStore,
    id: &str,
    registered: &crate::model_registry::LocalModel,
) -> Reply<Json<Removal>> {
    let _guard = store
        .lock
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Catalog unavailable"))?;
    let current = crate::model_registry::read_regular(&store.catalog)
        .ok()
        .flatten()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Cannot read catalog"))?;
    let mut catalog: serde_json::Value = serde_json::from_slice(&current)
        .map_err(|_| (StatusCode::UNPROCESSABLE_ENTITY, "Invalid catalog"))?;
    let models = catalog
        .get("models")
        .and_then(|value| value.as_array())
        .ok_or((StatusCode::UNPROCESSABLE_ENTITY, "Invalid catalog"))?;
    let index = models
        .iter()
        .position(|model| model.get("id").and_then(|value| value.as_str()) == Some(id))
        .ok_or((StatusCode::NOT_FOUND, "Model is not installed"))?;
    let path = models[index]
        .get("local")
        .and_then(|local| local.get("path"))
        .and_then(|path| path.as_str())
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Model has no managed local files",
        ))?;
    for key in ["roles", "aliases"] {
        if catalog
            .get(key)
            .and_then(|value| value.as_object())
            .is_some_and(|map| map.values().any(|value| value.as_str() == Some(id)))
        {
            return Err((
                StatusCode::CONFLICT,
                "Model is used by a role or alias; choose another model first",
            ));
        }
    }
    let root = store
        .server
        .parent()
        .and_then(|parent| parent.parent())
        .and_then(|root| std::fs::canonicalize(root.join("models")).ok())
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot find model storage",
        ))?;
    let model_dir = path
        .parent()
        .and_then(|dir| std::fs::canonicalize(dir).ok())
        .filter(|dir| dir.starts_with(&root) && *dir != root)
        .ok_or((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Model files are outside managed storage",
        ))?;
    let mut shared = Vec::new();
    for (position, model) in models.iter().enumerate() {
        if position != index {
            strings(model, &mut shared);
        }
    }
    let files: std::collections::BTreeSet<_> = registered
        .versions
        .values()
        .flat_map(|version| version.files.iter())
        .map(|file| {
            (
                crate::model_registry::bundle_path(&model_dir, file),
                file.sha256.to_ascii_lowercase(),
                file.id.clone(),
            )
        })
        .collect();
    if let Some(models) = catalog
        .get_mut("models")
        .and_then(|value| value.as_array_mut())
    {
        models.remove(index);
    }
    let text = serde_json::to_string_pretty(&catalog)
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Invalid catalog"))?;
    let revision = crate::catalog_management::revision(&current);
    commit(store, &current, &revision, &text)?;
    let mut removed = Vec::new();
    let mut failed = Vec::new();
    for (target, sha, file_id) in files {
        let name = target
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let shared_file = shared
            .iter()
            .any(|value| std::path::Path::new(value) == target);
        let staged = target
            .parent()
            .map(|parent| parent.join(format!(".rune-model-download-{file_id}-{sha}.part")));
        if let Some(staged) = staged {
            let _ = std::fs::remove_file(staged);
        }
        let regular = std::fs::symlink_metadata(&target)
            .is_ok_and(|meta| meta.is_file() && !meta.file_type().is_symlink());
        if shared_file || !regular {
            continue;
        }
        match std::fs::remove_file(&target) {
            Ok(()) => removed.push(name),
            Err(_) => failed.push(name),
        }
    }
    let _ = std::fs::remove_dir(model_dir.join("text_encoder"));
    let leftovers: Vec<_> = std::fs::read_dir(&model_dir)
        .map(|entries| entries.flatten().map(|entry| entry.file_name()).collect())
        .unwrap_or_default();
    if leftovers
        .iter()
        .all(|name| name == ".rune-model-install.lock")
    {
        let _ = std::fs::remove_file(model_dir.join(".rune-model-install.lock"));
        let _ = std::fs::remove_dir(&model_dir);
    }
    Ok(Json(Removal {
        revision: crate::catalog_management::revision(text.as_bytes()),
        removed,
        failed,
    }))
}

#[derive(Serialize)]
pub(super) struct Residency {
    changed: bool,
}

fn residency_error(message: &str) -> (StatusCode, &'static str) {
    if message.contains("exact local model") {
        (StatusCode::UNPROCESSABLE_ENTITY, "Select a local model")
    } else if message.contains("still loading") || message.contains("in use") {
        (StatusCode::CONFLICT, "Model is busy; retry shortly")
    } else {
        (StatusCode::BAD_GATEWAY, "Model could not be changed")
    }
}

pub(super) async fn load_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Reply<Json<Residency>> {
    store(&state)?;
    let inference = state.inference.clone();
    tokio::task::spawn_blocking(move || inference.load_model(&id))
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Load was interrupted"))?
        .map_err(|message| residency_error(&message))?;
    Ok(Json(Residency { changed: true }))
}

pub(super) async fn unload_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Reply<Json<Residency>> {
    store(&state)?;
    let inference = state.inference.clone();
    let changed = tokio::task::spawn_blocking(move || inference.unload_model(&id))
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Unload was interrupted"))?
        .map_err(|message| residency_error(&message))?;
    Ok(Json(Residency { changed }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        extract::Request,
    };
    use tower::ServiceExt;

    #[test]
    fn rejects_inline_provider_credentials() {
        assert!(!references_only(
            r#"{"models":[{"remote":{"api_key":"secret"}}]}"#
        ));
        assert!(references_only(
            r#"{"models":[{"remote":{"api_key":"{{PROVIDER_KEY}}"}}]}"#
        ));
        assert!(!references_only(
            r#"{"models":[{"remote":{"api_key":"{{KEY}}suffix"}}]}"#
        ));
    }

    #[tokio::test]
    async fn effective_roles_use_running_config_and_require_authentication() {
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        for (id, roles) in [("first", vec!["voice"]), ("second", vec!["VOICE"])] {
            let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
                "id": id, "name": id, "modality": "Text", "roles": roles,
                "engine": {"supported": ["llama-server"]}, "local": {"path": "model.gguf"}
            }))
            .unwrap();
            config.models.push(entry.try_into().unwrap());
        }
        config.roles.insert("agent".into(), "second".into());
        config.roles.insert("voice".into(), "second".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config.clone(),
        );
        state.management = Some(Arc::new(CatalogStore::new(
            "unused-server".into(),
            "unused-catalog".into(),
        )));
        let router = crate::api::create_router(state.clone());
        let request = |authorized| {
            let builder = Request::builder().uri("/v1/management/roles");
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request(false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let response = router.clone().oneshot(request(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"roles":{"agent":"second","voice":"second"}})
        );
        config.roles.remove("voice");
        config.roles.remove("agent");
        state.reload_catalog(config);
        let response = router.oneshot(request(true)).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(body, serde_json::json!({"roles":{"voice":"first"}}));
    }

    #[tokio::test]
    async fn model_candidates_require_auth_and_validate_purpose() {
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new(
            "unused-server".into(),
            "unused-catalog".into(),
        )));
        let router = crate::api::create_router(state);
        let request = |query: &str, authorized: bool| {
            let builder = Request::builder().uri(format!("/v1/management/model-candidates{query}"));
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request("", false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            router
                .clone()
                .oneshot(request("?purpose=unknown", true))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        let response = router
            .oneshot(request("?purpose=computer-use", true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1_000_000).await.unwrap())
                .unwrap();
        assert!(
            body.as_array()
                .is_some_and(|candidates| !candidates.is_empty())
        );
    }

    #[tokio::test]
    async fn engine_inventory_requires_authentication_and_reports_entries() {
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new(
            "unused-server".into(),
            "unused-catalog".into(),
        )));
        let jobs = state.engine_installation.clone();
        let router = crate::api::create_router(state);
        let request = |authorized| {
            let builder = Request::builder().uri("/v1/management/engines");
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request(false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let release_request = |authorized: bool, name: &str| {
            let builder = Request::builder()
                .method("POST")
                .uri(format!("/v1/management/engines/{name}/release-check"));
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(release_request(false, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            router
                .clone()
                .oneshot(release_request(true, "encoder"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        let installation_request = |method: &str, authorized: bool, name: &str| {
            let builder = Request::builder()
                .method(method)
                .uri(format!("/v1/management/engines/{name}/installation"));
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("POST", false, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("POST", true, "encoder"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("GET", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        *jobs.lock().unwrap() = Some(EngineInstallation {
            name: "needle".into(),
            operation: "install",
            status: "running",
            message: "Installing engine".into(),
        });
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("GET", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("POST", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        let update_request = |method: &str, authorized: bool, name: &str| {
            let builder = Request::builder()
                .method(method)
                .uri(format!("/v1/management/engines/{name}/update"));
            let builder = if authorized {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(update_request("POST", false, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            router
                .clone()
                .oneshot(update_request("POST", true, "encoder"))
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            router
                .clone()
                .oneshot(update_request("GET", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        *jobs.lock().unwrap() = Some(EngineInstallation {
            name: "needle".into(),
            operation: "update",
            status: "running",
            message: "Updating engine".into(),
        });
        assert_eq!(
            router
                .clone()
                .oneshot(update_request("GET", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            router
                .clone()
                .oneshot(installation_request("GET", true, "needle"))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        let response = router.oneshot(request(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 32768).await.unwrap()).unwrap();
        assert!(
            body["engines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["name"] == "encoder" && item["installed"] == true)
        );
    }

    #[tokio::test]
    #[ignore = "requires an isolated RUNE_HOME and network access"]
    async fn managed_needle_update_reports_current_without_replacing_binary() {
        let root = crate::engine_inventory::managed_root().unwrap();
        let config = crate::config::ModelRegistry::new();
        let inference = Arc::new(crate::inference::AppState::new(&config));
        let installer_state = inference.clone();
        tokio::task::spawn_blocking(move || {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (events, receiver) = std::sync::mpsc::channel();
            drop(receiver);
            crate::inference::engines::update::install_engine(
                installer_state,
                stop,
                crate::config::EngineKind::Needle,
                events,
            )
            .unwrap();
        })
        .await
        .unwrap();
        let binary = root.join("bin/needle").join(if cfg!(windows) {
            "needle.exe"
        } else {
            "needle"
        });
        let before = crate::inference::engines::download::file_sha256(&binary).unwrap();
        let runtime = crate::inference::engines::registry::needle_runtime().unwrap();
        let (events, receiver) = std::sync::mpsc::channel();
        drop(receiver);
        let current = crate::inference::engines::update::update_needle(
            inference.clone(),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            format!("revision {}", runtime.revision),
            events,
        )
        .unwrap();
        assert!(current.contains("up to date"));
        let mut config = config;
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            inference,
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new(
            "unused-server".into(),
            "unused-catalog".into(),
        )));
        let router = crate::api::create_router(state);
        let request = |method| {
            Request::builder()
                .method(method)
                .uri("/v1/management/engines/needle/update")
                .header("authorization", "Bearer test-key")
                .body(Body::empty())
                .unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request("POST"))
                .await
                .unwrap()
                .status(),
            StatusCode::ACCEPTED
        );
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let response = router.clone().oneshot(request("GET")).await.unwrap();
            let body: serde_json::Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 32768).await.unwrap())
                    .unwrap();
            if body["status"] != "running" {
                assert_eq!(body["status"], "completed", "{body}");
                assert!(
                    body["message"].as_str().unwrap().contains("up to date"),
                    "{body}"
                );
                break;
            }
            assert!(tokio::time::Instant::now() < deadline, "Update timed out");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(
            crate::inference::engines::download::file_sha256(&binary).unwrap(),
            before
        );
    }

    #[tokio::test]
    async fn management_requires_auth_and_rejects_stale_updates() {
        let dir = std::env::temp_dir().join(format!(
            "infer-catalog-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let server = dir.join("server.json");
        let catalog = dir.join("catalog.json");
        std::fs::write(&server, r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"api_key":"test-key","max_loaded_models":1,"idle_unload_seconds":300}}"#).unwrap();
        let original = r#"{"schema_version":3,"models":[]}"#;
        std::fs::write(&catalog, original).unwrap();
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new(server.clone(), catalog.clone())));
        let router = crate::api::create_router(state);
        let request = |key: bool| {
            let builder = Request::builder().uri("/v1/management/catalog");
            let builder = if key {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request(false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let response = router.clone().oneshot(request(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let document: CatalogDocument =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        std::fs::write(&catalog, format!("{original}\n")).unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/v1/management/catalog")
                    .header("authorization", "Bearer test-key")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&document).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            std::fs::read_to_string(&catalog).unwrap(),
            format!("{original}\n")
        );
        let current = std::fs::read(&catalog).unwrap();
        let updated = CatalogDocument {
            revision: crate::catalog_management::revision(&current),
            catalog: "{\"schema_version\":3,\"models\":[],\"aliases\":{\"alias\":\"example\"}}\n"
                .into(),
        };
        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/v1/management/catalog")
                    .header("authorization", "Bearer test-key")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&updated).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(std::fs::read_to_string(&catalog).unwrap(), updated.catalog);
        std::fs::remove_file(server).unwrap();
        std::fs::remove_file(catalog).unwrap();
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(dir).unwrap();
    }

    #[tokio::test]
    async fn removing_a_model_deletes_its_files_and_refuses_models_in_use() {
        let registered: crate::model_registry::LocalModel = serde_json::from_value(serde_json::json!({
            "id": "test-model", "name": "Test model", "description": "Test",
            "license": "MIT", "source_url": "https://example.invalid",
            "engine": "llama-server", "format": "gguf", "capabilities": ["Chat"],
            "latest": "v1",
            "versions": {"v1": {"release_number": 1, "files": [
                {"id": "model", "filename": "test.gguf", "url": "https://example.invalid/test.gguf",
                 "sha256": "00", "size_bytes": 4}
            ]}}
        }))
        .unwrap();
        let file = registered.versions[&registered.latest]
            .files
            .iter()
            .find(|file| file.id == "model")
            .unwrap()
            .filename
            .clone();
        let root = std::env::temp_dir().join(format!(
            "infer-remove-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config_dir = root.join("config");
        let model_dir = root.join("models").join("maker").join("family");
        std::fs::create_dir_all(&config_dir).unwrap();
        std::fs::create_dir_all(&model_dir).unwrap();
        let model_dir = std::fs::canonicalize(&model_dir).unwrap();
        let weights = model_dir.join(&file);
        std::fs::write(&weights, b"GGUF").unwrap();
        let server = config_dir.join("server.json");
        let catalog = config_dir.join("catalog.json");
        std::fs::write(&server, r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"api_key":"test-key","max_loaded_models":1,"idle_unload_seconds":300}}"#).unwrap();
        let entry = serde_json::json!({
            "id": registered.id, "name": registered.name, "modality": "Text",
            "engine": {"supported": ["llama-server"]},
            "local": {"path": weights.to_string_lossy()}
        });
        let write_catalog = |roles: serde_json::Value| {
            std::fs::write(
                &catalog,
                serde_json::json!({"schema_version": 3, "models": [entry], "roles": roles})
                    .to_string(),
            )
            .unwrap();
        };
        let store = CatalogStore::new(server.clone(), catalog.clone());
        write_catalog(serde_json::json!({"agent": registered.id}));
        let refused = remove_with(&store, &registered.id, &registered)
            .err()
            .unwrap();
        assert_eq!(refused.0, StatusCode::CONFLICT);
        assert!(weights.exists());
        let missing = remove_with(&store, "other-model", &registered)
            .err()
            .unwrap();
        assert_eq!(missing.0, StatusCode::NOT_FOUND);
        write_catalog(serde_json::json!({}));
        let done = remove_with(&store, &registered.id, &registered)
            .ok()
            .unwrap();
        assert_eq!(done.0.removed, vec![file.clone()]);
        assert!(!weights.exists());
        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&catalog).unwrap()).unwrap();
        assert!(after["models"].as_array().unwrap().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn loading_requires_auth_and_an_exact_local_model() {
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new("s".into(), "c".into())));
        let router = crate::api::create_router(state);
        let post = |action: &str, key: bool| {
            let builder = Request::builder()
                .method("POST")
                .uri(format!("/v1/management/models/missing/{action}"));
            let builder = if key {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        for action in ["load", "unload"] {
            assert_eq!(
                router
                    .clone()
                    .oneshot(post(action, false))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::UNAUTHORIZED
            );
            assert_eq!(
                router
                    .clone()
                    .oneshot(post(action, true))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::UNPROCESSABLE_ENTITY
            );
        }
    }
}
