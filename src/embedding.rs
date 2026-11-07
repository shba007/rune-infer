//! Process-wide settings for hosts embedding Infer without changing CWD.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Configure engine storage for an embedded Windows host before starting Infer.
/// An embedding host owns the process lifetime and must drain its server on exit.
/// Only one Infer home can be used by an embedding process.
pub fn initialize(root: &Path) -> Result<(), &'static str> {
    if !cfg!(windows) {
        return Err("Embedded Infer is currently supported on Windows only");
    }
    if !root.is_absolute() {
        return Err("Embedded Infer requires an absolute Infer home");
    }
    if let Some(existing) = ROOT.get() {
        return if existing == root {
            Ok(())
        } else {
            Err("Embedded Infer was already initialized with another home")
        };
    }
    ROOT.set(root.to_path_buf())
        .map_err(|_| "Embedded Infer was initialized concurrently")
}

pub(crate) fn root() -> Option<&'static Path> {
    ROOT.get().map(PathBuf::as_path)
}

pub struct LocalServer {
    pub base_url: String,
    pub api_key: String,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), String>>,
}

impl LocalServer {
    pub async fn start(root: &Path, model: &str) -> Result<Self, String> {
        let mut config = load_local_model(root, model)
            .map_err(|error| format!("Cannot load the local Infer model: {error:#}"))?;
        let api_key = crate::setup::generate_api_key();
        config.server.api_key = Some(api_key.clone());
        config.server.host = "127.0.0.1".into();
        config.server.port = 0;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| format!("Cannot start local Infer: {error}"))?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        initialize(root).map_err(str::to_owned)?;
        let server_path = root.join("config/server.json");
        let catalog_path = root.join("config/catalog.json");
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            crate::server::serve_embedded_with_management(
                config,
                server_path,
                catalog_path,
                listener,
                async {
                    let _ = stopped.await;
                },
            )
            .await
            .map_err(|error| format!("Local Infer stopped: {error:#}"))
        });
        Ok(Self {
            base_url: format!("http://{address}"),
            api_key,
            stop: Some(stop),
            task,
        })
    }

    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    pub async fn shutdown(mut self) -> Result<(), String> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(10), &mut self.task).await {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => Err(format!("Local Infer stopped unexpectedly: {error}")),
            Err(_) => {
                self.task.abort();
                let _ = self.task.await;
                Err("Local Infer did not stop in time".into())
            }
        }
    }
}

/// Prepare the upstream llama engine using the embedding host's home.
pub fn install_llama_observed(
    root: &Path,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    progress: impl FnMut(crate::inference::engines::download::DownloadProgress) + 'static,
) -> Result<PathBuf, String> {
    initialize(root).map_err(str::to_owned)?;
    crate::inference::engines::download::with_cancellation(cancel, || {
        crate::inference::engines::download::with_progress(progress, || {
            crate::setup::install_llama_server()
        })
    })
}

/// Load one existing local llama model without changing the host's CWD or env.
/// Relative model paths have the same meaning as in Infer's standalone home.
/// The returned snapshot is intentionally not hot-reloaded.
pub fn load_local_model(
    root: &Path,
    model_id: &str,
) -> anyhow::Result<crate::config::ModelRegistry> {
    use crate::config::EngineKind;
    use anyhow::{Context, ensure};

    ensure!(root.is_absolute(), "Infer home must be absolute");
    let env_path = root.join(".env");
    let mut overrides = Vec::new();
    if env_path.exists() {
        for entry in dotenvy::from_path_iter(&env_path).context("Cannot read Infer environment")? {
            let (key, value) = entry.context("Invalid Infer environment")?;
            if std::env::var_os(&key).is_none() {
                overrides.push((key, value));
            }
        }
    }
    let mut config = crate::config::ModelRegistry::from_files_with_env(
        root.join("config/server.json"),
        root.join("config/catalog.json"),
        &overrides,
    )?;
    config.models.retain(|model| model.id == model_id);
    let model = config
        .models
        .first_mut()
        .context("Configured local model is unavailable")?;
    ensure!(
        model.is_local(),
        "Embedded computer use requires a local model"
    );
    let engines = model.engine_plan();
    ensure!(
        !engines.is_empty()
            && engines.iter().all(|engine| matches!(
                engine,
                EngineKind::LlamaServer | EngineKind::LlamaServerPrism
            )),
        "Embedded computer use currently supports only llama-server engines"
    );
    fn absolute(root: &Path, value: &mut String) {
        if !Path::new(value).is_absolute() {
            *value = root.join(&*value).to_string_lossy().into_owned();
        }
    }
    absolute(root, &mut model.model_path);
    for path in [&mut model.mmproj_path, &mut model.mtp_path] {
        if let Some(path) = path.as_mut() {
            absolute(root, path);
        }
    }
    for lora in &mut model.lora {
        absolute(root, &mut lora.path);
    }
    ensure!(
        Path::new(&model.model_path).is_file(),
        "Configured local model file is missing"
    );
    for path in [&model.mmproj_path, &model.mtp_path].into_iter().flatten() {
        ensure!(
            Path::new(path).is_file(),
            "A configured local model companion file is missing"
        );
    }
    config.roles.clear();
    config.aliases.clear();
    Ok(config)
}
