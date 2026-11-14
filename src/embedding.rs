//! Process-wide settings for hosts embedding Infer without changing CWD.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ROOT: OnceLock<PathBuf> = OnceLock::new();
pub const WORKER_CONTRACT: &str = concat!("rune-infer-worker-v1 ", env!("CARGO_PKG_VERSION"));

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

pub(crate) fn configure_child(command: &mut std::process::Command, executable: &Path) {
    if root().is_none() {
        return;
    }
    if let Some(parent) = executable.parent().filter(|path| path.is_absolute()) {
        command.current_dir(parent);
    }
    if let Some(paths) = std::env::var_os("PATH") {
        let safe = std::env::split_paths(&paths).filter(|path| path.is_absolute());
        if let Ok(paths) = std::env::join_paths(safe) {
            command.env("PATH", paths);
        }
    }
}

pub(crate) fn system_tool(name: &str) -> std::io::Result<PathBuf> {
    if root().is_none() {
        return Ok(PathBuf::from(name));
    }
    let executable = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    std::env::var_os("PATH")
        .and_then(|paths| find_absolute_tool(&paths, &executable))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{name} was not found in an absolute PATH directory"),
            )
        })
}

fn find_absolute_tool(paths: &std::ffi::OsStr, executable: &str) -> Option<PathBuf> {
    std::env::split_paths(paths)
        .filter(|path| path.is_absolute())
        .map(|path| path.join(executable))
        .find(|path| path.is_file())
}

pub(crate) fn no_window(command: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

pub(crate) fn system_command(name: &str) -> std::io::Result<std::process::Command> {
    let path = system_tool(name)?;
    let mut command = std::process::Command::new(&path);
    configure_child(&mut command, &path);
    no_window(&mut command);
    Ok(command)
}

pub(crate) fn worker_executable(
    override_name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = match std::env::var_os(override_name) {
        Some(path) => PathBuf::from(path),
        None => crate::home::installed_infer_binary()?,
    };
    if !path.is_absolute() {
        return Err(
            format!("{override_name} must be an absolute path when Infer is embedded").into(),
        );
    }
    let path = std::fs::canonicalize(path)
        .map_err(|_| "Install the matching rune-infer executable to use this engine")?;
    let mut command = std::process::Command::new(&path);
    configure_child(&mut command, &path);
    command.arg("worker-contract");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let mut child = command.spawn()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(
                "The installed rune-infer worker did not respond to the compatibility check".into(),
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let output = child.wait_with_output()?;
    let version = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || version.trim() != WORKER_CONTRACT {
        return Err(
            "The installed rune-infer worker version does not match the embedded library".into(),
        );
    }
    Ok(path)
}

pub struct LocalServer {
    pub base_url: String,
    pub api_key: String,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), String>>,
}

impl LocalServer {
    pub async fn start(root: &Path, model: &str) -> Result<Self, String> {
        let config = load_local_model(root, model)
            .map_err(|error| format!("Cannot load the local Infer model: {error:#}"))?;
        Self::start_with_config(root, config).await
    }

    /// Start the complete configured catalog, including local and remote models.
    pub async fn start_catalog(root: &Path) -> Result<Self, String> {
        let config = load_catalog(root)
            .map_err(|error| format!("Cannot load the Infer catalog: {error:#}"))?;
        Self::start_with_config(root, config).await
    }

    async fn start_with_config(
        root: &Path,
        mut config: crate::config::ModelRegistry,
    ) -> Result<Self, String> {
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

fn read_catalog(root: &Path) -> anyhow::Result<crate::config::ModelRegistry> {
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
    crate::config::ModelRegistry::from_files_with_env(
        root.join("config/server.json"),
        root.join("config/catalog.json"),
        &overrides,
    )
}

fn absolute_local_paths(root: &Path, model: &mut crate::config::ModelConfig) {
    fn absolute(root: &Path, value: &mut String) {
        if !Path::new(value).is_absolute() {
            *value = root.join(&*value).to_string_lossy().into_owned();
        }
    }
    if !model.is_local() {
        return;
    }
    absolute(root, &mut model.model_path);
    for path in [
        &mut model.mmproj_path,
        &mut model.mtp_path,
        &mut model.vae_path,
        &mut model.text_encoder_path,
        &mut model.tokenizer_path,
        &mut model.llm_vision_path,
    ]
    .into_iter()
    .flatten()
    {
        absolute(root, path);
    }
    for lora in &mut model.lora {
        absolute(root, &mut lora.path);
    }
}

/// Load the configured catalog without changing the host's current directory or environment.
pub fn load_catalog(root: &Path) -> anyhow::Result<crate::config::ModelRegistry> {
    let mut config = read_catalog(root)?;
    for model in &mut config.models {
        absolute_local_paths(root, model);
    }
    Ok(config)
}

/// Prepare a fresh local Infer configuration without replacing existing files.
pub fn prepare_config(root: &Path) -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    static SETUP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = SETUP_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("Infer setup is unavailable"))?;
    ensure!(root.is_absolute(), "Infer home must be absolute");
    crate::model_registry::regular_dir(root)?;
    let config_dir = root.join("config");
    crate::model_registry::regular_dir(&config_dir)?;
    let server = config_dir.join("server.json");
    let catalog = config_dir.join("catalog.json");
    let server_exists = crate::model_registry::read_regular(&server)?.is_some();
    let catalog_exists = crate::model_registry::read_regular(&catalog)?.is_some();
    if server_exists && catalog_exists {
        read_catalog(root)?;
        return Ok(());
    }
    ensure!(
        !server_exists && !catalog_exists,
        "Infer configuration is incomplete; existing files were preserved"
    );
    ensure!(
        crate::model_registry::read_regular(&config_dir.join("models.json"))?.is_none(),
        "Legacy Infer configuration exists; existing files were preserved"
    );
    let server_bytes = serde_json::to_vec_pretty(&crate::schema::ServerFile {
        schema_version: crate::schema::SCHEMA_VERSION,
        server: crate::config::ServerConfig {
            host: "127.0.0.1".into(),
            ..Default::default()
        },
    })?;
    let catalog_bytes = serde_json::to_vec_pretty(&crate::schema::CatalogFile {
        schema_version: crate::schema::SCHEMA_VERSION,
        models: Vec::new(),
        roles: Default::default(),
        aliases: Default::default(),
    })?;
    crate::model_registry::write_new(&server, &server_bytes)
        .context("Cannot create Infer server configuration; existing files were preserved")?;
    crate::model_registry::write_new(&catalog, &catalog_bytes)
        .context("Cannot create Infer catalog; any existing configuration was preserved")?;
    Ok(())
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
    let mut config = read_catalog(root)?;
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
    absolute_local_paths(root, model);
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

#[cfg(test)]
mod catalog_tests {
    use super::*;

    #[cfg(windows)]
    #[tokio::test]
    #[ignore = "starts an isolated embedded Infer server"]
    async fn fresh_setup_serves_computer_use_candidates() {
        let root = std::env::temp_dir().join(format!(
            "infer-prepare-server-{}",
            crate::setup::generate_api_key()
        ));
        prepare_config(&root).unwrap();
        let server = LocalServer::start_catalog(&root).await.unwrap();
        let client = reqwest::Client::new();
        let url = format!(
            "{}/v1/management/model-candidates?purpose=computer-use",
            server.base_url
        );
        assert_eq!(client.get(&url).send().await.unwrap().status(), 401);
        let models: serde_json::Value = client
            .get(&url)
            .bearer_auth(&server.api_key)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(models.as_array().is_some_and(|items| !items.is_empty()));
        server.shutdown().await.unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prepare_config_creates_fresh_setup_and_preserves_existing_files() {
        let root = std::env::temp_dir().join(format!(
            "infer-prepare-{}",
            crate::setup::generate_api_key()
        ));
        prepare_config(&root).unwrap();
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        let first_server = std::fs::read(&server).unwrap();
        let first_catalog = std::fs::read(&catalog).unwrap();
        let config = load_catalog(&root).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(config.models.is_empty());
        prepare_config(&root).unwrap();
        assert_eq!(std::fs::read(&server).unwrap(), first_server);
        assert_eq!(std::fs::read(&catalog).unwrap(), first_catalog);
        std::fs::remove_file(&catalog).unwrap();
        assert!(prepare_config(&root).is_err());
        assert_eq!(std::fs::read(&server).unwrap(), first_server);
        assert!(!catalog.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prepare_config_refuses_legacy_files_without_modifying_them() {
        let root = std::env::temp_dir().join(format!(
            "infer-prepare-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir_all(root.join("config")).unwrap();
        let legacy = root.join("config/models.json");
        std::fs::write(&legacy, b"legacy fixture").unwrap();
        assert!(prepare_config(&root).is_err());
        assert_eq!(std::fs::read(&legacy).unwrap(), b"legacy fixture");
        assert!(!root.join("config/server.json").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embedded_system_tool_lookup_ignores_relative_path_entries() {
        let relative = std::env::join_paths([Path::new(".")]).unwrap();
        assert!(find_absolute_tool(&relative, "Cargo.toml").is_none());
        let absolute = std::env::join_paths([std::env::current_dir().unwrap()]).unwrap();
        assert!(find_absolute_tool(&absolute, "Cargo.toml").is_some());
    }

    #[test]
    fn complete_catalog_preserves_roles_and_anchors_only_local_paths() {
        let root = std::env::temp_dir().join(format!(
            "infer-embedded-catalog-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config_dir = root.join("config");
        std::fs::create_dir_all(&config_dir).unwrap();
        let variable = format!("INFER_EMBED_TEST_KEY_{}", std::process::id());
        std::fs::write(root.join(".env"), format!("{variable}=fixture-key\n")).unwrap();
        std::fs::write(config_dir.join("server.json"), r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"api_key":null,"max_loaded_models":1,"idle_unload_seconds":300}}"#).unwrap();
        let catalog = serde_json::json!({"schema_version":3,"models":[
            {"id":"local","name":"Local","modality":"Text","engine":{"supported":["llama-server"]},"local":{"path":"models/local.gguf"},"extra":{"mmproj_path":"models/projector.gguf"}},
            {"id":"remote","name":"Remote","modality":"Text","remote":{"provider":"openai","upstream_model":"cloud","api_key":format!("{{{{{variable}}}}}")}}
        ],"roles":{"agent":"local","voice":"remote"},"aliases":{"primary":"local"}});
        std::fs::write(
            config_dir.join("catalog.json"),
            serde_json::to_vec(&catalog).unwrap(),
        )
        .unwrap();
        let cwd = std::env::current_dir().unwrap();
        let config = load_catalog(&root).unwrap();
        assert_eq!(std::env::current_dir().unwrap(), cwd);
        assert_eq!(config.models.len(), 2);
        assert_eq!(
            Path::new(&config.models[0].model_path),
            root.join("models/local.gguf")
        );
        assert_eq!(
            Path::new(config.models[0].mmproj_path.as_deref().unwrap()),
            root.join("models/projector.gguf")
        );
        assert_eq!(config.models[1].upstream_model.as_deref(), Some("cloud"));
        assert_eq!(
            config.roles.get("voice").map(String::as_str),
            Some("remote")
        );
        assert_eq!(
            config.aliases.get("primary").map(String::as_str),
            Some("local")
        );
        assert!(load_catalog(Path::new("relative")).is_err());
        std::fs::remove_file(config_dir.join("server.json")).unwrap();
        std::fs::remove_file(config_dir.join("catalog.json")).unwrap();
        std::fs::remove_file(root.join(".env")).unwrap();
        std::fs::remove_dir(config_dir).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
