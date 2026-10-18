//! Resolution of the Rune Infer home directory (`~/.rune-infer`) and the
//! config file location.
//!
//! Layout inside the home directory (also used verbatim for a dev checkout,
//! which keeps these paths relative to the current directory):
//!
//! ```text
//! <home>/config/models.json   model catalog + server settings
//! <home>/.env                 secrets referenced from the config as {{VAR}}
//! <home>/bin/                 auto-installed engine binaries
//! <home>/logs/rune-infer.log  audit log
//! ```

use std::path::{Path, PathBuf};

pub const HOME_ENV: &str = "RUNE_HOME";
pub const DEFAULT_SERVER_REL: &str = "config/server.json";
pub const DEFAULT_CATALOG_REL: &str = "config/catalog.json";
pub const LEGACY_CONFIG_REL: &str = "config/models.json";

/// `$RUNE_HOME`, else `~/.rune-infer`.
pub fn rune_home() -> PathBuf {
    if let Some(custom) = std::env::var_os(HOME_ENV).filter(|v| !v.is_empty()) {
        return PathBuf::from(custom);
    }
    user_home().join(".rune-infer")
}

fn user_home() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Where the process should treat as its working root.
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Directory the process runs in (config/, bin/, logs/ and .env live here).
    pub root: PathBuf,
    /// Absolute path of server.json (schema 3).
    pub server_path: PathBuf,
    /// Absolute path of catalog.json (schema 3).
    pub catalog_path: PathBuf,
    /// Absolute path of legacy models.json (schema 1 & 2).
    pub legacy_config_path: PathBuf,
}

impl Workspace {
    /// Pick the workspace.
    ///
    /// * `--config <path>` given: honoured as-is; root is inferred.
    /// * `RUNE_HOME` set: that folder.
    /// * `./config/server.json` or `./config/models.json` exists in current dir: dev checkout.
    /// * otherwise `~/.rune-infer`.
    pub fn resolve(explicit_config: Option<&str>) -> std::io::Result<Self> {
        let cwd = std::env::current_dir()?;

        if let Some(raw) = explicit_config {
            let path = absolutize(&cwd, Path::new(raw));
            if path.is_dir() {
                let root = if path.file_name().is_some_and(|n| n == "config") {
                    path.parent().unwrap_or(&path).to_path_buf()
                } else {
                    path.clone()
                };
                let cfg_dir = if path.file_name().is_some_and(|n| n == "config") {
                    path
                } else {
                    root.join("config")
                };
                return Ok(Self {
                    server_path: cfg_dir.join("server.json"),
                    catalog_path: cfg_dir.join("catalog.json"),
                    legacy_config_path: cfg_dir.join("models.json"),
                    root,
                });
            }

            let parent = path.parent().unwrap_or(&cwd).to_path_buf();
            let root = if parent.file_name().is_some_and(|n| n == "config") {
                parent.parent().unwrap_or(&parent).to_path_buf()
            } else {
                parent.clone()
            };

            let (server_path, catalog_path, legacy_config_path) =
                match path.file_name().and_then(|n| n.to_str()) {
                    Some("server.json") => (
                        path.clone(),
                        parent.join("catalog.json"),
                        parent.join("models.json"),
                    ),
                    Some("catalog.json") => (
                        parent.join("server.json"),
                        path.clone(),
                        parent.join("models.json"),
                    ),
                    _ => (
                        parent.join("server.json"),
                        parent.join("catalog.json"),
                        path.clone(),
                    ),
                };

            return Ok(Self {
                root,
                server_path,
                catalog_path,
                legacy_config_path,
            });
        }

        // An explicit RUNE_HOME always wins over implicit ./config detection.
        let home_forced = std::env::var_os(HOME_ENV).is_some_and(|v| !v.is_empty());
        let local_server = cwd.join(DEFAULT_SERVER_REL);
        let local_catalog = cwd.join(DEFAULT_CATALOG_REL);
        let local_legacy = cwd.join(LEGACY_CONFIG_REL);

        if !home_forced && (local_server.exists() || local_legacy.exists()) {
            return Ok(Self {
                root: cwd,
                server_path: local_server,
                catalog_path: local_catalog,
                legacy_config_path: local_legacy,
            });
        }

        let root = rune_home();
        Ok(Self {
            server_path: root.join("config").join("server.json"),
            catalog_path: root.join("config").join("catalog.json"),
            legacy_config_path: root.join("config").join("models.json"),
            root,
        })
    }

    /// Create the root and make it the process working directory so the
    /// engines' relative `bin/`, `logs/` and `config/` paths resolve here.
    pub fn enter(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.root)?;
        std::env::set_current_dir(&self.root)
    }

    pub fn env_file(&self) -> PathBuf {
        self.root.join(".env")
    }

    pub fn config_exists(&self) -> bool {
        self.server_path.exists() && self.catalog_path.exists()
    }

    pub fn legacy_config_exists(&self) -> bool {
        self.legacy_config_path.exists()
    }
}

fn absolutize(cwd: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}
