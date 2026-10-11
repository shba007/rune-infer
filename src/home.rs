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
pub const DEFAULT_CONFIG_REL: &str = "config/models.json";

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
    /// Absolute path of the config file (may not exist yet).
    pub config_path: PathBuf,
}

impl Workspace {
    /// Pick the workspace.
    ///
    /// * `--config <path>` given: honoured as-is; the root is the config's
    ///   grandparent when it is named `config/<file>`, otherwise its parent.
    /// * `RUNE_HOME` set: that folder.
    /// * `./config/models.json` exists in the current directory: dev checkout.
    /// * otherwise `~/.rune-infer`.
    pub fn resolve(explicit_config: Option<&str>) -> std::io::Result<Self> {
        let cwd = std::env::current_dir()?;

        if let Some(raw) = explicit_config {
            let config_path = absolutize(&cwd, Path::new(raw));
            let parent = config_path.parent().unwrap_or(&cwd).to_path_buf();
            let root = if parent.file_name().is_some_and(|n| n == "config") {
                parent.parent().unwrap_or(&parent).to_path_buf()
            } else {
                parent
            };
            return Ok(Self { root, config_path });
        }

        // An explicit RUNE_HOME always wins over implicit ./config detection.
        let home_forced = std::env::var_os(HOME_ENV).is_some_and(|v| !v.is_empty());
        let local = cwd.join(DEFAULT_CONFIG_REL);
        if !home_forced && local.exists() {
            return Ok(Self {
                root: cwd,
                config_path: local,
            });
        }

        let root = rune_home();
        Ok(Self {
            config_path: root.join("config").join("models.json"),
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
        self.config_path.exists()
    }
}

fn absolutize(cwd: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}
