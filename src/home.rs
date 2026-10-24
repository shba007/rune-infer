//! Resolution of the Rune Infer home directory (`~/.rune/infer`) and the
//! config file location.
//!
//! Layout inside the home directory (also used verbatim for a dev checkout,
//! which keeps these paths relative to the current directory):
//!
//! ```text
//! <home>/config/server.json   server settings
//! <home>/config/catalog.json  model catalog
//! <home>/.env                 secrets referenced from the config as {{VAR}}
//! <home>/bin/                 auto-installed engine binaries
//! <home>/logs/rune-infer.log  audit log
//! ```

use std::path::{Path, PathBuf};

pub const HOME_ENV: &str = "RUNE_HOME";
pub const DEFAULT_SERVER_REL: &str = "config/server.json";
pub const DEFAULT_CATALOG_REL: &str = "config/catalog.json";
pub const LEGACY_CONFIG_REL: &str = "config/models.json";

mod migration;
#[cfg(test)]
pub(crate) use migration::migrate_legacy_home;

/// `$RUNE_HOME`, else `~/.rune/infer`; never fall back to the current directory.
pub fn rune_home() -> std::io::Result<PathBuf> {
    if let Some(custom) = std::env::var_os(HOME_ENV).filter(|v| !v.is_empty()) {
        return Ok(absolutize(&std::env::current_dir()?, Path::new(&custom)));
    }
    Ok(user_home()?.join(".rune").join("infer"))
}

fn user_home() -> std::io::Result<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("HOME").filter(|v| !v.is_empty()))
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Cannot resolve the user home; set RUNE_HOME or use --config explicitly",
            )
        })
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
    /// * otherwise `~/.rune/infer`.
    pub fn resolve(explicit_config: Option<&str>) -> std::io::Result<Self> {
        let cwd = std::env::current_dir()?;
        let custom = std::env::var_os(HOME_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        // Home lookup is intentionally deferred so explicit paths work without it.
        Self::resolve_at(
            explicit_config,
            &cwd,
            custom.as_deref(),
            user_home().ok().as_deref(),
        )
    }

    fn resolve_at(
        explicit_config: Option<&str>,
        cwd: &Path,
        custom: Option<&Path>,
        user: Option<&Path>,
    ) -> std::io::Result<Self> {
        if let Some(raw) = explicit_config {
            let path = absolutize(cwd, Path::new(raw));
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

            let parent = path.parent().unwrap_or(cwd).to_path_buf();
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
        let home_forced = custom.is_some();
        let local_server = cwd.join(DEFAULT_SERVER_REL);
        let local_catalog = cwd.join(DEFAULT_CATALOG_REL);
        let local_legacy = cwd.join(LEGACY_CONFIG_REL);

        if !home_forced && (local_server.exists() || local_legacy.exists()) {
            return Ok(Self {
                root: cwd.to_path_buf(),
                server_path: local_server,
                catalog_path: local_catalog,
                legacy_config_path: local_legacy,
            });
        }

        let root = if let Some(custom) = custom {
            absolutize(cwd, custom)
        } else {
            let user = user.filter(|p| p.is_absolute()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Cannot resolve the user home; set RUNE_HOME or use --config explicitly",
                )
            })?;
            let root = user.join(".rune").join("infer");
            migration::migrate_defaults(user, &root)?;
            root
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "infer home spaces {}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn legacy(&self) -> PathBuf {
            let root = self.0.join(".rune-infer");
            std::fs::create_dir_all(root.join("config/custom")).unwrap();
            std::fs::write(root.join("config/server.json"), r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":18083,"max_loaded_models":1,"idle_unload_seconds":300,"api_key":"{{INFER_MIGRATION_TEST_KEY}}"}}"#).unwrap();
            std::fs::write(root.join("config/catalog.json"), r#"{"schema_version":3,"models":[{"id":"fixture","name":"Fixture","modality":"Text","engine":{"supported":["encoder"]},"local":{"path":"{{INFER_MIGRATION_MODEL_DIR}}/model.gguf"}}]}"#).unwrap();
            std::fs::write(root.join(".env"), "INFER_MIGRATION_TEST_KEY=private-fixture-value\nINFER_MIGRATION_MODEL_DIR=models/../models\n").unwrap();
            std::fs::write(root.join("config/custom/settings.txt"), "customization").unwrap();
            for dir in ["models", "bin/engine", "logs", "unrelated-project"] {
                std::fs::create_dir_all(root.join(dir)).unwrap();
            }
            std::fs::write(root.join("models/model.gguf"), "weights stay here").unwrap();
            std::fs::write(root.join("bin/engine/server"), "engine").unwrap();
            std::fs::write(root.join("logs/log.txt"), "history").unwrap();
            std::fs::write(root.join("logs/.active_pids"), "do not copy").unwrap();
            root
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn fresh_default_and_overrides_do_not_need_cwd_or_user_home_fallbacks() {
        let f = Fixture::new();
        let cwd = f.0.join("unrelated");
        std::fs::create_dir(&cwd).unwrap();
        let ws = Workspace::resolve_at(None, &cwd, None, Some(&f.0)).unwrap();
        assert_eq!(ws.root, f.0.join(".rune/infer"));
        assert_eq!(ws.server_path, ws.root.join("config/server.json"));
        let plan = crate::setup::SetupPlan {
            exposure: crate::setup::Exposure::Local,
            port: 18084,
            server_api_key: None,
            cloud: vec![],
            local: vec![],
            gpu_available: false,
        };
        crate::setup::write_setup(&plan, &ws.server_path, &ws.catalog_path, &ws.env_file())
            .unwrap();
        assert!(ws.config_exists());
        assert_eq!(
            crate::config::ModelRegistry::from_files(&ws.server_path, &ws.catalog_path)
                .unwrap()
                .server
                .port,
            18084
        );
        assert!(Workspace::resolve_at(None, &cwd, None, None).is_err());
        let custom = f.0.join("custom infer");
        let ws = Workspace::resolve_at(None, &cwd, Some(&custom), None).unwrap();
        assert_eq!(ws.root, custom);
        std::fs::create_dir_all(cwd.join("config")).unwrap();
        std::fs::write(cwd.join("config/server.json"), "{}").unwrap();
        assert_eq!(
            Workspace::resolve_at(None, &cwd, None, None).unwrap().root,
            cwd
        );
        assert_eq!(
            Workspace::resolve_at(None, &cwd, Some(&custom), None)
                .unwrap()
                .root,
            custom
        );
        std::fs::create_dir_all(f.0.join("explicit/config")).unwrap();
        let path = f.0.join("explicit/config/server.json");
        assert_eq!(
            Workspace::resolve_at(Some(path.to_str().unwrap()), &cwd, Some(&custom), None)
                .unwrap()
                .root,
            f.0.join("explicit")
        );
    }

    #[test]
    fn migration_preserves_settings_secrets_weights_and_customizations_and_repeats_safely() {
        let f = Fixture::new();
        let old = f.legacy();
        let new = f.0.join(".rune/infer");
        std::fs::create_dir_all(f.0.join(".rune/tools/plugins")).unwrap();
        std::fs::write(f.0.join(".rune/rune-kit.exe"), "kit").unwrap();
        assert!(migrate_legacy_home(&old, &new).unwrap());
        assert_eq!(
            std::fs::read(old.join(".env")).unwrap(),
            std::fs::read(new.join(".env")).unwrap()
        );
        let vars = vec![
            (
                "INFER_MIGRATION_TEST_KEY".into(),
                "private-fixture-value".into(),
            ),
            (
                "INFER_MIGRATION_MODEL_DIR".into(),
                "models/../models".into(),
            ),
        ];
        let registry = crate::config::ModelRegistry::from_files_with_env(
            new.join("config/server.json"),
            new.join("config/catalog.json"),
            &vars,
        )
        .unwrap();
        assert_eq!(registry.server.port, 18083);
        assert_eq!(
            Path::new(&registry.models[0].model_path)
                .canonicalize()
                .unwrap(),
            old.canonicalize().unwrap().join("models/model.gguf")
        );
        assert!(Path::new(&registry.models[0].model_path).exists());
        assert!(new.join("bin/engine/server").exists());
        assert!(new.join("logs/log.txt").exists());
        assert!(!new.join("logs/.active_pids").exists());
        assert!(new.join("config/custom/settings.txt").exists());
        assert!(!new.join("models").exists());
        assert!(!new.join("unrelated-project").exists());
        assert_eq!(
            std::fs::read(f.0.join(".rune/rune-kit.exe")).unwrap(),
            b"kit"
        );
        std::fs::write(new.join("config/custom/settings.txt"), "new customization").unwrap();
        assert!(!migrate_legacy_home(&old, &new).unwrap());
        assert_eq!(
            std::fs::read(new.join("config/custom/settings.txt")).unwrap(),
            b"new customization"
        );
        std::fs::write(old.join("config/custom/changed.txt"), "late edit").unwrap();
        assert!(migrate_legacy_home(&old, &new).is_err());
        assert!(!new.join("config/custom/changed.txt").exists());
    }

    #[test]
    fn conflicts_and_invalid_config_leave_originals_and_destination_untouched() {
        let f = Fixture::new();
        let old = f.legacy();
        let new = f.0.join(".rune/infer");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(new.join(".env"), "destination secret").unwrap();
        assert!(migrate_legacy_home(&old, &new).is_err());
        assert_eq!(
            std::fs::read(new.join(".env")).unwrap(),
            b"destination secret"
        );
        let other = f.0.join("other/infer");
        std::fs::write(old.join("config/catalog.json"), "private-broken-json").unwrap();
        let err = migrate_legacy_home(&old, &other).unwrap_err().to_string();
        assert!(!err.contains("private-broken-json"));
        assert!(!other.exists());
        assert!(old.join(".env").exists());
    }

    #[test]
    fn unified_schema_two_is_converted_only_in_the_new_home() {
        let f = Fixture::new();
        let old = f.legacy();
        let new = f.0.join(".rune/infer");
        let server: serde_json::Value =
            serde_json::from_slice(&std::fs::read(old.join("config/server.json")).unwrap())
                .unwrap();
        let catalog: serde_json::Value =
            serde_json::from_slice(&std::fs::read(old.join("config/catalog.json")).unwrap())
                .unwrap();
        std::fs::write(old.join("config/models.json"),serde_json::to_vec(&serde_json::json!({"schema_version":2,"server":server["server"],"models":catalog["models"]})).unwrap()).unwrap();
        std::fs::remove_file(old.join("config/server.json")).unwrap();
        std::fs::remove_file(old.join("config/catalog.json")).unwrap();
        assert!(migrate_legacy_home(&old, &new).unwrap());
        assert!(old.join("config/models.json").exists());
        assert!(!old.join("config/server.json").exists());
        assert!(new.join("config/server.json").exists());
        assert!(!migrate_legacy_home(&old, &new).unwrap());
    }

    #[test]
    fn visible_home_migrates_to_hidden_home_and_preserves_both_products() {
        let f = Fixture::new();
        let old = f.legacy();
        let visible = f.0.join("rune/infer");
        assert!(migrate_legacy_home(&old, &visible).unwrap());
        std::fs::write(
            visible.join("config/custom/settings.txt"),
            "latest customization",
        )
        .unwrap();
        std::fs::create_dir_all(f.0.join(".rune/tools/plugins")).unwrap();
        std::fs::write(f.0.join(".rune/rune-kit.exe"), "kit binary").unwrap();
        std::fs::write(f.0.join(".rune/tools/plugins/keep"), "kit plugin").unwrap();
        let cwd = f.0.join("unrelated working directory");
        std::fs::create_dir(&cwd).unwrap();
        let ws = Workspace::resolve_at(None, &cwd, None, Some(&f.0)).unwrap();
        assert_eq!(ws.root, f.0.join(".rune/infer"));
        assert_eq!(
            std::fs::read(ws.root.join(".env")).unwrap(),
            std::fs::read(visible.join(".env")).unwrap()
        );
        assert_eq!(
            std::fs::read_to_string(ws.root.join("config/custom/settings.txt")).unwrap(),
            "latest customization"
        );
        assert!(old.join("models/model.gguf").exists());
        assert!(visible.join("bin/engine/server").exists());
        assert!(ws.root.join("bin/engine/server").exists());
        assert!(!ws.root.join("models").exists());
        assert_eq!(
            std::fs::read_to_string(f.0.join(".rune/rune-kit.exe")).unwrap(),
            "kit binary"
        );
        assert_eq!(
            std::fs::read_to_string(f.0.join(".rune/tools/plugins/keep")).unwrap(),
            "kit plugin"
        );
        assert_eq!(
            Workspace::resolve_at(None, &cwd, None, Some(&f.0))
                .unwrap()
                .root,
            ws.root
        );
        std::fs::write(visible.join("config/custom/new.txt"), "later edit").unwrap();
        assert!(Workspace::resolve_at(None, &cwd, None, Some(&f.0)).is_err());
    }

    #[test]
    fn visible_only_and_ambiguous_sources_never_overwrite_data() {
        let f = Fixture::new();
        let old = f.legacy();
        let visible = f.0.join("rune/infer");
        migrate_legacy_home(&old, &visible).unwrap();
        std::fs::remove_file(visible.join(".legacy-home-migration.json")).unwrap();
        let hidden = f.0.join(".rune/infer");
        assert!(migration::migrate_defaults(&f.0, &hidden).is_err());
        assert!(!hidden.exists());
        // Hide the older configuration without deleting it; the visible home
        // now independently qualifies for migration.
        std::fs::rename(old.join("config"), old.join("saved-config")).unwrap();
        assert!(migration::migrate_defaults(&f.0, &hidden).unwrap());
        assert!(!migration::migrate_defaults(&f.0, &hidden).unwrap());
        assert!(old.join("saved-config/server.json").exists());
    }
}
