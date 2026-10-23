//! Copy the legacy installation without changing the original or moving weights.
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MARKER: &str = ".legacy-home-migration.json";
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize)]
struct Receipt {
    source: PathBuf,
    files: BTreeMap<String, String>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::other(format!(
        "{message}. Originals are unchanged; resolve the conflict or select the legacy home explicitly with RUNE_HOME"
    ))
}

fn plain_metadata(path: &Path) -> io::Result<std::fs::Metadata> {
    let meta = std::fs::symlink_metadata(path)?;
    // Do not follow links/junctions into unrelated projects or model directories.
    #[cfg(windows)]
    let link = {
        use std::os::windows::fs::MetadataExt;
        meta.file_attributes() & 0x400 != 0
    };
    #[cfg(not(windows))]
    let link = meta.file_type().is_symlink();
    if link {
        return Err(invalid(
            "Legacy migration does not follow links or junctions",
        ));
    }
    Ok(meta)
}

fn copy_tree(source: &Path, target: &Path) -> io::Result<()> {
    if plain_metadata(source)?.is_dir() {
        std::fs::create_dir(target)?;
        for entry in std::fs::read_dir(source)? {
            let entry = entry?;
            if entry.file_name() == ".active_pids" {
                continue;
            }
            copy_tree(&entry.path(), &target.join(entry.file_name()))?;
        }
    } else {
        std::fs::copy(source, target)?;
    }
    Ok(())
}

fn fingerprint(source: &Path) -> io::Result<BTreeMap<String, String>> {
    fn visit(root: &Path, path: &Path, files: &mut BTreeMap<String, String>) -> io::Result<()> {
        if plain_metadata(path)?.is_dir() {
            for entry in std::fs::read_dir(path)? {
                visit(root, &entry?.path(), files)?;
            }
        } else {
            files.insert(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                hex::encode(Sha256::digest(std::fs::read(path)?)),
            );
        }
        Ok(())
    }
    let mut files = BTreeMap::new();
    for name in ["config", ".env"] {
        if source.join(name).try_exists()? {
            visit(source, &source.join(name), &mut files)?;
        }
    }
    Ok(files)
}

fn environment(root: &Path) -> io::Result<Vec<(String, String)>> {
    let env = root.join(".env");
    if !env.try_exists()? {
        return Ok(vec![]);
    }
    let values = dotenvy::from_path_iter(env)
        .map_err(|_| invalid("Cannot read legacy environment file"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("Invalid legacy environment file"))?;
    // Match dotenv startup behavior: the existing process environment wins.
    Ok(values
        .into_iter()
        .filter(|(key, _)| std::env::var_os(key).is_none())
        .collect())
}

fn validate(root: &Path) -> io::Result<()> {
    crate::config::ModelRegistry::from_files_with_env(
        root.join("config/server.json"),
        root.join("config/catalog.json"),
        &environment(root)?,
    )
    .map(|_| ())
    // Parsing/validation errors may contain user values; never echo them.
    .map_err(|_| {
        invalid("Migrated configuration failed validation (check schema and environment variables)")
    })
}

fn rebase(
    value: &mut serde_json::Value,
    source: &Path,
    env: &[(String, String)],
) -> io::Result<()> {
    match value {
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                if (key == "path" || key.ends_with("_path")) && value.is_string() {
                    let raw = value.as_str().unwrap();
                    let quoted = serde_json::to_string(raw)?;
                    let expanded = crate::config::interpolate_with(&quoted, &|name| {
                        env.iter()
                            .find(|(key, _)| key == name)
                            .map(|(_, value)| value.clone())
                            .or_else(|| std::env::var(name).ok())
                    })
                    .map_err(|_| invalid("Cannot resolve a legacy path template"))?;
                    let resolved: String = serde_json::from_str(&expanded)?;
                    if !resolved.is_empty() && Path::new(&resolved).is_relative() {
                        // Preserve templates and .env bytes; retain references to the old
                        // root instead of copying weights or arbitrary user files.
                        let root = source.to_string_lossy();
                        // Win32 verbatim paths disable slash/.. normalization. Templates
                        // can expand to relative paths containing either, so do not
                        // persist canonicalize()'s verbatim prefix in model references.
                        #[cfg(windows)]
                        let root = if let Some(unc) = root.strip_prefix(r"\\?\UNC\") {
                            format!(r"\\{unc}")
                        } else {
                            root.strip_prefix(r"\\?\").unwrap_or(&root).to_string()
                        };
                        *value = serde_json::Value::String(
                            Path::new(&root[..])
                                .join(raw)
                                .to_string_lossy()
                                .into_owned(),
                        );
                    }
                } else {
                    rebase(value, source, env)?;
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                rebase(value, source, env)?;
            }
        }
        _ => (),
    }
    Ok(())
}

/// Automatically called only for the implicit default home. Explicit overrides
/// and development checkouts bypass it. A receipt makes repeat runs non-destructive.
pub(crate) fn migrate_legacy_home(source: &Path, destination: &Path) -> io::Result<bool> {
    if !source.try_exists()? {
        return Ok(false);
    }
    plain_metadata(source)?;
    let config = source.join("config");
    if !["server.json", "catalog.json", "models.json"]
        .iter()
        .any(|name| config.join(name).exists())
    {
        return Ok(false);
    }
    let receipt = Receipt {
        source: source.canonicalize()?,
        files: fingerprint(source)?,
    };
    if destination.try_exists()? {
        plain_metadata(destination)?;
        let prior: Receipt = std::fs::read(destination.join(MARKER))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .ok_or_else(|| {
                invalid("Both legacy and destination homes exist; destination was not overwritten")
            })?;
        if prior.source != receipt.source || prior.files != receipt.files {
            return Err(invalid(
                "Legacy configuration changed since migration; destination was not overwritten",
            ));
        }
        validate(destination)?;
        return Ok(false);
    }
    let parent = destination
        .parent()
        .ok_or_else(|| invalid("Invalid destination home"))?;
    std::fs::create_dir_all(parent)?;
    let stage = parent.join(format!(
        ".infer-migration-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&stage)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stage, std::fs::Permissions::from_mode(0o700))?;
    }
    let result = (|| {
        for name in ["config", ".env", "bin", "logs"] {
            if source.join(name).try_exists()? {
                copy_tree(&source.join(name), &stage.join(name))?;
            }
        }
        if !stage.join("config/server.json").exists() && !stage.join("config/catalog.json").exists()
        {
            crate::migrate::run(&stage.join("config/models.json"), false)
                .map_err(|_| invalid("Legacy schema conversion failed"))?;
        }
        let env = environment(&stage)?;
        let catalog = stage.join("config/catalog.json");
        let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(&catalog)?)
            .map_err(|_| invalid("Invalid legacy catalog JSON"))?;
        rebase(&mut value, &receipt.source, &env)?;
        std::fs::write(&catalog, serde_json::to_string_pretty(&value)? + "\n")?;
        validate(&stage)?;
        if fingerprint(source)? != receipt.files {
            return Err(invalid(
                "Legacy files changed during migration; retry after stopping the old server",
            ));
        }
        std::fs::write(stage.join(MARKER), serde_json::to_vec_pretty(&receipt)?)?;
        if destination.try_exists()? {
            return Err(invalid(
                "Destination appeared during migration; it was not overwritten",
            ));
        }
        std::fs::rename(&stage, destination)?;
        Ok(true)
    })();
    if stage.exists() {
        std::fs::remove_dir_all(&stage)?;
    }
    result
}
