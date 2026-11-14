//! Local model discovery and safe staging for the Infer-owned download job.
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Found {
    id: String,
    name: String,
    bytes: u64,
    path: String,
}

type Listed = (String, u64, PathBuf);
fn find_files(dir: &Path, cancel: Option<&AtomicBool>) -> Vec<Listed> {
    let mut found = Vec::new();
    let mut pending = vec![(dir.to_path_buf(), 0u8)];
    let mut visited = 0usize;
    while let Some((folder, depth)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&folder) else {
            continue;
        };
        for entry in entries.flatten() {
            if cancel.is_some_and(|token| token.load(Ordering::Relaxed)) {
                return found;
            }
            visited += 1;
            if visited > 50_000 {
                return found;
            }
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() {
                if depth < 5 {
                    pending.push((entry.path(), depth + 1));
                }
            } else if kind.is_file()
                && let Ok(meta) = entry.metadata()
            {
                found.push((
                    entry.file_name().to_string_lossy().to_ascii_lowercase(),
                    meta.len(),
                    entry.path(),
                ));
            }
        }
    }
    found
}

fn file_hash(path: &Path, cancel: &AtomicBool) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err("Import cancelled".into());
        }
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

type Matched = Vec<(crate::model_registry::File, PathBuf)>;
fn matches_in(model: &crate::model_registry::LocalModel, files: &[Listed]) -> Option<Matched> {
    model
        .versions
        .get(&model.latest)?
        .files
        .iter()
        .map(|file| {
            files
                .iter()
                .find(|(name, size, _)| {
                    *name == file.filename.to_ascii_lowercase() && *size == file.size_bytes
                })
                .map(|(_, _, path)| (file.clone(), path.clone()))
        })
        .collect()
}

fn validate_source(dir: &Path) -> Result<(), String> {
    if !dir.is_absolute() || !dir.is_dir() {
        return Err("Choose an existing folder on the Infer server".into());
    }
    Ok(())
}

fn existing_regular(path: &Path) -> Result<Option<u64>, String> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => Ok(Some(meta.len())),
        Ok(_) => Err("Model destination is not a regular file; it was preserved".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

pub fn scan(dir: &Path) -> Result<Vec<Found>, String> {
    validate_source(dir)?;
    let files = find_files(dir, None);
    let mut found = Vec::new();
    for model in crate::model_registry::bundled()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|m| m.engine == "llama-server")
    {
        if let Some(matched) = matches_in(&model, &files)
            && let Some((_, path)) = matched.iter().find(|(file, _)| file.id == "model")
        {
            let bytes = matched
                .iter()
                .try_fold(0u64, |sum, (file, _)| sum.checked_add(file.size_bytes))
                .ok_or("Model size is invalid")?;
            found.push(Found {
                id: model.id,
                name: model.name,
                bytes,
                path: path.display().to_string(),
            });
        }
    }
    Ok(found)
}

pub(super) fn stage(
    root: &Path,
    model: &crate::model_registry::LocalModel,
    dir: &Path,
    cancel: &AtomicBool,
) -> Result<(), String> {
    validate_source(dir)?;
    let matched = matches_in(model, &find_files(dir, Some(cancel)))
        .ok_or("This folder no longer contains the complete model files")?;
    let main = matched
        .iter()
        .find(|(file, _)| file.id == "model")
        .ok_or("Model has no main file")?;
    let relative = model
        .catalog
        .as_ref()
        .and_then(|catalog| catalog.local.path.clone())
        .or_else(|| {
            model
                .install_dir
                .as_ref()
                .map(|install| format!("{install}/{}", main.0.filename))
        })
        .ok_or("Model has no install location")?;
    let parts = crate::model_registry::validate_install_path(&relative, &main.0.filename)
        .map_err(|e| e.to_string())?;
    let mut target = root.to_path_buf();
    crate::model_registry::regular_dir(&target).map_err(|e| e.to_string())?;
    target.push("models");
    crate::model_registry::regular_dir(&target).map_err(|e| e.to_string())?;
    for part in &parts[..parts.len() - 1] {
        target.push(part);
        crate::model_registry::regular_dir(&target).map_err(|e| e.to_string())?;
    }
    let _lock = crate::model_registry::model_install_lock(&target).map_err(|e| e.to_string())?;
    for (file, source) in matched {
        if cancel.load(Ordering::Relaxed) {
            return Err("Import cancelled".into());
        }
        if existing_regular(&target.join(&file.filename))?.is_some() {
            continue;
        }
        let staged = target.join(format!(
            ".rune-model-download-{}-{}.part",
            file.id,
            file.sha256.to_ascii_lowercase()
        ));
        if let Some(existing) = existing_regular(&staged)? {
            if existing == file.size_bytes
                && file_hash(&staged, cancel)?.eq_ignore_ascii_case(&file.sha256)
            {
                continue;
            }
            return Err(
                "Existing staged model file conflicts with this import; it was preserved".into(),
            );
        }
        let temp = target.join(format!(
            ".rune-model-import-{}",
            crate::setup::generate_api_key()
        ));
        let copy_result = (|| -> Result<(), String> {
            let mut input = std::fs::File::open(&source).map_err(|e| e.to_string())?;
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
                .map_err(|e| e.to_string())?;
            let mut buffer = vec![0u8; 1024 * 1024];
            let mut digest = Sha256::new();
            loop {
                if cancel.load(Ordering::Relaxed) {
                    return Err("Import cancelled".into());
                }
                let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                output
                    .write_all(&buffer[..count])
                    .map_err(|e| e.to_string())?;
            }
            output.sync_all().map_err(|e| e.to_string())?;
            drop(output);
            let meta = std::fs::metadata(&temp).map_err(|e| e.to_string())?;
            if meta.len() != file.size_bytes
                || !hex::encode(digest.finalize()).eq_ignore_ascii_case(&file.sha256)
            {
                return Err(format!(
                    "{} does not match the reviewed model file",
                    file.filename
                ));
            }
            crate::model_registry::move_no_replace(&temp, &staged).map_err(|error| {
                format!(
                    "Cannot publish staged model file without overwriting existing data: {error}"
                )
            })?;
            Ok(())
        })();
        if temp.exists() {
            let _ = std::fs::remove_file(&temp);
        }
        copy_result?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PathBuf, crate::model_registry::LocalModel) {
        let root =
            std::env::temp_dir().join(format!("infer-import-{}", crate::setup::generate_api_key()));
        std::fs::create_dir_all(root.join("source")).unwrap();
        let source = root.join("source/model.gguf");
        std::fs::write(&source, b"GGUFxxx").unwrap();
        let hash = crate::inference::engines::download::file_sha256(&source).unwrap();
        let model = serde_json::from_value(serde_json::json!({
            "id":"fixture", "install_dir":"author/family", "name":"Fixture", "description":"test", "license":"test", "source_url":"https://example.com", "engine":"llama-server", "format":"gguf", "capabilities":["Chat"], "latest":"v1",
            "versions":{"v1":{"release_number":1,"files":[{"id":"model","filename":"model.gguf","url":"https://example.com/model.gguf","sha256":hash,"size_bytes":7}]}}
        })).unwrap();
        (root, model)
    }

    #[test]
    fn import_copies_reviewed_file_without_aliasing_source() {
        let (root, model) = fixture();
        let source = root.join("source");
        let target = root.join("home");
        let cancel = AtomicBool::new(false);
        stage(&target, &model, &source, &cancel).unwrap();
        let hash = &model.versions["v1"].files[0].sha256;
        let staged = target
            .join("models/author/family")
            .join(format!(".rune-model-download-model-{hash}.part"));
        std::fs::write(source.join("model.gguf"), b"changed").unwrap();
        assert_eq!(std::fs::read(&staged).unwrap(), b"GGUFxxx");
        std::fs::write(source.join("model.gguf"), b"GGUFxxx").unwrap();
        stage(&target, &model, &source, &cancel).unwrap();
        assert_eq!(std::fs::read(&staged).unwrap(), b"GGUFxxx");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn import_respects_model_installation_lock() {
        let (root, model) = fixture();
        let source = root.join("source");
        let target = root.join("home");
        let folder = target.join("models/author/family");
        std::fs::create_dir_all(&folder).unwrap();
        let lock = crate::model_registry::model_install_lock(&folder).unwrap();
        assert!(stage(&target, &model, &source, &AtomicBool::new(false)).is_err());
        drop(lock);
        stage(&target, &model, &source, &AtomicBool::new(false)).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn import_preserves_conflicting_staged_files_and_honors_cancellation() {
        let (root, model) = fixture();
        let source = root.join("source");
        let target = root.join("home");
        let folder = target.join("models/author/family");
        std::fs::create_dir_all(&folder).unwrap();
        let hash = &model.versions["v1"].files[0].sha256;
        let staged = folder.join(format!(".rune-model-download-model-{hash}.part"));
        std::fs::write(&staged, b"conflicting contents").unwrap();
        assert!(stage(&target, &model, &source, &AtomicBool::new(false)).is_err());
        assert_eq!(std::fs::read(&staged).unwrap(), b"conflicting contents");
        assert!(stage(&target, &model, &source, &AtomicBool::new(true)).is_err());
        assert_eq!(std::fs::read(&staged).unwrap(), b"conflicting contents");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn import_job_registers_the_verified_model_in_the_selected_server_catalog() {
        let (root, model) = fixture();
        let home = root.join("home");
        let config_home = root.join("configured-server");
        crate::embedding::prepare_config(&config_home).unwrap();
        let bin = home.join("bin/llama-upstream");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            bin.join(if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            }),
            b"fixture",
        )
        .unwrap();
        let server = config_home.join("config/server.json");
        let catalog = config_home.join("config/catalog.json");
        let downloads = super::super::Downloads::default();
        downloads
            .start_selected(
                home.clone(),
                server,
                catalog.clone(),
                model,
                Some(root.join("source")),
            )
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let status = downloads.status();
            if status.phase == "complete" {
                break;
            }
            assert_ne!(status.phase, "failed", "{:?}", status.error);
            assert!(std::time::Instant::now() < deadline, "Import timed out");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            std::fs::read(home.join("models/author/family/model.gguf")).unwrap(),
            b"GGUFxxx"
        );
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(catalog).unwrap()).unwrap();
        assert!(
            data["models"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| entry["id"] == "fixture")
        );
        assert!(!home.join("config/catalog.json").exists());
        drop(downloads);
        std::fs::remove_dir_all(root).unwrap();
    }
}
