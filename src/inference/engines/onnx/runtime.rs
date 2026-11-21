use super::verify_runtime;
use crate::inference::engines::download;
use std::{
    error::Error,
    path::{Path, PathBuf},
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RuntimeManifest<'a> {
    #[serde(borrow)]
    pub(super) runtime_library: (&'a str, &'a str),
    #[serde(borrow)]
    pub(super) providers: Vec<(&'a str, &'a str)>,
    ort_archive_dir: &'a str,
    cuda_directory: &'a str,
    #[serde(borrow)]
    dependencies: Vec<Dependency<'a>>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Dependency<'a> {
    id: &'a str,
    bin_dir: &'a str,
    license_path: &'a str,
    license_name: &'a str,
    #[serde(borrow)]
    files: Vec<(&'a str, &'a str)>,
}

#[derive(serde::Deserialize)]
struct BundledRegistry<'a> {
    #[serde(borrow)]
    extensions: Extensions<'a>,
}

#[derive(serde::Deserialize)]
struct Extensions<'a> {
    #[serde(borrow)]
    onnx: RuntimeManifest<'a>,
}

pub(super) fn manifest() -> &'static RuntimeManifest<'static> {
    static MANIFEST: std::sync::LazyLock<RuntimeManifest<'static>> =
        std::sync::LazyLock::new(|| {
            let registry: BundledRegistry<'_> =
                serde_json::from_str(crate::registry_document::ENGINES)
                    .expect("invalid bundled ONNX runtime metadata");
            registry.extensions.onnx
        });
    &MANIFEST
}

fn archive_checksum(id: &str) -> Result<&'static str, Box<dyn Error>> {
    super::super::registry::bundled_onnx()
        .files
        .get(id)
        .map(|file| file.sha256.as_str())
        .ok_or_else(|| format!("missing bundled ONNX component {id}").into())
}

fn checked_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(format!(
            "Runtime location is not a regular directory: {}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn check_files(directory: &Path, files: &[(&str, &str)]) -> Result<(), Box<dyn Error>> {
    checked_directory(directory)?;
    for (name, expected) in files {
        let path = directory.join(name);
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || download::file_sha256(&path)? != *expected
        {
            return Err(format!("ONNX runtime component is unverified: {}", path.display()).into());
        }
    }
    Ok(())
}

pub(super) fn managed_gpu_directory(runtime: &Path) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let directory = runtime
        .parent()
        .ok_or("ONNX Runtime has no parent directory")?
        .join(manifest().cuda_directory);
    if !directory.exists() {
        return Ok(None);
    }
    for dependency in &manifest().dependencies {
        check_files(&directory, &dependency.files)?;
    }
    Ok(Some(std::fs::canonicalize(directory)?))
}

pub(crate) fn runtime_version(runtime: &Path) -> Option<String> {
    verify_runtime(runtime, true).ok()?;
    let version = &super::super::registry::bundled_onnx().version;
    match managed_gpu_directory(runtime) {
        Ok(Some(_)) => Some(format!("ONNX Runtime {version} CPU + CUDA")),
        Ok(None) => Some(format!("ONNX Runtime {version} CPU")),
        Err(_) => None,
    }
}

fn unpack(
    client: &reqwest::blocking::Client,
    stage: &Path,
    name: &str,
    url: &str,
    hash: &str,
    size: u64,
) -> Result<PathBuf, Box<dyn Error>> {
    let archive = stage.join(format!("{name}.zip"));
    download::download_verified_sized(client, url, &archive, "[ONNX Runtime]", hash, size)?;
    let destination = stage.join(name);
    std::fs::create_dir(&destination)?;
    download::extract_archive(&archive, &destination, "[ONNX Runtime]")?;
    Ok(destination)
}

fn copy_checked(
    source: &Path,
    destination: &Path,
    files: &[(&str, &str)],
) -> Result<(), Box<dyn Error>> {
    check_files(source, files)?;
    for (name, _) in files {
        std::fs::copy(source.join(name), destination.join(name))?;
    }
    check_files(destination, files)
}

fn assemble_cuda(
    stage: &Path,
    mut obtain: impl FnMut(&Path, &str, &str) -> Result<PathBuf, Box<dyn Error>>,
) -> Result<PathBuf, Box<dyn Error>> {
    let dependencies = stage.join(manifest().cuda_directory);
    std::fs::create_dir(&dependencies)?;
    let licenses = dependencies.join("licenses");
    std::fs::create_dir(&licenses)?;
    for dependency in &manifest().dependencies {
        let unpacked = obtain(stage, dependency.id, archive_checksum(dependency.id)?)?;
        copy_checked(
            &unpacked.join(dependency.bin_dir),
            &dependencies,
            &dependency.files,
        )?;
        std::fs::copy(
            unpacked.join(dependency.license_path),
            licenses.join(dependency.license_name),
        )?;
    }
    Ok(dependencies)
}

pub(crate) fn ensure_runtime_installed() -> Result<PathBuf, Box<dyn Error>> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("ONNX Runtime installation currently supports Windows x64 only".into());
    }
    if let Some(path) = std::env::var_os("ONNX_RUNTIME_LIBRARY_PATH") {
        let path = std::fs::canonicalize(path)?;
        verify_runtime(&path, false)?;
        return Ok(path);
    }
    let client = download::create_download_client()?;
    let mut registry = None;
    install_at(download::engine_bin()?, |stage, name, hash| {
        if registry.is_none() {
            registry = Some(super::super::registry::onnx_files(&client)?);
        }
        let file = registry
            .as_ref()
            .and_then(|files| files.get(name))
            .ok_or("Missing ONNX registry component")?;
        if file.sha256 != hash {
            return Err("ONNX registry checksum mismatch".into());
        }
        unpack(&client, stage, name, &file.url, hash, file.size_bytes)
    })
}

fn install_at(
    bin: PathBuf,
    mut obtain: impl FnMut(&Path, &str, &str) -> Result<PathBuf, Box<dyn Error>>,
) -> Result<PathBuf, Box<dyn Error>> {
    std::fs::create_dir_all(&bin)?;
    let bin = std::fs::canonicalize(bin)?;
    let root = bin.join("onnxruntime");
    if root.exists() {
        checked_directory(&root)?;
    }
    let existing = root.join(manifest().runtime_library.0);
    if existing.exists() {
        verify_runtime(&existing, true)?;
    }
    let gpu = root.join(manifest().cuda_directory);
    if existing.is_file() && gpu.exists() {
        managed_gpu_directory(&existing)?;
        return Ok(std::fs::canonicalize(existing)?);
    }
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stage = bin.join(format!(".onnx-stage-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage)?;
    let result = (|| -> Result<PathBuf, Box<dyn Error>> {
        if !existing.is_file() {
            if root.exists() {
                return Err("Existing ONNX Runtime directory is incomplete; preserve it before installation".into());
            }
            let unpacked = obtain(&stage, "ort", archive_checksum("ort")?)?;
            let core = stage.join("onnxruntime");
            std::fs::create_dir(&core)?;
            let archive_root = unpacked.join(manifest().ort_archive_dir);
            let libraries = archive_root.join("lib");
            copy_checked(
                &libraries,
                &core,
                std::slice::from_ref(&manifest().runtime_library),
            )?;
            copy_checked(&libraries, &core, &manifest().providers)?;
            for name in ["LICENSE", "ThirdPartyNotices.txt"] {
                std::fs::copy(archive_root.join(name), core.join(name))?;
            }
            std::fs::rename(&core, &root)?;
        }
        verify_runtime(&existing, true)?;
        let dependencies = assemble_cuda(&stage, &mut obtain)?;
        if gpu.exists() {
            return Err(
                "ONNX CUDA destination appeared during installation; no files were overwritten"
                    .into(),
            );
        }
        std::fs::rename(&dependencies, &gpu)?;
        managed_gpu_directory(&existing)?;
        Ok(std::fs::canonicalize(&existing)?)
    })();
    if let Ok(resolved) = std::fs::canonicalize(&stage)
        && resolved.parent() == Some(bin.as_path())
    {
        let _ = std::fs::remove_dir_all(resolved);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_cuda_bundle_is_rejected_without_modifying_files() {
        let root = std::env::temp_dir().join(format!(
            "rune-onnx-bundle-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        let runtime = root.join("onnxruntime.dll");
        std::fs::write(&runtime, b"existing runtime").unwrap();
        assert!(managed_gpu_directory(&runtime).unwrap().is_none());
        let cuda = root.join(manifest().cuda_directory);
        std::fs::create_dir(&cuda).unwrap();
        let marker = cuda.join("cudnn64_9.dll");
        std::fs::write(&marker, b"unverified").unwrap();
        assert!(managed_gpu_directory(&runtime).is_err());
        assert_eq!(std::fs::read(&marker).unwrap(), b"unverified");
        assert_eq!(std::fs::read(&runtime).unwrap(), b"existing runtime");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn copy_checked_rejects_changed_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "rune-onnx-copy-{}",
            crate::setup::generate_api_key()
        ));
        let source = root.join("source");
        let destination = root.join("destination");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(source.join("test.dll"), b"reviewed").unwrap();
        let hash = download::file_sha256(&source.join("test.dll")).unwrap();
        let files = [("test.dll", hash.as_str())];
        copy_checked(&source, &destination, &files).unwrap();
        std::fs::write(source.join("test.dll"), b"changed").unwrap();
        assert!(copy_checked(&source, &destination, &files).is_err());
        assert_eq!(
            std::fs::read(destination.join("test.dll")).unwrap(),
            b"reviewed"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "uses locally downloaded, checksum-pinned ONNX and NVIDIA archives"]
    fn installs_verified_bundle_from_local_archives() {
        let archives = PathBuf::from(
            std::env::var_os("RUNE_ONNX_TEST_ARCHIVES").expect("set RUNE_ONNX_TEST_ARCHIVES"),
        );
        let destination = PathBuf::from(
            std::env::var_os("RUNE_ONNX_TEST_DESTINATION").expect("set RUNE_ONNX_TEST_DESTINATION"),
        );
        std::fs::create_dir(&destination).unwrap();
        let path = install_at(destination, |stage, name, expected| {
            let filename = match name {
                "ort" => "onnxruntime-win-x64-gpu_cuda12-1.31.1.zip",
                "cudnn" => "cudnn-9.27.0.42.zip",
                "cublas" => "cublas.zip",
                "cudart" => "cudart.zip",
                _ => return Err("unknown test archive".into()),
            };
            let archive = archives.join(filename);
            if download::file_sha256(&archive)? != expected {
                return Err("test archive checksum mismatch".into());
            }
            let unpacked = stage.join(name);
            std::fs::create_dir(&unpacked)?;
            download::extract_archive(&archive, &unpacked, "[ONNX test]")?;
            Ok(unpacked)
        })
        .unwrap();
        assert_eq!(
            runtime_version(&path).as_deref(),
            Some("ONNX Runtime 1.31.1 CPU + CUDA")
        );
        assert!(
            path.parent()
                .unwrap()
                .join("cuda-12/licenses/cudnn.txt")
                .is_file()
        );
        println!("Verified fixture runtime: {}", path.display());
    }
}
