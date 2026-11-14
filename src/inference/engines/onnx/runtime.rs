//! Reviewed Windows ONNX Runtime and CUDA dependency bundle.
use super::{PROVIDER_CUDA_SHA, PROVIDER_SHARED_SHA, RUNTIME_SHA, verify_runtime};
use crate::inference::engines::download;
use std::{
    error::Error,
    path::{Path, PathBuf},
};

const ORT_ARCHIVE_SHA: &str = "5cbf65edc93dc7bb3296249b1b6e1b2228e575902af64d6163d0da2103cef9d5";
const CUDNN_ARCHIVE_SHA: &str = "06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142";
const CUBLAS_ARCHIVE_SHA: &str = "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661";
const CUDART_ARCHIVE_SHA: &str = "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891";

const ORT_FILES: [(&str, &str); 3] = [
    ("onnxruntime.dll", RUNTIME_SHA),
    ("onnxruntime_providers_shared.dll", PROVIDER_SHARED_SHA),
    ("onnxruntime_providers_cuda.dll", PROVIDER_CUDA_SHA),
];
const CUBLAS_FILES: [(&str, &str); 2] = [
    (
        "cublas64_12.dll",
        "52ce5ba0ec5327d39be6021f0f9362d2ed4d64d7206f9b5afc06398d827b8e82",
    ),
    (
        "cublasLt64_12.dll",
        "71705a7cf0923f4ab034d0d2620bbfc989ff669ac3d79f868b7a7c597ca559c5",
    ),
];
const CUDART_FILES: [(&str, &str); 1] = [(
    "cudart64_12.dll",
    "760c38928bbe5759f7b31ed6692599eb7ec83cedd5702e84c2b72028a89837e1",
)];
const CUDNN_FILES: [(&str, &str); 10] = [
    (
        "cudnn_adv64_9.dll",
        "712ace96559e60b5be81a1097acb7609f601ff4cd0cdef848b013d671753b4b4",
    ),
    (
        "cudnn_cnn64_9.dll",
        "dff4c446c855b3ab2d486aebf4c11388cf74da14d7bd0bfafb61c3a75481313d",
    ),
    (
        "cudnn_engines_precompiled64_9.dll",
        "6b603ac45fc018a96f014d0c6cdbc8f480dce2dbe33b8ca38228a4e4c531f3fe",
    ),
    (
        "cudnn_engines_runtime_compiled64_9.dll",
        "f92f4776261e4409aa9f84357c40bc3626700292a1d22025746388bea43ae11b",
    ),
    (
        "cudnn_engines_tensor_ir64_9.dll",
        "d7380a7c4ad90590968e97b4d8b43a616e7e0070bfdeeee04c4d0be85ad52394",
    ),
    (
        "cudnn_ext64_9.dll",
        "f20888488b701e429830e19cf59022e90a0c5dd398723162ea9f320b3ca06636",
    ),
    (
        "cudnn_graph64_9.dll",
        "445e8c3e44a6a6035749380009ae35ae8c2bc246428229b79545a25238bb56e2",
    ),
    (
        "cudnn_heuristic64_9.dll",
        "0634af738266cd36c912e99cd3807ee452450e963bf885be204fdcfb322cfc70",
    ),
    (
        "cudnn_ops64_9.dll",
        "390bfe103a244c714fca5a1320950ba51a3173a7fbd6664ed535ce7367891ccd",
    ),
    (
        "cudnn64_9.dll",
        "7268ef53e32c2c574372326dfc9d94a26095f4ed9c89dba87cfb3ab1e40a6470",
    ),
];

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
        .join("cuda-12");
    if !directory.exists() {
        return Ok(None);
    }
    check_files(&directory, &CUDNN_FILES)?;
    check_files(&directory, &CUBLAS_FILES)?;
    check_files(&directory, &CUDART_FILES)?;
    Ok(Some(std::fs::canonicalize(directory)?))
}

pub(crate) fn runtime_version(runtime: &Path) -> Option<String> {
    verify_runtime(runtime, true).ok()?;
    match managed_gpu_directory(runtime) {
        Ok(Some(_)) => Some("ONNX Runtime 1.31.1 CPU + CUDA".into()),
        Ok(None) => Some("ONNX Runtime 1.31.1 CPU".into()),
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
    let dependencies = stage.join("cuda-12");
    std::fs::create_dir(&dependencies)?;
    let licenses = dependencies.join("licenses");
    std::fs::create_dir(&licenses)?;
    let cudnn = obtain(stage, "cudnn", CUDNN_ARCHIVE_SHA)?;
    copy_checked(&cudnn.join("nvidia/cudnn/bin"), &dependencies, &CUDNN_FILES)?;
    std::fs::copy(
        cudnn.join("nvidia_cudnn_cu12-9.27.0.42.dist-info/licenses/License.txt"),
        licenses.join("cudnn.txt"),
    )?;
    let cublas = obtain(stage, "cublas", CUBLAS_ARCHIVE_SHA)?;
    copy_checked(
        &cublas.join("nvidia/cublas/bin"),
        &dependencies,
        &CUBLAS_FILES,
    )?;
    std::fs::copy(
        cublas.join("nvidia_cublas_cu12-12.9.2.10.dist-info/licenses/License.txt"),
        licenses.join("cublas.txt"),
    )?;
    let cudart = obtain(stage, "cudart", CUDART_ARCHIVE_SHA)?;
    copy_checked(
        &cudart.join("nvidia/cuda_runtime/bin"),
        &dependencies,
        &CUDART_FILES,
    )?;
    std::fs::copy(
        cudart.join("nvidia_cuda_runtime_cu12-12.9.79.dist-info/licenses/License.txt"),
        licenses.join("cuda-runtime.txt"),
    )?;
    check_files(&dependencies, &CUDNN_FILES)?;
    check_files(&dependencies, &CUBLAS_FILES)?;
    check_files(&dependencies, &CUDART_FILES)?;
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
    let existing = root.join("onnxruntime.dll");
    if existing.exists() {
        verify_runtime(&existing, true)?;
    }
    let gpu = root.join("cuda-12");
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
            let unpacked = obtain(&stage, "ort", ORT_ARCHIVE_SHA)?;
            let core = stage.join("onnxruntime");
            std::fs::create_dir(&core)?;
            copy_checked(
                &unpacked.join("onnxruntime-win-x64-gpu_cuda12-1.31.1/lib"),
                &core,
                &ORT_FILES,
            )?;
            for name in ["LICENSE", "ThirdPartyNotices.txt"] {
                std::fs::copy(
                    unpacked
                        .join("onnxruntime-win-x64-gpu_cuda12-1.31.1")
                        .join(name),
                    core.join(name),
                )?;
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
        let cuda = root.join("cuda-12");
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
