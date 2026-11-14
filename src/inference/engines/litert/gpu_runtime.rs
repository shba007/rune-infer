//! Checksum-pinned WebGPU/DXC sidecars, published separately from the CPU DLL.
use super::super::download;
use std::{
    error::Error,
    path::{Path, PathBuf},
};

const FILES: [(&str, &str); 4] = [
    ("libLiteRt.dll", super::RUNTIME_SHA),
    (
        "libLiteRtWebGpuAccelerator.dll",
        "ade56eff38715f3e7b3ada0cfb736db4cccb6bed4668a891a37ebcf4eee38548",
    ),
    (
        "dxcompiler.dll",
        "5888d3e590f5bfd8743484e7f31313996734adec5741506271a5c6368cbe8ba9",
    ),
    (
        "dxil.dll",
        "0cf5fa4b90f74550166108b3870c883f4a85243976b246b1da3ea90ad5b92015",
    ),
];

pub(super) fn bundle_path() -> Result<PathBuf, String> {
    Ok(download::engine_bin()?.join("litert/gpu-2.3.0"))
}

pub(super) fn validate_bundle(directory: &Path) -> Result<(), Box<dyn Error>> {
    for (name, hash) in FILES {
        let path = directory.join(name);
        if download::file_sha256(&path)? != hash {
            return Err(
                format!("LiteRT GPU component checksum mismatch: {}", path.display()).into(),
            );
        }
    }
    Ok(())
}

pub(super) fn ensure_installed(cpu: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("LiteRT GPU currently requires Windows x64".into());
    }
    super::validate_runtime(cpu)?;
    let destination = bundle_path()?;
    if destination.exists() {
        validate_bundle(&destination)?;
        return Ok(std::fs::canonicalize(destination.join("libLiteRt.dll"))?);
    }
    let root = destination.parent().ok_or("Missing LiteRT directory")?;
    std::fs::create_dir_all(root)?;
    let root = std::fs::canonicalize(root)?;
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stage = root.join(format!(".gpu-stage-{}-{id}", std::process::id()));
    std::fs::create_dir(&stage)?;
    let result = (|| {
        let client = download::create_download_client()?;
        let files = super::super::registry::litert_files(&client)?;
        let gpu = files.get("gpu").ok_or("LiteRT GPU component missing")?;
        let dxc = files.get("dxc").ok_or("LiteRT DXC component missing")?;
        std::fs::copy(cpu, stage.join("libLiteRt.dll"))?;
        download::download_verified_sized(
            &client,
            &gpu.url,
            &stage.join(FILES[1].0),
            "[LiteRT GPU]",
            &gpu.sha256,
            gpu.size_bytes,
        )?;
        let archive = stage.join("dxc.zip");
        download::download_verified_sized(
            &client,
            &dxc.url,
            &archive,
            "[LiteRT DXC]",
            &dxc.sha256,
            dxc.size_bytes,
        )?;
        let extracted = stage.join("dxc");
        std::fs::create_dir(&extracted)?;
        download::extract_archive(&archive, &extracted, "[LiteRT DXC]")?;
        for name in ["dxcompiler.dll", "dxil.dll"] {
            std::fs::copy(extracted.join("bin/x64").join(name), stage.join(name))?;
        }
        validate_bundle(&stage)?;
        // Publish only the verified four-file bundle; preserve existing files.
        let publish = stage.join("bundle");
        std::fs::create_dir(&publish)?;
        for (name, _) in FILES {
            std::fs::rename(stage.join(name), publish.join(name))?;
        }
        if destination.exists() {
            validate_bundle(&destination)?;
        } else {
            std::fs::rename(&publish, &destination)?;
        }
        validate_bundle(&destination)?;
        Ok(std::fs::canonicalize(destination.join("libLiteRt.dll"))?)
    })();
    if let Ok(resolved) = std::fs::canonicalize(&stage)
        && resolved.parent() == Some(root.as_path())
    {
        let _ = std::fs::remove_dir_all(resolved);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incomplete_or_modified_gpu_bundles_are_rejected() {
        let root = std::env::temp_dir().join(format!(
            "rune-litert-bundle-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        assert!(validate_bundle(&root).is_err());
        for (file, _) in FILES {
            std::fs::write(root.join(file), b"unverified").unwrap();
        }
        assert!(validate_bundle(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
