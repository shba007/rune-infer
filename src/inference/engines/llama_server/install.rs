use super::RuntimeFlavor;
use crate::inference::engines::download;
use std::error::Error;
use std::path::PathBuf;

pub fn ensure_binary_installed(flavor: RuntimeFlavor) -> Result<PathBuf, Box<dyn Error>> {
    let exe_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    if let Ok(custom) = std::env::var("LLAMA_SERVER_PATH") {
        let p = PathBuf::from(custom);
        if p.exists() {
            return Ok(p);
        }
    }

    let root_bin = PathBuf::from("bin").join(exe_name);
    if root_bin.exists() {
        return Ok(root_bin);
    }

    let folder_name = match flavor {
        RuntimeFlavor::Upstream => "llama-upstream",
        RuntimeFlavor::Prism => "llama-prism",
    };
    let base_dir = PathBuf::from("bin").join(folder_name);
    let exe_path = base_dir.join(exe_name);

    if exe_path.exists() {
        #[cfg(windows)]
        {
            let cuda_dll = base_dir.join("ggml-cuda.dll");
            if cuda_dll.exists() {
                let has_cudart = std::fs::read_dir(&base_dir).is_ok_and(|entries| {
                    entries.flatten().any(|e| {
                        let name = e.file_name().to_string_lossy().to_lowercase();
                        name.starts_with("cudart64_") && name.ends_with(".dll")
                    })
                });
                if has_cudart {
                    return Ok(exe_path);
                }
            } else {
                return Ok(exe_path);
            }
        }
        #[cfg(not(windows))]
        return Ok(exe_path);
    }

    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    println!(
        "[LlamaServerEngine] Runtime '{:?}' resolving dependencies from official releases...",
        flavor
    );
    std::fs::create_dir_all(&base_dir)?;

    let client = download::create_download_client()?;
    let release_url = match flavor {
        RuntimeFlavor::Upstream => {
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
        }
        RuntimeFlavor::Prism => {
            "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases?per_page=5"
        }
    };

    let mut req = client
        .get(release_url)
        .header("Accept", "application/vnd.github+json");

    if let Ok(token) = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")) {
        req = req.header("Authorization", format!("Bearer {}", token.trim()));
    }

    let release_data: serde_json::Value = match req.send().and_then(|r| r.error_for_status()) {
        Ok(resp) => resp.json()?,
        Err(_) => {
            let html_url = match flavor {
                RuntimeFlavor::Upstream => "https://github.com/ggml-org/llama.cpp/releases",
                RuntimeFlavor::Prism => "https://github.com/PrismML-Eng/llama.cpp/releases",
            };
            let resp = client.get(html_url).send()?;
            let tag = resp
                .url()
                .as_str()
                .split("/tag/")
                .nth(1)
                .ok_or("Could not determine tag from redirect")?;

            let mut direct_urls = Vec::new();
            let host_cuda = download::detect_host_cuda_version();

            if cfg!(target_os = "windows") {
                let cuda_ver = host_cuda
                    .map(|c| format!("{}.{}", c.0, c.1))
                    .unwrap_or_else(|| "12.4".to_string());
                let base_dl =
                    format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}");
                let bin_asset = format!("llama-{tag}-bin-win-cuda-{cuda_ver}-x64.zip");
                let cudart_asset = format!("cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip");
                direct_urls.push((bin_asset.clone(), format!("{base_dl}/{bin_asset}")));
                direct_urls.push((cudart_asset.clone(), format!("{base_dl}/{cudart_asset}")));
            } else if cfg!(target_os = "macos") {
                let arch = if cfg!(target_arch = "aarch64") {
                    "arm64"
                } else {
                    "x64"
                };
                let bin_asset = format!("llama-{tag}-bin-macos-{arch}.tar.gz");
                direct_urls.push((
                    bin_asset.clone(),
                    format!(
                        "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{bin_asset}"
                    ),
                ));
            } else {
                let bin_asset = format!("llama-{tag}-bin-ubuntu-x64.tar.gz");
                direct_urls.push((
                    bin_asset.clone(),
                    format!(
                        "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{bin_asset}"
                    ),
                ));
            }

            for (name, url) in direct_urls {
                download::download_and_extract(
                    &client,
                    &url,
                    &base_dir,
                    &name,
                    "[LlamaServerEngine]",
                )?;
            }
            download::flatten_dlls(&base_dir);
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
            download::make_executable(&exe_path)?;
            return Ok(exe_path);
        }
    };

    let target_release = if let Some(releases_arr) = release_data.as_array() {
        releases_arr
            .iter()
            .find(|rel| {
                rel["assets"].as_array().is_some_and(|assets| {
                    assets
                        .iter()
                        .any(|a| a["name"].as_str().unwrap_or("").contains("bin-"))
                })
            })
            .unwrap_or(&release_data[0])
    } else {
        &release_data
    };

    let assets = target_release["assets"]
        .as_array()
        .ok_or("Invalid release data: missing assets")?;

    let host_cuda = download::detect_host_cuda_version();
    let mut download_urls = Vec::new();

    if cfg!(target_os = "windows") {
        for asset in assets {
            let name = asset["name"].as_str().unwrap_or("");
            if !download::matches_arch(name) {
                continue;
            }
            if !name.starts_with("cudart")
                && (name.contains("bin-win-cuda") || name.contains("bin-cuda"))
            {
                if let Some(parsed_ver) = download::extract_cuda_version(name) {
                    if host_cuda.is_none() || host_cuda == Some(parsed_ver) {
                        download_urls.push((
                            name.to_string(),
                            asset["browser_download_url"].as_str().unwrap().to_string(),
                        ));
                        if let Some(cda) = assets.iter().find(|a| {
                            let cn = a["name"].as_str().unwrap_or("");
                            cn.contains("cudart")
                                && download::extract_cuda_version(cn) == Some(parsed_ver)
                        }) {
                            download_urls.push((
                                cda["name"].as_str().unwrap().to_string(),
                                cda["browser_download_url"].as_str().unwrap().to_string(),
                            ));
                        }
                        break;
                    }
                }
            }
        }
    } else if cfg!(target_os = "macos") {
        let arch = if cfg!(target_arch = "aarch64") {
            "arm64"
        } else {
            "x64"
        };
        if let Some(asset) = assets.iter().find(|a| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .contains(&format!("bin-macos-{arch}"))
        }) {
            download_urls.push((
                asset["name"].as_str().unwrap().to_string(),
                asset["browser_download_url"].as_str().unwrap().to_string(),
            ));
        }
    } else {
        if let Some(asset) = assets.iter().find(|a| {
            let n = a["name"].as_str().unwrap_or("");
            download::matches_arch(n) && (n.contains("bin-linux") || n.contains("bin-ubuntu"))
        }) {
            download_urls.push((
                asset["name"].as_str().unwrap().to_string(),
                asset["browser_download_url"].as_str().unwrap().to_string(),
            ));
        }
    }

    for (name, url) in download_urls {
        download::download_and_extract(&client, &url, &base_dir, &name, "[LlamaServerEngine]")?;
    }

    download::flatten_dlls(&base_dir);
    if !exe_path.exists() {
        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            let _ = std::fs::copy(&nested, &exe_path);
        }
    }
    download::make_executable(&exe_path)?;
    Ok(exe_path)
}
