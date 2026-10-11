use super::RuntimeFlavor;
use crate::inference::engines::download;
use std::error::Error;
use std::path::{Path, PathBuf};

fn copy_companion_cuda_dlls(target_dir: &Path) {
    #[cfg(windows)]
    {
        let companion_dirs = [
            "bin/llama-prism",
            "bin/sd-server",
            "bin/crispasr",
            "bin/audiocpp",
            "bin/nemo-speech",
            "bin",
        ];
        for dir in &companion_dirs {
            let p = Path::new(dir);
            if let Ok(entries) = std::fs::read_dir(p) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let name = entry.file_name().to_string_lossy().to_lowercase();
                        if name.ends_with(".dll") {
                            let dest = target_dir.join(entry.file_name());
                            if !dest.exists() {
                                let _ = std::fs::copy(&path, &dest);
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = target_dir;
}

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
        copy_companion_cuda_dlls(&base_dir);
        return Ok(exe_path);
    }

    if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
        let _ = std::fs::copy(&nested, &exe_path);
        copy_companion_cuda_dlls(&base_dir);
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

    tracing::info!(
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
                RuntimeFlavor::Upstream => "https://github.com/ggml-org/llama.cpp/releases/latest",
                RuntimeFlavor::Prism => "https://github.com/PrismML-Eng/llama.cpp/releases/latest",
            };
            let resp = client.get(html_url).send()?;
            let final_url = resp.url().as_str();
            let tag = final_url.split("/tag/").nth(1).ok_or_else(|| {
                format!("Could not determine release tag from redirect: {final_url}")
            })?;

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
                let _ = download::download_and_extract(
                    &client,
                    &url,
                    &base_dir,
                    &name,
                    "[LlamaServerEngine]",
                );
            }
            download::flatten_dlls(&base_dir);
            copy_companion_cuda_dlls(&base_dir);
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
            download::make_executable(&exe_path)?;
            if exe_path.exists() {
                return Ok(exe_path);
            }
            return Err(
                "Fallback tag extraction did not yield a valid llama-server executable.".into(),
            );
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
        let mut selected_cuda_asset = None;
        let mut selected_cudart_asset = None;

        if let Some(host) = host_cuda {
            for asset in assets {
                let name = asset["name"].as_str().unwrap_or("");
                if !download::matches_arch(name) {
                    continue;
                }
                let is_cuda_bin = !name.starts_with("cudart")
                    && (name.contains("bin-win-cuda")
                        || name.contains("bin-cuda")
                        || name.contains("bin-win-cu"));
                if is_cuda_bin {
                    if let Some(parsed_ver) = download::extract_cuda_version(name) {
                        if parsed_ver.0 == host.0 {
                            selected_cuda_asset =
                                Some((name, asset["browser_download_url"].as_str().unwrap()));
                            if let Some(cda) = assets.iter().find(|a| {
                                let cn = a["name"].as_str().unwrap_or("");
                                cn.contains("cudart")
                                    && download::extract_cuda_version(cn) == Some(parsed_ver)
                            }) {
                                selected_cudart_asset = Some((
                                    cda["name"].as_str().unwrap(),
                                    cda["browser_download_url"].as_str().unwrap(),
                                ));
                            }
                            if parsed_ver == host {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some((name, url)) = selected_cuda_asset {
            download_urls.push((name.to_string(), url.to_string()));
            if let Some((c_name, c_url)) = selected_cudart_asset {
                download_urls.push((c_name.to_string(), c_url.to_string()));
            }
        } else {
            let fallback_asset = assets.iter().find(|a| {
                let n = a["name"].as_str().unwrap_or("");
                download::matches_arch(n)
                    && (n.contains("bin-win-vulkan")
                        || n.contains("bin-win-cpu")
                        || n.contains("bin-win-x64"))
            });
            if let Some(fallback) = fallback_asset {
                download_urls.push((
                    fallback["name"].as_str().unwrap().to_string(),
                    fallback["browser_download_url"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                ));
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

    if download_urls.is_empty() {
        return Err("No compatible llama-server release asset found for this platform. Please set LLAMA_SERVER_PATH in .env".into());
    }

    for (name, url) in download_urls {
        download::download_and_extract(&client, &url, &base_dir, &name, "[LlamaServerEngine]")?;
    }

    download::flatten_dlls(&base_dir);
    copy_companion_cuda_dlls(&base_dir);

    if !exe_path.exists() {
        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            let _ = std::fs::copy(&nested, &exe_path);
        }
    }
    download::make_executable(&exe_path)?;

    if !exe_path.exists() {
        return Err(format!(
            "Could not locate '{}' in '{}' after extraction",
            exe_name,
            base_dir.display()
        )
        .into());
    }

    Ok(exe_path)
}
