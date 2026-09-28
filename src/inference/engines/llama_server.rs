use super::download;
use crate::inference::process::{ProcessGuard, configure_death_signal};
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::Usage;
use std::error::Error;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct ScratchDirGuard(PathBuf);
impl Drop for ScratchDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFlavor {
    Upstream,
    Prism,
}

pub struct LlamaServerEngine {
    id: String,
    port: u16,
    client: reqwest::blocking::Client,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl LlamaServerEngine {
    pub fn new(
        id: String,
        model_path: impl AsRef<Path>,
        mmproj_path: Option<impl AsRef<Path>>,
        mtp_path: Option<impl AsRef<Path>>,
        n_ctx: Option<u32>,
        n_gpu_layers: Option<u32>,
        mtp_heads: Option<u32>,
        extra_args: Option<String>,
        flavor: RuntimeFlavor,
    ) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::ensure_binary_installed(flavor)?;
        let port = Self::find_free_port()?;

        println!(
            "[LlamaServerEngine] Launching managed llama-server ({:?}) for '{}' on port {}...",
            flavor, id, port
        );

        let mut cmd = Command::new(&bin_path);
        configure_death_signal(&mut cmd);

        cmd.arg("-m")
            .arg(model_path.as_ref())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg(n_ctx.unwrap_or(32768).to_string())
            .arg("-ngl")
            .arg(n_gpu_layers.unwrap_or(99).to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(ref mmproj) = mmproj_path {
            let path = mmproj.as_ref();
            if path.exists() {
                println!(
                    "[LlamaServerEngine] Attaching multimodal projector: {}",
                    path.display()
                );
                cmd.arg("--mmproj").arg(path);
            } else {
                cmd.arg("--no-mmproj");
            }
        } else {
            cmd.arg("--no-mmproj");
        }

        if let Some(ref mtp) = mtp_path {
            let path = mtp.as_ref();
            if path.exists() {
                let heads = mtp_heads.unwrap_or(2).max(1);
                println!(
                    "[LlamaServerEngine] Attaching separate MTP draft model: {} ({} draft heads)",
                    path.display(),
                    heads
                );
                cmd.arg("--spec-draft-model").arg(path);
                cmd.arg("--spec-type").arg("draft-mtp");
                cmd.arg("--spec-draft-n-max").arg(heads.to_string());
            } else {
                eprintln!(
                    "[LlamaServerEngine] Warning: MTP path '{}' not found, running without MTP",
                    path.display()
                );
            }
        } else if let Some(heads) = mtp_heads {
            if heads > 0 {
                println!(
                    "[LlamaServerEngine] Enabling embedded MTP draft heads: {}",
                    heads
                );
                cmd.arg("--spec-type").arg("draft-mtp");
                cmd.arg("--spec-draft-n-max").arg(heads.to_string());
            }
        }

        if let Some(extras) = extra_args {
            for arg in extras.split_whitespace() {
                cmd.arg(arg);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn llama-server binary at '{}': {e}",
                bin_path.display()
            )
        })?;

        let stderr_pipe = child.stderr.take();
        let last_stderr_lines = Arc::new(Mutex::new(Vec::new()));
        if let Some(pipe) = stderr_pipe {
            let lines_clone = last_stderr_lines.clone();
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(pipe);
                for line in reader.lines().flatten() {
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 30 {
                        l.remove(0);
                    }
                    l.push(line);
                }
            });
        }

        let guard = ProcessGuard::new(child, format!("llama-server-{}", id));
        Self::wait_for_server(port, &guard, &last_stderr_lines)?;

        let client = reqwest::blocking::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(10)
            .timeout(Duration::from_secs(300))
            .build()?;

        println!("[LlamaServerEngine] Managed llama-server is ready and healthy.");

        Ok(Self {
            id,
            port,
            client,
            _process: guard,
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    fn ensure_binary_installed(flavor: RuntimeFlavor) -> Result<PathBuf, Box<dyn Error>> {
        let exe_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };

        if let Ok(custom) = std::env::var("LLAMA_SERVER_PATH") {
            let p = PathBuf::from(custom);
            if p.exists() {
                println!(
                    "[LlamaServerEngine] Using binary from LLAMA_SERVER_PATH: {}",
                    p.display()
                );
                return Ok(p);
            }
        }

        let root_bin = PathBuf::from("bin").join(exe_name);
        if root_bin.exists() {
            println!(
                "[LlamaServerEngine] Using existing binary from {}",
                root_bin.display()
            );
            return Ok(root_bin);
        }

        let folder_name = match flavor {
            RuntimeFlavor::Upstream => "llama-upstream",
            RuntimeFlavor::Prism => "llama-prism",
        };
        let base_dir = PathBuf::from("bin").join(folder_name);
        let exe_path = base_dir.join(exe_name);

        if exe_path.exists() {
            let is_healthy = {
                #[cfg(windows)]
                {
                    let cuda_dll = base_dir.join("ggml-cuda.dll");
                    if cuda_dll.exists() {
                        std::fs::read_dir(&base_dir).map_or(false, |entries| {
                            entries.flatten().any(|e| {
                                let name = e.file_name().to_string_lossy().to_lowercase();
                                name.starts_with("cudart64_") && name.ends_with(".dll")
                            })
                        })
                    } else {
                        true
                    }
                }
                #[cfg(not(windows))]
                true
            };

            if is_healthy {
                return Ok(exe_path);
            } else {
                println!(
                    "[LlamaServerEngine] Existing installation in '{}' missing CUDA DLLs; repairing...",
                    base_dir.display()
                );
            }
        }

        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let candidate = dir.join(exe_name);
                if candidate.exists() {
                    println!(
                        "[LlamaServerEngine] Found binary in system PATH: {}",
                        candidate.display()
                    );
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
            Err(err) => {
                eprintln!(
                    "[LlamaServerEngine] Direct GitHub API request failed ({err}). Falling back to release redirect..."
                );

                let html_url = match flavor {
                    RuntimeFlavor::Upstream => "https://github.com/ggml-org/llama.cpp/releases",
                    RuntimeFlavor::Prism => "https://github.com/PrismML-Eng/llama.cpp/releases",
                };

                let resp = client.get(html_url).send().map_err(|e| {
                    format!(
                        "Could not resolve release assets from GitHub: {e}.\n\
                        Please download llama-server manually and place it at '{}' or set LLAMA_SERVER_PATH.",
                        exe_path.display()
                    )
                })?;

                let final_url = resp.url().as_str();
                let tag = final_url
                    .split("/tag/")
                    .nth(1)
                    .ok_or_else(|| format!("Could not determine tag from URL: {final_url}"))?;

                println!("[LlamaServerEngine] Resolved latest tag: {tag}");

                let host_cuda = download::detect_host_cuda_version();
                let mut direct_urls = Vec::new();

                if cfg!(target_os = "windows") {
                    let cuda_ver = host_cuda
                        .map(|c| format!("{}.{}", c.0, c.1))
                        .unwrap_or_else(|| "12.4".to_string());
                    let bin_asset = format!("llama-{tag}-bin-win-cuda-{cuda_ver}-x64.zip");
                    let cudart_asset = format!("cudart-llama-bin-win-cuda-{cuda_ver}-x64.zip");
                    let base_dl =
                        format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}");

                    println!(
                        "[LlamaServerEngine] Attempting direct download of CUDA {cuda_ver} assets..."
                    );
                    direct_urls.push((bin_asset.clone(), format!("{base_dl}/{bin_asset}")));
                    direct_urls.push((cudart_asset.clone(), format!("{base_dl}/{cudart_asset}")));
                } else if cfg!(target_os = "macos") {
                    let arch_label = if cfg!(target_arch = "aarch64") {
                        "arm64"
                    } else {
                        "x64"
                    };
                    let bin_asset = format!("llama-{tag}-bin-macos-{arch_label}.tar.gz");
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

                if !exe_path.exists() {
                    if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                        let _ = std::fs::copy(&nested, &exe_path);
                    }
                }

                download::make_executable(&exe_path)?;

                if exe_path.exists() {
                    println!(
                        "[LlamaServerEngine] ✓ Runtime installed via direct fallback at {}",
                        exe_path.display()
                    );
                    return Ok(exe_path);
                }

                return Err(format!(
                    "Fallback extraction completed, but '{}' was not found.\n\
                    Place 'llama-server.exe' in 'bin/' or set LLAMA_SERVER_PATH in .env.",
                    exe_path.display()
                )
                .into());
            }
        };

        let target_release = if let Some(releases_arr) = release_data.as_array() {
            releases_arr
                .iter()
                .find(|rel| {
                    rel["assets"].as_array().map_or(false, |assets| {
                        assets
                            .iter()
                            .any(|a| a["name"].as_str().unwrap_or("").contains("bin-"))
                    })
                })
                .unwrap_or_else(|| &release_data[0])
        } else {
            &release_data
        };

        let assets = target_release["assets"]
            .as_array()
            .ok_or("Invalid GitHub API response: missing 'assets'")?;

        let host_cuda = download::detect_host_cuda_version();
        let mut download_urls = Vec::new();

        if cfg!(target_os = "windows") {
            struct Candidate<'a> {
                parsed_ver: (u32, u32),
                bin_name: &'a str,
                bin_url: &'a str,
                cudart_name: Option<&'a str>,
                cudart_url: Option<&'a str>,
            }

            let mut candidates: Vec<Candidate> = Vec::new();

            for asset in assets {
                let name = asset["name"].as_str().unwrap_or("");
                if !download::matches_arch(name) {
                    continue;
                }

                if !name.starts_with("cudart")
                    && (name.contains("bin-win-cuda") || name.contains("bin-cuda"))
                {
                    if let Some(parsed_ver) = download::extract_cuda_version(name) {
                        let cudart = assets.iter().find(|a| {
                            let cname = a["name"].as_str().unwrap_or("");
                            cname.contains("cudart")
                                && download::matches_arch(cname)
                                && download::extract_cuda_version(cname) == Some(parsed_ver)
                        });

                        candidates.push(Candidate {
                            parsed_ver,
                            bin_name: name,
                            bin_url: asset["browser_download_url"].as_str().unwrap_or(""),
                            cudart_name: cudart.and_then(|a| a["name"].as_str()),
                            cudart_url: cudart.and_then(|a| a["browser_download_url"].as_str()),
                        });
                    }
                }
            }

            candidates.sort_by(|a, b| b.parsed_ver.cmp(&a.parsed_ver));

            let selected = if let Some(max_cuda) = host_cuda {
                println!(
                    "[LlamaServerEngine] Host GPU driver supports up to CUDA {}.{}",
                    max_cuda.0, max_cuda.1
                );
                candidates
                    .iter()
                    .find(|c| c.parsed_ver <= max_cuda)
                    .or_else(|| candidates.first())
            } else {
                candidates
                    .iter()
                    .find(|c| c.parsed_ver == (12, 4))
                    .or_else(|| candidates.first())
            };

            if let Some(c) = selected {
                println!(
                    "[LlamaServerEngine] Selected CUDA {}.{} build ({})",
                    c.parsed_ver.0, c.parsed_ver.1, c.bin_name
                );
                download_urls.push((c.bin_name.to_string(), c.bin_url.to_string()));
                if let (Some(cname), Some(curl)) = (c.cudart_name, c.cudart_url) {
                    download_urls.push((cname.to_string(), curl.to_string()));
                }
            } else {
                if let Some(asset) = assets.iter().find(|a| {
                    let name = a["name"].as_str().unwrap_or("");
                    download::matches_arch(name)
                        && (name.contains("bin-win-vulkan")
                            || name.contains("bin-win-avx2")
                            || name.contains("bin-win-cpu"))
                }) {
                    println!(
                        "[LlamaServerEngine] Selected Windows CPU/Vulkan build: {}",
                        asset["name"].as_str().unwrap_or("")
                    );
                    download_urls.push((
                        asset["name"].as_str().unwrap().to_string(),
                        asset["browser_download_url"].as_str().unwrap().to_string(),
                    ));
                }
            }
        } else if cfg!(target_os = "macos") {
            let arch_label = if cfg!(target_arch = "aarch64") {
                "arm64"
            } else {
                "x64"
            };
            if let Some(asset) = assets.iter().find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                name.contains(&format!("bin-macos-{}", arch_label))
            }) {
                println!("[LlamaServerEngine] Selected macOS {} build", arch_label);
                download_urls.push((
                    asset["name"].as_str().unwrap().to_string(),
                    asset["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        } else {
            let arch_label = if cfg!(target_arch = "aarch64") {
                "arm64"
            } else {
                "x64"
            };
            let cuda_asset = if host_cuda.is_some() {
                assets.iter().find(|a| {
                    let name = a["name"].as_str().unwrap_or("");
                    download::matches_arch(name)
                        && (name.contains("bin-linux-cuda") || name.contains("bin-ubuntu-cuda"))
                })
            } else {
                None
            };

            let fallback_asset = assets.iter().find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                download::matches_arch(name)
                    && (name.contains("bin-ubuntu") || name.contains("bin-linux"))
            });

            if let Some(asset) = cuda_asset.or(fallback_asset) {
                println!("[LlamaServerEngine] Selected Linux {} build", arch_label);
                download_urls.push((
                    asset["name"].as_str().unwrap().to_string(),
                    asset["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        }

        if download_urls.is_empty() {
            return Err("No compatible release assets found for this platform".into());
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

        if !exe_path.exists() {
            return Err(format!(
                "Executable '{}' not found in '{}'",
                exe_name,
                base_dir.display()
            )
            .into());
        }

        println!(
            "[LlamaServerEngine] ✓ Runtime '{:?}' installed successfully at {}",
            flavor,
            exe_path.display()
        );
        Ok(exe_path)
    }

    fn wait_for_server(
        port: u16,
        guard_mutex: &Arc<Mutex<ProcessGuard>>,
        stderr_lines: &Arc<Mutex<Vec<String>>>,
    ) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()?;
        let health_url = format!("http://127.0.0.1:{port}/health");
        let start = Instant::now();
        let timeout = Duration::from_secs(120);

        while start.elapsed() < timeout {
            if let Ok(mut guard) = guard_mutex.try_lock() {
                if let Ok(Some(status)) = guard.try_wait() {
                    let logs = stderr_lines.lock().unwrap().join("\n");
                    let detail = if logs.trim().is_empty() {
                        String::new()
                    } else {
                        format!(
                            "\n--- [llama-server stderr] ---\n{}\n-----------------------------",
                            logs
                        )
                    };
                    return Err(
                        format!("Server terminated prematurely with {status}{detail}").into(),
                    );
                }
            }

            if let Ok(resp) = client.get(&health_url).send() {
                if resp.status().is_success() {
                    return Ok(());
                }
            }

            std::thread::sleep(Duration::from_millis(500));
        }
        Err("Timed out waiting for managed llama-server".into())
    }

    fn parse_xml_tool_call(block: &str) -> Option<serde_json::Value> {
        let fn_start = block.find("<function=")?;
        let after_fn = &block[fn_start + "<function=".len()..];
        let fn_end = after_fn.find('>')?;
        let fn_name = after_fn[..fn_end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if fn_name.is_empty() {
            return None;
        }

        let mut args = serde_json::Map::new();
        let mut cursor = &after_fn[fn_end + 1..];

        while let Some(p_start) = cursor.find("<parameter=") {
            let after_p = &cursor[p_start + "<parameter=".len()..];
            let p_end = after_p.find('>')?;
            let param_name = after_p[..p_end]
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            let val_start = &after_p[p_end + 1..];

            let val_end = val_start
                .find("<parameter=")
                .or_else(|| val_start.find("</parameter>"))
                .or_else(|| val_start.find("</function>"))
                .unwrap_or(val_start.len());

            let raw_val = val_start[..val_end].trim();
            let val = serde_json::from_str::<serde_json::Value>(raw_val)
                .unwrap_or_else(|_| serde_json::Value::String(raw_val.to_string()));

            if !param_name.is_empty() {
                args.insert(param_name, val);
            }

            cursor = &val_start[val_end..];
            if let Some(stripped) = cursor.strip_prefix("</parameter>") {
                cursor = stripped;
            }
        }

        Some(serde_json::json!({
            "name": fn_name,
            "arguments": serde_json::Value::Object(args)
        }))
    }

    fn extract_tool_calls(raw: &str) -> Option<(Option<String>, serde_json::Value)> {
        if raw.contains("<tool_call>") {
            let mut calls = Vec::new();
            let mut remaining = raw;
            let mut text_parts = Vec::new();

            while let Some(start) = remaining.find("<tool_call>") {
                let before = &remaining[..start];
                if !before.trim().is_empty() {
                    text_parts.push(before.trim());
                }
                let after_start = &remaining[start + "<tool_call>".len()..];
                let (block, next_rem) = match after_start.find("</tool_call>") {
                    Some(end) => (
                        &after_start[..end],
                        &after_start[end + "</tool_call>".len()..],
                    ),
                    None => (after_start, ""),
                };
                remaining = next_rem;
                let block = block.trim();

                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(block) {
                    if let Some(arr) = parsed.as_array() {
                        calls.extend(arr.clone());
                    } else if parsed.is_object() {
                        calls.push(parsed);
                    }
                    continue;
                }

                if let Some(call) = Self::parse_xml_tool_call(block) {
                    calls.push(call);
                }
            }

            if !remaining.trim().is_empty() {
                text_parts.push(remaining.trim());
            }

            if !calls.is_empty() {
                let clean_text = text_parts.join("\n\n").trim().to_string();
                let content_opt = if clean_text.is_empty() {
                    None
                } else {
                    Some(clean_text)
                };
                return Some((content_opt, serde_json::Value::Array(calls)));
            }
        }

        let mut cleaned = raw;
        if let Some(think_end) = raw.find("</think>") {
            cleaned = &raw[think_end + "</think>".len()..];
        }

        if let (Some(first_b), Some(last_b)) = (cleaned.find('['), cleaned.rfind(']')) {
            if first_b < last_b {
                if let Ok(val) =
                    serde_json::from_str::<serde_json::Value>(&cleaned[first_b..=last_b])
                {
                    if val.is_array() {
                        return Some((None, val));
                    }
                }
            }
        }
        if let (Some(first_b), Some(last_b)) = (cleaned.find('{'), cleaned.rfind('}')) {
            if first_b < last_b {
                if let Ok(val) =
                    serde_json::from_str::<serde_json::Value>(&cleaned[first_b..=last_b])
                {
                    if val.is_object() {
                        return Some((None, serde_json::Value::Array(vec![val])));
                    }
                }
            }
        }

        None
    }
}

impl InferenceEngine for LlamaServerEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ToolCall {
                prompt,
                schema,
                images,
                messages,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let stream_mode = on_token.is_some();

                let scratch_path = std::env::temp_dir().join("rune_infer").join(format!(
                    "ipc_{}_{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                ));
                let _ = std::fs::create_dir_all(&scratch_path);
                let _guard = ScratchDirGuard(scratch_path.clone());

                let mut server_messages = Vec::new();

                if !messages.is_empty() {
                    for (m_idx, msg) in messages.iter().enumerate() {
                        let (text, _) = msg.split_text_and_media();
                        let mut parts = Vec::new();

                        if !text.is_empty() {
                            parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }

                        if msg.role == "user"
                            && m_idx == messages.iter().position(|m| m.role == "user").unwrap_or(0)
                        {
                            for (f_idx, frame_bytes) in images.iter().enumerate() {
                                let frame_file = scratch_path.join(format!("frame_{f_idx}.jpg"));
                                if std::fs::write(&frame_file, frame_bytes).is_ok() {
                                    parts.push(serde_json::json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": frame_file.to_string_lossy().to_string()
                                        }
                                    }));
                                }
                            }
                        }

                        let mut msg_obj = serde_json::json!({
                            "role": msg.role,
                        });

                        if msg.role == "assistant" && msg.tool_calls.is_some() {
                            if !text.is_empty() {
                                msg_obj["content"] = serde_json::Value::String(text);
                            } else {
                                msg_obj["content"] = serde_json::Value::Null;
                            }
                            msg_obj["tool_calls"] = serde_json::json!(msg.tool_calls);
                        } else if msg.role == "tool" {
                            msg_obj["content"] = serde_json::Value::String(text);
                            if let Some(ref tid) = msg.tool_call_id {
                                msg_obj["tool_call_id"] = serde_json::json!(tid);
                            }
                            if let Some(ref name) = msg.name {
                                msg_obj["name"] = serde_json::json!(name);
                            }
                        } else {
                            msg_obj["content"] = serde_json::Value::Array(parts);
                        }

                        server_messages.push(msg_obj);
                    }
                } else {
                    let mut parts = Vec::new();
                    let clean_prompt = prompt
                        .replace("<|im_start|>", "")
                        .replace("<|im_end|>", "")
                        .replace("<__media__>", "")
                        .replace("[media]", "")
                        .trim()
                        .to_string();

                    parts.push(serde_json::json!({
                        "type": "text",
                        "text": clean_prompt
                    }));

                    for (f_idx, frame_bytes) in images.iter().enumerate() {
                        let frame_file = scratch_path.join(format!("frame_{f_idx}.jpg"));
                        if std::fs::write(&frame_file, frame_bytes).is_ok() {
                            parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": frame_file.to_string_lossy().to_string()
                                }
                            }));
                        }
                    }

                    server_messages.push(serde_json::json!({
                        "role": "user",
                        "content": parts
                    }));
                }

                let has_tools = match schema {
                    serde_json::Value::Object(o) => !o.is_empty(),
                    serde_json::Value::Array(a) => !a.is_empty(),
                    _ => false,
                };

                let mut body = serde_json::json!({
                    "messages": server_messages,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": stream_mode
                });
                if has_tools {
                    body["tools"] = schema.clone();
                }

                let resp = self
                    .client
                    .post(&endpoint)
                    .json(&body)
                    .send()
                    .map_err(|e| {
                        format!("Failed to connect to managed llama-server at {endpoint}: {e}")
                    })?;

                if !resp.status().is_success() {
                    return Err(format!("llama-server returned HTTP {}", resp.status()).into());
                }

                let mut full_output = String::new();
                let mut is_thinking = false;
                let mut streamed_tokens = 0u32;
                let mut server_usage: Option<Usage> = None;

                if stream_mode {
                    let reader = std::io::BufReader::new(resp);
                    for line in reader.lines() {
                        let line = line?;
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                break;
                            }
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(u) = v.get("usage") {
                                    let pt = u
                                        .get("prompt_tokens")
                                        .and_then(|x| x.as_u64())
                                        .unwrap_or(0)
                                        as u32;
                                    let ct = u
                                        .get("completion_tokens")
                                        .and_then(|x| x.as_u64())
                                        .unwrap_or(0)
                                        as u32;
                                    server_usage = Some(Usage::new(pt, ct));
                                }

                                let reasoning_opt = v["choices"][0]["delta"]["reasoning_content"]
                                    .as_str()
                                    .or_else(|| v["choices"][0]["delta"]["reasoning"].as_str())
                                    .or_else(|| v["choices"][0]["delta"]["thought"].as_str());

                                if let Some(reasoning) = reasoning_opt {
                                    if !reasoning.is_empty() {
                                        streamed_tokens += 1;
                                        if !is_thinking {
                                            is_thinking = true;
                                            full_output.push_str("<think>\n");
                                            if let Some(ref mut cb) = on_token {
                                                if !cb("<think>\n") {
                                                    break;
                                                }
                                            }
                                        }
                                        full_output.push_str(reasoning);
                                        if let Some(ref mut cb) = on_token {
                                            if !cb(reasoning) {
                                                break;
                                            }
                                        }
                                    }
                                }

                                let content_opt = v["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .or_else(|| v["content"].as_str());

                                if let Some(content) = content_opt {
                                    if !content.is_empty() {
                                        streamed_tokens += 1;
                                        if is_thinking {
                                            is_thinking = false;
                                            full_output.push_str("\n</think>\n\n");
                                            if let Some(ref mut cb) = on_token {
                                                if !cb("\n</think>\n\n") {
                                                    break;
                                                }
                                            }
                                        }
                                        full_output.push_str(content);
                                        if let Some(ref mut cb) = on_token {
                                            if !cb(content) {
                                                break;
                                            }
                                        }
                                    }
                                }

                                if v["choices"][0]["finish_reason"].is_string()
                                    || v["stop"].as_bool().unwrap_or(false)
                                {
                                    if is_thinking {
                                        is_thinking = false;
                                        full_output.push_str("\n</think>\n\n");
                                        if let Some(ref mut cb) = on_token {
                                            let _ = cb("\n</think>\n\n");
                                        }
                                    }
                                    if v["stop"].as_bool().unwrap_or(false) {
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if is_thinking {
                        full_output.push_str("\n</think>\n\n");
                        if let Some(ref mut cb) = on_token {
                            let _ = cb("\n</think>\n\n");
                        }
                    }
                } else {
                    let result: serde_json::Value = resp.json()?;
                    if let Some(u) = result.get("usage") {
                        let pt =
                            u.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                        let ct = u
                            .get("completion_tokens")
                            .and_then(|x| x.as_u64())
                            .unwrap_or(0) as u32;
                        server_usage = Some(Usage::new(pt, ct));
                    }

                    let choice = &result["choices"][0];

                    if let Some(tc) = choice["message"]
                        .get("tool_calls")
                        .filter(|v| v.is_array() && !v.as_array().unwrap().is_empty())
                    {
                        let content_str = choice["message"]["content"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| s.to_string());
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: content_str,
                                tool_calls: tc.clone(),
                            },
                            usage: server_usage.unwrap_or_else(|| Usage::new(0, 0)),
                        });
                    }

                    let content = choice["message"]["content"]
                        .as_str()
                        .or_else(|| result["content"].as_str())
                        .unwrap_or("");

                    let reasoning = choice["message"]["reasoning_content"]
                        .as_str()
                        .or_else(|| choice["message"]["reasoning"].as_str())
                        .or_else(|| choice["message"]["thought"].as_str())
                        .unwrap_or("");

                    if !reasoning.is_empty() && !content.contains("<think>") {
                        full_output = format!("<think>\n{}\n</think>\n\n{}", reasoning, content);
                    } else {
                        full_output = content.to_string();
                    }
                }

                let final_usage = server_usage.unwrap_or_else(|| {
                    let est_prompt = (prompt.len() / 4).max(1) as u32;
                    let est_completion = if stream_mode {
                        streamed_tokens
                    } else {
                        (full_output.len() / 4).max(1) as u32
                    };
                    Usage::new(est_prompt, est_completion)
                });

                if let Some((clean_content, parsed_tools)) = Self::extract_tool_calls(&full_output)
                {
                    return Ok(InferenceOutput {
                        response: InferenceTaskResponse::ToolCall {
                            content: clean_content,
                            tool_calls: parsed_tools,
                        },
                        usage: final_usage,
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Text(full_output.trim().to_string()),
                    usage: final_usage,
                })
            }
            _ => Err("LlamaServerEngine does not support this task type".into()),
        }
    }
}
