use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use std::error::Error;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct ProcessGuard(Child);

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        println!(
            "[BonsaiEngine] Shutting down llama-server process (PID: {})...",
            self.0.id()
        );
        let _ = self.0.kill();
        let _ = self.0.wait();
        println!("[BonsaiEngine] Process terminated and VRAM released.");
    }
}

pub struct BonsaiEngine {
    id: String,
    port: u16,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl BonsaiEngine {
    pub fn new(
        id: String,
        model_path: impl AsRef<Path>,
        mmproj_path: Option<impl AsRef<Path>>,
        n_ctx: Option<u32>,
        n_gpu_layers: Option<u32>,
    ) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::ensure_binary_installed()?;
        let port = Self::find_free_port()?;

        println!(
            "[BonsaiEngine] Launching managed llama-server for '{}' on port {}...",
            id, port
        );

        let mut cmd = Command::new(&bin_path);
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
                    "[BonsaiEngine] Attaching multimodal projector: {}",
                    path.display()
                );
                cmd.arg("--mmproj").arg(path);
            } else {
                eprintln!(
                    "[BonsaiEngine] WARNING: mmproj path does not exist: {}",
                    path.display()
                );
            }
        }

        let child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn llama-server binary at '{}': {e}",
                bin_path.display()
            )
        })?;

        let guard = ProcessGuard(child);

        // Wait for the server to load weights and report healthy
        Self::wait_for_server(port, &guard)?;

        println!("[BonsaiEngine] Managed llama-server is ready and healthy.");

        Ok(Self {
            id,
            port,
            _process: Arc::new(Mutex::new(guard)),
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    /// Detect the maximum CUDA version supported by the installed NVIDIA display driver
    fn detect_host_cuda_version() -> Option<(u32, u32)> {
        let output = Command::new("nvidia-smi").output().ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let idx = text.find("CUDA Version:")?;
        let rest = &text[idx + "CUDA Version:".len()..];
        let ver_str = rest.split_whitespace().next()?;
        Self::parse_cuda_version(ver_str)
    }

    fn parse_cuda_version(s: &str) -> Option<(u32, u32)> {
        let clean = s.trim_start_matches("cu").trim_start_matches('v');
        let mut parts = clean.split('.');
        let major: u32 = parts.next()?.parse().ok()?;
        let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
        Some((major, minor))
    }

    fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        let base_dir = PathBuf::from("bin").join("llama-prism");
        let exe_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let exe_path = base_dir.join(exe_name);

        if exe_path.exists() {
            return Ok(exe_path);
        }

        println!("[BonsaiEngine] PrismML runtime not found. Resolving latest compatible binary...");
        std::fs::create_dir_all(&base_dir)?;

        let client = reqwest::blocking::Client::builder()
            .user_agent("rune-infer/0.1.0")
            .timeout(Duration::from_secs(300))
            .build()?;

        let release_url = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest";
        let resp = client.get(release_url).send()?.error_for_status()?;
        let release_data: serde_json::Value = resp.json()?;

        let assets = release_data["assets"]
            .as_array()
            .ok_or("Invalid GitHub API response: missing 'assets'")?;

        let arch = if cfg!(target_arch = "x86_64") {
            "x64"
        } else if cfg!(target_arch = "aarch64") {
            "arm64"
        } else {
            "x64"
        };

        let host_cuda = Self::detect_host_cuda_version();
        if let Some((maj, min)) = host_cuda {
            println!("[BonsaiEngine] Host GPU driver supports up to CUDA {maj}.{min}");
        } else {
            println!(
                "[BonsaiEngine] nvidia-smi not detected; selecting latest release CUDA version"
            );
        }

        let mut download_urls = Vec::new();

        if cfg!(target_os = "windows") {
            struct CudaCandidate<'a> {
                version_str: String,
                parsed_ver: (u32, u32),
                bin_name: &'a str,
                bin_url: &'a str,
                cudart_name: &'a str,
                cudart_url: &'a str,
            }

            let mut candidates: Vec<CudaCandidate> = Vec::new();

            for asset in assets {
                let name = asset["name"].as_str().unwrap_or("");
                if name.starts_with("llama-")
                    && name.contains("bin-win-cuda")
                    && name.contains(arch)
                {
                    if let Some(rest) = name.split("cuda-").nth(1) {
                        let ver_str = rest.split('-').next().unwrap_or("");
                        if let Some(parsed_ver) = Self::parse_cuda_version(ver_str) {
                            if let Some(cudart) = assets.iter().find(|a| {
                                let cname = a["name"].as_str().unwrap_or("");
                                cname.starts_with("cudart-")
                                    && cname.contains("bin-win-cuda")
                                    && cname.contains(ver_str)
                                    && cname.contains(arch)
                            }) {
                                candidates.push(CudaCandidate {
                                    version_str: ver_str.to_string(),
                                    parsed_ver,
                                    bin_name: name,
                                    bin_url: asset["browser_download_url"].as_str().unwrap_or(""),
                                    cudart_name: cudart["name"].as_str().unwrap_or(""),
                                    cudart_url: cudart["browser_download_url"]
                                        .as_str()
                                        .unwrap_or(""),
                                });
                            }
                        }
                    }
                }
            }

            candidates.sort_by(|a, b| b.parsed_ver.cmp(&a.parsed_ver));

            let selected = if let Some(max_cuda) = host_cuda {
                candidates
                    .iter()
                    .find(|c| c.parsed_ver <= max_cuda)
                    .or_else(|| candidates.first())
            } else {
                candidates.first()
            };

            if let Some(c) = selected {
                println!(
                    "[BonsaiEngine] Selected CUDA {} build ({}, {})",
                    c.version_str, c.bin_name, c.cudart_name
                );
                download_urls.push((c.bin_name.to_string(), c.bin_url.to_string()));
                download_urls.push((c.cudart_name.to_string(), c.cudart_url.to_string()));
            }
        } else if cfg!(target_os = "macos") {
            if let Some(asset) = assets.iter().find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                name.contains("bin-macos-arm64") && name.ends_with(".tar.gz")
            }) {
                download_urls.push((
                    asset["name"].as_str().unwrap().to_string(),
                    asset["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        } else {
            if let Some(asset) = assets.iter().find(|a| {
                let name = a["name"].as_str().unwrap_or("");
                name.contains("bin-linux-cuda") || name.contains("bin-ubuntu")
            }) {
                download_urls.push((
                    asset["name"].as_str().unwrap().to_string(),
                    asset["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        }

        if download_urls.is_empty() {
            return Err(
                "No compatible release asset found for this platform on PrismML-Eng/llama.cpp"
                    .into(),
            );
        }

        for (name, url) in download_urls {
            println!("[BonsaiEngine] Downloading {}...", name);
            let temp_file = base_dir.join(&name);
            let mut archive_resp = client.get(&url).send()?.error_for_status()?;
            let mut file = std::fs::File::create(&temp_file)?;
            std::io::copy(&mut archive_resp, &mut file)?;

            println!("[BonsaiEngine] Extracting {}...", name);
            let status = Command::new("tar")
                .args([
                    "-xf",
                    temp_file.to_str().unwrap(),
                    "-C",
                    base_dir.to_str().unwrap(),
                ])
                .status();

            let _ = std::fs::remove_file(&temp_file);

            if let Err(e) = status {
                return Err(format!("Failed to execute 'tar' to extract {name}: {e}").into());
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(&exe_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&exe_path, perms);
            }
        }

        if !exe_path.exists() {
            return Err(format!(
                "Extraction completed, but '{}' was not found in '{}'",
                exe_name,
                base_dir.display()
            )
            .into());
        }

        println!("[BonsaiEngine] ✓ PrismML runtime installed successfully.");
        Ok(exe_path)
    }

    fn wait_for_server(port: u16, guard: &ProcessGuard) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()?;
        let health_url = format!("http://127.0.0.1:{port}/health");
        let start = Instant::now();
        let timeout = Duration::from_secs(120);

        while start.elapsed() < timeout {
            if let Ok(Some(exit_status)) = unsafe {
                let child_ptr = &guard.0 as *const Child as *mut Child;
                (*child_ptr).try_wait()
            } {
                return Err(format!(
                    "llama-server terminated unexpectedly with status: {exit_status}"
                )
                .into());
            }

            if let Ok(resp) = client.get(&health_url).send() {
                if resp.status().is_success() {
                    return Ok(());
                }
            }

            std::thread::sleep(Duration::from_millis(500));
        }

        Err("Timed out waiting for llama-server to become healthy (120s limit exceeded)".into())
    }

    fn extract_tool_call_json(raw: &str) -> String {
        if let Some(start) = raw.find("<tool_call>") {
            let content_start = start + "<tool_call>".len();
            if let Some(end) = raw[content_start..].find("</tool_call>") {
                return raw[content_start..content_start + end].trim().to_string();
            }
            return raw[content_start..].trim().to_string();
        }

        let mut cleaned = raw;
        if let Some(think_end) = raw.find("</think>") {
            cleaned = &raw[think_end + "</think>".len()..];
        }

        if let (Some(first_b), Some(last_b)) = (cleaned.find('['), cleaned.rfind(']')) {
            if first_b < last_b {
                return cleaned[first_b..=last_b].trim().to_string();
            }
        }
        if let (Some(first_b), Some(last_b)) = (cleaned.find('{'), cleaned.rfind('}')) {
            if first_b < last_b {
                return cleaned[first_b..=last_b].trim().to_string();
            }
        }

        cleaned.trim().to_string()
    }
}

impl InferenceEngine for BonsaiEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceTaskResponse, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ToolCall {
                prompt,
                schema,
                images,
                messages,
            } => {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(300))
                    .build()?;

                // Modern llama-server (libmtmd) processes vision via /v1/chat/completions
                let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
                let stream_mode = on_token.is_some();

                let mut server_messages = Vec::new();

                if !messages.is_empty() {
                    for msg in messages {
                        let (text, img_urls) = msg.split_text_and_images();
                        let mut parts = Vec::new();

                        if !text.is_empty() {
                            parts.push(serde_json::json!({
                                "type": "text",
                                "text": text
                            }));
                        }

                        for url in &img_urls {
                            let data_url = if url.starts_with("data:") {
                                url.clone()
                            } else if let Ok(bytes) = std::fs::read(url) {
                                let b64 = base64::Engine::encode(
                                    &base64::engine::general_purpose::STANDARD,
                                    &bytes,
                                );
                                format!("data:image/jpeg;base64,{}", b64)
                            } else {
                                url.clone()
                            };

                            parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }));
                        }

                        server_messages.push(serde_json::json!({
                            "role": msg.role,
                            "content": parts
                        }));
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

                    for bytes in images {
                        let b64 = base64::Engine::encode(
                            &base64::engine::general_purpose::STANDARD,
                            bytes,
                        );
                        parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/jpeg;base64,{}", b64)
                            }
                        }));
                    }

                    server_messages.push(serde_json::json!({
                        "role": "user",
                        "content": parts
                    }));
                }

                let body = serde_json::json!({
                    "messages": server_messages,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": stream_mode
                });

                let resp = client.post(&endpoint).json(&body).send().map_err(|e| {
                    format!("Failed to connect to managed llama-server at {endpoint}: {e}")
                })?;

                if !resp.status().is_success() {
                    return Err(format!("llama-server returned HTTP {}", resp.status()).into());
                }

                let mut full_output = String::new();
                let mut is_thinking = false;

                if stream_mode {
                    let reader = std::io::BufReader::new(resp);
                    for line in reader.lines() {
                        let line = line?;
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                break;
                            }
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                // 1. Capture isolated reasoning/thinking tokens (OpenAI reasoning_content format)
                                let reasoning_opt = v["choices"][0]["delta"]["reasoning_content"]
                                    .as_str()
                                    .or_else(|| v["choices"][0]["delta"]["reasoning"].as_str())
                                    .or_else(|| v["choices"][0]["delta"]["thought"].as_str());

                                if let Some(reasoning) = reasoning_opt {
                                    if !reasoning.is_empty() {
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

                                // 2. Capture regular content tokens
                                let content_opt = v["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .or_else(|| v["content"].as_str());

                                if let Some(content) = content_opt {
                                    if !content.is_empty() {
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
                    let choice = &result["choices"][0];
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

                let has_tools = match schema {
                    serde_json::Value::Object(o) => !o.is_empty(),
                    serde_json::Value::Array(a) => !a.is_empty(),
                    _ => false,
                };

                if has_tools && full_output.contains("<tool_call>") {
                    let clean_json = Self::extract_tool_call_json(&full_output);
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&clean_json) {
                        return Ok(InferenceTaskResponse::ToolCall(parsed));
                    }
                }

                Ok(InferenceTaskResponse::Text(full_output.trim().to_string()))
            }
        }
    }
}
