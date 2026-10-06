use super::download;
use crate::config::ModelConfig;
use crate::inference::process::{ProcessGuard, configure_death_signal};
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ImageData, ImageGenerationResponse, Usage};
use std::error::Error;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct SdServerEngine {
    id: String,
    port: u16,
    client: reqwest::blocking::Client,
    _process: Arc<Mutex<ProcessGuard>>,
    default_steps: u32,
    default_cfg: f32,
}

impl SdServerEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::ensure_binary_installed()?;
        let port = Self::find_free_port()?;

        println!(
            "[SdServerEngine] Launching managed sd-server for '{}' on port {}...",
            model.id, port
        );

        let mut cmd = Command::new(&bin_path);
        configure_death_signal(&mut cmd);

        cmd.arg("--diffusion-model")
            .arg(&model.model_path)
            .arg("--listen-ip")
            .arg("127.0.0.1")
            .arg("--listen-port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(ref llm) = model.text_encoder_path {
            if Path::new(llm).exists() {
                println!("[SdServerEngine] Attaching text encoder / LLM: {}", llm);
                cmd.arg("--llm").arg(llm);
            }
        }

        if let Some(ref vae) = model.vae_path {
            if Path::new(vae).exists() {
                println!("[SdServerEngine] Attaching VAE: {}", vae);
                cmd.arg("--vae").arg(vae);
            }
        }

        if let Some(ref vision) = model.llm_vision_path {
            if Path::new(vision).exists() {
                println!("[SdServerEngine] Attaching vision tower: {}", vision);
                cmd.arg("--llm_vision").arg(vision);
            }
        }

        let steps = model.default_steps.unwrap_or(20);
        let cfg = model.default_cfg_scale.unwrap_or(4.5);
        let sampler = model.default_sample_method.as_deref().unwrap_or("euler");

        cmd.arg("--steps").arg(steps.to_string());
        cmd.arg("--cfg-scale").arg(cfg.to_string());
        cmd.arg("--sampling-method").arg(sampler);

        if let Some(ref r) = model.runtime {
            if let Some(ref extras) = r.extra_args {
                for arg in extras.split_whitespace() {
                    cmd.arg(arg);
                }
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn sd-server binary at '{}': {e}",
                bin_path.display()
            )
        })?;

        let stderr_pipe = child.stderr.take();
        let last_stderr_lines = Arc::new(Mutex::new(Vec::new()));
        if let Some(pipe) = stderr_pipe {
            let lines_clone = last_stderr_lines.clone();
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    // Stream logs so you can see live progress and errors
                    eprintln!("[sd-server] {}", line);
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 30 {
                        l.remove(0);
                    }
                    l.push(line);
                }
            });
        }

        let guard = ProcessGuard::new(child, format!("sd-server-{}", model.id));
        Self::wait_for_server(port, &guard, &last_stderr_lines)?;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600)) // 10 min timeout for high-res DiT inference
            .build()?;

        println!("[SdServerEngine] Managed sd-server is ready and healthy.");

        Ok(Self {
            id: model.id.clone(),
            port,
            client,
            _process: guard,
            default_steps: steps,
            default_cfg: cfg,
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        let exe_name = if cfg!(windows) {
            "sd-server.exe"
        } else {
            "sd-server"
        };

        if let Ok(custom) = std::env::var("SD_SERVER_PATH") {
            let p = PathBuf::from(custom);
            if p.exists() {
                return Ok(p);
            }
        }

        let base_dir = PathBuf::from("bin").join("sd-server");
        let exe_path = base_dir.join(exe_name);
        if exe_path.exists() {
            return Ok(exe_path);
        }

        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            return Ok(nested);
        }

        let root_bin = PathBuf::from("bin").join(exe_name);
        if root_bin.exists() {
            return Ok(root_bin);
        }

        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let candidate = dir.join(exe_name);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }

        println!("[SdServerEngine] Resolving sd-server runtime from GitHub...");
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url =
            "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=5";
        let mut req = client
            .get(release_url)
            .header("Accept", "application/vnd.github+json");

        if let Ok(token) = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")) {
            req = req.header("Authorization", format!("Bearer {}", token.trim()));
        }

        let release_val: serde_json::Value = match req.send().and_then(|r| r.error_for_status()) {
            Ok(resp) => resp.json()?,
            Err(err) => {
                eprintln!(
                    "[SdServerEngine] Direct GitHub API request failed ({err}). Falling back to release redirect..."
                );
                let html_url = "https://github.com/leejet/stable-diffusion.cpp/releases";
                let resp = client.get(html_url).send().map_err(|e| {
                    format!(
                        "Could not resolve release assets from GitHub: {e}.\n\
                        Please download 'sd-server' manually and place it at '{}' or set SD_SERVER_PATH in .env.",
                        exe_path.display()
                    )
                })?;
                let final_url = resp.url().as_str();
                let tag = final_url
                    .split("/tag/")
                    .nth(1)
                    .ok_or_else(|| format!("Could not determine tag from URL: {final_url}"))?;
                println!("[SdServerEngine] Resolved latest tag: {tag}");
                let api_fallback = format!(
                    "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/{tag}"
                );
                client
                    .get(&api_fallback)
                    .header("Accept", "application/vnd.github+json")
                    .send()?
                    .error_for_status()?
                    .json()?
            }
        };

        let assets = if let Some(arr) = release_val.as_array() {
            arr.first()
                .and_then(|r| r["assets"].as_array())
                .ok_or("No release assets found")?
        } else {
            release_val["assets"]
                .as_array()
                .ok_or("Invalid release payload: missing assets")?
        };

        let host_cuda = download::detect_host_cuda_version();
        let mut download_urls = Vec::new();

        if cfg!(target_os = "windows") {
            let wants_cuda = host_cuda.is_some_and(|(major, _)| major >= 12);
            let cuda_asset = if wants_cuda || host_cuda.is_none() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.contains("bin-win-cuda") && !n.starts_with("cudart")
                })
            } else {
                None
            };

            let cudart_asset = if cuda_asset.is_some() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.starts_with("cudart-sd-bin-win")
                })
            } else {
                None
            };

            if let (Some(ca), Some(cda)) = (cuda_asset, cudart_asset) {
                if let Some(cuda) = host_cuda {
                    println!(
                        "[SdServerEngine] Host supports CUDA {}.{}. Selected CUDA 12 build (backward-compatible)",
                        cuda.0, cuda.1
                    );
                } else {
                    println!(
                        "[SdServerEngine] Selected Windows CUDA 12 build with cudart dependencies"
                    );
                }
                download_urls.push((
                    ca["name"].as_str().unwrap().to_string(),
                    ca["browser_download_url"].as_str().unwrap().to_string(),
                ));
                download_urls.push((
                    cda["name"].as_str().unwrap().to_string(),
                    cda["browser_download_url"].as_str().unwrap().to_string(),
                ));
            } else {
                let fallback = assets
                    .iter()
                    .find(|a| {
                        let n = a["name"].as_str().unwrap_or("").to_lowercase();
                        n.contains("bin-win-vulkan") || n.contains("bin-win-cpu")
                    })
                    .ok_or("No matching Windows binary asset found")?;
                download_urls.push((
                    fallback["name"].as_str().unwrap().to_string(),
                    fallback["browser_download_url"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                ));
            }
        } else if cfg!(target_os = "macos") {
            let asset = assets
                .iter()
                .find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.contains("bin-darwin-macos") || n.contains("bin-macos")
                })
                .ok_or("No matching macOS asset found")?;
            download_urls.push((
                asset["name"].as_str().unwrap().to_string(),
                asset["browser_download_url"].as_str().unwrap().to_string(),
            ));
        } else {
            let asset = assets
                .iter()
                .find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.contains("bin-linux-ubuntu") && (n.contains("vulkan") || !n.contains("rocm"))
                })
                .ok_or("No matching Linux asset found")?;
            download_urls.push((
                asset["name"].as_str().unwrap().to_string(),
                asset["browser_download_url"].as_str().unwrap().to_string(),
            ));
        }

        for (archive_name, download_url) in download_urls {
            download::download_and_extract(
                &client,
                &download_url,
                &base_dir,
                &archive_name,
                "[SdServerEngine]",
            )?;
        }

        download::flatten_dlls(&base_dir);

        if !exe_path.exists() {
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
        }

        download::make_executable(&exe_path)?;

        if !exe_path.exists() {
            return Err(format!("Could not locate '{}' after extraction", exe_name).into());
        }

        println!(
            "[SdServerEngine] ✓ sd-server installed successfully at {}",
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
        let check_url = format!("http://127.0.0.1:{port}/v1/models");
        let start = Instant::now();
        let timeout = Duration::from_secs(120);

        while start.elapsed() < timeout {
            if let Ok(mut guard) = guard_mutex.try_lock() {
                if let Ok(Some(status)) = guard.try_wait() {
                    let logs = stderr_lines.lock().unwrap().join("\n");
                    return Err(format!(
                        "sd-server terminated prematurely with {status}\n--- [sd-server stderr] ---\n{logs}\n---------------------------"
                    ).into());
                }
            }

            if let Ok(resp) = client.get(&check_url).send() {
                if resp.status().is_success() {
                    return Ok(());
                }
            }

            std::thread::sleep(Duration::from_millis(500));
        }

        Err("Timed out waiting for managed sd-server".into())
    }
}

impl InferenceEngine for SdServerEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ImageGeneration {
                prompt,
                negative_prompt,
                size,
                response_format,
                steps,
                cfg_scale,
                seed,
                sample_method,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/images/generations", self.port);
                let (width, height) = if let Some(s) = size {
                    let parts: Vec<&str> = if s.contains('x') {
                        s.split('x').collect()
                    } else if s.contains('×') {
                        s.split('×').collect()
                    } else {
                        s.split('*').collect()
                    };
                    if parts.len() == 2 {
                        (
                            parts[0].trim().parse::<u32>().unwrap_or(1024),
                            parts[1].trim().parse::<u32>().unwrap_or(1024),
                        )
                    } else {
                        (1024, 1024)
                    }
                } else {
                    (1024, 1024)
                };

                let final_steps = steps.unwrap_or(self.default_steps);
                let final_cfg = cfg_scale.unwrap_or(self.default_cfg);
                let size_str = format!("{}x{}", width, height);

                // Build <sd_cpp_extra_args> snippet for standard stable-diffusion.cpp compatibility
                let mut extra_map = serde_json::Map::new();
                extra_map.insert("steps".to_string(), serde_json::json!(final_steps));
                extra_map.insert("cfg_scale".to_string(), serde_json::json!(final_cfg));

                if let Some(s) = seed {
                    if *s >= 0 {
                        extra_map.insert("seed".to_string(), serde_json::json!(s));
                    }
                }
                if let Some(np) = negative_prompt {
                    if !np.trim().is_empty() {
                        extra_map
                            .insert("negative_prompt".to_string(), serde_json::json!(np.trim()));
                    }
                }
                if let Some(sm) = sample_method {
                    if !sm.trim().is_empty() {
                        extra_map
                            .insert("sampling_method".to_string(), serde_json::json!(sm.trim()));
                    }
                }

                let extra_json_str = serde_json::Value::Object(extra_map).to_string();
                let final_prompt = if prompt.contains("<sd_cpp_extra_args>") {
                    prompt.clone()
                } else {
                    format!(
                        "{}<sd_cpp_extra_args>{}</sd_cpp_extra_args>",
                        prompt.trim(),
                        extra_json_str
                    )
                };

                let mut body = serde_json::json!({
                    "prompt": final_prompt,
                    "size": size_str,
                    "width": width,
                    "height": height,
                    "steps": final_steps,
                    "cfg_scale": final_cfg,
                    "response_format": response_format.as_deref().unwrap_or("b64_json"),
                });

                if let Some(s) = seed {
                    if *s >= 0 {
                        body["seed"] = serde_json::json!(s);
                    }
                }
                if let Some(np) = negative_prompt {
                    if !np.trim().is_empty() {
                        body["negative_prompt"] = serde_json::json!(np.trim());
                    }
                }
                if let Some(sm) = sample_method {
                    if !sm.trim().is_empty() {
                        body["sampling_method"] = serde_json::json!(sm.trim());
                    }
                }

                println!(
                    "[SdServerEngine] Generating image ({} steps | CFG: {:.1} | size: {}{}{})...",
                    final_steps,
                    final_cfg,
                    size_str,
                    seed.map(|s| format!(" | seed: {s}")).unwrap_or_default(),
                    negative_prompt
                        .as_ref()
                        .map(|np| format!(" | neg: \"{}\"", np.trim()))
                        .unwrap_or_default()
                );

                let resp = self.client.post(&endpoint).json(&body).send()?;
                if !resp.status().is_success() {
                    let err_text = resp.text().unwrap_or_default();
                    return Err(format!("sd-server returned error: {err_text}").into());
                }

                let result: serde_json::Value = resp.json()?;
                let mut data_vec = Vec::new();
                if let Some(arr) = result["data"].as_array() {
                    for item in arr {
                        data_vec.push(ImageData {
                            b64_json: item["b64_json"].as_str().map(|s| s.to_string()),
                            url: item["url"].as_str().map(|s| s.to_string()),
                        });
                    }
                }

                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Image(ImageGenerationResponse {
                        created,
                        data: data_vec,
                    }),
                    usage: Usage::new(prompt.len() as u32 / 4, 1),
                })
            }
            _ => Err("SdServerEngine only supports ImageGeneration tasks".into()),
        }
    }
}
