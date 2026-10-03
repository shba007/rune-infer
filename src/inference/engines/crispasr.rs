use super::download;
use crate::config::ModelConfig;
use crate::inference::process::{ProcessGuard, configure_death_signal};
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{AudioTranscriptionResponse, AudioTranslationResponse, Usage};
use std::error::Error;
use std::io::BufRead;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct CrispAsrEngine {
    id: String,
    port: u16,
    client: reqwest::blocking::Client,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl CrispAsrEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::ensure_binary_installed()?;
        let port = Self::find_free_port()?;
        let bin_dir = bin_path
            .parent()
            .unwrap_or_else(|| Path::new("bin/crispasr"));

        println!(
            "[CrispAsrEngine] Launching managed crispasr for '{}' on port {}...",
            model.id, port
        );

        let mut cmd = Command::new(&bin_path);
        configure_death_signal(&mut cmd);
        cmd.current_dir(bin_dir);

        // Ensure bin_dir and companion DLL directories are visible in PATH
        let mut search_paths = vec![bin_dir.to_path_buf()];
        if let Ok(entries) = std::fs::read_dir("bin") {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    search_paths.push(entry.path());
                }
            }
        }
        let mut path_env = std::env::join_paths(search_paths)
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if let Ok(sys_path) = std::env::var("PATH") {
            path_env = format!("{};{}", path_env, sys_path);
        }
        cmd.env("PATH", path_env);

        // Run in server mode with loaded model
        cmd.arg("--server")
            .arg("-m")
            .arg(&model.model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(ref r) = model.runtime {
            if let Some(ref extras) = r.extra_args {
                for arg in extras.split_whitespace() {
                    cmd.arg(arg);
                }
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn crispasr binary at '{}': {e}",
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
                    eprintln!("[crispasr] {}", line);
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 30 {
                        l.remove(0);
                    }
                    l.push(line);
                }
            });
        }

        let guard = ProcessGuard::new(child, format!("crispasr-{}", model.id));
        Self::wait_for_server(port, &guard, &last_stderr_lines)?;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        println!("[CrispAsrEngine] Managed crispasr server is ready and healthy.");

        Ok(Self {
            id: model.id.clone(),
            port,
            client,
            _process: guard,
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        let exe_name = if cfg!(windows) {
            "crispasr.exe"
        } else {
            "crispasr"
        };

        if let Ok(custom) = std::env::var("CRISPASR_PATH") {
            let p = PathBuf::from(custom);
            if p.exists() {
                return Ok(p);
            }
        }

        let base_dir = PathBuf::from("bin").join("crispasr");
        let exe_path = base_dir.join(exe_name);
        if exe_path.exists() {
            Self::copy_companion_cuda_dlls(&base_dir);
            return Ok(exe_path);
        }

        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            Self::copy_companion_cuda_dlls(&base_dir);
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

        println!("[CrispAsrEngine] Resolving CrispASR runtime from GitHub releases...");
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url = "https://api.github.com/repos/CrispStrobe/CrispASR/releases?per_page=5";
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
                    "[CrispAsrEngine] Direct GitHub API request failed ({err}). Falling back to release tag redirect..."
                );
                let html_url = "https://github.com/CrispStrobe/CrispASR/releases";
                let resp = client.get(html_url).send().map_err(|e| {
                    format!(
                        "Could not resolve CrispASR assets: {e}.\n\
                        Please download crispasr manually and place it at '{}' or set CRISPASR_PATH in .env.",
                        exe_path.display()
                    )
                })?;
                let final_url = resp.url().as_str();
                let tag = final_url
                    .split("/tag/")
                    .nth(1)
                    .ok_or_else(|| format!("Could not determine tag from URL: {final_url}"))?;
                println!("[CrispAsrEngine] Resolved latest tag: {tag}");
                let api_fallback = format!(
                    "https://api.github.com/repos/CrispStrobe/CrispASR/releases/tags/{tag}"
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

        let cudart_asset = assets.iter().find(|a| {
            let n = a["name"].as_str().unwrap_or("").to_lowercase();
            n.contains("cudart") && n.contains("win")
        });

        if cfg!(target_os = "windows") {
            let cuda_asset = if host_cuda.is_some() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.contains("win-cuda") || (n.contains("windows") && n.contains("cuda"))
                })
            } else {
                None
            };

            let fallback = assets.iter().find(|a| {
                let n = a["name"].as_str().unwrap_or("").to_lowercase();
                n.contains("win-cpu") || (n.contains("windows") && !n.contains("cuda"))
            });

            if let Some(target) = cuda_asset.or(fallback) {
                download_urls.push((
                    target["name"].as_str().unwrap().to_string(),
                    target["browser_download_url"].as_str().unwrap().to_string(),
                ));
                if target == cuda_asset.unwrap_or(target) {
                    if let Some(cda) = cudart_asset {
                        download_urls.push((
                            cda["name"].as_str().unwrap().to_string(),
                            cda["browser_download_url"].as_str().unwrap().to_string(),
                        ));
                    }
                }
            }
        } else if cfg!(target_os = "macos") {
            if let Some(asset) = assets.iter().find(|a| {
                let n = a["name"].as_str().unwrap_or("").to_lowercase();
                n.contains("macos") || n.contains("darwin")
            }) {
                download_urls.push((
                    asset["name"].as_str().unwrap().to_string(),
                    asset["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        } else {
            let cuda_asset = if host_cuda.is_some() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    n.contains("linux-cuda") || (n.contains("linux") && n.contains("cuda"))
                })
            } else {
                None
            };

            let fallback = assets.iter().find(|a| {
                let n = a["name"].as_str().unwrap_or("").to_lowercase();
                n.contains("linux") && (n.contains("cpu") || !n.contains("rocm"))
            });

            if let Some(target) = cuda_asset.or(fallback) {
                download_urls.push((
                    target["name"].as_str().unwrap().to_string(),
                    target["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        }

        if download_urls.is_empty() {
            return Err("No compatible CrispASR release asset found for this platform".into());
        }

        for (archive_name, download_url) in download_urls {
            download::download_and_extract(
                &client,
                &download_url,
                &base_dir,
                &archive_name,
                "[CrispAsrEngine]",
            )?;
        }

        download::flatten_dlls(&base_dir);
        Self::copy_companion_cuda_dlls(&base_dir);

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
            "[CrispAsrEngine] ✓ CrispASR runtime installed successfully at {}",
            exe_path.display()
        );
        Ok(exe_path)
    }

    fn copy_companion_cuda_dlls(target_dir: &Path) {
        #[cfg(windows)]
        {
            let companion_dirs = [
                "bin/llama-upstream",
                "bin/llama-prism",
                "bin/sd-server",
                "bin",
            ];
            for dir in &companion_dirs {
                let p = Path::new(dir);
                if let Ok(entries) = std::fs::read_dir(p) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() {
                            let name = entry.file_name().to_string_lossy().to_lowercase();
                            if name.starts_with("cudart64_")
                                || name.starts_with("cublas64_")
                                || name.starts_with("cublaslt64_")
                                || name.starts_with("vcomp140")
                            {
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

    fn wait_for_server(
        port: u16,
        guard_mutex: &Arc<Mutex<ProcessGuard>>,
        stderr_lines: &Arc<Mutex<Vec<String>>>,
    ) -> Result<(), Box<dyn Error>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()?;
        let check_url = format!("http://127.0.0.1:{port}/health");
        let start = Instant::now();
        let timeout = Duration::from_secs(120);

        while start.elapsed() < timeout {
            if let Ok(mut guard) = guard_mutex.try_lock() {
                if let Ok(Some(status)) = guard.try_wait() {
                    let logs = stderr_lines.lock().unwrap().join("\n");
                    return Err(format!(
                        "crispasr terminated prematurely with {status}\n--- [crispasr stderr] ---\n{logs}\n---------------------------"
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

        Err("Timed out waiting for managed crispasr server".into())
    }
}

impl InferenceEngine for CrispAsrEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::AudioTranscription {
                audio_bytes,
                filename,
                prompt,
                language,
                temperature,
                response_format,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/audio/transcriptions", self.port);

                let mut form = reqwest::blocking::multipart::Form::new().part(
                    "file",
                    reqwest::blocking::multipart::Part::bytes(audio_bytes.clone())
                        .file_name(filename.clone()),
                );

                if let Some(p) = prompt {
                    form = form.text("prompt", p.clone());
                }
                if let Some(l) = language {
                    form = form.text("language", l.clone());
                }
                if let Some(t) = temperature {
                    form = form.text("temperature", t.to_string());
                }
                if let Some(rf) = response_format {
                    form = form.text("response_format", rf.clone());
                }

                let resp = self.client.post(&endpoint).multipart(form).send()?;
                if !resp.status().is_success() {
                    let err = resp.text().unwrap_or_default();
                    return Err(format!("crispasr transcription failed: {err}").into());
                }

                let parsed: serde_json::Value = resp.json()?;
                let text = parsed
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let transcript_resp = AudioTranscriptionResponse {
                    text: text.clone(),
                    language: parsed
                        .get("language")
                        .and_then(|l| l.as_str())
                        .map(|s| s.to_string()),
                    duration: parsed.get("duration").and_then(|d| d.as_f64()),
                    segments: parsed.get("segments").cloned(),
                    words: parsed.get("words").cloned(),
                };

                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (text.len() / 4).max(1) as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Transcription(transcript_resp),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::AudioTranslation {
                audio_bytes,
                filename,
                prompt,
                temperature,
                response_format,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/audio/translations", self.port);

                let mut form = reqwest::blocking::multipart::Form::new().part(
                    "file",
                    reqwest::blocking::multipart::Part::bytes(audio_bytes.clone())
                        .file_name(filename.clone()),
                );

                if let Some(p) = prompt {
                    form = form.text("prompt", p.clone());
                }
                if let Some(t) = temperature {
                    form = form.text("temperature", t.to_string());
                }
                if let Some(rf) = response_format {
                    form = form.text("response_format", rf.clone());
                }

                let resp = self.client.post(&endpoint).multipart(form).send()?;
                if !resp.status().is_success() {
                    let err = resp.text().unwrap_or_default();
                    return Err(format!("crispasr translation failed: {err}").into());
                }

                let parsed: serde_json::Value = resp.json()?;
                let text = parsed
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (text.len() / 4).max(1) as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Translation(AudioTranslationResponse { text }),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::AudioSpeech {
                input,
                voice,
                response_format,
                speed,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/audio/speech", self.port);

                let mut body = serde_json::json!({
                    "input": input,
                });
                if let Some(v) = voice {
                    body["voice"] = serde_json::json!(v);
                }
                if let Some(rf) = response_format {
                    body["response_format"] = serde_json::json!(rf);
                }
                if let Some(s) = speed {
                    body["speed"] = serde_json::json!(s);
                }

                let resp = self.client.post(&endpoint).json(&body).send()?;
                if !resp.status().is_success() {
                    let err = resp.text().unwrap_or_default();
                    return Err(format!("crispasr speech synthesis failed: {err}").into());
                }

                let audio_data = resp.bytes()?.to_vec();
                let prompt_tokens = (input.len() / 4).max(1) as u32;
                let completion_tokens = (audio_data.len() / 3200).max(1) as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Audio(audio_data),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::SpeechToSpeech {
                audio_bytes,
                target_language,
                source_language,
                response_format,
            } => {
                let endpoint = format!("http://127.0.0.1:{}/v1/audio/translations", self.port);

                let mut form = reqwest::blocking::multipart::Form::new().part(
                    "file",
                    reqwest::blocking::multipart::Part::bytes(audio_bytes.clone())
                        .file_name("input.wav".to_string()),
                );

                if let Some(sl) = source_language {
                    form = form.text("source_language", sl.clone());
                }
                form = form.text("target_language", target_language.clone());

                let translated_text = match self.client.post(&endpoint).multipart(form).send() {
                    Ok(resp) if resp.status().is_success() => {
                        let parsed: serde_json::Value = resp.json().unwrap_or_default();
                        parsed
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Speech translation completed.")
                            .to_string()
                    }
                    _ => "Speech translation completed.".to_string(),
                };

                let speech_endpoint = format!("http://127.0.0.1:{}/v1/audio/speech", self.port);
                let body = serde_json::json!({
                    "input": translated_text,
                    "voice": "alloy",
                    "response_format": response_format.as_deref().unwrap_or("wav")
                });

                let audio_data = match self.client.post(&speech_endpoint).json(&body).send() {
                    Ok(resp) if resp.status().is_success() => {
                        resp.bytes().map(|b| b.to_vec()).unwrap_or_default()
                    }
                    _ => Vec::new(),
                };

                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (audio_data.len() / 3200).max(1) as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Audio(audio_data),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            _ => Err("CrispAsrEngine only supports Audio tasks".into()),
        }
    }
}
