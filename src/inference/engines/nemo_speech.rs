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

pub struct NemoSpeechEngine {
    id: String,
    port: u16,
    client: reqwest::blocking::Client,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl NemoSpeechEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::ensure_binary_installed()?;
        let port = Self::find_free_port()?;
        let bin_dir = bin_path
            .parent()
            .unwrap_or_else(|| Path::new("bin/nemo-speech"));

        // Inspect PE imports and auto-resolve any missing runtime DLLs
        Self::repair_dependencies(&bin_path, bin_dir);

        tracing::info!(
            "[NemoSpeechEngine] Launching managed nemo-speech for '{}' on port {}...",
            model.id,
            port
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
        cmd.env("PATH", &path_env);
        cmd.env("Path", &path_env);

        // Launch in server mode with loaded ASR model
        cmd.arg("serve")
            .arg("--asr-model")
            .arg(&model.model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
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
                "Failed to spawn nemo-speech binary at '{}': {e}",
                bin_path.display()
            )
        })?;

        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let last_output_lines = Arc::new(Mutex::new(Vec::new()));

        if let Some(pipe) = stdout_pipe {
            let lines_clone = last_output_lines.clone();
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    tracing::info!(target: "nemo_speech", "[nemo-speech stdout] {}", line);
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 50 {
                        l.remove(0);
                    }
                    l.push(format!("[stdout] {line}"));
                }
            });
        }

        if let Some(pipe) = stderr_pipe {
            let lines_clone = last_output_lines.clone();
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    tracing::info!(target: "nemo_speech", "[nemo-speech stderr] {}", line);
                    let mut l = lines_clone.lock().unwrap();
                    if l.len() >= 50 {
                        l.remove(0);
                    }
                    l.push(format!("[stderr] {line}"));
                }
            });
        }

        let guard = ProcessGuard::new(child, format!("nemo-speech-{}", model.id));
        Self::wait_for_server(port, &guard, &last_output_lines)?;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        tracing::info!("[NemoSpeechEngine] Managed nemo-speech server is ready and healthy.");

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
            "nemo-speech.exe"
        } else {
            "nemo-speech"
        };

        if let Ok(custom) = std::env::var("NEMO_SPEECH_PATH") {
            let p = PathBuf::from(custom);
            if p.exists() {
                return Ok(p);
            }
        }

        let base_dir = PathBuf::from("bin").join("nemo-speech");
        let exe_path = base_dir.join(exe_name);

        // If binary is in a nested archive directory, promote it to base_dir
        if !exe_path.exists() {
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
        }

        if exe_path.exists() {
            Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
            Self::copy_companion_cuda_dlls(&base_dir);
            Self::repair_dependencies(&exe_path, &base_dir);
            return Ok(exe_path);
        }

        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            let nested_dir = nested.parent().unwrap_or(&base_dir);
            Self::flatten_all_dlls_recursive(&base_dir, nested_dir);
            Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
            Self::copy_companion_cuda_dlls(nested_dir);
            Self::repair_dependencies(&nested, nested_dir);
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

        println!("[NemoSpeechEngine] Resolving NeMo-Speech.cpp runtime from GitHub releases...");
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url = "https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases?per_page=5";
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
                    "[NemoSpeechEngine] Direct GitHub API request failed ({err}). Falling back to release tag redirect..."
                );
                let html_url = "https://github.com/NVIDIA/NeMo-Speech.cpp/releases";
                let resp = client.get(html_url).send().map_err(|e| {
                    format!(
                        "Could not resolve NeMo-Speech.cpp assets: {e}.\n\
                        Please download nemo-speech manually and place it at '{}' or set NEMO_SPEECH_PATH in .env.",
                        exe_path.display()
                    )
                })?;
                let final_url = resp.url().as_str();
                let tag = final_url
                    .split("/tag/")
                    .nth(1)
                    .ok_or_else(|| format!("Could not determine tag from URL: {final_url}"))?;
                println!("[NemoSpeechEngine] Resolved latest tag: {tag}");
                let api_fallback = format!(
                    "https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases/tags/{tag}"
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
                .ok_or("No release assets found in NeMo-Speech.cpp")?
        } else {
            release_val["assets"]
                .as_array()
                .ok_or("Invalid release payload: missing assets")?
        };

        let host_cuda = download::detect_host_cuda_version();
        let mut download_urls = Vec::new();

        if cfg!(target_os = "windows") {
            let cuda_asset = if host_cuda.is_some() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    (n.contains("bin-win-cuda")
                        || n.contains("win-cuda")
                        || (n.contains("windows") && n.contains("cuda")))
                        && !n.contains("cudart")
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
                    n.contains("linux-cuda") || n.contains("ubuntu-cuda")
                })
            } else {
                None
            };

            let fallback = assets.iter().find(|a| {
                let n = a["name"].as_str().unwrap_or("").to_lowercase();
                n.contains("linux") || n.contains("ubuntu")
            });

            if let Some(target) = cuda_asset.or(fallback) {
                download_urls.push((
                    target["name"].as_str().unwrap().to_string(),
                    target["browser_download_url"].as_str().unwrap().to_string(),
                ));
            }
        }

        if download_urls.is_empty() {
            return Err(
                "No compatible NeMo-Speech.cpp release asset found for this platform".into(),
            );
        }

        for (archive_name, download_url) in download_urls {
            download::download_and_extract(
                &client,
                &download_url,
                &base_dir,
                &archive_name,
                "[NemoSpeechEngine]",
            )?;
        }

        Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
        download::flatten_dlls(&base_dir);
        Self::copy_companion_cuda_dlls(&base_dir);
        Self::repair_dependencies(&exe_path, &base_dir);

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
            "[NemoSpeechEngine] ✓ NeMo-Speech.cpp runtime installed successfully at {}",
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
                "bin/crispasr",
                "bin/audiocpp",
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

    fn flatten_all_dlls_recursive(src_dir: &Path, dest_dir: &Path) {
        if let Ok(entries) = std::fs::read_dir(src_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext.eq_ignore_ascii_case("dll") {
                            if let Some(fname) = path.file_name() {
                                let dest = dest_dir.join(fname);
                                if path != dest {
                                    let _ = std::fs::copy(&path, &dest);
                                }
                            }
                        }
                    }
                } else if path.is_dir() && path != dest_dir {
                    Self::flatten_all_dlls_recursive(&path, dest_dir);
                }
            }
        }
    }

    #[cfg(windows)]
    fn repair_dependencies(exe_path: &Path, target_dir: &Path) {
        Self::copy_companion_cuda_dlls(target_dir);

        let imported_dlls = match Self::get_pe_imported_dlls(exe_path) {
            Ok(list) => list,
            Err(_) => return,
        };

        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let system32 = PathBuf::from(&system_root).join("System32");

        let mut search_dirs: Vec<PathBuf> = vec![
            PathBuf::from("bin/llama-upstream"),
            PathBuf::from("bin/llama-prism"),
            PathBuf::from("bin/sd-server"),
            PathBuf::from("bin/crispasr"),
            PathBuf::from("bin/audiocpp"),
            PathBuf::from("bin"),
        ];

        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            search_dirs.push(PathBuf::from(cuda_path).join("bin"));
        }
        let cuda_base = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
        if let Ok(entries) = std::fs::read_dir(cuda_base) {
            for entry in entries.flatten() {
                let bin = entry.path().join("bin");
                if bin.exists() {
                    search_dirs.push(bin);
                }
            }
        }

        for dll in imported_dlls {
            let dest_file = target_dir.join(&dll);
            if dest_file.exists() || system32.join(&dll).exists() {
                continue;
            }

            let mut found = false;
            for s_dir in &search_dirs {
                let candidate = s_dir.join(&dll);
                if candidate.exists() {
                    if std::fs::copy(&candidate, &dest_file).is_ok() {
                        tracing::info!(
                            "[NemoSpeechEngine] Resolved dependency '{}' from '{}'",
                            dll,
                            s_dir.display()
                        );
                        found = true;
                        break;
                    }
                }
            }

            if !found && !dll.to_lowercase().starts_with("api-ms-win") {
                tracing::warn!(
                    "[NemoSpeechEngine] Warning: Missing dependency '{}' not found in companion or system directories",
                    dll
                );
            }
        }
    }

    #[cfg(not(windows))]
    fn repair_dependencies(_exe_path: &Path, _target_dir: &Path) {}

    #[cfg(windows)]
    fn get_pe_imported_dlls(exe_path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
        let bytes = std::fs::read(exe_path)?;
        if bytes.len() < 0x40 || &bytes[0..2] != b"MZ" {
            return Ok(Vec::new());
        }
        let pe_offset = u32::from_le_bytes(bytes[0x3C..0x40].try_into()?) as usize;
        if bytes.len() < pe_offset + 24 || &bytes[pe_offset..pe_offset + 4] != b"PE\0\0" {
            return Ok(Vec::new());
        }

        let num_sections =
            u16::from_le_bytes(bytes[pe_offset + 6..pe_offset + 8].try_into()?) as usize;
        let opt_hdr_size =
            u16::from_le_bytes(bytes[pe_offset + 20..pe_offset + 22].try_into()?) as usize;
        let opt_hdr_offset = pe_offset + 24;

        if bytes.len() < opt_hdr_offset + opt_hdr_size {
            return Ok(Vec::new());
        }

        let magic = u16::from_le_bytes(bytes[opt_hdr_offset..opt_hdr_offset + 2].try_into()?);
        let import_dir_offset = if magic == 0x20B {
            opt_hdr_offset + 120
        } else if magic == 0x10B {
            opt_hdr_offset + 104
        } else {
            return Ok(Vec::new());
        };

        if bytes.len() < import_dir_offset + 8 {
            return Ok(Vec::new());
        }

        let import_rva =
            u32::from_le_bytes(bytes[import_dir_offset..import_dir_offset + 4].try_into()?);
        if import_rva == 0 {
            return Ok(Vec::new());
        }

        let sections_offset = opt_hdr_offset + opt_hdr_size;
        let mut sections = Vec::new();
        for i in 0..num_sections {
            let sec_offset = sections_offset + i * 40;
            if bytes.len() < sec_offset + 40 {
                break;
            }
            let virtual_size =
                u32::from_le_bytes(bytes[sec_offset + 8..sec_offset + 12].try_into()?);
            let virtual_address =
                u32::from_le_bytes(bytes[sec_offset + 12..sec_offset + 16].try_into()?);
            let raw_data_size =
                u32::from_le_bytes(bytes[sec_offset + 16..sec_offset + 20].try_into()?);
            let raw_data_ptr =
                u32::from_le_bytes(bytes[sec_offset + 20..sec_offset + 24].try_into()?);
            sections.push((virtual_address, virtual_size, raw_data_size, raw_data_ptr));
        }

        let rva_to_file_offset = |rva: u32| -> Option<usize> {
            for &(va, vs, rs, ptr) in &sections {
                let span = vs.max(rs);
                if rva >= va && rva < va + span {
                    let delta = rva - va;
                    if delta < rs {
                        return Some((ptr + delta) as usize);
                    }
                }
            }
            None
        };

        let import_file_offset = match rva_to_file_offset(import_rva) {
            Some(offset) => offset,
            None => return Ok(Vec::new()),
        };

        let mut dll_names = Vec::new();
        let mut desc_offset = import_file_offset;

        while bytes.len() >= desc_offset + 20 {
            let name_rva =
                u32::from_le_bytes(bytes[desc_offset + 12..desc_offset + 16].try_into()?);
            let first_thunk =
                u32::from_le_bytes(bytes[desc_offset + 16..desc_offset + 20].try_into()?);

            if name_rva == 0 && first_thunk == 0 {
                break;
            }

            if let Some(name_file_offset) = rva_to_file_offset(name_rva) {
                let mut name_bytes = Vec::new();
                let mut curr = name_file_offset;
                while curr < bytes.len() && bytes[curr] != 0 {
                    name_bytes.push(bytes[curr]);
                    curr += 1;
                }
                if let Ok(name_str) = String::from_utf8(name_bytes) {
                    if !name_str.trim().is_empty() {
                        dll_names.push(name_str);
                    }
                }
            }
            desc_offset += 20;
        }

        Ok(dll_names)
    }

    fn wait_for_server(
        port: u16,
        guard_mutex: &Arc<Mutex<ProcessGuard>>,
        output_lines: &Arc<Mutex<Vec<String>>>,
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
                    let logs = output_lines.lock().unwrap().join("\n");
                    tracing::error!(
                        target: "nemo_speech",
                        "nemo-speech terminated prematurely with {status}:\n{logs}"
                    );
                    return Err(format!(
                        "nemo-speech terminated prematurely with {status}\n--- [nemo-speech output] ---\n{logs}\n---------------------------"
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

        Err("Timed out waiting for managed nemo-speech server".into())
    }
}

impl InferenceEngine for NemoSpeechEngine {
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
                    return Err(format!("nemo-speech transcription failed: {err}").into());
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
                    return Err(format!("nemo-speech translation failed: {err}").into());
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
            _ => Err(
                "NemoSpeechEngine only supports Audio transcription and translation tasks".into(),
            ),
        }
    }
}
