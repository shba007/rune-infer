use super::download;
use crate::config::ModelConfig;
use crate::inference::process::{
    ProcessGuard, SafeBlockingClient, configure_death_signal, http_probe_success,
};
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

fn release_asset<'a>(
    assets: &'a [serde_json::Value],
    os: &str,
    arch: &str,
    host_cuda: Option<(u32, u32)>,
) -> Option<&'a serde_json::Value> {
    let compatible = |asset: &&serde_json::Value, cuda: bool| {
        let Some(name) = asset["name"].as_str() else {
            return false;
        };
        let name = name.to_ascii_lowercase();
        let suffix = if os == "windows" { ".zip" } else { ".tar.gz" };
        let Some(stem) = name.strip_suffix(suffix) else {
            return false;
        };
        if !stem.starts_with("nemo-speech-") {
            return false;
        }
        let parts: Vec<_> = stem.split('-').collect();
        let platform_matches = match os {
            "windows" => parts.contains(&"windows") || parts.contains(&"win"),
            "linux" => parts.contains(&"linux") || parts.contains(&"ubuntu"),
            "macos" => parts.contains(&"macos") || parts.contains(&"darwin"),
            _ => false,
        };
        let arch_matches = match arch {
            "x86_64" => {
                parts
                    .iter()
                    .any(|part| matches!(*part, "x86_64" | "x64" | "amd64"))
                    && !parts
                        .iter()
                        .any(|part| matches!(*part, "aarch64" | "arm64"))
            }
            "aarch64" => {
                parts
                    .iter()
                    .any(|part| matches!(*part, "aarch64" | "arm64"))
                    && !parts
                        .iter()
                        .any(|part| matches!(*part, "x86_64" | "x64" | "amd64"))
            }
            _ => false,
        };
        let backend_matches = if cuda {
            host_cuda.is_some_and(|host| {
                parts
                    .last()
                    .is_some_and(|backend| backend.starts_with("cuda"))
                    && download::extract_cuda_version(stem).is_none_or(|required| required <= host)
            })
        } else {
            stem.ends_with("-cpu") || (os == "macos" && stem.ends_with("-metal"))
        };
        platform_matches && arch_matches && backend_matches
    };
    let cuda = if os != "macos" && host_cuda.is_some() {
        assets.iter().find(|asset| compatible(asset, true))
    } else {
        None
    };
    cuda.or_else(|| assets.iter().find(|asset| compatible(asset, false)))
}

pub struct NemoSpeechEngine {
    id: String,
    port: u16,
    client: SafeBlockingClient,
    _process: Arc<Mutex<ProcessGuard>>,
}

impl NemoSpeechEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::resolve_binary(false)?;
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

        let abs_bin_path = if bin_path.is_absolute() {
            bin_path.clone()
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.join(&bin_path)
        } else {
            bin_path.clone()
        };

        let mut cmd = Command::new(&abs_bin_path);
        configure_death_signal(&mut cmd);
        cmd.current_dir(bin_dir);

        // Ensure bin_dir and companion DLL directories are visible in PATH
        let mut search_paths = vec![bin_dir.to_path_buf()];

        if let Ok(sys_path) = std::env::var("PATH") {
            search_paths.extend(std::env::split_paths(&sys_path));
        }
        if let Ok(path_env) = std::env::join_paths(search_paths) {
            cmd.env("PATH", &path_env);
            #[cfg(windows)]
            cmd.env("Path", &path_env);
        }

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
            client: SafeBlockingClient::new(client),
            _process: guard,
        })
    }

    fn find_free_port() -> Result<u16, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    pub(crate) fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        Self::resolve_binary(true)
    }

    fn resolve_binary(install: bool) -> Result<PathBuf, Box<dyn Error>> {
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

        let base_dir = download::engine_bin().join("nemo-speech");
        let exe_path = base_dir.join(exe_name);

        // If binary is in a nested archive directory, promote it to base_dir
        if !exe_path.exists() {
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
        }

        if exe_path.exists() {
            Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
            Self::repair_dependencies(&exe_path, &base_dir);
            return Ok(exe_path);
        }

        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
            let nested_dir = nested.parent().unwrap_or(&base_dir);
            Self::flatten_all_dlls_recursive(&base_dir, nested_dir);
            Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
            Self::repair_dependencies(&nested, nested_dir);
            return Ok(nested);
        }

        let root_bin = download::engine_bin().join(exe_name);
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

        if !install {
            return Err(download::engine_unavailable("nemo-speech").into());
        }

        tracing::info!(
            "[NemoSpeechEngine] Resolving NeMo-Speech.cpp runtime from GitHub releases..."
        );
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url = "https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases";
        let release_val = download::resolve_release(&client, release_url)?;

        Self::install_release(&client, &release_val, base_dir)
    }

    // Update staging bypasses environment overrides and existing installations.
    pub(crate) fn install_release(
        client: &reqwest::blocking::Client,
        release: &serde_json::Value,
        base_dir: PathBuf,
    ) -> Result<PathBuf, Box<dyn Error>> {
        std::fs::create_dir_all(&base_dir)?;
        let exe_name = if cfg!(windows) {
            "nemo-speech.exe"
        } else {
            "nemo-speech"
        };
        let exe_path = base_dir.join(exe_name);
        let assets = download::preferred_release(release)?["assets"]
            .as_array()
            .ok_or("Selected release has no engine assets")?;

        let asset = release_asset(
            assets,
            std::env::consts::OS,
            std::env::consts::ARCH,
            download::detect_host_cuda_version(),
        )
        .ok_or_else(|| {
            format!(
                "No compatible NeMo-Speech.cpp runtime archive for {}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })?;
        let (archive_name, download_url) = download::extract_release_asset_info(asset)?;
        let sha256 = download::asset_sha256(std::slice::from_ref(asset), &archive_name)?;
        download::download_and_extract(
            client,
            &download_url,
            &base_dir,
            &archive_name,
            "[NemoSpeechEngine]",
            &sha256,
        )?;

        Self::flatten_all_dlls_recursive(&base_dir, &base_dir);
        download::flatten_dlls(&base_dir);

        if !exe_path.exists() {
            if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
                let _ = std::fs::copy(&nested, &exe_path);
            }
        }

        Self::repair_dependencies(&exe_path, &base_dir);
        download::make_executable(&exe_path)?;

        if !exe_path.exists() {
            return Err(format!("Could not locate '{}' after extraction", exe_name).into());
        }

        tracing::info!(
            "[NemoSpeechEngine] ✓ NeMo-Speech.cpp runtime installed successfully at {}",
            exe_path.display()
        );
        Ok(exe_path)
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
        let imported_dlls = match Self::get_pe_imported_dlls(exe_path) {
            Ok(list) => list,
            Err(_) => return,
        };

        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let system32 = PathBuf::from(&system_root).join("System32");

        let mut search_dirs: Vec<PathBuf> = vec![target_dir.to_path_buf()];

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

            if http_probe_success(port, "/health") {
                return Ok(());
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
                    speakers: None,
                    speaker_activity: None,
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

#[cfg(test)]
mod asset_tests {
    use super::*;

    #[test]
    fn runtime_selection_matches_platform_architecture_and_available_backend() {
        let names = [
            "nemo-speech-0.3.0-windows-x86_64-cuda.zip.sha256",
            "nemo-speech-0.3.0-windows-aarch64-cuda.zip",
            "nemo-speech-0.3.0-windows-x86_64-cuda.zip",
            "nemo-speech-0.3.0-windows-x86_64-vulkan.zip",
            "nemo-speech-0.3.0-windows-x86_64-cpu.zip",
            "nemo-speech-0.3.0-linux-aarch64-cuda13.tar.gz",
            "nemo-speech-0.3.0-linux-aarch64-cuda12.tar.gz",
            "nemo-speech-0.3.0-linux-aarch64-cpu.tar.gz",
            "nemo-speech-0.3.0-linux-x86_64-cuda.tar.gz",
            "nemo-speech-0.3.0-linux-x86_64-cpu.tar.gz",
            "nemo-speech-0.3.0-macos-x86_64-cpu.tar.gz",
            "nemo-speech-0.3.0-macos-aarch64-cpu.tar.gz",
            "itn_configs.tar.bz2",
        ];
        let assets: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({"name":name}))
            .collect();
        for (os, arch, cuda, expected) in [
            ("windows", "x86_64", Some((13, 0)), Some(2)),
            ("windows", "x86_64", None, Some(4)),
            ("windows", "aarch64", Some((13, 0)), Some(1)),
            ("windows", "aarch64", None, None),
            ("linux", "aarch64", Some((12, 4)), Some(6)),
            ("linux", "aarch64", Some((13, 0)), Some(5)),
            ("linux", "aarch64", None, Some(7)),
            ("linux", "x86_64", Some((12, 4)), Some(8)),
            ("linux", "x86_64", None, Some(9)),
            ("macos", "x86_64", None, Some(10)),
            ("macos", "aarch64", None, Some(11)),
            ("windows", "riscv64", Some((13, 0)), None),
            ("freebsd", "x86_64", None, None),
        ] {
            assert_eq!(
                release_asset(&assets, os, arch, cuda).and_then(|asset| asset["name"].as_str()),
                expected.map(|index| names[index]),
                "{os}/{arch} {cuda:?}"
            );
        }
        for invalid in [names[0], names[1], names[3], names[12]] {
            assert!(
                release_asset(
                    &[serde_json::json!({"name":invalid})],
                    "windows",
                    "x86_64",
                    None
                )
                .is_none()
            );
        }
    }
}
