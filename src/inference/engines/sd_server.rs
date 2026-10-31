use super::download;
use crate::config::ModelConfig;
use crate::inference::process::{
    ProcessGuard, SafeBlockingClient, configure_death_signal, http_probe_success,
};
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

fn valid_commit(value: &str) -> bool {
    (7..=40).contains(&value.len()) && value.bytes().all(|c| c.is_ascii_hexdigit())
}

// Some upstream binaries embed "version unknown" but still report their commit.
// Use that executable evidence rather than a filename or installed folder name.
pub(crate) fn reported_version(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let (version, commit) = line
            .trim()
            .strip_prefix("stable-diffusion.cpp version ")?
            .split_once(", commit ")?;
        let commit = commit.trim();
        (!version.trim().is_empty() && valid_commit(commit))
            .then(|| format!("commit {}", commit.to_ascii_lowercase()))
    })
}

pub(crate) fn release_commit(release: &str) -> Option<&str> {
    let (build, commit) = release.strip_prefix("master-")?.split_once('-')?;
    if build.is_empty() || !build.bytes().all(|c| c.is_ascii_digit()) || !valid_commit(commit) {
        return None;
    }
    Some(commit)
}

pub(crate) fn revision_matches(installed: &str, release: &str) -> Option<bool> {
    let installed = installed.strip_prefix("commit ")?;
    if !valid_commit(installed) {
        return None;
    }
    let commit = release_commit(release)?;
    let installed = installed.to_ascii_lowercase();
    let commit = commit.to_ascii_lowercase();
    Some(installed.starts_with(&commit) || commit.starts_with(&installed))
}

fn release_assets<'a>(
    assets: &'a [serde_json::Value],
    os: &str,
    arch: &str,
    host_cuda: Option<(u32, u32)>,
) -> Result<Vec<&'a serde_json::Value>, Box<dyn Error>> {
    let matching_arch = |part: &str| match arch {
        "x86_64" => matches!(part, "x64" | "x86_64" | "amd64"),
        "aarch64" => matches!(part, "arm64" | "aarch64"),
        _ => false,
    };
    let platform = |asset: &'a serde_json::Value| -> Option<String> {
        let name = asset["name"].as_str()?.to_ascii_lowercase();
        let stem = name
            .strip_suffix(".zip")
            .or_else(|| name.strip_suffix(".tar.gz"))?;
        let platform = stem.strip_prefix("sd-")?.split_once("-bin-")?.1;
        let parts: Vec<_> = platform.split('-').collect();
        let arches: Vec<_> = parts
            .iter()
            .filter(|part| matches!(**part, "x64" | "x86_64" | "amd64" | "arm64" | "aarch64"))
            .collect();
        if arches.len() != 1 || !matching_arch(arches[0]) {
            return None;
        }
        let compatible = match os {
            "windows" => {
                parts.len() == 3 && matches!(parts[0], "win" | "windows") && matching_arch(parts[2])
            }
            "macos" => {
                (parts.len() == 4
                    && parts[0] == "darwin"
                    && parts[1] == "macos"
                    && matching_arch(parts[3]))
                    || (parts.len() == 2 && parts[0] == "macos" && matching_arch(parts[1]))
            }
            "linux" => {
                let base = if parts.first() == Some(&"linux") && parts.get(1) == Some(&"ubuntu") {
                    // Published Ubuntu archives include the distribution version.
                    parts.get(3..)?
                } else if parts.first() == Some(&"linux") {
                    parts.get(1..)?
                } else {
                    return None;
                };
                (base.len() == 1 && matching_arch(base[0]))
                    || (base.len() == 2
                        && matching_arch(base[0])
                        && matches!(base[1], "cpu" | "vulkan"))
            }
            _ => false,
        };
        compatible.then(|| platform.to_string())
    };
    if os == "windows" {
        if let Some(host) = host_cuda {
            let mut cuda: Vec<_> = assets
                .iter()
                .filter_map(|asset| {
                    let platform = platform(asset)?;
                    platform.split('-').nth(1)?.strip_prefix("cuda")?;
                    let required = download::extract_cuda_version(&platform)?;
                    (required <= host).then_some((required, platform, asset))
                })
                .collect();
            cuda.sort_by_key(|entry| std::cmp::Reverse(entry.0));
            for (_, platform, binary) in cuda {
                let runtime = format!("cudart-sd-bin-{}.zip", platform.replacen("cuda", "cu", 1));
                if let Some(companion) = assets.iter().find(|asset| {
                    asset["name"]
                        .as_str()
                        .is_some_and(|name| name.eq_ignore_ascii_case(&runtime))
                }) {
                    // Dependencies first; an interrupted install cannot leave a bare executable.
                    return Ok(vec![companion, binary]);
                }
            }
        }
    }
    assets
        .iter()
        .find(|asset| {
            platform(asset).is_some_and(|platform| {
                os != "windows"
                    || platform
                        .split('-')
                        .nth(1)
                        .is_some_and(|backend| matches!(backend, "cpu" | "vulkan"))
            })
        })
        .map(|asset| vec![asset])
        .ok_or_else(|| {
            format!(
        "No compatible SD-server archive with verified platform and runtime pairing for {os}/{arch}"
    ).into()
        })
}

pub struct SdServerEngine {
    id: String,
    port: u16,
    client: SafeBlockingClient,
    _process: Arc<Mutex<ProcessGuard>>,
    default_steps: u32,
    default_cfg: f32,
}

impl SdServerEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let bin_path = Self::resolve_binary(false)?;
        let port = Self::find_free_port()?;

        tracing::info!(
            "[SdServerEngine] Launching managed sd-server for '{}' on port {}...",
            model.id,
            port
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
                tracing::info!("[SdServerEngine] Attaching text encoder / LLM: {}", llm);
                cmd.arg("--llm").arg(llm);
            }
        }

        if let Some(ref vae) = model.vae_path {
            if Path::new(vae).exists() {
                tracing::info!("[SdServerEngine] Attaching VAE: {}", vae);
                cmd.arg("--vae").arg(vae);
            }
        }

        if let Some(ref vision) = model.llm_vision_path {
            if Path::new(vision).exists() {
                tracing::info!("[SdServerEngine] Attaching vision tower: {}", vision);
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
                    tracing::info!("[sd-server] {}", line);
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

        tracing::info!("[SdServerEngine] Managed sd-server is ready and healthy.");

        Ok(Self {
            id: model.id.clone(),
            port,
            client: SafeBlockingClient::new(client),
            _process: guard,
            default_steps: steps,
            default_cfg: cfg,
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

        let base_dir = download::engine_bin().join("sd-server");
        let exe_path = base_dir.join(exe_name);
        if exe_path.exists() {
            return Ok(exe_path);
        }

        if let Some(nested) = download::find_executable_recursive(&base_dir, exe_name) {
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
            return Err(download::engine_unavailable("sd-server").into());
        }

        tracing::info!("[SdServerEngine] Resolving sd-server runtime from GitHub...");
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases";
        let release_val = download::resolve_release(&client, release_url)?;

        Self::install_release(&client, &release_val, base_dir)
    }

    pub(crate) fn install_release(
        client: &reqwest::blocking::Client,
        release: &serde_json::Value,
        base_dir: PathBuf,
    ) -> Result<PathBuf, Box<dyn Error>> {
        std::fs::create_dir_all(&base_dir)?;
        let exe_name = if cfg!(windows) {
            "sd-server.exe"
        } else {
            "sd-server"
        };
        let exe_path = base_dir.join(exe_name);
        let assets = download::preferred_release(release)?["assets"]
            .as_array()
            .ok_or("Selected release has no engine assets")?;

        let selected = release_assets(
            assets,
            std::env::consts::OS,
            std::env::consts::ARCH,
            download::detect_host_cuda_version(),
        )?;
        let download_urls = selected
            .into_iter()
            .map(download::extract_release_asset_info)
            .collect::<Result<Vec<_>, _>>()?;

        for (name, _) in &download_urls {
            download::asset_sha256(assets, name)?;
        }

        for (archive_name, download_url) in download_urls {
            download::download_and_extract(
                client,
                &download_url,
                &base_dir,
                &archive_name,
                "[SdServerEngine]",
                &download::asset_sha256(assets, &archive_name)?,
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

        tracing::info!(
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

            if http_probe_success(port, "/v1/models") {
                return Ok(());
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

                tracing::info!(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_packages_match_platform_and_pair_cuda_dependencies_exactly() {
        let names = [
            "sd-master-37bc88b-bin-win-cuda13-x64.zip.sha256",
            "sd-master-37bc88b-bin-win-cuda12-arm64.zip",
            "cudart-sd-bin-win-cu13-x64.zip",
            "sd-master-37bc88b-bin-win-cuda13-x64.zip",
            "cudart-sd-bin-win-cu12-x64.zip",
            "sd-master-37bc88b-bin-win-cuda12-x64.zip",
            "sd-master-37bc88b-bin-win-cpu-x64.zip",
            "sd-master-37bc88b-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
            "sd-master-37bc88b-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
            "sd-master-37bc88b-bin-Linux-Ubuntu-24.04-x86_64.zip",
            "sd-master-37bc88b-bin-Darwin-macOS-26.6.2-arm64.zip",
        ];
        let assets: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({"name":name}))
            .collect();
        for (os, arch, cuda, expected) in [
            ("windows", "x86_64", Some((13, 0)), vec![2, 3]),
            ("windows", "x86_64", Some((12, 8)), vec![4, 5]),
            ("windows", "x86_64", Some((11, 8)), vec![6]),
            ("windows", "x86_64", None, vec![6]),
            ("windows", "aarch64", Some((12, 8)), vec![]),
            ("linux", "x86_64", None, vec![8]),
            ("linux", "x86_64", Some((13, 0)), vec![8]),
            ("linux", "aarch64", None, vec![]),
            ("macos", "aarch64", None, vec![10]),
            ("macos", "x86_64", None, vec![]),
            ("windows", "riscv64", None, vec![]),
            ("android", "x86_64", None, vec![]),
        ] {
            let result = release_assets(&assets, os, arch, cuda);
            if expected.is_empty() {
                assert!(result.is_err(), "{os}/{arch}/{cuda:?}");
            } else {
                let selected: Vec<_> = result
                    .unwrap()
                    .iter()
                    .map(|asset| asset["name"].as_str().unwrap())
                    .collect();
                assert_eq!(
                    selected,
                    expected
                        .iter()
                        .map(|index| names[*index])
                        .collect::<Vec<_>>(),
                    "{os}/{arch}/{cuda:?}"
                );
            }
        }
        // No cross-version/architecture pairing; use a safe fallback or refuse.
        let missing_runtime: Vec<_> = assets
            .iter()
            .filter(|asset| asset["name"] != names[4])
            .cloned()
            .collect();
        assert_eq!(
            release_assets(&missing_runtime, "windows", "x86_64", Some((12, 8))).unwrap()[0]["name"],
            names[6]
        );
        assert!(release_assets(&assets[5..6], "windows", "x86_64", Some((12, 8))).is_err());
        let malformed = serde_json::json!({"name":"sd-master-37bc88b-bin-win-cpu-arm64-x64.zip"});
        assert!(release_assets(&[malformed], "windows", "x86_64", None).is_err());
    }

    #[test]
    fn sd_identity_uses_reported_commits_and_never_infers_release_order() {
        for (output, expected) in [
            (
                "stable-diffusion.cpp version unknown, commit be48649\r\n",
                Some("commit be48649"),
            ),
            (
                "stable-diffusion.cpp version 1.0.0, commit BE48649ABC\n",
                Some("commit be48649abc"),
            ),
            ("stable-diffusion.cpp version unknown, commit unknown", None),
            (
                "stable-diffusion.cpp version unknown, commit be48649-dirty",
                None,
            ),
            ("stable-diffusion.cpp version unknown, commit bad", None),
            ("other-engine version unknown, commit be48649", None),
            ("Usage: sd-server --version", None),
        ] {
            assert_eq!(reported_version(output).as_deref(), expected, "{output}");
        }
        for (installed, release, expected) in [
            ("commit be48649", "master-1017-be48649", Some(true)),
            ("commit be48649abc", "master-1017-BE48649", Some(true)),
            ("commit be48649", "master-1017-be48649abc", Some(true)),
            ("commit be48649", "master-1028-37bc88b", Some(false)),
            ("commit 37bc88b", "master-1017-be48649", Some(false)),
            ("Unknown", "master-1028-37bc88b", None),
            ("commit be48649", "master-bad-be48649", None),
            ("commit be48649", "master-1017-unknown", None),
            ("commit be48649", "v1.0.0", None),
        ] {
            assert_eq!(
                revision_matches(installed, release),
                expected,
                "{installed}/{release}"
            );
        }
    }
}
