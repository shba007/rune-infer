//! Shared engine inventory for the dashboard and authenticated management API.
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub(crate) fn managed_root() -> Result<PathBuf, String> {
    if let Some(root) = crate::embedding::root() {
        return Ok(root.to_path_buf());
    }
    if std::env::var_os(crate::home::HOME_ENV)
        .filter(|value| !value.is_empty())
        .is_some_and(|value| !std::path::Path::new(&value).is_absolute())
    {
        return Err("RUNE_HOME must be an absolute path".into());
    }
    crate::home::rune_home().map_err(|_| "Cannot resolve Infer's managed home".into())
}

fn override_vars(name: &str) -> &'static [&'static str] {
    match name {
        "llama-server" | "llama-server-prism" => &["LLAMA_SERVER_PATH"],
        "audio-cpp" => &["AUDIOCPP_PATH", "AUDIO8_PATH"],
        "sd-server" => &["SD_SERVER_PATH"],
        "crispasr" => &["CRISPASR_PATH"],
        "nemo-speech" => &["NEMO_SPEECH_PATH"],
        "whisper-cpp" => &["WHISPER_CPP_PATH"],
        "litert" => &["LITERT_LIBRARY_PATH"],
        "onnx" => &["ONNX_RUNTIME_LIBRARY_PATH"],
        _ => &[],
    }
}

pub(crate) fn installation_available(row: &EngineRow) -> bool {
    !row.installed
        && !override_vars(row.name)
            .iter()
            .any(|name| std::env::var_os(name).is_some())
        && install_engine_kind(row.name).is_some()
        && (!matches!(row.name, "litert" | "onnx") || cfg!(all(windows, target_arch = "x86_64")))
        && cfg!(any(
            windows,
            target_os = "linux",
            target_os = "android",
            target_os = "macos"
        ))
}

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub(crate) struct EngineRow {
    pub(crate) name: &'static str,
    pub(crate) path: PathBuf,
    pub(crate) managed: bool,
    pub(crate) installed: bool,
    pub(crate) version: String,
    pub(crate) latest: String,
    pub(crate) build_tag: Option<String>,
    pub(crate) registry_versions: Option<std::collections::BTreeMap<u64, u64>>,
    pub(crate) registry_revisions: Option<std::collections::BTreeMap<String, u64>>,
    pub(crate) published_at: Option<String>,
    pub(crate) gpu_upgrade: bool,
    pub(crate) check_detail: Option<String>,
}

pub(crate) fn discover_engines(root: &std::path::Path) -> Vec<EngineRow> {
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let definitions = [
        ("llama-server", "llama-upstream", "llama-server", true),
        ("llama-server-prism", "llama-prism", "llama-server", true),
        ("audio-cpp", "audiocpp", "audiocpp_cli", true),
        ("sd-server", "sd-server", "sd-server", true),
        ("crispasr", "crispasr", "crispasr", true),
        ("nemo-speech", "nemo-speech", "nemo-speech", true),
        ("needle", "needle", "needle", false),
        ("whisper-cpp", "whisper-cpp", "whisper-cli", true),
        ("litert", "litert", "libLiteRt", false),
        ("onnx", "onnxruntime", "onnxruntime", false),
    ];
    let mut rows: Vec<EngineRow> = definitions
        .into_iter()
        .map(|(name, directory, executable, version_supported)| {
            let filename = if name == "onnx" {
                "onnxruntime.dll".into()
            } else if name == "litert" {
                "libLiteRt.dll".into()
            } else {
                format!("{executable}{suffix}")
            };
            let custom = override_vars(name)
                .iter()
                .find_map(std::env::var_os)
                .map(PathBuf::from);
            let base = root.join("bin").join(directory);
            let direct = base.join(&filename);
            let managed = direct.is_file().then_some(direct).or_else(|| {
                crate::inference::engines::download::find_executable_recursive(&base, &filename)
            });
            let root_binary = root.join("bin").join(&filename);
            let root_binary = root_binary.is_file().then_some(root_binary);
            let candidate = if let Some(path) = custom {
                path.is_absolute().then_some((path, false))
            } else if name == "llama-server" || name == "llama-server-prism" {
                root_binary.map(|p| (p, false))
                    .or_else(|| managed.map(|p| (p, true)))
            } else {
                managed.map(|p| (p, true)).or_else(|| {
                    if matches!(name, "needle" | "litert" | "onnx") {
                        None
                    } else {
                        root_binary.map(|p| (p, false))
                    }
                })
            };
            let (path, managed, installed) = match candidate {
                Some((path, managed)) => {
                    let installed = path.is_file();
                    (path, managed, installed)
                }
                None => (base.join(&filename), matches!(name, "litert" | "onnx"), false),
            };
            let version = if name == "onnx" && installed {
                crate::inference::engines::onnx::runtime::runtime_version(&path)
            } else if name == "litert" && installed {
                crate::inference::engines::litert::runtime_version(&path)
            } else if name == "needle" && installed && managed {
                crate::inference::engines::needle::installed_revision(&path)
            } else if version_supported && managed {
                probe_engine_version(&path).and_then(|version| {
                    if name == "sd-server" {
                        crate::inference::engines::sd_server::reported_version(&version)
                    } else {
                        Some(version)
                    }
                })
            } else {
                None
            }
            .unwrap_or_else(|| {
                if installed {
                    "Unknown".into()
                } else {
                    "Not installed".into()
                }
            });
            EngineRow {
                name,
                path,
                managed,
                installed,
                version,
                latest: if name == "onnx" {
                    "1.31.1 · pinned".into()
                } else if name == "litert" {
                    "2.3.0 · pinned".into()
                } else if managed || !installed {
                    "Checking…".into()
                } else {
                    "External".into()
                },
                build_tag: None,
                registry_versions: None,
registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: match name {
                    "litert" => Some("Pinned Windows x64 CPU + WebGPU/DXC runtime · GPU requires a parity-verified model export".into()),
                    "onnx" => Some("Pinned Windows x64 ONNX Runtime + CUDA 12/cuDNN · approximately 1.7 GB download".into()),
                    _ => None,
                },
            }
        })
        .collect();
    rows.push(EngineRow {
        name: "encoder",
        path: "Built into Rune Infer".into(),
        managed: false,
        installed: true,
        version: "Built in".into(),
        latest: "Built in".into(),
        build_tag: None,
        registry_versions: None,
        registry_revisions: None,
        published_at: None,
        gpu_upgrade: false,
        check_detail: Some("Built-in encoder uses synthetic outputs".into()),
    });
    rows
}

fn probe_engine_version(path: &std::path::Path) -> Option<String> {
    use std::process::{Command, Stdio};
    let mut command = Command::new(path);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                use std::io::Read;
                let mut stdout = String::new();
                let mut stderr = String::new();
                child
                    .stdout
                    .take()?
                    .take(4096)
                    .read_to_string(&mut stdout)
                    .ok()?;
                child
                    .stderr
                    .take()?
                    .take(4096)
                    .read_to_string(&mut stderr)
                    .ok()?;
                return parse_engine_version(&stdout, &stderr);
            }
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if start.elapsed() >= Duration::from_secs(3) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(40)),
        }
    }
}

pub(crate) fn parse_engine_version(stdout: &str, stderr: &str) -> Option<String> {
    let lines: Vec<_> = stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines
        .iter()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim().eq_ignore_ascii_case("version") && !value.trim().is_empty() {
                Some(value.trim().to_owned())
            } else {
                None
            }
        })
        .or_else(|| lines.first().map(|line| (*line).to_owned()))
}

pub(crate) fn install_engine_kind(name: &str) -> Option<crate::config::EngineKind> {
    use crate::config::EngineKind;
    Some(match name {
        "llama-server" => EngineKind::LlamaServer,
        "llama-server-prism" => EngineKind::LlamaServerPrism,
        "audio-cpp" => EngineKind::AudioCpp,
        "sd-server" => EngineKind::SdServer,
        "crispasr" => EngineKind::Crispasr,
        "nemo-speech" => EngineKind::NemoSpeech,
        "needle" => EngineKind::Needle,
        "whisper-cpp" => EngineKind::WhisperCpp,
        "litert" => EngineKind::Litert,
        "onnx" => EngineKind::Onnx,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_managed_engine_under_the_supplied_root() {
        let root = std::env::temp_dir().join(format!(
            "rune-infer-inventory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let folder = root.join("bin").join("needle");
        std::fs::create_dir_all(&folder).unwrap();
        let file = folder.join(if cfg!(windows) {
            "needle.exe"
        } else {
            "needle"
        });
        std::fs::write(&file, b"fixture").unwrap();
        let engines = discover_engines(&root);
        let needle = engines
            .iter()
            .find(|engine| engine.name == "needle")
            .unwrap();
        assert!(needle.installed && needle.managed);
        assert_eq!(needle.path, file);
        std::fs::remove_dir_all(root).unwrap();
    }
}
