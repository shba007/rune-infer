use super::download;
use crate::inference::process::configure_death_signal;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::Usage;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct NeedleEngine {
    id: String,
    bin_path: PathBuf,
    model_path: PathBuf,
}

impl NeedleEngine {
    pub fn new(id: String, model_path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.exists() {
            return Err(format!("Model path does not exist: {}", model_path.display()).into());
        }

        let bin_path = Self::resolve_binary(false)?;

        Ok(Self {
            id,
            bin_path,
            model_path,
        })
    }

    pub(crate) fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        Self::resolve_binary(true)
    }

    fn resolve_binary(install: bool) -> Result<PathBuf, Box<dyn Error>> {
        let base_dir = download::engine_bin().join("needle");
        let exe_name = if cfg!(windows) {
            "needle.exe"
        } else {
            "needle"
        };
        let exe_path = base_dir.join(exe_name);

        if exe_path.exists() {
            return Ok(exe_path);
        }

        if !install {
            return Err(download::engine_unavailable("needle").into());
        }

        tracing::info!("[NeedleEngine] Needle 3 binary not found. Resolving from Hugging Face...");
        std::fs::create_dir_all(&base_dir)?;

        platform_directory(std::env::consts::OS, std::env::consts::ARCH)?;

        let client = download::create_download_client()?;
        let metadata = latest_metadata(&client)?;
        Self::install_release(&client, &metadata, base_dir)
    }

    pub(crate) fn install_release(
        client: &reqwest::blocking::Client,
        metadata: &serde_json::Value,
        base_dir: PathBuf,
    ) -> Result<PathBuf, Box<dyn Error>> {
        std::fs::create_dir_all(&base_dir)?;
        let platform = platform_directory(std::env::consts::OS, std::env::consts::ARCH)?;
        let exe_name = if cfg!(windows) {
            "needle.exe"
        } else {
            "needle"
        };
        let exe_path = base_dir.join(exe_name);
        let file = format!("{platform}/{exe_name}");
        let (revision, sha256) = verified_binary_metadata(metadata, &file)?;
        let download_url =
            format!("https://huggingface.co/Cactus-Compute/needle3/resolve/{revision}/{file}");
        download::download_verified(client, &download_url, &exe_path, "[NeedleEngine]", sha256)?;
        download::make_executable(&exe_path)?;
        let receipt = serde_json::json!({"revision":revision, "sha256":sha256, "file":file});
        let mut receipt_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(base_dir.join(".needle-install.json"))?;
        use std::io::Write;
        receipt_file.write_all(&serde_json::to_vec(&receipt)?)?;

        tracing::info!(
            "[NeedleEngine] ✓ Needle 3 installed at {}",
            exe_path.display()
        );
        Ok(exe_path)
    }
}

pub(crate) fn latest_metadata(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(client
        .get("https://huggingface.co/api/models/Cactus-Compute/needle3?blobs=true")
        .send()?
        .error_for_status()?
        .json()?)
}

#[cfg(feature = "tui")]
pub(crate) fn verify_forward_update(
    client: &reqwest::blocking::Client,
    installed: &str,
    selected: &str,
) -> Result<(), Box<dyn Error>> {
    if !valid_revision(installed) || !valid_revision(selected) {
        return Err("Invalid Needle revision; update refused".into());
    }
    let history: serde_json::Value = client
        .get(format!(
            "https://huggingface.co/api/models/Cactus-Compute/needle3/commits/{selected}"
        ))
        .send()?
        .error_for_status()?
        .json()?;
    verify_revision_history(&history, installed, selected)
}

#[cfg(any(feature = "tui", test))]
fn verify_revision_history(
    history: &serde_json::Value,
    installed: &str,
    selected: &str,
) -> Result<(), Box<dyn Error>> {
    let entries = history
        .as_array()
        .ok_or("Invalid Needle revision history")?;
    let has_revision = |revision: &str| {
        entries.iter().any(|entry| {
            entry["id"]
                .as_str()
                .is_some_and(|id| id.eq_ignore_ascii_case(revision))
        })
    };
    if !valid_revision(installed)
        || !valid_revision(selected)
        || installed.eq_ignore_ascii_case(selected)
        || !entries
            .iter()
            .all(|entry| entry["id"].as_str().is_some_and(valid_revision))
        || !has_revision(selected)
        || !has_revision(installed)
    {
        return Err(
            "Available Needle revision history cannot prove a forward update; update refused"
                .into(),
        );
    }
    Ok(())
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|c| c.is_ascii_hexdigit())
}

pub(crate) fn platform_directory(os: &str, arch: &str) -> Result<&'static str, Box<dyn Error>> {
    match (os, arch) {
        ("windows", "x86_64") => Ok("windows-x86_64"),
        ("windows", "aarch64") => Ok("windows-arm64"),
        ("linux", "x86_64") => Ok("linux-x86_64"),
        ("linux", "aarch64") => Ok("linux-arm64"),
        ("linux", "riscv64") => Ok("linux-riscv64"),
        ("android", "aarch64") => Ok("android-arm64"),
        ("android", "riscv64") => Ok("android-riscv64"),
        ("macos", "aarch64") => Ok("macos-arm64"),
        _ => Err(format!("No verified Needle binary is available for {os}/{arch}").into()),
    }
}

fn verified_binary_metadata<'a>(
    metadata: &'a serde_json::Value,
    file: &str,
) -> Result<(&'a str, &'a str), Box<dyn Error>> {
    let revision = metadata["sha"]
        .as_str()
        .filter(|v| valid_revision(v))
        .ok_or("Needle metadata has no immutable revision; installation refused")?;
    let digest = metadata["siblings"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .find(|item| item["rfilename"].as_str() == Some(file))
        })
        .and_then(|item| item["lfs"]["sha256"].as_str())
        .filter(|v| v.len() == 64 && v.bytes().all(|c| c.is_ascii_hexdigit()))
        .ok_or("Needle metadata has no valid SHA-256 for this platform; installation refused")?;
    Ok((revision, digest))
}

#[cfg(any(feature = "tui", test))]
pub(crate) fn installed_revision(binary: &Path) -> Option<String> {
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(binary.parent()?.join(".needle-install.json")).ok()?)
            .ok()?;
    let revision = receipt["revision"].as_str().filter(|v| valid_revision(v))?;
    let digest = receipt["sha256"].as_str()?;
    (download::file_sha256(binary)
        .ok()?
        .eq_ignore_ascii_case(digest))
    .then(|| format!("revision {revision}"))
}

impl InferenceEngine for NeedleEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ToolCall { prompt, schema, .. } => {
                let schema_str = schema.to_string();
                let temp_tools_path = std::env::temp_dir().join(format!(
                    "needle_tools_{}_{}.json",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                ));

                std::fs::write(&temp_tools_path, &schema_str)?;

                let mut cmd = Command::new(&self.bin_path);
                configure_death_signal(&mut cmd);
                cmd.arg("--model")
                    .arg(&self.model_path)
                    .arg("--tools")
                    .arg(&temp_tools_path)
                    .arg("--prompt")
                    .arg(prompt);

                let output = cmd.output();
                let _ = std::fs::remove_file(&temp_tools_path);

                let output = output.map_err(|e| format!("Failed to run Needle binary: {e}"))?;
                let raw_output = String::from_utf8_lossy(&output.stdout).trim().to_string();

                let prompt_tokens = ((prompt.len() + schema_str.len()) / 4).max(1) as u32;
                let completion_tokens = (raw_output.len() / 4).max(1) as u32;

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw_output) {
                    if let Some(calls) = val.get("function_calls").filter(|c| c.is_array()) {
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: None,
                                tool_calls: calls.clone(),
                            },
                            usage: Usage::new(prompt_tokens, completion_tokens),
                        });
                    } else if val.is_array() || val.is_object() {
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall {
                                content: None,
                                tool_calls: val,
                            },
                            usage: Usage::new(prompt_tokens, completion_tokens),
                        });
                    }
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Text(raw_output),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            _ => Err("NeedleEngine only supports ToolCall tasks".into()),
        }
    }
}

#[cfg(test)]
mod install_tests {
    use super::*;
    #[test]
    fn needle_history_requires_selected_revision_and_verified_ancestor() {
        let installed = "a".repeat(40);
        let selected = "b".repeat(40);
        let history = serde_json::json!([{"id":selected},{"id":installed}]);
        assert!(verify_revision_history(&history, &installed, &selected).is_ok());
        // Date ordering is not used to infer ancestry: this is a revision-scoped history.
        assert!(
            verify_revision_history(
                &serde_json::json!([{"id":installed},{"id":selected}]),
                &installed,
                &selected
            )
            .is_ok()
        );
        for invalid in [
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::json!([{"id":selected}]),
            serde_json::json!([{"id":installed}]),
            serde_json::json!([{"id":selected},{"id":"main"},{"id":installed}]),
        ] {
            assert!(verify_revision_history(&invalid, &installed, &selected).is_err());
        }
        assert!(verify_revision_history(&history, &installed, &installed).is_err());
        assert!(verify_revision_history(&history, "main", &selected).is_err());
    }

    #[test]
    fn missing_engine_returns_an_actionable_error_without_creating_installation() {
        let root = std::env::temp_dir().join(format!("rune-needle-missing-{}", std::process::id()));
        let bin = root.join("bin");
        std::fs::create_dir_all(&root).unwrap();
        let error = download::with_install_bin(bin.clone(), || {
            NeedleEngine::new("test".into(), &root)
                .err()
                .unwrap()
                .to_string()
        });
        assert!(error.contains("Engine 'needle' is not available"));
        assert!(error.contains("dashboard"));
        assert!(!bin.exists());
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn platform_mapping_refuses_unavailable_binaries() {
        assert_eq!(
            platform_directory("windows", "x86_64").unwrap(),
            "windows-x86_64"
        );
        assert_eq!(
            platform_directory("macos", "aarch64").unwrap(),
            "macos-arm64"
        );
        assert!(platform_directory("macos", "x86_64").is_err());
        assert!(platform_directory("unknown", "unknown").is_err());
    }
    #[test]
    fn metadata_requires_matching_platform_immutable_revision_and_sha256() {
        let file = "windows-x86_64/needle.exe";
        let mut metadata = serde_json::json!({"sha":"a".repeat(40), "siblings":[{"rfilename":file,"lfs":{"sha256":"b".repeat(64)}}]});
        assert!(verified_binary_metadata(&metadata, file).is_ok());
        assert!(verified_binary_metadata(&metadata, "macos-x86_64/needle").is_err());
        metadata["sha"] = serde_json::json!("main");
        assert!(verified_binary_metadata(&metadata, file).is_err());
        metadata["sha"] = serde_json::json!("a".repeat(40));
        metadata["siblings"][0]["lfs"]["sha256"] = serde_json::json!("invalid");
        assert!(verified_binary_metadata(&metadata, file).is_err());
    }
    #[test]
    fn installed_revision_is_reported_only_when_binary_matches_receipt() {
        let root = std::env::temp_dir().join(format!(
            "rune-needle-receipt-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        let binary = root.join("needle.exe");
        std::fs::write(&binary, "fixture").unwrap();
        let digest = download::file_sha256(&binary).unwrap();
        std::fs::write(
            root.join(".needle-install.json"),
            serde_json::to_vec(&serde_json::json!({"revision":"a".repeat(40),"sha256":digest}))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            installed_revision(&binary),
            Some(format!("revision {}", "a".repeat(40)))
        );
        std::fs::write(&binary, "changed").unwrap();
        assert_eq!(installed_revision(&binary), None);
        std::fs::remove_dir_all(root).unwrap();
    }
}
