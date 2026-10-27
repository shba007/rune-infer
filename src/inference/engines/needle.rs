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

        let bin_path = Self::ensure_binary_installed()?;

        Ok(Self {
            id,
            bin_path,
            model_path,
        })
    }

    pub(crate) fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
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

        tracing::info!("[NeedleEngine] Needle 3 binary not found. Resolving from Hugging Face...");
        std::fs::create_dir_all(&base_dir)?;

        let platform = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("windows", "x86_64") => "windows-x86_64",
            ("windows", "aarch64") => "windows-arm64",
            ("linux", "x86_64") => "linux-x86_64",
            ("linux", "aarch64") => "linux-arm64",
            ("linux", "arm") => "linux-armv7",
            ("linux", "riscv64") => "linux-riscv64",
            ("macos", "aarch64") => "macos-arm64",
            ("macos", "x86_64") => "macos-arm64",
            _ => "linux-x86_64",
        };

        let client = download::create_download_client()?;
        let metadata: serde_json::Value = client
            .get("https://huggingface.co/api/models/Cactus-Compute/needle3?blobs=true")
            .send()?
            .error_for_status()?
            .json()?;
        let revision = metadata["sha"]
            .as_str()
            .filter(|sha| sha.len() == 40 && sha.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or("Needle metadata has no immutable revision; installation refused.")?;
        let file = format!("{platform}/{exe_name}");
        let sha256 = metadata["siblings"]
            .as_array()
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|e| e["rfilename"].as_str() == Some(&file))
            })
            .and_then(|entry| entry["lfs"]["sha256"].as_str())
            .ok_or("Needle metadata has no SHA-256 for this platform; installation refused.")?;
        let download_url =
            format!("https://huggingface.co/Cactus-Compute/needle3/resolve/{revision}/{file}");
        download::download_verified(&client, &download_url, &exe_path, "[NeedleEngine]", sha256)?;
        download::make_executable(&exe_path)?;

        tracing::info!(
            "[NeedleEngine] ✓ Needle 3 installed at {}",
            exe_path.display()
        );
        Ok(exe_path)
    }
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
