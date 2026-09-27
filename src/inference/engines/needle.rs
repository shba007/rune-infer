use crate::inference::process::configure_death_signal;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::Usage;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

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

    fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        let base_dir = PathBuf::from("bin").join("needle");
        let exe_name = if cfg!(windows) {
            "needle.exe"
        } else {
            "needle"
        };
        let exe_path = base_dir.join(exe_name);

        if exe_path.exists() {
            return Ok(exe_path);
        }

        println!("[NeedleEngine] Needle 3 binary not found. Resolving from Hugging Face...");
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

        let download_url = format!(
            "https://huggingface.co/Cactus-Compute/needle3/resolve/main/{}/{}",
            platform, exe_name
        );

        println!("[NeedleEngine] Downloading from {}...", download_url);
        let client = reqwest::blocking::Client::builder()
            .user_agent("rune-infer/0.1.0")
            .timeout(Duration::from_secs(180))
            .build()?;

        let mut resp = client.get(&download_url).send()?.error_for_status()?;
        let mut file = std::fs::File::create(&exe_path)?;
        std::io::copy(&mut resp, &mut file)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(&exe_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&exe_path, perms);
            }
        }

        println!(
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
                            response: InferenceTaskResponse::ToolCall(calls.clone()),
                            usage: Usage::new(prompt_tokens, completion_tokens),
                        });
                    } else if val.is_array() || val.is_object() {
                        return Ok(InferenceOutput {
                            response: InferenceTaskResponse::ToolCall(val),
                            usage: Usage::new(prompt_tokens, completion_tokens),
                        });
                    }
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Text(raw_output),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
        }
    }
}
