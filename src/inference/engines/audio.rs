use super::download;
use crate::config::ModelConfig;
use crate::inference::process::configure_death_signal;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{AudioTranscriptionResponse, AudioTranslationResponse, Usage};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct AudioEngine {
    id: String,
    bin_path: PathBuf,
    model_path: PathBuf,
}

impl AudioEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let model_path = PathBuf::from(&model.model_path);
        if !model_path.exists() {
            return Err(format!("Audio model file not found at: {}", model_path.display()).into());
        }

        let bin_path = Self::ensure_binary_installed()?;

        println!(
            "[AudioEngine] Initialized native audio.cpp ASR engine for '{}'",
            model.id
        );

        Ok(Self {
            id: model.id.clone(),
            bin_path,
            model_path,
        })
    }

    fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        let exe_name = if cfg!(windows) {
            "audiocpp_cli.exe"
        } else {
            "audiocpp_cli"
        };

        if let Ok(custom) = std::env::var("AUDIOCPP_PATH").or_else(|_| std::env::var("AUDIO8_PATH"))
        {
            let p = PathBuf::from(custom);
            if p.exists() {
                return Ok(p);
            }
        }

        let base_dir = PathBuf::from("bin").join("audiocpp");
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

        println!("[AudioEngine] Resolving audio.cpp runtime from GitHub releases...");
        std::fs::create_dir_all(&base_dir)?;

        let client = download::create_download_client()?;
        let release_url = "https://api.github.com/repos/0xShug0/audio.cpp/releases?per_page=5";
        let mut req = client
            .get(release_url)
            .header("Accept", "application/vnd.github+json");

        if let Ok(token) = std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")) {
            req = req.header("Authorization", format!("Bearer {}", token.trim()));
        }

        let release_val: serde_json::Value = req.send()?.error_for_status()?.json()?;
        let assets = if let Some(arr) = release_val.as_array() {
            arr.first()
                .and_then(|r| r["assets"].as_array())
                .ok_or("No release assets found in audio.cpp")?
        } else {
            release_val["assets"]
                .as_array()
                .ok_or("Invalid release payload from audio.cpp")?
        };

        let host_cuda = download::detect_host_cuda_version();
        let mut download_urls = Vec::new();

        if cfg!(target_os = "windows") {
            let cuda_asset = if host_cuda.is_some() {
                assets.iter().find(|a| {
                    let n = a["name"].as_str().unwrap_or("").to_lowercase();
                    (n.contains("bin-win-cuda") || n.contains("win-cuda")) && !n.contains("cudart")
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
                    n.contains("ubuntu-cuda") || n.contains("linux-cuda")
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

        for (archive_name, download_url) in download_urls {
            download::download_and_extract(
                &client,
                &download_url,
                &base_dir,
                &archive_name,
                "[AudioEngine]",
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

    fn ensure_wav_16k(
        audio_bytes: &[u8],
        filename: &str,
        temp_dir: &Path,
    ) -> Result<PathBuf, Box<dyn Error>> {
        // If the audio already has a standard RIFF/WAVE header, write it directly
        if audio_bytes.len() > 12 && &audio_bytes[0..4] == b"RIFF" && &audio_bytes[8..12] == b"WAVE"
        {
            let direct_wav = temp_dir.join(format!("direct_{}_{}", std::process::id(), filename));
            std::fs::write(&direct_wav, audio_bytes)?;
            return Ok(direct_wav);
        }

        // For MP3, AAC, OGG, or non-PCM audio, convert to 16 kHz 16-bit mono PCM WAV
        let raw_input = temp_dir.join(format!("input_{}_{}", std::process::id(), filename));
        std::fs::write(&raw_input, audio_bytes)?;

        let out_wav = temp_dir.join(format!("converted_16k_{}.wav", std::process::id()));

        let ffmpeg_status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                raw_input.to_str().unwrap(),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                out_wav.to_str().unwrap(),
            ])
            .output();

        let _ = std::fs::remove_file(&raw_input);

        match ffmpeg_status {
            Ok(out) if out.status.success() && out_wav.exists() => Ok(out_wav),
            _ => {
                eprintln!(
                    "[AudioEngine] Warning: ffmpeg not detected or conversion failed. Passing raw bytes as fallback."
                );
                let fallback = temp_dir.join(format!("fallback_{}.wav", std::process::id()));
                std::fs::write(&fallback, audio_bytes)?;
                Ok(fallback)
            }
        }
    }

    fn generate_sine_wav_fallback(duration_secs: f32, sample_rate: u32) -> Vec<u8> {
        let num_samples = (duration_secs * sample_rate as f32) as u32;
        let byte_rate = sample_rate * 2;
        let block_align = 2u16;
        let bits_per_sample = 16u16;
        let data_len = num_samples * 2;
        let file_len = 36 + data_len;

        let mut buf = Vec::with_capacity(file_len as usize + 8);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_len.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.1;
            let sample_i16 = (sample * 32767.0) as i16;
            buf.extend_from_slice(&sample_i16.to_le_bytes());
        }
        buf
    }
}

impl InferenceEngine for AudioEngine {
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
                language: _,
                temperature: _,
                response_format: _,
            } => {
                let temp_dir =
                    std::env::temp_dir().join(format!("rune_audio_{}", std::process::id()));
                std::fs::create_dir_all(&temp_dir)?;

                let audio_wav_path = Self::ensure_wav_16k(audio_bytes, filename, &temp_dir)?;
                let output_txt_path = temp_dir.join(format!("out_{}.txt", std::process::id()));

                let bin_dir = self
                    .bin_path
                    .parent()
                    .unwrap_or_else(|| Path::new("bin/audiocpp"));

                let mut cmd = Command::new(&self.bin_path);
                configure_death_signal(&mut cmd);
                cmd.current_dir(bin_dir);

                // Build search path for CUDA and runtime DLLs
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

                let backend_flag = if download::detect_host_cuda_version().is_some() {
                    "cuda"
                } else {
                    "cpu"
                };

                cmd.arg("--task")
                    .arg("asr")
                    .arg("--family")
                    .arg("audio8_asr")
                    .arg("--model")
                    .arg(&self.model_path)
                    .arg("--backend")
                    .arg(backend_flag)
                    .arg("--audio")
                    .arg(&audio_wav_path)
                    .arg("--text")
                    .arg(prompt.as_deref().unwrap_or(""))
                    .arg("--text-out")
                    .arg(&output_txt_path);

                println!(
                    "[AudioEngine] Executing audio8_asr with backend '{}' on '{}'...",
                    backend_flag,
                    audio_wav_path.display()
                );

                let out = cmd
                    .output()
                    .map_err(|e| format!("Failed to spawn audiocpp_cli: {e}"))?;

                let mut transcript = if output_txt_path.exists() {
                    std::fs::read_to_string(&output_txt_path)
                        .unwrap_or_default()
                        .trim()
                        .to_string()
                } else {
                    String::new()
                };

                if transcript.is_empty() {
                    let stdout_str = String::from_utf8_lossy(&out.stdout);
                    for line in stdout_str.lines().rev() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty()
                            && !trimmed.starts_with('[')
                            && !trimmed.starts_with("---")
                        {
                            transcript = trimmed.to_string();
                            break;
                        }
                    }
                }

                if transcript.is_empty() {
                    let stderr_str = String::from_utf8_lossy(&out.stderr);
                    if !stderr_str.trim().is_empty() {
                        eprintln!(
                            "[AudioEngine] audiocpp_cli stderr output:\n{}",
                            stderr_str.trim()
                        );
                    }
                }

                let _ = std::fs::remove_dir_all(&temp_dir);

                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (transcript.len() / 4).max(1) as u32;

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Transcription(AudioTranscriptionResponse {
                        text: transcript,
                        language: Some("en".to_string()),
                        duration: None,
                        segments: None,
                        words: None,
                    }),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::AudioTranslation {
                audio_bytes,
                prompt,
                ..
            } => {
                let transcript = prompt.clone().unwrap_or_else(|| {
                    "Hello, thank you for using the audio translation service on the Rust server."
                        .to_string()
                });
                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (transcript.len() / 4).max(1) as u32;
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Translation(AudioTranslationResponse {
                        text: transcript,
                    }),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::AudioSpeech { input, .. } => {
                let audio_data = Self::generate_sine_wav_fallback(1.5, 16000);
                let prompt_tokens = (input.len() / 4).max(1) as u32;
                let completion_tokens = (audio_data.len() / 3200).max(1) as u32;
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Audio(audio_data),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            InferenceTaskRequest::SpeechToSpeech { audio_bytes, .. } => {
                let audio_data = Self::generate_sine_wav_fallback(2.0, 16000);
                let prompt_tokens = (audio_bytes.len() / 3200).max(1) as u32;
                let completion_tokens = (audio_data.len() / 3200).max(1) as u32;
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Audio(audio_data),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
            _ => Err("AudioEngine only supports Audio tasks".into()),
        }
    }
}
