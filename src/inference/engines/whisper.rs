use super::download;
use crate::config::ModelConfig;
use crate::inference::process::configure_death_signal;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{AudioTranscriptionResponse, AudioTranslationResponse, Usage};
use std::{
    error::Error,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

const RELEASES_URL: &str = "https://api.github.com/repos/ggml-org/whisper.cpp/releases";

pub(crate) fn resolve_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    super::release_cache::resolved_release(
        &format!(
            "whisper:{}/{}:{RELEASES_URL}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
        || {
            let releases = download::fetch_releases(client, RELEASES_URL)?;
            resolve_release_with(&releases, |tag| {
                download::fetch_release_tag(client, RELEASES_URL, tag)
            })
        },
    )
}

fn stable_build_tag(release: &serde_json::Value) -> Result<Option<String>, Box<dyn Error>> {
    let mut tag = None;
    for line in release["body"].as_str().unwrap_or("").lines() {
        let Some(link) = line
            .trim()
            .strip_prefix("**Nightly build:**")
            .or_else(|| line.trim().strip_prefix("Nightly build:"))
        else {
            continue;
        };
        let (label, url) = link
            .trim()
            .strip_prefix('[')
            .and_then(|link| link.split_once("]("))
            .ok_or("Invalid Whisper stable release build link")?;
        let url = url
            .strip_suffix(')')
            .ok_or("Invalid Whisper stable release build link")?;
        if tag.is_some()
            || label.strip_prefix('b').is_none_or(|number| {
                number.is_empty()
                    || number.len() > 20
                    || !number.bytes().all(|byte| byte.is_ascii_digit())
            })
            || url != format!("https://github.com/ggml-org/whisper.cpp/releases/tag/{label}")
        {
            return Err("Invalid or conflicting Whisper stable release build link".into());
        }
        tag = Some(label.to_owned());
    }
    Ok(tag)
}

fn resolve_release_with(
    releases: &serde_json::Value,
    mut fetch_build: impl FnMut(&str) -> Result<serde_json::Value, Box<dyn Error>>,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let releases = releases.as_array().ok_or("Invalid Whisper release list")?;
    let has_stable = releases.iter().any(download::is_stable_release);
    for release in releases {
        if !download::is_published_release(release)
            || (has_stable && !download::is_stable_release(release))
        {
            continue;
        }
        let mut selected = release.clone();
        if platform_asset(&selected).is_none() && download::is_stable_release(&selected) {
            if let Some(tag) = stable_build_tag(&selected)? {
                let build = fetch_build(&tag)?;
                let commit = selected["target_commitish"].as_str().filter(|value| {
                    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
                if commit.is_none()
                    || build["target_commitish"].as_str() != commit
                    || !download::is_published_release(&build)
                    || build["tag_name"].as_str() != Some(&tag)
                {
                    return Err(
                        "Whisper binary build does not match the stable release commit".into(),
                    );
                }
                selected["assets"] = build["assets"].clone();
                selected["rune_build_tag"] = tag.into();
                selected["rune_build_published_at"] = build["published_at"].clone();
            }
        }
        if platform_asset(&selected).is_some() {
            return Ok(selected);
        }
    }
    Err("No verified whisper-cli release asset is available for this platform".into())
}

pub struct WhisperCppEngine {
    id: String,
    binary: PathBuf,
    model: PathBuf,
    no_gpu: bool,
}
impl WhisperCppEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            id: model.id.clone(),
            binary: std::fs::canonicalize(Self::resolve_binary(false)?)?,
            model: std::fs::canonicalize(&model.model_path)?,
            no_gpu: model.runtime.as_ref().is_some_and(|r| r.gpu_layers == 0),
        })
    }

    pub(crate) fn ensure_binary_installed() -> Result<PathBuf, Box<dyn Error>> {
        Self::resolve_binary(true)
    }

    fn resolve_binary(install: bool) -> Result<PathBuf, Box<dyn Error>> {
        if let Ok(path) = std::env::var("WHISPER_CPP_PATH") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Ok(path);
            }
        }
        let executable = if cfg!(windows) {
            "whisper-cli.exe"
        } else {
            "whisper-cli"
        };
        let base = download::engine_bin().join("whisper-cpp");
        let direct = base.join(executable);
        if direct.is_file() {
            return Ok(direct);
        }
        if let Some(path) = download::find_executable_recursive(&base, executable) {
            return Ok(path);
        }
        if let Some(path) = std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|p| p.join(executable))
                .find(|p| p.is_file())
        }) {
            return Ok(path);
        }
        if !install {
            return Err(download::engine_unavailable("whisper-cpp").into());
        }

        let client = download::create_download_client()?;
        let release = resolve_release(&client)?;
        let asset = platform_asset(&release)
            .ok_or("No verified whisper-cli release asset is available for this platform")?;
        let (archive, url) = download::extract_release_asset_info(asset)?;
        let digest = download::asset_sha256(std::slice::from_ref(asset), &archive)?;
        std::fs::create_dir_all(&base)?;
        download::download_and_extract(&client, &url, &base, &archive, "[WhisperCpp]", &digest)?;
        download::flatten_dlls(&base);
        let binary = if direct.is_file() {
            direct
        } else {
            download::find_executable_recursive(&base, executable)
                .ok_or("Verified archive does not contain whisper-cli")?
        };
        download::make_executable(&binary)?;
        Ok(binary)
    }
}
pub(crate) fn platform_asset(release: &serde_json::Value) -> Option<&serde_json::Value> {
    let name = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "whisper-bin-x64.zip",
        ("linux", "x86_64") => "whisper-bin-ubuntu-x64.tar.gz",
        ("linux", "aarch64") => "whisper-bin-ubuntu-arm64.tar.gz",
        _ => return None,
    };
    release["assets"].as_array()?.iter().find(|a| {
        a["name"].as_str() == Some(name)
            && download::asset_sha256(std::slice::from_ref(a), name).is_ok()
    })
}
impl InferenceEngine for WhisperCppEngine {
    fn id(&self) -> &str {
        &self.id
    }
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        let (bytes, prompt, language, temperature, translate) = match task {
            InferenceTaskRequest::AudioTranscription {
                audio_bytes,
                prompt,
                language,
                temperature,
                ..
            } => (audio_bytes, prompt, language.as_deref(), temperature, false),
            InferenceTaskRequest::AudioTranslation {
                audio_bytes,
                prompt,
                temperature,
                ..
            } => (audio_bytes, prompt, None, temperature, true),
            _ => {
                return Err(
                    "whisper.cpp supports speech transcription and translation only".into(),
                );
            }
        };
        if !is_pcm_wav_16k(bytes) {
            return Err("whisper.cpp requires a 16 kHz mono 16-bit PCM WAV input; convert the audio and retry".into());
        }
        let temp = RequestDirectory::new()?;
        let input = temp.0.join("input.wav");
        let output = temp.0.join("transcript");
        std::fs::write(&input, bytes)?;
        let mut command = Command::new(&self.binary);
        configure_death_signal(&mut command);
        command
            .arg("--model")
            .arg(&self.model)
            .arg("--file")
            .arg(&input)
            .arg("--output-json")
            .arg("--output-file")
            .arg(&output)
            .arg("--language")
            .arg(language.unwrap_or("auto"));
        if let Some(value) = prompt {
            command.arg("--prompt").arg(value);
        }
        if let Some(value) = temperature {
            command.arg("--temperature").arg(value.to_string());
        }
        if translate {
            command.arg("--translate");
        }
        if self.no_gpu {
            command.arg("--no-gpu");
        }
        let result = command.output()?;
        if !result.status.success() {
            return Err(format!(
                "whisper.cpp inference failed: {}",
                String::from_utf8_lossy(&result.stderr)
            )
            .into());
        }
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.with_extension("json"))?)?;
        let (text, segments) = parse_transcript(&json)?;
        let response = if translate {
            InferenceTaskResponse::Translation(AudioTranslationResponse { text })
        } else {
            InferenceTaskResponse::Transcription(AudioTranscriptionResponse {
                text,
                language: json["result"]["language"].as_str().map(str::to_string),
                duration: None,
                segments: Some(serde_json::Value::Array(segments)),
                words: None,
                speakers: None,
                speaker_activity: None,
            })
        };
        Ok(InferenceOutput {
            response,
            usage: Usage::default(),
        })
    }
}

fn parse_transcript(
    json: &serde_json::Value,
) -> Result<(String, Vec<serde_json::Value>), Box<dyn Error>> {
    let entries = json["transcription"]
        .as_array()
        .ok_or("whisper.cpp returned no transcription")?;
    let mut text = String::new();
    let mut segments = Vec::new();
    for (id, entry) in entries.iter().enumerate() {
        let part = entry["text"]
            .as_str()
            .ok_or("whisper.cpp returned a malformed segment")?;
        text.push_str(part);
        let start = entry["offsets"]["from"]
            .as_f64()
            .ok_or("whisper.cpp returned a segment without timestamps")?
            / 1000.0;
        let end = entry["offsets"]["to"]
            .as_f64()
            .ok_or("whisper.cpp returned a segment without timestamps")?
            / 1000.0;
        segments.push(serde_json::json!({"id": id, "start": start, "end": end, "text": part}));
    }
    Ok((text.trim().to_string(), segments))
}

fn is_pcm_wav_16k(bytes: &[u8]) -> bool {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return false;
    }
    let mut pos = 12usize;
    while let Some(header) = bytes.get(pos..pos.saturating_add(8)) {
        let len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let Some(end) = pos.checked_add(8).and_then(|p| p.checked_add(len)) else {
            return false;
        };
        if end > bytes.len() {
            return false;
        }
        if &header[..4] == b"fmt " && len >= 16 {
            let fmt = &bytes[pos + 8..end];
            return u16::from_le_bytes([fmt[0], fmt[1]]) == 1
                && u16::from_le_bytes([fmt[2], fmt[3]]) == 1
                && u32::from_le_bytes(fmt[4..8].try_into().unwrap()) == 16000
                && u16::from_le_bytes([fmt[14], fmt[15]]) == 16;
        }
        pos = end.saturating_add(len % 2);
    }
    false
}
struct RequestDirectory(PathBuf);
impl RequestDirectory {
    fn new() -> Result<Self, Box<dyn Error>> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "rune-whisper-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err("Cannot create a unique whisper.cpp request directory".into())
    }
}
impl Drop for RequestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolves_verified_stable_builds_and_legacy_packages() {
        let asset = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("windows", "x86_64") => "whisper-bin-x64.zip",
            ("linux", "x86_64") => "whisper-bin-ubuntu-x64.tar.gz",
            ("linux", "aarch64") => "whisper-bin-ubuntu-arm64.tar.gz",
            _ => return,
        };
        let releases = serde_json::json!([
            {"tag_name":"v2.0.0", "draft":false, "prerelease":false, "assets":[]},
            {"tag_name":"v1.9.0-rc1", "draft":false, "prerelease":true, "assets":[{"name":asset,"digest":format!("sha256:{}", "a".repeat(64))}]},
            {"tag_name":"v1.8.0", "draft":false, "prerelease":false, "assets":[{"name":asset,"digest":format!("sha256:{}", "b".repeat(64))}]}
        ]);
        assert_eq!(
            resolve_release_with(&releases, |_| panic!("legacy releases need no build fetch"))
                .unwrap()["tag_name"],
            "v1.8.0"
        );
        let commit = "a".repeat(40);
        let stable = serde_json::json!({"tag_name":"v1.9.5","draft":false,"prerelease":false,
            "target_commitish":commit,"assets":[],
            "body":"**Nightly build:** [b5454](https://github.com/ggml-org/whisper.cpp/releases/tag/b5454)"});
        let build = serde_json::json!({"tag_name":"b5454","draft":false,"prerelease":true,
            "target_commitish":commit,"published_at":"2026-10-06T17:00:00Z",
            "assets":[{"name":asset,"digest":format!("sha256:{}", "b".repeat(64))}]});
        let selected = resolve_release_with(&serde_json::json!([stable.clone()]), |tag| {
            assert_eq!(tag, "b5454");
            Ok(build.clone())
        })
        .unwrap();
        assert_eq!(selected["tag_name"], "v1.9.5");
        assert_eq!(selected["rune_build_tag"], "b5454");
        assert_eq!(platform_asset(&selected).unwrap()["name"], asset);
        for (field, value) in [
            ("target_commitish", serde_json::json!("b".repeat(40))),
            ("tag_name", serde_json::json!("b9999")),
            ("draft", serde_json::json!(true)),
        ] {
            let mut invalid = build.clone();
            invalid[field] = value;
            assert!(
                resolve_release_with(
                    &serde_json::json!([stable.clone()]),
                    |_| Ok(invalid.clone())
                )
                .is_err(),
                "{field}"
            );
        }
        for body in [
            "**Nightly build:** [b5454](https://example.com/releases/tag/b5454)",
            "**Nightly build:** [b5454](https://github.com/ggml-org/llama.cpp/releases/tag/b5454)",
            "**Nightly build:** [b5454/other](https://github.com/ggml-org/whisper.cpp/releases/tag/b5454/other)",
            "**Nightly build:** [b5454](https://github.com/ggml-org/whisper.cpp/releases/tag/b9999)",
            "**Nightly build:** invalid",
            "**Nightly build:** [b5454](https://github.com/ggml-org/whisper.cpp/releases/tag/b5454)\n**Nightly build:** [b5454](https://github.com/ggml-org/whisper.cpp/releases/tag/b5454)",
        ] {
            let mut invalid = stable.clone();
            invalid["body"] = body.into();
            assert!(
                resolve_release_with(&serde_json::json!([invalid]), |_| panic!(
                    "invalid links must not be fetched"
                ))
                .is_err(),
                "{body}"
            );
        }
        let mut unsigned = selected.clone();
        unsigned["assets"][0]["digest"] = serde_json::Value::Null;
        assert!(platform_asset(&unsigned).is_none());
    }

    #[test]
    #[ignore = "reads official GitHub release metadata; does not download or install engine packages"]
    fn live_stable_release_metadata() {
        let release = resolve_release(&download::create_download_client().unwrap()).unwrap();
        assert!(download::is_stable_release(&release));
        let asset = platform_asset(&release).unwrap();
        let (name, _) = download::extract_release_asset_info(asset).unwrap();
        let digest = download::asset_sha256(std::slice::from_ref(asset), &name).unwrap();
        println!(
            "Whisper stable {} · binary build {} · asset {name} · SHA-256 {digest}",
            release["tag_name"], release["rune_build_tag"]
        );
    }

    #[test]
    fn cli_json_is_converted_to_api_text_and_timestamps() {
        let raw = serde_json::json!({"result":{"language":"en"},"transcription":[
            {"text":" Hello", "offsets":{"from":250,"to":1500}},
            {"text":" world", "offsets":{"from":1500,"to":2000}}
        ]});
        let (text, segments) = parse_transcript(&raw).unwrap();
        assert_eq!(text, "Hello world");
        assert_eq!(segments[0]["start"], 0.25);
        assert_eq!(segments[1]["end"], 2.0);
        assert!(parse_transcript(&serde_json::json!({"transcription":[{"text":"bad"}]})).is_err());
    }

    #[test]
    fn validates_wav_format_instead_of_accepting_any_riff() {
        let mut wav = vec![0u8; 44];
        wav[..4].copy_from_slice(b"RIFF");
        wav[8..12].copy_from_slice(b"WAVE");
        wav[12..16].copy_from_slice(b"fmt ");
        wav[16..20].copy_from_slice(&16u32.to_le_bytes());
        wav[20..22].copy_from_slice(&1u16.to_le_bytes());
        wav[22..24].copy_from_slice(&1u16.to_le_bytes());
        wav[24..28].copy_from_slice(&16000u32.to_le_bytes());
        wav[34..36].copy_from_slice(&16u16.to_le_bytes());
        assert!(is_pcm_wav_16k(&wav));
        wav[22] = 2;
        assert!(!is_pcm_wav_16k(&wav));
        assert!(!is_pcm_wav_16k(b"RIFF"));
    }
}
