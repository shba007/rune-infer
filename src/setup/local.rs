//! Local setup installation and sequential, real-model inference checks.
use super::LocalCandidate;
use crate::config::{EngineKind, Modality, ModelConfig, ServerConfig};
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use base64::Engine;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone)]
struct CheckedFile {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
}
impl CheckedFile {
    fn read(path: &Path) -> Result<Self, String> {
        let meta = path
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if !meta.is_file() || meta.len() == 0 {
            return Err(format!("{} must be a nonempty file", path.display()));
        }
        Ok(Self {
            path: path.to_path_buf(),
            len: meta.len(),
            modified: meta.modified().map_err(|e| e.to_string())?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PreparedLocal {
    pub engines: Vec<(EngineKind, PathBuf)>,
    models: Vec<serde_json::Value>,
    files: Vec<CheckedFile>,
    gpu: bool,
}
impl PreparedLocal {
    /// Refuse stale verification if selections or checked files changed before save.
    pub fn validate(&self, models: &[LocalCandidate], gpu: bool) -> Result<(), String> {
        if self.gpu != gpu || models.iter().any(|m| m.engines.len() != 1) {
            return Err("Local engine settings changed; run the model checks again.".into());
        }
        let choices = planned_models(models, gpu)?;
        if choices != self.models {
            return Err("Local model selections changed; run the checks again.".into());
        }
        for file in &self.files {
            let now = CheckedFile::read(&file.path)?;
            if file.len != now.len || file.modified != now.modified {
                return Err(format!(
                    "{} changed; run the model check again.",
                    file.path.display()
                ));
            }
        }
        Ok(())
    }
}

fn planned_models(models: &[LocalCandidate], gpu: bool) -> Result<Vec<serde_json::Value>, String> {
    models
        .iter()
        .map(|model| {
            serde_json::to_value(super::local_entry(model, gpu)).map_err(|e| e.to_string())
        })
        .collect()
}

pub enum LocalPreparationEvent {
    Progress(String),
    Finished(Result<PreparedLocal, String>),
}

pub fn validate_local_files(model: &LocalCandidate) -> Result<(), String> {
    if !model.needs.is_empty() {
        return Err(format!(
            "{} needs extra.{} before it can be checked",
            model.display_name,
            model.needs.join(", extra.")
        ));
    }
    for path in model_files(model) {
        CheckedFile::read(&path)?;
    }
    Ok(())
}
fn model_files(model: &LocalCandidate) -> Vec<PathBuf> {
    let mut files = vec![model.path.clone()];
    for (key, value) in &model.extra {
        if key.ends_with("_path") {
            if let Some(path) = value.as_str() {
                files.push(PathBuf::from(path));
            }
        }
    }
    files
}

pub fn prepare_local_models(
    models: &[LocalCandidate],
    gpu: bool,
    progress: impl FnMut(String),
) -> Result<PreparedLocal, String> {
    prepare_local_models_with(
        models,
        gpu,
        super::install_engine,
        |model| smoke_model(model, gpu),
        progress,
    )
}

pub(crate) fn prepare_local_models_with(
    models: &[LocalCandidate],
    gpu: bool,
    mut install: impl FnMut(EngineKind) -> Result<PathBuf, String>,
    mut smoke: impl FnMut(&LocalCandidate) -> Result<(), String>,
    mut progress: impl FnMut(String),
) -> Result<PreparedLocal, String> {
    let mut report = PreparedLocal {
        engines: Vec::new(),
        models: Vec::new(),
        files: Vec::new(),
        gpu,
    };
    // Validate the entire selection before installing anything.
    for model in models {
        if model.engines.len() != 1 || model.engine() == EngineKind::Encoder {
            return Err("Choose one real inference engine per model.".into());
        }
        validate_local_files(model)?;
        smoke_task(&model.modality)?;
        ModelConfig::try_from(super::local_entry(model, gpu)).map_err(|e| e.to_string())?;
        for path in model_files(model) {
            report.files.push(CheckedFile::read(&path)?);
        }
    }
    for model in models {
        let engine = model.engine();
        if !report.engines.iter().any(|(kind, _)| *kind == engine) {
            progress(format!("Installing or locating {engine}"));
            let path = install(engine)?;
            report.files.push(CheckedFile::read(&path)?);
            report.engines.push((engine, path));
        }
    }
    for (i, model) in models.iter().enumerate() {
        progress(format!(
            "Checking {}/{}: {} · {}",
            i + 1,
            models.len(),
            model.display_name,
            model.engine()
        ));
        smoke(model).map_err(|e| format!("{} · {}: {e}", model.display_name, model.engine()))?;
    }
    report.models = planned_models(models, gpu)?;
    report.validate(models, gpu)?;
    Ok(report)
}

fn smoke_model(model: &LocalCandidate, gpu: bool) -> Result<(), String> {
    let mut config =
        ModelConfig::try_from(super::local_entry(model, gpu)).map_err(|e| e.to_string())?;
    // Internal check id avoids legacy mock-name behavior without changing the saved id.
    config.id = "rune-setup-check".into();
    let task = smoke_task(&model.modality)?;
    let engine: std::sync::Arc<dyn crate::inference::traits::InferenceEngine> =
        if model.engine() == EngineKind::AudioCpp {
            std::sync::Arc::new(
                crate::inference::engines::audio::AudioEngine::new_for_setup(&config)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            crate::inference::registry::ModelRegistry::create_engine(
                &config,
                &ServerConfig::default(),
                crate::inference::registry::ModelRegistry::detect_used_gpu_vram(),
            )
            .map_err(|e| e.to_string())?
            .0
        };
    let output = engine.execute(&task, None).map_err(|e| e.to_string())?;
    // Drop the only engine reference, unloading this model before the next check.
    drop(engine);
    validate_smoke_output(&model.modality, &output.response)
}

fn smoke_task(modality: &Modality) -> Result<InferenceTaskRequest, String> {
    use serde_json::json;
    let value = match modality {
        Modality::Text | Modality::VisionText | Modality::Omni | Modality::Decision => {
            let mut images = Vec::new();
            if matches!(modality, Modality::VisionText | Modality::Omni) {
                let image = image::RgbImage::from_pixel(32, 32, image::Rgb([220, 20, 20]));
                let mut bytes = std::io::Cursor::new(Vec::new());
                image
                    .write_to(&mut bytes, image::ImageFormat::Png)
                    .map_err(|e| e.to_string())?;
                images.push(bytes.into_inner());
            }
            json!({"type":"tool_call","prompt":if images.is_empty() {"Reply with OK."} else {"Name the color in the image."},"schema":[],"images":images,"audios":[],"messages":[],"max_tokens":32,"temperature":0,"thinking":false,"chat_template_kwargs":{"enable_thinking":false}})
        }
        Modality::Embedding => json!({"type":"embedding","input":["Rune setup check"]}),
        Modality::ImageGeneration => {
            json!({"type":"image_generation","prompt":"A red square","size":"512x512","steps":1,"seed":0,"response_format":"b64_json"})
        }
        Modality::SpeechToText => {
            json!({"type":"audio_transcription","audio_bytes":silence_wav(),"filename":"setup.wav","response_format":"json"})
        }
        Modality::TextToSpeech => {
            json!({"type":"audio_speech","input":"Hello.","response_format":"wav"})
        }
        _ => {
            return Err(format!(
                "No real setup inference check is available for {}",
                modality.as_str()
            ));
        }
    };
    serde_json::from_value(value).map_err(|e| format!("Cannot build setup check: {e}"))
}

fn silence_wav() -> Vec<u8> {
    let size = 16000u32 * 2;
    let mut wav = b"RIFF".to_vec();
    wav.extend((36 + size).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(16000u32.to_le_bytes());
    wav.extend(32000u32.to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend(size.to_le_bytes());
    wav.resize(44 + size as usize, 0);
    wav
}

fn validate_smoke_output(
    modality: &Modality,
    response: &InferenceTaskResponse,
) -> Result<(), String> {
    let valid = match (modality, response) {
        (
            Modality::Text | Modality::VisionText | Modality::Omni | Modality::Decision,
            InferenceTaskResponse::Text(text),
        ) => !text.trim().is_empty(),
        (
            Modality::Text | Modality::VisionText | Modality::Omni | Modality::Decision,
            InferenceTaskResponse::ToolCall {
                content,
                tool_calls,
            },
        ) => {
            content.as_ref().is_some_and(|s| !s.trim().is_empty())
                || tool_calls.as_array().is_some_and(|a| !a.is_empty())
        }
        (Modality::Embedding, InferenceTaskResponse::Embedding(result)) => result
            .data
            .iter()
            .any(|e| !e.embedding.is_empty() && e.embedding.iter().all(|n| n.is_finite())),
        (Modality::SpeechToText, InferenceTaskResponse::Transcription(_)) => true, // silence legitimately has no transcript
        (Modality::TextToSpeech, InferenceTaskResponse::Audio(bytes)) => {
            bytes.len() > 44 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE")
        }
        (Modality::ImageGeneration, InferenceTaskResponse::Image(result)) => {
            result.data.iter().any(|i| {
                i.b64_json
                    .as_ref()
                    .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok())
                    .is_some_and(|b| image::load_from_memory(&b).is_ok())
            })
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err("Engine did not return a usable response for the selected model.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::testing::Gguf;

    fn fixture() -> (PathBuf, Vec<LocalCandidate>, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rune-local-check-{}",
            crate::setup::generate_api_key()
        ));
        Gguf::llm("llama", "First").write(&root.join("first.gguf"));
        Gguf::llm("llama", "Second").write(&root.join("second.gguf"));
        let binary = root.join("fixture-engine");
        std::fs::write(&binary, "fixture executable").unwrap();
        let models = crate::setup::discover_models(&root).unwrap().models;
        (root, models, binary)
    }

    #[test]
    fn installs_shared_engine_once_checks_each_model_and_rejects_stale_proof() {
        let (root, models, binary) = fixture();
        let mut installations = 0;
        let mut checked = Vec::new();
        let proof = prepare_local_models_with(
            &models,
            false,
            |_| {
                installations += 1;
                Ok(binary.clone())
            },
            |m| {
                checked.push(m.path.clone());
                Ok(())
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(installations, 1);
        assert_eq!(
            checked,
            models.iter().map(|m| m.path.clone()).collect::<Vec<_>>()
        );
        proof.validate(&models, false).unwrap();
        assert!(proof.validate(&models, true).is_err());
        let mut changed = models.clone();
        changed[0].engines = vec![EngineKind::LlamaServerPrism];
        assert!(proof.validate(&changed, false).is_err());
        std::fs::write(&models[0].path, "changed model").unwrap();
        assert!(proof.validate(&models, false).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_assets_or_synthetic_engines_never_install_and_smoke_failures_never_pass() {
        let (root, mut models, binary) = fixture();
        let mut installs = 0;
        models[0].extra.insert(
            "mmproj_path".into(),
            root.join("missing-projector")
                .to_string_lossy()
                .to_string()
                .into(),
        );
        assert!(
            prepare_local_models_with(
                &models,
                false,
                |_| {
                    installs += 1;
                    Ok(binary.clone())
                },
                |_| Ok(()),
                |_| {}
            )
            .is_err()
        );
        assert_eq!(installs, 0);
        models[0].extra.clear();
        models[0].engines = vec![EngineKind::Encoder];
        assert!(
            prepare_local_models_with(
                &models,
                false,
                |_| panic!("must not install synthetic engine"),
                |_| Ok(()),
                |_| {}
            )
            .is_err()
        );
        models[0].engines = vec![EngineKind::LlamaServer];
        let mut checked = 0;
        let error = prepare_local_models_with(
            &models,
            false,
            |_| Ok(binary.clone()),
            |_| {
                checked += 1;
                Err("load failed".into())
            },
            |_| {},
        )
        .unwrap_err();
        assert!(error.contains("load failed"));
        assert_eq!(checked, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn chosen_engine_is_exactly_one_and_must_be_compatible() {
        let (root, mut models, _) = fixture();
        models.truncate(1);
        models[0].engines.push(EngineKind::LlamaServerPrism);
        let chosen =
            crate::setup::select_local_engines(&models, &[EngineKind::LlamaServerPrism]).unwrap();
        assert_eq!(chosen[0].engines, vec![EngineKind::LlamaServerPrism]);
        assert!(crate::setup::select_local_engines(&models, &[EngineKind::AudioCpp]).is_err());
        assert!(crate::setup::select_local_engines(&models, &[]).is_err());
        assert!(crate::setup::select_local_engines(&models, &[EngineKind::Encoder]).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn smoke_inputs_cover_discovered_modalities_and_require_valid_outputs() {
        for modality in [
            Modality::Text,
            Modality::VisionText,
            Modality::Omni,
            Modality::Decision,
            Modality::Embedding,
            Modality::ImageGeneration,
            Modality::SpeechToText,
            Modality::TextToSpeech,
        ] {
            let task = smoke_task(&modality).unwrap();
            if matches!(modality, Modality::VisionText | Modality::Omni) {
                let InferenceTaskRequest::ToolCall { images, .. } = task else {
                    panic!("vision needs image check")
                };
                assert_eq!(images.len(), 1);
                assert!(image::load_from_memory(&images[0]).is_ok());
            }
            assert!(
                validate_smoke_output(&modality, &InferenceTaskResponse::Error("failed".into()))
                    .is_err()
            );
        }
        assert_eq!(silence_wav().len(), 32044);
        assert!(
            validate_smoke_output(&Modality::Text, &InferenceTaskResponse::Text(String::new()))
                .is_err()
        );
        assert!(
            validate_smoke_output(&Modality::Text, &InferenceTaskResponse::Text("OK".into()))
                .is_ok()
        );
        assert!(
            validate_smoke_output(
                &Modality::TextToSpeech,
                &InferenceTaskResponse::Audio(vec![0; 100])
            )
            .is_err()
        );
    }
}
