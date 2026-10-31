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
    tasks: Vec<Vec<String>>,
}
impl PreparedLocal {
    pub(super) fn entries(
        &self,
        models: &[LocalCandidate],
        gpu: bool,
    ) -> Result<Vec<crate::schema::ModelEntry>, String> {
        self.validate(models, gpu)?;
        if self.tasks.len() != models.len() {
            return Err("Incomplete local verification; run the checks again.".into());
        }
        Ok(models
            .iter()
            .zip(&self.tasks)
            .map(|(model, tasks)| {
                let mut entry = super::local_entry(model, gpu);
                entry.tasks = tasks.clone();
                entry
            })
            .collect())
    }
    /// Refuse stale verification if selections or checked files changed before save.
    pub fn validate(&self, models: &[LocalCandidate], gpu: bool) -> Result<(), String> {
        if self.gpu != gpu || models.iter().any(|m| m.engines.is_empty()) {
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
    Download(crate::inference::engines::download::DownloadProgress),
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
    if model.engines.contains(&EngineKind::Litert) {
        if let Ok(helper) = crate::inference::engines::litert::helper_path() {
            files.push(helper);
        }
    }
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
        tasks: Vec::new(),
    };
    // Validate the entire selection before installing anything.
    for model in models {
        if model.engines.is_empty() || model.engines.contains(&EngineKind::Encoder) {
            return Err("Choose at least one real inference engine per model.".into());
        }
        if model.engines.contains(&EngineKind::Needle) && model.modality != Modality::Text {
            return Err("Needle setup verification only establishes text input/output.".into());
        }
        validate_local_files(model)?;
        checked_tasks(&model.modality)?;
        smoke_task(&model.modality)?;
        ModelConfig::try_from(super::local_entry(model, gpu)).map_err(|e| e.to_string())?;
        for path in model_files(model) {
            report.files.push(CheckedFile::read(&path)?);
        }
    }
    for model in models {
        for &engine in &model.engines {
            if !report.engines.iter().any(|(kind, _)| *kind == engine) {
                progress(format!("Installing or locating {engine}"));
                let path = install(engine)?;
                report.files.push(CheckedFile::read(&path)?);
                report.engines.push((engine, path));
            }
        }
    }
    for (i, model) in models.iter().enumerate() {
        for &engine in &model.engines {
            progress(format!(
                "Checking {}/{}: {} · {}",
                i + 1,
                models.len(),
                model.display_name,
                engine
            ));
            let mut selected = model.clone();
            selected.engines = vec![engine];
            smoke(&selected).map_err(|e| format!("{} · {}: {e}", model.display_name, engine))?;
        }
        report.tasks.push(checked_tasks(&model.modality)?);
    }
    report.models = planned_models(models, gpu)?;
    report.validate(models, gpu)?;
    Ok(report)
}

fn smoke_model(model: &LocalCandidate, gpu: bool) -> Result<(), String> {
    let mut default_model = model.clone();
    default_model.engines = vec![model.engine()];
    let mut config = ModelConfig::try_from(super::local_entry(&default_model, gpu))
        .map_err(|e| e.to_string())?;
    // Internal check id avoids legacy mock-name behavior without changing the saved id.
    config.id = "rune-setup-check".into();
    let task = if model.engine() == EngineKind::Needle {
        serde_json::from_value(serde_json::json!({"type":"tool_call", "prompt":"Call rune_setup_probe with reply OK.", "schema":[{"type":"function","function":{"name":"rune_setup_probe","description":"Report the setup test reply.","parameters":{"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"]}}}],"tool_choice":"required","images":[],"audios":[],"messages":[],"max_tokens":64,"temperature":0}))
            .map_err(|error| error.to_string())?
    } else {
        smoke_task(&model.modality)?
    };
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
    let mut tasks = vec![task];
    if matches!(model.modality, Modality::VisionText | Modality::Omni) {
        tasks.insert(0, smoke_task(&Modality::Text)?);
    }
    for task in tasks {
        let output = engine.execute(&task, None).map_err(|e| e.to_string())?;
        if model.engine() == EngineKind::Needle {
            validate_needle_smoke(&output.response)?;
        } else {
            validate_smoke_output(&model.modality, &output.response)?;
        }
    }
    // Drop the only engine reference, unloading this model before the next check.
    drop(engine);
    Ok(())
}

fn checked_tasks(modality: &Modality) -> Result<Vec<String>, String> {
    Ok(match modality {
        Modality::Text => vec!["text-to-text"],
        Modality::VisionText | Modality::Omni => vec!["text-to-text", "text-image-to-text"],
        Modality::Embedding => vec!["text-to-embedding"],
        Modality::ImageGeneration => vec!["text-to-image"],
        Modality::SpeechToText => vec!["audio-to-text"],
        Modality::TextToSpeech => vec!["text-to-audio"],
        Modality::Decision => vec!["text-to-decision"],
        Modality::DepthEstimation => vec!["image-to-depth"],
        _ => {
            return Err(format!(
                "No task verification is available for {}",
                modality.as_str()
            ));
        }
    }
    .into_iter()
    .map(str::to_owned)
    .collect())
}

fn validate_needle_smoke(response: &InferenceTaskResponse) -> Result<(), String> {
    if let InferenceTaskResponse::ToolCall { tool_calls, .. } = response {
        let calls = tool_calls
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![tool_calls.clone()]);
        for call in calls {
            let function = call.get("function").unwrap_or(&call);
            if function["name"].as_str() != Some("rune_setup_probe") {
                continue;
            }
            let args = match &function["arguments"] {
                serde_json::Value::String(value) => {
                    serde_json::from_str(value).unwrap_or(serde_json::Value::Null)
                }
                value => value.clone(),
            };
            if args["reply"]
                .as_str()
                .is_some_and(|reply| !reply.trim().is_empty())
            {
                return Ok(());
            }
        }
    }
    Err("Needle did not return a complete rune_setup_probe tool call with JSON arguments.".into())
}

fn smoke_task(modality: &Modality) -> Result<InferenceTaskRequest, String> {
    use serde_json::json;
    let value = match modality {
        Modality::DepthEstimation => {
            let image = image::RgbImage::from_fn(64, 64, |x, y| {
                image::Rgb([(x * 4) as u8, (y * 4) as u8, 120])
            });
            let mut bytes = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut bytes, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            json!({"type":"image_depth", "image_bytes":bytes.into_inner(), "preview":true})
        }
        Modality::Text | Modality::VisionText | Modality::Omni => {
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
        Modality::Decision => json!({"type":"decision","text":"The status is ready.","images":[],
            "questions":[{"name":"status","type":"choice","instruction":"Choose the status.","criteria":["ready","waiting"]}]}),
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
        (Modality::DepthEstimation, InferenceTaskResponse::Depth(result)) => {
            result.width == crate::inference::engines::litert::WIDTH
                && result.height == crate::inference::engines::litert::HEIGHT
                && result.values.len() == (result.width * result.height) as usize
                && result.values.iter().all(|value| value.is_finite())
                && result.max > result.min
                && result.units == "relative"
                && result
                    .preview_png_base64
                    .as_ref()
                    .and_then(|value| base64::engine::general_purpose::STANDARD.decode(value).ok())
                    .is_some_and(|value| image::load_from_memory(&value).is_ok())
        }
        (
            Modality::Text | Modality::VisionText | Modality::Omni,
            InferenceTaskResponse::Text(text),
        ) => !text.trim().is_empty(),
        (
            Modality::Text | Modality::VisionText | Modality::Omni,
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
        (Modality::Decision, InferenceTaskResponse::Decision(result)) => !result.answers.is_empty(),
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

    #[test]
    #[ignore = "requires existing MiniCPM weights, projector and llama-server; performs real inference and saves only a temporary catalog"]
    fn live_minicpm_setup_saves_checked_tasks() {
        let catalog: crate::schema::CatalogFile =
            serde_json::from_slice(&std::fs::read("config/catalog.json").unwrap()).unwrap();
        let entry = catalog
            .models
            .into_iter()
            .find(|model| model.id == "minicpm-o-4-5-q8-0")
            .unwrap();
        assert_eq!(
            entry.engine.as_ref().unwrap().plan(),
            [EngineKind::LlamaServer]
        );
        assert!(
            Path::new(if cfg!(windows) {
                "bin/llama-upstream/llama-server.exe"
            } else {
                "bin/llama-upstream/llama-server"
            })
            .is_file()
        );
        let path = &entry.local.as_ref().unwrap().path;
        let mut candidate = crate::setup::discover_models(Path::new(path))
            .unwrap()
            .models
            .into_iter()
            .find(|model| model.path == Path::new(path))
            .unwrap();
        candidate.modality = Modality::VisionText;
        // Bound the temporary acceptance run to the same 4096-token window as the API probe.
        candidate.context_length = Some(4096);
        candidate.engines = vec![EngineKind::LlamaServer];
        candidate.extra = entry.extra;
        candidate.needs.clear();
        let models = vec![candidate];
        let proof = prepare_local_models(&models, true, |message| println!("{message}")).unwrap();
        let root = std::env::temp_dir().join(format!(
            "rune-live-setup-{}",
            crate::setup::generate_api_key()
        ));
        let plan = crate::setup::SetupPlan {
            exposure: crate::setup::Exposure::Local,
            port: 8080,
            server_api_key: None,
            cloud: Vec::new(),
            local: models,
            gpu_available: true,
        };
        let server = root.join("config/server.json");
        let catalog = root.join("config/catalog.json");
        crate::setup::write_verified_setup(
            &plan,
            Some(&proof),
            &server,
            &catalog,
            &root.join(".env"),
        )
        .unwrap();
        let loaded = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        assert_eq!(
            loaded.models[0].tasks,
            ["text-to-text", "text-image-to-text"]
        );
        assert_eq!(loaded.models[0].modality, Modality::VisionText);
        println!(
            "Saved and reloaded verified tasks: {:?}",
            loaded.models[0].tasks
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    fn fixture() -> (PathBuf, Vec<LocalCandidate>, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rune-local-check-{}",
            crate::setup::generate_api_key()
        ));
        Gguf::llm("llama", "First").write(&root.join("first.gguf"));
        Gguf::llm("llama", "Second").write(&root.join("second.gguf"));
        let binary = root.join("fixture-engine");
        std::fs::write(&binary, "fixture executable").unwrap();
        let mut models = crate::setup::discover_models(&root).unwrap().models;
        for model in &mut models {
            model.engines.truncate(1);
        }
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
        let plan = super::super::SetupPlan {
            exposure: super::super::Exposure::Local,
            port: 8080,
            server_api_key: None,
            cloud: Vec::new(),
            local: models.clone(),
            gpu_available: false,
        };
        let (server, catalog, env) = (
            root.join("config/server.json"),
            root.join("config/catalog.json"),
            root.join(".env"),
        );
        assert!(super::super::write_verified_setup(&plan, None, &server, &catalog, &env).is_err());
        assert!(!catalog.exists());
        super::super::write_setup(&plan, &server, &catalog, &env).unwrap();
        let mut existing: crate::schema::CatalogFile =
            serde_json::from_str(&std::fs::read_to_string(&catalog).unwrap()).unwrap();
        existing.models[0].aliases = vec!["owner-alias".into()];
        existing.models[0].roles = vec!["owner-role".into()];
        std::fs::write(&catalog, serde_json::to_vec(&existing).unwrap()).unwrap();
        assert_eq!(
            super::super::write_verified_setup(&plan, Some(&proof), &server, &catalog, &env)
                .unwrap(),
            2
        );
        let loaded = crate::config::ModelRegistry::from_files(&server, &catalog).unwrap();
        assert!(
            loaded
                .models
                .iter()
                .all(|model| model.tasks == ["text-to-text"])
        );
        assert_eq!(loaded.models[0].aliases, ["owner-alias"]);
        assert_eq!(loaded.models[0].roles, ["owner-role"]);
        assert_eq!(
            super::super::write_verified_setup(&plan, Some(&proof), &server, &catalog, &env)
                .unwrap(),
            0
        );
        let snapshot =
            [&server, &catalog, &env].map(|path| std::fs::read(path).unwrap_or_default());
        let mut conflict: crate::schema::CatalogFile =
            serde_json::from_slice(&snapshot[1]).unwrap();
        conflict.models[0].tasks.clear();
        conflict.models[0].local.as_mut().unwrap().gpu_layers = 1;
        let conflict_bytes = serde_json::to_vec(&conflict).unwrap();
        std::fs::write(&catalog, &conflict_bytes).unwrap();
        assert!(
            super::super::write_verified_setup(&plan, Some(&proof), &server, &catalog, &env)
                .is_err()
        );
        assert_eq!(std::fs::read(&catalog).unwrap(), conflict_bytes);
        assert_eq!(std::fs::read(&server).unwrap(), snapshot[0]);
        std::fs::write(&catalog, &snapshot[1]).unwrap();
        assert!(proof.validate(&models, true).is_err());
        let mut changed = models.clone();
        changed[0].engines = vec![EngineKind::LlamaServerPrism];
        assert!(proof.validate(&changed, false).is_err());
        std::fs::write(&models[0].path, "changed model").unwrap();
        assert!(proof.validate(&models, false).is_err());
        assert!(
            super::super::write_verified_setup(&plan, Some(&proof), &server, &catalog, &env)
                .is_err()
        );
        assert_eq!(
            [&server, &catalog, &env].map(|path| std::fs::read(path).unwrap_or_default()),
            snapshot
        );
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
    fn chosen_engines_must_be_nonempty_and_compatible() {
        let (root, mut models, _) = fixture();
        models.truncate(1);
        models[0].engines.push(EngineKind::LlamaServerPrism);
        let chosen =
            crate::setup::select_local_engines(&models, &[vec![EngineKind::LlamaServerPrism]])
                .unwrap();
        assert_eq!(chosen[0].engines, vec![EngineKind::LlamaServerPrism]);
        assert!(crate::setup::select_local_engines(&models, &[vec![]]).is_err());
        let both = crate::setup::select_local_engines(
            &models,
            &[vec![EngineKind::LlamaServerPrism, EngineKind::LlamaServer]],
        )
        .unwrap();
        assert_eq!(both[0].engine(), EngineKind::LlamaServerPrism);
        assert_eq!(both[0].engines.len(), 2);
        assert!(
            crate::setup::select_local_engines(&models, &[vec![EngineKind::AudioCpp]]).is_err()
        );
        assert!(crate::setup::select_local_engines(&models, &[]).is_err());
        assert!(crate::setup::select_local_engines(&models, &[vec![EngineKind::Encoder]]).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installs_and_checks_every_selected_engine() {
        let (root, mut models, binary) = fixture();
        for model in &mut models {
            model.engines.push(EngineKind::LlamaServerPrism);
        }
        let mut installed = Vec::new();
        let mut checked = Vec::new();
        let proof = prepare_local_models_with(
            &models,
            false,
            |engine| {
                installed.push(engine);
                Ok(binary.clone())
            },
            |model| {
                checked.push(model.engine());
                Ok(())
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(
            installed,
            [EngineKind::LlamaServer, EngineKind::LlamaServerPrism]
        );
        assert_eq!(
            checked,
            models
                .iter()
                .flat_map(|model| model.engines.iter().copied())
                .collect::<Vec<_>>()
        );
        proof.validate(&models, false).unwrap();
        models[0].engines.reverse();
        assert!(proof.validate(&models, false).is_err());
        models[0].engines.reverse();
        std::fs::write(&binary, "changed engine executable").unwrap();
        assert!(proof.validate(&models, false).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn needle_probe_requires_a_complete_requested_tool_call() {
        use serde_json::json;
        let response = |tool_calls| InferenceTaskResponse::ToolCall {
            content: None,
            tool_calls,
        };
        assert!(
            validate_needle_smoke(&response(
                json!([{"name":"rune_setup_probe","arguments":{"reply":"OK"}}])
            ))
            .is_ok()
        );
        for calls in [
            json!([]),
            json!([{"name":"other","arguments":{"reply":"OK"}}]),
            json!([{"name":"rune_setup_probe","arguments":"{"}]),
        ] {
            assert!(validate_needle_smoke(&response(calls)).is_err());
        }
        assert!(validate_needle_smoke(&InferenceTaskResponse::Text("OK".into())).is_err());
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
