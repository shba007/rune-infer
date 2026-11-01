//! Request task checks shared by chat and primitive inference.
use crate::config::{EngineKind, ModelConfig};
use crate::inference::types::InferenceTaskRequest;
use crate::types::{ApiError, ChatCompletionRequest, ErrorResponse, MediaItem};
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

pub(super) struct Requirements {
    inputs: Vec<&'static str>,
    output: &'static str,
    operation: &'static str,
    llama_audio_error: Option<&'static str>,
    audiocpp_error: Option<&'static str>,
    sd_edit_error: Option<&'static str>,
}

impl Requirements {
    #[allow(clippy::result_large_err)]
    pub(super) fn check(
        &self,
        state: &super::AppState,
        model: &ModelConfig,
        engine: Option<&std::sync::Arc<dyn crate::inference::traits::InferenceEngine>>,
    ) -> Result<(), Response> {
        let registry = state
            .inference
            .registry
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let resident = registry.engines.get(&model.id);
        let selected = engine.or(resident);
        let kind = if let Some(selected) = selected {
            registry.engine_leases.iter().find_map(|(kind, lease)| {
                lease
                    .upgrade()
                    .filter(|actual| std::sync::Arc::ptr_eq(actual, selected))
                    .map(|_| *kind)
            })
        } else {
            model.engine_plan().first().copied()
        };
        drop(registry);
        self.validate(model, kind)
    }
    pub(super) fn chat(request: &ChatCompletionRequest) -> Self {
        let mut result = Self::new(&[], "text", "chat");
        result.messages(&request.messages);
        result
    }

    fn messages(&mut self, messages: &[crate::types::ChatMessage]) {
        for message in messages {
            let (text, media) = message.split_text_and_media();
            if !text.trim().is_empty() || message.tool_calls.is_some() {
                self.add("text");
            }
            for media in media {
                self.add(match media {
                    MediaItem::Image(_) => "image",
                    MediaItem::Audio { .. } => "audio",
                    MediaItem::Video(_) => "video",
                });
            }
        }
        if self.inputs.is_empty() {
            self.add("text");
        }
    }

    pub(super) fn new(
        inputs: &[&'static str],
        output: &'static str,
        operation: &'static str,
    ) -> Self {
        Self {
            inputs: inputs.to_vec(),
            output,
            operation,
            llama_audio_error: None,
            audiocpp_error: None,
            sd_edit_error: None,
        }
    }

    fn add(&mut self, input: &'static str) {
        if !self.inputs.contains(&input) {
            self.inputs.push(input);
        }
    }

    pub(super) fn task(task: &InferenceTaskRequest) -> Self {
        use InferenceTaskRequest::*;
        let mut requirements = match task {
            ImageDepth { .. } => Self::new(&["image"], "depth", "depth"),
            ToolCall {
                prompt,
                images,
                audios,
                messages,
                ..
            } => {
                let mut result = Self::new(&[], "text", "chat");
                if !prompt.trim().is_empty() {
                    result.add("text");
                }
                if !images.is_empty() {
                    result.add("image");
                }
                if !audios.is_empty() {
                    result.add("audio");
                }
                result.messages(messages);
                result
            }
            ImageGeneration { .. } => Self::new(&["text"], "image", "generation"),
            ImageEdit { .. } => Self::new(&["text", "image"], "image", "edit"),
            AudioTranscription { .. } => Self::new(&["audio"], "text", "transcription"),
            AudioTranslation { .. } => Self::new(&["audio"], "text", "translation"),
            AudioSpeech { .. } => Self::new(&["text"], "audio", "speech"),
            SpeechToSpeech { .. } => Self::new(&["audio"], "audio", "speech-to-speech"),
            AudioDiarization { .. } => Self::new(&["audio"], "diarization", "diarization"),
            Embedding { .. } => Self::new(&["text"], "embedding", "embedding"),
            Moderation { .. } => Self::new(&["text"], "moderation", "moderation"),
            NluIntent { .. } => Self::new(&["text"], "text", "nlu"),
            DocumentOcr { .. } => Self::new(&["image"], "text", "ocr"),
            ImageCaption { .. } => Self::new(&["image"], "text", "caption"),
            DetectObjects {
                prompt, categories, ..
            } => {
                let text = prompt.as_ref().is_some_and(|p| !p.trim().is_empty())
                    || categories.as_ref().is_some_and(|c| !c.is_empty());
                Self::new(
                    if text { &["text", "image"] } else { &["image"] },
                    "detection",
                    "detection",
                )
            }
            ImageEmbedding { .. } => Self::new(&["image"], "embedding", "image-embedding"),
            RecognizeObjects { .. } => Self::new(&["text", "image"], "detection", "recognition"),
            ImageUpscale { .. } | ImageRestoration { .. } => {
                Self::new(&["image"], "image", "processing")
            }
            ImageStyleTransfer { .. } => Self::new(&["image"], "image", "processing"),
            ImageGrounding { .. } => Self::new(&["text", "image"], "detection", "grounding"),
            Decision { images, text, .. } => Self::new(
                if !images.is_empty() && text.trim().is_empty() {
                    &["image"]
                } else if images.is_empty() {
                    &["text"]
                } else {
                    &["text", "image"]
                },
                "decision",
                "decision",
            ),
        };
        requirements.llama_audio_error =
            crate::inference::engines::llama_server::audio::validate_task(task).err();
        requirements.audiocpp_error = crate::inference::engines::audio::validate_task(task).err();
        requirements.sd_edit_error =
            crate::inference::engines::sd_server::edit::validate_task(task).err();
        requirements
    }

    #[allow(clippy::result_large_err)]
    pub(super) fn validate(
        &self,
        model: &ModelConfig,
        engine: Option<EngineKind>,
    ) -> Result<(), Response> {
        let refuse = |status, code, message| {
            Err((
                status,
                Json(ErrorResponse {
                    error: ApiError::new(message)
                        .with_type("invalid_request_error")
                        .with_code(code),
                }),
            )
                .into_response())
        };
        let audio_operation = match self.operation {
            "transcription" => Some(crate::audio_operations::AudioOperation::Transcription),
            "translation" => Some(crate::audio_operations::AudioOperation::SpeechTranslation),
            "speech" => Some(crate::audio_operations::AudioOperation::SpeechGeneration),
            _ => None,
        };
        if let Some(operation) = audio_operation {
            use crate::audio_operations::SupportStatus;
            let support = model.resolved_audio_operation(operation, engine);
            if support.status != SupportStatus::Supported {
                let (status, code) = if support.status == SupportStatus::Unsupported {
                    (StatusCode::BAD_REQUEST, "model_operation_unsupported")
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "model_operation_unverified",
                    )
                };
                return refuse(
                    status,
                    code,
                    format!(
                        "Model '{}' cannot execute {}: {}",
                        model.id,
                        operation.as_str(),
                        support
                            .reason
                            .unwrap_or_else(|| "Operation support is not established".into())
                    ),
                );
            }
        }
        if model.tasks.is_empty() {
            return refuse(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_tasks_unverified",
                format!("Model '{}' has no verified input/output tasks", model.id),
            );
        }
        let supported = model.tasks.iter().any(|task| {
            task.split_once("-to-").is_some_and(|(inputs, outputs)| {
                outputs == self.output
                    && self
                        .inputs
                        .iter()
                        .all(|input| inputs.split('-').any(|value| value == *input))
            })
        });
        if !supported {
            return refuse(
                StatusCode::BAD_REQUEST,
                "model_task_unsupported",
                format!(
                    "Model '{}' does not support {}-to-{}",
                    model.id,
                    self.inputs.join("-"),
                    self.output
                ),
            );
        }
        if let Some(engine) = engine {
            use EngineKind::*;
            let implemented = match engine {
                LlamaServer | LlamaServerPrism => matches!(
                    self.operation,
                    "chat"
                        | "caption"
                        | "embedding"
                        | "grounding"
                        | "decision"
                        | "transcription"
                        | "speech"
                ),
                SdServer => matches!(self.operation, "generation" | "edit"),
                AudioCpp => matches!(self.operation, "transcription" | "speech" | "diarization"),
                NemoSpeech | WhisperCpp => {
                    matches!(self.operation, "transcription" | "translation")
                }
                Crispasr => matches!(
                    self.operation,
                    "transcription" | "translation" | "speech" | "speech-to-speech"
                ),
                Needle => {
                    self.operation == "chat" && self.inputs.iter().all(|input| *input == "text")
                }
                Encoder => false,
                Litert => self.operation == "depth",
            };
            if !implemented {
                return refuse(
                    StatusCode::BAD_REQUEST,
                    "engine_task_unsupported",
                    format!("Engine '{engine}' does not implement {}", self.operation),
                );
            }
            if engine == AudioCpp
                && let Some(message) = self.audiocpp_error
            {
                return refuse(
                    StatusCode::BAD_REQUEST,
                    "engine_audio_option_unsupported",
                    message.into(),
                );
            }
            if matches!(engine, LlamaServer | LlamaServerPrism) {
                if let Some(message) = self.llama_audio_error {
                    return refuse(
                        StatusCode::BAD_REQUEST,
                        "engine_audio_option_unsupported",
                        message.into(),
                    );
                }
            }
            if matches!(engine, LlamaServer | LlamaServerPrism)
                && (self.output == "audio"
                    || self
                        .inputs
                        .iter()
                        .any(|input| matches!(*input, "image" | "audio" | "video")))
                && !model
                    .mmproj_path
                    .as_deref()
                    .is_some_and(|path| std::path::Path::new(path).is_file())
            {
                return refuse(
                    StatusCode::BAD_REQUEST,
                    "model_projector_missing",
                    format!("Model '{}' requires a usable projector", model.id),
                );
            }
            if matches!(engine, LlamaServer | LlamaServerPrism)
                && (self.output == "audio"
                    || self
                        .inputs
                        .iter()
                        .any(|input| matches!(*input, "image" | "audio" | "video")))
            {
                let projector = model
                    .mmproj_path
                    .as_deref()
                    .and_then(|path| crate::detect::read_gguf(std::path::Path::new(path)).ok());
                let supports = projector.is_some_and(|info| {
                    (self.output != "audio" || info.has_gen_audio_encoder == Some(true))
                        && (!self
                            .inputs
                            .iter()
                            .any(|input| matches!(*input, "image" | "video"))
                            || info.has_vision_encoder == Some(true))
                        && (!self.inputs.contains(&"audio") || info.has_audio_encoder == Some(true))
                });
                if !supports {
                    return refuse(
                        StatusCode::BAD_REQUEST,
                        "model_projector_unsupported",
                        format!(
                            "Model '{}' projector does not establish support for the requested inputs/output",
                            model.id
                        ),
                    );
                }
            }
            if engine == SdServer {
                if let Some(message) = self.sd_edit_error {
                    return refuse(
                        StatusCode::BAD_REQUEST,
                        "invalid_image_edit",
                        message.into(),
                    );
                }
                for (field, path) in [
                    ("extra.vae_path", &model.vae_path),
                    ("extra.text_encoder_path", &model.text_encoder_path),
                    ("extra.llm_vision_path", &model.llm_vision_path),
                ] {
                    if let Some(path) = path
                        && !crate::inference::engines::sd_server::companion_path_available(
                            std::path::Path::new(path),
                        )
                    {
                        return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse {
                            error: ApiError::new(format!("Model '{}' companion '{field}' is missing or is not a supported model file/directory: {path}", model.id))
                                .with_type("invalid_request_error")
                                .with_param(field)
                                .with_code("model_companion_missing"),
                        })).into_response());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod companion_tests {
    use super::*;

    #[test]
    fn lfm_speech_requires_explicit_generator_metadata_and_valid_options() {
        let root = std::env::temp_dir().join(format!(
            "rune-speech-projector-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("projector.gguf");
        let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
            "id":"speech-fixture","name":"Speech fixture","tasks":["text-to-audio"],
            "engine":["llama-server"],"local":{"path":"Cargo.toml"},
            "audio_operations":{"speech_generation":{"status":"supported","verified_with":[{"engine":"llama-server","version":"fixture"}]}},
            "extra":{"mmproj_path":path.to_str().unwrap()}
        }))
        .unwrap();
        let model = ModelConfig::try_from(entry).unwrap();
        let task: InferenceTaskRequest =
            serde_json::from_value(serde_json::json!({"type":"audio_speech","input":"Hello"}))
                .unwrap();
        let check = Requirements::task(&task);
        assert!(
            check
                .validate(&model, Some(EngineKind::LlamaServer))
                .is_err()
        );
        for generator in [false, true] {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            bytes.extend(2u64.to_le_bytes());
            for value in ["general.architecture", "clip"] {
                bytes.extend((value.len() as u64).to_le_bytes());
                bytes.extend(value.as_bytes());
                if value == "general.architecture" {
                    bytes.extend(8u32.to_le_bytes());
                }
            }
            let key = "clip.has_gen_audio_encoder";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(7u32.to_le_bytes());
            bytes.push(u8::from(generator));
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(
                check
                    .validate(&model, Some(EngineKind::LlamaServer))
                    .is_ok(),
                generator
            );
        }
        let bad: InferenceTaskRequest = serde_json::from_value(
            serde_json::json!({"type":"audio_speech","input":"Hello","voice":"alloy"}),
        )
        .unwrap();
        assert!(
            Requirements::task(&bad)
                .validate(&model, Some(EngineKind::LlamaServer))
                .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn sd_companion_directory_and_actionable_error() {
        let root = std::env::temp_dir().join(format!(
            "rune-task-companion-{}",
            crate::setup::generate_api_key()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
            "id":"image-fixture", "name":"Image fixture", "tasks":["text-to-image"],
            "engine":["sd-server"], "local":{"path":"unused.gguf"},
            "extra":{"text_encoder_path": root.to_str().unwrap()}
        }))
        .unwrap();
        let model = ModelConfig::try_from(entry).unwrap();
        let requirements = Requirements::new(&["text"], "image", "generation");
        let response = requirements
            .validate(&model, Some(EngineKind::SdServer))
            .unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 16384)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["code"], "model_companion_missing");
        assert_eq!(error["error"]["param"], "extra.text_encoder_path");
        std::fs::write(root.join("model.safetensors.index.json"), b"{}").unwrap();
        assert!(
            requirements
                .validate(&model, Some(EngineKind::SdServer))
                .is_ok()
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
