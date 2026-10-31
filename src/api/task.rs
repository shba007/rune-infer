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
        }
    }

    fn add(&mut self, input: &'static str) {
        if !self.inputs.contains(&input) {
            self.inputs.push(input);
        }
    }

    pub(super) fn task(task: &InferenceTaskRequest) -> Self {
        use InferenceTaskRequest::*;
        match task {
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
        }
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
                    "chat" | "caption" | "embedding" | "grounding" | "decision"
                ),
                SdServer => self.operation == "generation",
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
            if matches!(engine, LlamaServer | LlamaServerPrism)
                && self
                    .inputs
                    .iter()
                    .any(|input| matches!(*input, "image" | "audio" | "video"))
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
                && self
                    .inputs
                    .iter()
                    .any(|input| matches!(*input, "image" | "audio" | "video"))
            {
                let projector = model
                    .mmproj_path
                    .as_deref()
                    .and_then(|path| crate::detect::read_gguf(std::path::Path::new(path)).ok());
                let supports = projector.is_some_and(|info| {
                    (!self
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
                            "Model '{}' projector does not establish support for these inputs",
                            model.id
                        ),
                    );
                }
            }
            if engine == SdServer
                && [
                    &model.vae_path,
                    &model.text_encoder_path,
                    &model.llm_vision_path,
                ]
                .into_iter()
                .flatten()
                .any(|path| !std::path::Path::new(path).is_file())
            {
                return refuse(
                    StatusCode::BAD_REQUEST,
                    "model_companion_missing",
                    format!("Model '{}' has a missing configured companion", model.id),
                );
            }
        }
        Ok(())
    }
}
