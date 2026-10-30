//! Explicit transport doubles. These never load models or establish model support.
use rune_infer::{
    api::{AppState, create_router},
    config::{EngineConfig, EngineKind, ModelRegistry},
    inference::{
        AppState as InferenceState,
        engines::encoder::EncoderEngine,
        traits::InferenceEngine,
        types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse},
    },
    proxy::ProxyService,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

struct TransportDouble {
    response: EncoderEngine,
    projector: PathBuf,
}

impl Drop for TransportDouble {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.projector);
    }
}

impl InferenceEngine for TransportDouble {
    fn id(&self) -> &str {
        self.response.id()
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn std::error::Error>> {
        if let InferenceTaskRequest::AudioTranscription { filename, .. } = task {
            assert!(!filename.contains(['/', '\\']));
            return Ok(InferenceOutput {
                response: InferenceTaskResponse::Transcription(serde_json::from_value(
                    serde_json::json!({"text": filename}),
                )?),
                usage: rune_infer::types::Usage::new(0, 0),
            });
        }
        self.response.execute(task, on_token)
    }
}

pub fn create_mock_app(mut catalog: ModelRegistry) -> axum::Router {
    let mut doubles = Vec::new();
    for model in &mut catalog.models {
        assert!(model.is_local() && !model.tasks.is_empty());
        let projector = std::env::temp_dir().join(format!(
            "rune-transport-{}.gguf",
            rune_infer::setup::generate_api_key()
        ));
        // Metadata-only GGUF simulates a local backend's projector contract.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        for key in ["clip.has_vision_encoder", "clip.has_audio_encoder"] {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(7u32.to_le_bytes()); // GGUF bool
            bytes.push(1);
        }
        std::fs::write(&projector, bytes).unwrap();
        model.mmproj_path = Some(projector.to_string_lossy().into_owned());
        let kind = if model.tasks == ["audio-to-text"] {
            EngineKind::AudioCpp
        } else {
            EngineKind::LlamaServer
        };
        model.engine = Some(EngineConfig {
            supported: vec![kind],
            preferred: None,
        });
        doubles.push(Arc::new(TransportDouble {
            response: EncoderEngine::new(model).unwrap(),
            projector,
        }));
    }
    let inference = Arc::new(InferenceState::new(&catalog));
    for (model, double) in catalog.models.iter().zip(doubles) {
        inference.registry.lock().unwrap().finish_load(
            model,
            double,
            model.engine_plan()[0],
            Instant::now(),
            None,
        );
    }
    create_router(AppState::new(
        inference,
        Arc::new(ProxyService::new(&catalog)),
        catalog,
    ))
}
