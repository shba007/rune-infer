pub mod engines;
pub mod process;
pub mod registry;
pub mod traits;
pub mod types;

use std::sync::Arc;
use std::sync::Mutex;
use traits::InferenceEngine;
use types::{InferenceOutput, InferenceTaskRequest};

pub struct AppState {
    pub registry: Mutex<registry::ModelRegistry>,
}

impl AppState {
    pub fn new(config: &crate::config::ModelRegistry) -> Self {
        Self {
            registry: Mutex::new(registry::ModelRegistry::new(config.clone())),
        }
    }

    pub fn get_engine(&self, id: &str) -> Result<Arc<dyn InferenceEngine>, String> {
        let mut registry = self
            .registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.get_or_load(id)
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let registry = self
            .registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.loaded_models()
    }

    pub fn execute_task(
        &self,
        engine_id: &str,
        task: &InferenceTaskRequest,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, String> {
        let engine = self.get_engine(engine_id)?;
        engine.execute(task, on_token).map_err(|e| e.to_string())
    }
}
