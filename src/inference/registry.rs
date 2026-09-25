use crate::config::{ModelConfig, ModelRegistry as ConfigRegistry};
use crate::inference::traits::InferenceEngine;
use std::collections::HashMap;
use std::sync::Arc;

pub struct ModelRegistry {
    config: ConfigRegistry,
    engines: HashMap<String, Arc<dyn InferenceEngine>>,
    lru_keys: Vec<String>,
}

impl ModelRegistry {
    pub fn new(config: ConfigRegistry) -> Self {
        println!(
            "[ModelRegistry] Initialized model catalog with {} models configured.",
            config.models.len()
        );
        println!("[ModelRegistry] Lazy loading enabled: No models are preloaded into VRAM/RAM.");
        Self {
            config,
            engines: HashMap::new(),
            lru_keys: Vec::new(),
        }
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.engines.keys().cloned().collect()
    }

    pub fn get_or_load(&mut self, id: &str) -> Result<Arc<dyn InferenceEngine>, String> {
        if let Some(engine) = self.engines.get(id).cloned() {
            self.touch_lru(id);
            return Ok(engine);
        }

        let model_config = self
            .config
            .find(id)
            .ok_or_else(|| format!("Model '{}' not found in models.json", id))?
            .clone();

        let max_loaded = self.config.server.max_loaded_models;
        if max_loaded > 0 && self.engines.len() >= max_loaded {
            self.evict_lru();
        }

        println!(
            "[ModelRegistry] Lazy-loading model '{}' ({}) on demand...",
            model_config.id, model_config.name
        );

        let engine = Self::create_engine(&model_config)
            .map_err(|e| format!("Failed to load model '{}': {}", id, e))?;

        println!(
            "[ModelRegistry] ✓ Successfully loaded engine: \"{}\"",
            model_config.id
        );

        self.engines.insert(id.to_string(), engine.clone());
        self.lru_keys.push(id.to_string());

        Ok(engine)
    }

    fn touch_lru(&mut self, id: &str) {
        if let Some(pos) = self.lru_keys.iter().position(|k| k == id) {
            let key = self.lru_keys.remove(pos);
            self.lru_keys.push(key);
        }
    }

    fn evict_lru(&mut self) {
        if self.lru_keys.is_empty() {
            return;
        }
        let victim_id = self.lru_keys.remove(0);
        if let Some(_removed) = self.engines.remove(&victim_id) {
            println!(
                "[ModelRegistry] Evicted least-recently-used model '{}' to respect max_loaded_models ({})",
                victim_id, self.config.server.max_loaded_models
            );
        }
    }

    fn create_engine(
        model: &ModelConfig,
    ) -> Result<Arc<dyn InferenceEngine>, Box<dyn std::error::Error>> {
        if model.architecture == "ternary-bonsai-vl" {
            return Err(format!(
                "Model '{}' requires custom PrismML fork (ternary-bonsai-vl); not supported on standard llama.cpp",
                model.id
            ).into());
        }

        let model_path = std::path::Path::new(&model.model_path);
        if !model_path.exists() {
            return Err(format!(
                "Model '{}' file not found at path: {}",
                model.id, model.model_path
            )
            .into());
        }

        let n_ctx = model.runtime.as_ref().map(|r| r.context_length);
        let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers);

        match model.architecture.as_str() {
            "needle" | "cactus-needle" | "cactus-needle-3" => {
                let engine = crate::inference::engines::needle::NeedleEngine::new(
                    model.id.clone(),
                    model_path,
                )?;
                Ok(Arc::new(engine))
            }
            "qwen" | "qwen3" => {
                let engine = crate::inference::engines::qwen::QwenEngine::new(
                    model.id.clone(),
                    model_path,
                    n_ctx,
                    gpu_layers,
                )?;
                Ok(Arc::new(engine))
            }
            _ => {
                if model.vision || model.modality == crate::config::Modality::VisionText {
                    let mmproj_str = model.mmproj_path.as_deref().ok_or_else(|| {
                        format!("Vision model '{}' requires 'mmproj_path'", model.id)
                    })?;
                    let mmproj_path = std::path::Path::new(mmproj_str);
                    if !mmproj_path.exists() {
                        return Err(format!(
                            "mmproj file not found for '{}': {}",
                            model.id, mmproj_str
                        )
                        .into());
                    }

                    let engine = crate::inference::engines::qwen2vl::Qwen2VlEngine::new(
                        model.id.clone(),
                        model_path,
                        mmproj_path,
                        n_ctx,
                        gpu_layers,
                    )?;
                    Ok(Arc::new(engine))
                } else {
                    let engine = crate::inference::engines::qwen35::Qwen35Engine::new(
                        model.id.clone(),
                        model_path,
                        n_ctx,
                        gpu_layers,
                    )?;
                    Ok(Arc::new(engine))
                }
            }
        }
    }
}
