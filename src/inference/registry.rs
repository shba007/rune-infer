use crate::config::{ModelConfig, ModelRegistry as ConfigRegistry, ServerConfig};
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

        let vram_before = Self::detect_used_gpu_vram();

        let engine = Self::create_engine(&model_config, &self.config.server, vram_before)
            .map_err(|e| format!("Failed to load model '{}': {}", id, e))?;

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

    pub fn format_bytes(bytes: u64) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        const GB: f64 = MB * 1024.0;
        const TB: f64 = GB * 1024.0;

        let b = bytes as f64;
        if b >= TB {
            format!("{:.2} TiB", b / TB)
        } else if b >= GB {
            format!("{:.2} GiB", b / GB)
        } else if b >= MB {
            format!("{:.2} MiB", b / MB)
        } else if b >= KB {
            format!("{:.2} KiB", b / KB)
        } else {
            format!("{} B", bytes)
        }
    }

    pub fn format_params(params: u64) -> String {
        let p = params as f64;
        if p >= 1_000_000_000.0 {
            let val = p / 1_000_000_000.0;
            if (val.fract() * 10.0).round() == 0.0 {
                format!("{:.0}B", val)
            } else {
                format!("{:.1}B", val)
            }
        } else if p >= 1_000_000.0 {
            let val = p / 1_000_000.0;
            if (val.fract() * 10.0).round() == 0.0 {
                format!("{:.0}M", val)
            } else {
                format!("{:.1}M", val)
            }
        } else if p >= 1_000.0 {
            format!("{:.0}K", p / 1_000.0)
        } else {
            format!("{}", params)
        }
    }

    pub fn format_tokens(tokens: u32) -> String {
        if tokens >= 1_000_000 {
            format!("{:.1}M tokens", tokens as f64 / 1_000_000.0)
        } else if tokens >= 1024 && tokens % 1024 == 0 {
            format!("{}K tokens", tokens / 1024)
        } else if tokens >= 1000 {
            format!("{:.1}K tokens", tokens as f64 / 1000.0)
        } else {
            format!("{} tokens", tokens)
        }
    }

    fn detect_total_gpu_vram() -> Option<u64> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mib: u64 = text.lines().next()?.trim().parse().ok()?;
        Some(mib * 1024 * 1024)
    }

    fn detect_used_gpu_vram() -> Option<u64> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mib: u64 = text.lines().next()?.trim().parse().ok()?;
        Some(mib * 1024 * 1024)
    }

    fn estimate_bytes_per_token(model: &ModelConfig) -> u64 {
        if let Some(custom) = model.kv_bytes_per_token {
            return custom;
        }

        match model.total_params.unwrap_or(0) {
            p if p < 2_000_000_000 => 16 * 1024,
            p if p < 10_000_000_000 => 32 * 1024,
            p if p < 35_000_000_000 => 64 * 1024,
            _ => 128 * 1024,
        }
    }

    fn calculate_vram_and_context(
        model: &ModelConfig,
        server: &ServerConfig,
    ) -> (u32, u64, u64, u64, u64) {
        let total_vram = Self::detect_total_gpu_vram().unwrap_or(24 * 1024 * 1024 * 1024);
        let budget_ratio = server.vram_budget_ratio.clamp(0.0, 1.0);
        let budget_bytes = (total_vram as f64 * budget_ratio) as u64;

        // Ground-truth text model weight size
        let model_vram = std::fs::metadata(&model.model_path)
            .map(|m| m.len())
            .unwrap_or_else(|_| {
                if let (Some(params), Some(bpw)) = (model.total_params, model.bits_per_weight) {
                    ((params as f64 * bpw) / 8.0) as u64
                } else {
                    4 * 1024 * 1024 * 1024
                }
            });

        // Vision projector weight size calculated independently
        let projector_vram = if let Some(ref mmproj) = model.mmproj_path {
            std::fs::metadata(mmproj).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let bytes_per_token = Self::estimate_bytes_per_token(model);
        let max_capacity = model.max_context_length.unwrap_or(131072);
        let configured_ctx = model
            .runtime
            .as_ref()
            .map(|r| r.context_length)
            .unwrap_or(8192);

        let n_ctx = if configured_ctx <= 0 {
            let overhead_scratch = 768 * 1024 * 1024;
            let remaining_kv =
                budget_bytes.saturating_sub(model_vram + projector_vram + overhead_scratch);
            let raw_tokens = (remaining_kv / bytes_per_token) as u32;

            let rounded = (raw_tokens / 1024) * 1024;
            rounded.clamp(2048, max_capacity)
        } else {
            (configured_ctx as u32).min(max_capacity)
        };

        let context_vram = (n_ctx as u64) * bytes_per_token;
        (n_ctx, context_vram, model_vram, projector_vram, total_vram)
    }

    fn create_engine(
        model: &ModelConfig,
        server: &ServerConfig,
        vram_before: Option<u64>,
    ) -> Result<Arc<dyn InferenceEngine>, Box<dyn std::error::Error>> {
        let model_path = std::path::Path::new(&model.model_path);
        if !model_path.exists() {
            return Err(format!(
                "Model '{}' file not found at path: {}",
                model.id, model.model_path
            )
            .into());
        }

        let (n_ctx, context_vram, model_vram, projector_vram, _total_vram) =
            Self::calculate_vram_and_context(model, server);
        let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers);

        let engine: Arc<dyn InferenceEngine> = match model.architecture.as_str() {
            "needle" | "cactus-needle" | "cactus-needle-3" => {
                let engine = crate::inference::engines::needle::NeedleEngine::new(
                    model.id.clone(),
                    model_path,
                )?;
                Arc::new(engine)
            }
            "qwen" | "qwen3" => {
                let engine = crate::inference::engines::qwen::QwenEngine::new(
                    model.id.clone(),
                    model_path,
                    Some(n_ctx),
                    gpu_layers,
                )?;
                Arc::new(engine)
            }
            "ternary-bonsai-vl" | "bonsai" | "bonsai-vl" | "ternary-bonsai" => {
                let mmproj_path = model.mmproj_path.as_deref().map(std::path::Path::new);
                let engine = crate::inference::engines::bonsai::BonsaiEngine::new(
                    model.id.clone(),
                    model_path,
                    mmproj_path,
                    Some(n_ctx),
                    gpu_layers,
                )?;
                Arc::new(engine)
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
                        Some(n_ctx),
                        gpu_layers,
                    )?;
                    Arc::new(engine)
                } else {
                    let engine = crate::inference::engines::qwen35::Qwen35Engine::new(
                        model.id.clone(),
                        model_path,
                        Some(n_ctx),
                        gpu_layers,
                    )?;
                    Arc::new(engine)
                }
            }
        };

        let param_info = if let Some(p) = model.total_params {
            format!("{} params", Self::format_params(p))
        } else {
            "Weights".to_string()
        };

        let total_est = model_vram + projector_vram + context_vram;
        let real_str =
            if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                let actual_used = after.saturating_sub(before);
                format!("Real: {}", Self::format_bytes(actual_used))
            } else {
                "Real: N/A".to_string()
            };

        let capabilities_str = model.resolved_capabilities().join(", ");

        println!(
            "[ModelRegistry] ✓ Successfully loaded engine: \"{}\" ({})",
            model.id, model.name
        );
        println!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
        println!(
            "[ModelRegistry]   • Model:        {} | {}",
            param_info,
            Self::format_bytes(model_vram)
        );

        if projector_vram > 0 || model.vision {
            let res_str = model
                .max_resolution
                .as_deref()
                .unwrap_or("4096×4096 (Dynamic 4K)");
            println!(
                "[ModelRegistry]   • Projector:  {} | {}",
                res_str,
                Self::format_bytes(projector_vram)
            );
        }

        println!(
            "[ModelRegistry]   • Context:      {} | {}",
            Self::format_tokens(n_ctx),
            Self::format_bytes(context_vram)
        );
        println!(
            "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
            Self::format_bytes(total_est),
            real_str
        );

        Ok(engine)
    }
}
