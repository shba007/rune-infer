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
        } else if tokens >= 1024 && tokens.is_multiple_of(1024) {
            format!("{}K tokens", tokens / 1024)
        } else if tokens >= 1000 {
            format!("{:.1}K tokens", tokens as f64 / 1000.0)
        } else {
            format!("{} tokens", tokens)
        }
    }

    fn detect_total_gpu_vram() -> Option<u64> {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = text.lines().next() {
                    if let Ok(mib) = line.trim().parse::<u64>() {
                        return Some(mib * 1024 * 1024);
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    if let Ok(bytes) = text.trim().parse::<u64>() {
                        return Some(bytes);
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return Some(kb * 1024);
                            }
                        }
                    }
                }
            }
        }

        None
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

    fn estimate_total_layers(model: &ModelConfig) -> u32 {
        match model.total_params.unwrap_or(0) {
            0 => 64,
            p if p < 4_000_000_000 => 28,
            p if p < 10_000_000_000 => 32,
            p if p < 18_000_000_000 => 48,
            p if p < 36_000_000_000 => 64,
            p if p < 80_000_000_000 => 80,
            _ => 96,
        }
    }

    fn parse_cache_type_multiplier(args: &str, flag_long: &str, flag_short: &str) -> f64 {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        for (i, &t) in tokens.iter().enumerate() {
            let val = if t == flag_long || t == flag_short {
                tokens.get(i + 1).copied()
            } else if let Some(stripped) = t.strip_prefix(&format!("{flag_long}=")) {
                Some(stripped)
            } else {
                t.strip_prefix(&format!("{flag_short}="))
            };

            if let Some(v) = val {
                let clean = v.to_lowercase();
                if clean.contains("q4_0") || clean.contains("q4_1") || clean.contains("iq4") {
                    return 0.28;
                } else if clean.contains("q5_0") || clean.contains("q5_1") {
                    return 0.35;
                } else if clean.contains("q8_0") || clean.contains("fp8") || clean.contains("f8") {
                    return 0.53;
                } else if clean.contains("f16") || clean.contains("bf16") {
                    return 1.0;
                }
            }
        }
        1.0
    }

    fn estimate_bytes_per_token(model: &ModelConfig) -> u64 {
        let extra_args = model
            .runtime
            .as_ref()
            .and_then(|r| r.extra_args.as_deref())
            .unwrap_or("");

        if extra_args.contains("--no-kv-offload") {
            return 0;
        }

        let base = if let Some(custom) = model.kv_bytes_per_token {
            custom
        } else {
            match model.total_params.unwrap_or(0) {
                p if p < 2_000_000_000 => 16 * 1024,
                p if p < 10_000_000_000 => 32 * 1024,
                p if p < 35_000_000_000 => 64 * 1024,
                _ => 128 * 1024,
            }
        };

        let k_mult = Self::parse_cache_type_multiplier(extra_args, "--cache-type-k", "-ctk");
        let v_mult = Self::parse_cache_type_multiplier(extra_args, "--cache-type-v", "-ctv");
        let quant_mult = (k_mult + v_mult) / 2.0;

        ((base as f64 * quant_mult).max(1024.0)) as u64
    }

    fn calculate_vram_and_context(
        model: &ModelConfig,
        server: &ServerConfig,
    ) -> (u32, u64, u64, u64, u64, u64) {
        let total_vram = Self::detect_total_gpu_vram().unwrap_or(24 * 1024 * 1024 * 1024);
        let budget_ratio = server.vram_budget_ratio.clamp(0.0, 1.0);
        let budget_bytes = (total_vram as f64 * budget_ratio) as u64;

        let total_model_bytes = std::fs::metadata(&model.model_path)
            .map(|m| m.len())
            .unwrap_or_else(|_| {
                if let (Some(params), Some(bpw)) = (model.total_params, model.bits_per_weight) {
                    ((params as f64 * bpw) / 8.0) as u64
                } else {
                    4 * 1024 * 1024 * 1024
                }
            });

        let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers).unwrap_or(99);
        let total_layers = Self::estimate_total_layers(model);
        let model_vram = if gpu_layers >= total_layers || gpu_layers >= 90 {
            total_model_bytes
        } else {
            let offload_ratio = (gpu_layers as f64 / total_layers as f64).clamp(0.0, 1.0);
            ((total_model_bytes as f64) * (0.05 + 0.95 * offload_ratio))
                .min(total_model_bytes as f64) as u64
        };

        let projector_vram = if let Some(ref mmproj) = model.mmproj_path {
            std::fs::metadata(mmproj).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let mtp_vram = if let Some(ref mtp) = model.mtp_path {
            std::fs::metadata(mtp).map(|m| m.len()).unwrap_or(0)
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

        let mtp_headroom = if model.has_mtp() {
            1536 * 1024 * 1024
        } else {
            0
        };

        let extra_args = model
            .runtime
            .as_ref()
            .and_then(|r| r.extra_args.as_deref())
            .unwrap_or("");
        let has_flash_attn = extra_args.contains("-fa") || extra_args.contains("--flash-attn");
        let overhead_scratch = if has_flash_attn {
            512 * 1024 * 1024
        } else {
            768 * 1024 * 1024
        };

        let n_ctx = if configured_ctx <= 0 {
            let remaining_kv = budget_bytes.saturating_sub(
                model_vram + projector_vram + mtp_vram + overhead_scratch + mtp_headroom,
            );
            if let Some(raw_tokens) = remaining_kv.checked_div(bytes_per_token) {
                let rounded = ((raw_tokens as u32) / 1024) * 1024;
                rounded.clamp(2048, max_capacity)
            } else {
                max_capacity
            }
        } else {
            (configured_ctx as u32).min(max_capacity)
        };

        let context_vram = (n_ctx as u64) * bytes_per_token;
        (
            n_ctx,
            context_vram,
            model_vram,
            projector_vram,
            mtp_vram,
            total_vram,
        )
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

        let model_vram = std::fs::metadata(&model.model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let param_info = if let Some(p) = model.total_params {
            format!("{} params", Self::format_params(p))
        } else {
            "Weights".to_string()
        };

        let capabilities_str = model.resolved_capabilities().join(", ");

        if model.modality == crate::config::Modality::ImageGeneration {
            let engine = crate::inference::engines::sd_server::SdServerEngine::new(model)?;
            let text_encoder_vram = model
                .text_encoder_path
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .unwrap_or(0);

            let vae_vram = model
                .vae_path
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .unwrap_or(0);

            let overhead = 1536 * 1024 * 1024;
            let total_est = model_vram + text_encoder_vram + vae_vram + overhead;
            let real_str =
                if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                    let actual_used = after.saturating_sub(before);
                    format!("Real: {}", Self::format_bytes(actual_used))
                } else {
                    "Real: N/A".to_string()
                };

            let res_str = model
                .max_resolution
                .as_deref()
                .unwrap_or("1024×1024 (Native DiT)");
            let steps = model.default_steps.unwrap_or(30);
            let cfg = model.default_cfg_scale.unwrap_or(4.0);
            let sampler = model.default_sample_method.as_deref().unwrap_or("euler");

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
            if text_encoder_vram > 0 {
                println!(
                    "[ModelRegistry]   • Text Encoder: {}",
                    Self::format_bytes(text_encoder_vram)
                );
            }
            if vae_vram > 0 {
                println!(
                    "[ModelRegistry]   • VAE:          {}",
                    Self::format_bytes(vae_vram)
                );
            }
            println!("[ModelRegistry]   • Resolution:   {}", res_str);
            println!(
                "[ModelRegistry]   • Sampling:     {} steps | {:.1} CFG | {}",
                steps, cfg, sampler
            );
            println!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(total_est),
                real_str
            );

            return Ok(Arc::new(engine));
        }

        if model.architecture.eq_ignore_ascii_case("audio8")
            || model.architecture.eq_ignore_ascii_case("audiocpp")
            || model.id.to_lowercase().contains("audio8")
        {
            let engine = crate::inference::engines::audio::AudioEngine::new(model)?;
            println!(
                "[ModelRegistry] ✓ Successfully loaded Audio engine (audio.cpp): \"{}\"",
                model.id
            );
            return Ok(Arc::new(engine));
        }

        if model.architecture.eq_ignore_ascii_case("parakeet")
            || model.architecture.eq_ignore_ascii_case("nemo-speech")
            || model.architecture.eq_ignore_ascii_case("fastconformer")
            || model.architecture.eq_ignore_ascii_case("fastconformer-tdt")
            || model.id.to_lowercase().contains("parakeet")
            || model.id.to_lowercase().contains("nemotron-speech")
        {
            let engine = crate::inference::engines::nemo_speech::NemoSpeechEngine::new(model)?;
            let real_str =
                if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                    let actual_used = after.saturating_sub(before);
                    format!("Real: {}", Self::format_bytes(actual_used))
                } else {
                    "Real: N/A".to_string()
                };

            println!(
                "[ModelRegistry] ✓ Successfully loaded NeMo-Speech engine: \"{}\" ({})",
                model.id, model.name
            );
            println!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
            println!(
                "[ModelRegistry]   • Model:        {} | {}",
                param_info,
                Self::format_bytes(model_vram)
            );
            println!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(model_vram + 1024 * 1024 * 1024),
                real_str
            );
            return Ok(Arc::new(engine));
        }

        if model.modality == crate::config::Modality::SpeechToText
            || model.modality == crate::config::Modality::TextToSpeech
            || model.architecture.eq_ignore_ascii_case("crispasr")
            || model.architecture.eq_ignore_ascii_case("voxtral4b")
        {
            let engine = crate::inference::engines::crispasr::CrispAsrEngine::new(model)?;
            let real_str =
                if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                    let actual_used = after.saturating_sub(before);
                    format!("Real: {}", Self::format_bytes(actual_used))
                } else {
                    "Real: N/A".to_string()
                };

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
            println!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(model_vram + 1024 * 1024 * 1024),
                real_str
            );
            return Ok(Arc::new(engine));
        }

        let is_encoder_path = model.model_path.ends_with(".safetensors")
            || model.model_path.ends_with(".bin")
            || model.model_path.ends_with(".onnx");

        let is_encoder_task = model.modality == crate::config::Modality::Embedding
            || model.modality == crate::config::Modality::Moderation
            || model.modality == crate::config::Modality::Nlu
            || model.architecture.eq_ignore_ascii_case("modernbert")
            || model.architecture.eq_ignore_ascii_case("encoder")
            || model.id.contains("embedding")
            || model.id.contains("granite")
            || model.id.contains("guard")
            || model.id.contains("intent")
            || model.id.contains("mmbert");

        if is_encoder_task
            || (is_encoder_path && model.modality != crate::config::Modality::ImageGeneration)
        {
            let engine = crate::inference::engines::encoder::EncoderEngine::new(model)?;
            println!(
                "[ModelRegistry] ✓ Successfully loaded encoder engine: \"{}\" ({})",
                model.id, model.name
            );
            return Ok(Arc::new(engine));
        }

        let (n_ctx, context_vram, model_vram_on_gpu, projector_vram, mtp_vram, _total_vram) =
            Self::calculate_vram_and_context(model, server);
        let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers);
        let mtp_path = model.mtp_path.as_deref().map(std::path::Path::new);
        let mtp_heads = model.mtp_heads();
        let extra_args = model.runtime.as_ref().and_then(|r| r.extra_args.clone());

        let engine: Arc<dyn InferenceEngine> = match model.architecture.as_str() {
            "cactus-needle-3" => {
                let engine = crate::inference::engines::needle::NeedleEngine::new(
                    model.id.clone(),
                    model_path,
                )?;
                Arc::new(engine)
            }
            "ternary-bonsai" | "bonsai" | "bonsai-vl" | "ternary-bonsai-vl" => {
                let mmproj_path = model.mmproj_path.as_deref().map(std::path::Path::new);
                let engine = crate::inference::engines::llama_server::LlamaServerEngine::new(
                    model.id.clone(),
                    model_path,
                    mmproj_path,
                    mtp_path,
                    Some(n_ctx),
                    gpu_layers,
                    mtp_heads,
                    extra_args,
                    crate::inference::engines::llama_server::RuntimeFlavor::Prism,
                )?;
                Arc::new(engine)
            }
            _ => {
                let mmproj_path = model.mmproj_path.as_deref().map(std::path::Path::new);
                let engine = crate::inference::engines::llama_server::LlamaServerEngine::new(
                    model.id.clone(),
                    model_path,
                    mmproj_path,
                    mtp_path,
                    Some(n_ctx),
                    gpu_layers,
                    mtp_heads,
                    extra_args,
                    crate::inference::engines::llama_server::RuntimeFlavor::Upstream,
                )?;
                Arc::new(engine)
            }
        };

        let total_est = model_vram_on_gpu + projector_vram + mtp_vram + context_vram;
        let real_str =
            if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                let actual_used = after.saturating_sub(before);
                format!("Real: {}", Self::format_bytes(actual_used))
            } else {
                "Real: N/A".to_string()
            };

        let model_mem_str = if model_vram_on_gpu < model_vram {
            format!(
                "{} | {} (GPU: {})",
                param_info,
                Self::format_bytes(model_vram),
                Self::format_bytes(model_vram_on_gpu)
            )
        } else {
            format!("{} | {}", param_info, Self::format_bytes(model_vram))
        };

        println!(
            "[ModelRegistry] ✓ Successfully loaded engine: \"{}\" ({})",
            model.id, model.name
        );
        println!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
        println!("[ModelRegistry]   • Model:        {}", model_mem_str);

        if projector_vram > 0 || model.vision {
            let res_str = model
                .max_resolution
                .as_deref()
                .unwrap_or("4096×4096 (Dynamic 4K)");
            println!(
                "[ModelRegistry]   • Projector:    {} | {}",
                res_str,
                Self::format_bytes(projector_vram)
            );
        }

        if mtp_vram > 0 || model.has_mtp() {
            let heads_desc = mtp_heads
                .map(|h| format!("{} heads", h))
                .unwrap_or_else(|| "auto".to_string());
            println!(
                "[ModelRegistry]   • MTP Draft:    {} | {}",
                heads_desc,
                Self::format_bytes(mtp_vram)
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
