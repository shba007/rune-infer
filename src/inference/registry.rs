use crate::config::{EngineKind, ModelConfig, ModelRegistry as ConfigRegistry, ServerConfig};
use crate::inference::traits::InferenceEngine;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub enum LoadState {
    Loading,
    Ready(Result<Arc<dyn InferenceEngine>, String>),
}

pub struct LoadBarrier {
    pub mutex: Mutex<LoadState>,
    pub condvar: Condvar,
}

impl LoadBarrier {
    pub fn new() -> Self {
        Self {
            mutex: Mutex::new(LoadState::Loading),
            condvar: Condvar::new(),
        }
    }
}

impl Default for LoadBarrier {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ModelRegistry {
    pub(crate) config: ConfigRegistry,
    pub(crate) engines: HashMap<String, Arc<dyn InferenceEngine>>,
    pub(crate) engine_configs: HashMap<String, ModelConfig>,
    pub(crate) lru_keys: Vec<String>,
    pub(crate) last_used: HashMap<String, Instant>,
    pub(crate) loading: HashMap<String, Arc<LoadBarrier>>,
    pub(crate) load_durations: HashMap<String, u64>,
    pub(crate) last_evicted: Option<String>,
    pub(crate) load_evictions: HashMap<String, Option<String>>,
    pub(crate) engine_leases: Vec<(EngineKind, std::sync::Weak<dyn InferenceEngine>)>,
    pub(crate) updating_engines: std::collections::HashSet<EngineKind>,
}

impl ModelRegistry {
    pub fn new(config: ConfigRegistry) -> Self {
        tracing::info!(
            "[ModelRegistry] Initialized model catalog with {} models configured.",
            config.models.len()
        );
        tracing::info!(
            "[ModelRegistry] Lazy loading enabled: No models are preloaded into VRAM/RAM."
        );
        Self {
            config,
            engines: HashMap::new(),
            engine_configs: HashMap::new(),
            lru_keys: Vec::new(),
            last_used: HashMap::new(),
            loading: HashMap::new(),
            load_durations: HashMap::new(),
            last_evicted: None,
            load_evictions: HashMap::new(),
            engine_leases: Vec::new(),
            updating_engines: std::collections::HashSet::new(),
        }
    }

    /// Unload engines that have been idle for at least `idle`, returning their
    /// ids. An engine that is still referenced elsewhere is mid-request: it is
    /// never evicted and its idle clock restarts.
    pub fn evict_idle(&mut self, idle: Duration) -> Vec<String> {
        let now = Instant::now();
        let mut victims = Vec::new();
        for (id, engine) in &self.engines {
            if Arc::strong_count(engine) > 1 {
                self.last_used.insert(id.clone(), now);
                continue;
            }
            let since = self
                .last_used
                .get(id)
                .map_or(Duration::MAX, |t| now.duration_since(*t));
            if since >= idle {
                victims.push(id.clone());
            }
        }
        for id in &victims {
            self.engines.remove(id);
            self.engine_configs.remove(id);
            self.last_used.remove(id);
            self.lru_keys.retain(|k| k != id);
            crate::telemetry::model_evicted(id, &format!("idle for {}s", idle.as_secs()));
        }
        victims
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.engines.keys().cloned().collect()
    }

    pub fn loading_models(&self) -> Vec<String> {
        self.loading.keys().cloned().collect()
    }

    pub fn reload_catalog(&mut self, new_config: ConfigRegistry) {
        self.config = new_config;
    }

    /// Check if a model is already loaded and ready in memory.
    pub fn get_loaded(&mut self, id: &str) -> Option<Arc<dyn InferenceEngine>> {
        if let Some(engine) = self.engines.get(id).cloned() {
            self.touch_lru(id);
            self.last_used.insert(id.to_string(), Instant::now());
            return Some(engine);
        }
        if let Some(canonical) = self.config.find(id).map(|m| m.id.clone()) {
            if let Some(engine) = self.engines.get(&canonical).cloned() {
                self.touch_lru(&canonical);
                self.last_used.insert(canonical, Instant::now());
                return Some(engine);
            }
        }
        None
    }

    /// Check if a model is currently in the process of being loaded.
    pub fn get_loading(&self, id: &str) -> Option<Arc<LoadBarrier>> {
        if let Some(barrier) = self.loading.get(id).cloned() {
            return Some(barrier);
        }
        if let Some(canonical) = self.config.find(id).map(|m| &m.id) {
            return self.loading.get(canonical).cloned();
        }
        None
    }

    /// Prepare state to load a model: verifies configuration, evicts LRU if capacity
    /// is reached, registers a LoadBarrier, and returns configurations needed to create
    /// the engine outside the lock.
    pub fn prepare_load(
        &mut self,
        id: &str,
    ) -> Result<(ModelConfig, ServerConfig, Option<u64>, Arc<LoadBarrier>), String> {
        let model_config = self
            .config
            .find(id)
            .ok_or_else(|| format!("Model '{}' not found in models.json", id))?
            .clone();

        self.last_evicted = None;
        let max_loaded = self.config.server.max_loaded_models;
        if max_loaded > 0 && self.engines.len() >= max_loaded {
            self.evict_lru();
        }

        tracing::info!(
            "[ModelRegistry] Lazy-loading model '{}' ({}) on demand...",
            model_config.id,
            model_config.name
        );

        let vram_before = Self::detect_used_gpu_vram();
        crate::telemetry::model_loading(&model_config.id);

        let barrier = Arc::new(LoadBarrier::new());
        self.loading
            .insert(model_config.id.clone(), barrier.clone());

        Ok((
            model_config,
            self.config.server.clone(),
            vram_before,
            barrier,
        ))
    }

    /// Record successful load into the registry and telemetry, clearing the loading entry.
    pub fn finish_load(
        &mut self,
        model: &ModelConfig,
        engine: Arc<dyn InferenceEngine>,
        engine_kind: EngineKind,
        load_started: Instant,
        vram_before: Option<u64>,
    ) {
        // The catalog may have reloaded while the engine was starting.
        let canonical_id = model.id.clone();

        let vram_detail = match (vram_before, Self::detect_used_gpu_vram()) {
            (Some(before), Some(after)) => format!(
                "{engine_kind} · +{} VRAM",
                Self::format_bytes(after.saturating_sub(before))
            ),
            _ => engine_kind.to_string(),
        };
        let elapsed_ms = load_started.elapsed().as_millis() as u64;
        self.load_durations.insert(canonical_id.clone(), elapsed_ms);
        self.load_evictions
            .insert(canonical_id.clone(), self.last_evicted.clone());
        crate::telemetry::model_loaded(&canonical_id, elapsed_ms as u128, &vram_detail);

        self.engine_leases
            .push((engine_kind, Arc::downgrade(&engine)));
        self.engines.insert(canonical_id.clone(), engine);
        self.engine_configs
            .insert(canonical_id.clone(), model.clone());
        self.touch_lru(&canonical_id);
        self.last_used.insert(canonical_id.clone(), Instant::now());
        self.loading.remove(&canonical_id);
    }

    pub fn get_load_duration_ms(&self, id: &str) -> Option<u64> {
        self.load_durations.get(id).copied()
    }

    pub fn last_evicted(&self) -> Option<String> {
        self.last_evicted.clone()
    }

    pub fn get_load_eviction(&self, id: &str) -> Option<String> {
        self.load_evictions.get(id).cloned().flatten()
    }

    pub fn is_model_loaded(&self, id: &str) -> bool {
        self.engines.contains_key(id)
    }

    /// Record load failure in telemetry and clear the loading entry.
    pub fn fail_load(&mut self, id: &str, err_msg: &str) {
        let canonical_id = self
            .config
            .find(id)
            .map(|m| m.id.clone())
            .unwrap_or_else(|| id.to_string());
        crate::telemetry::model_failed(&canonical_id, err_msg);
        self.loading.remove(&canonical_id);
    }

    pub fn get_or_load(&mut self, id: &str) -> Result<Arc<dyn InferenceEngine>, String> {
        if let Some(engine) = self.get_loaded(id) {
            return Ok(engine);
        }

        let (model_config, server_config, vram_before, barrier) = self.prepare_load(id)?;
        let load_started = Instant::now();
        let canonical_id = model_config.id.clone();

        match Self::create_engine(&model_config, &server_config, vram_before) {
            Ok((engine, engine_kind)) => {
                self.finish_load(
                    &model_config,
                    engine.clone(),
                    engine_kind,
                    load_started,
                    vram_before,
                );
                let mut st = barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                *st = LoadState::Ready(Ok(engine.clone()));
                barrier.condvar.notify_all();
                Ok(engine)
            }
            Err(e) => {
                let err_msg = format!("Failed to load model '{}': {}", canonical_id, e);
                self.fail_load(&canonical_id, &err_msg);
                let mut st = barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                *st = LoadState::Ready(Err(err_msg.clone()));
                barrier.condvar.notify_all();
                Err(err_msg)
            }
        }
    }

    pub fn touch_lru(&mut self, id: &str) {
        if let Some(pos) = self.lru_keys.iter().position(|k| k == id) {
            let key = self.lru_keys.remove(pos);
            self.lru_keys.push(key);
        } else {
            self.lru_keys.push(id.to_string());
        }
    }

    pub fn evict_lru(&mut self) -> Option<String> {
        if self.lru_keys.is_empty() {
            return None;
        }
        // Prefer evicting an engine that is not currently referenced by an active request
        let victim_idx = self
            .lru_keys
            .iter()
            .position(|id| {
                self.engines
                    .get(id)
                    .is_none_or(|eng| Arc::strong_count(eng) <= 1)
            })
            .unwrap_or(0);

        let victim_id = self.lru_keys.remove(victim_idx);
        self.last_used.remove(&victim_id);
        if let Some(_removed) = self.engines.remove(&victim_id) {
            self.engine_configs.remove(&victim_id);
            self.last_evicted = Some(victim_id.clone());
            crate::telemetry::model_evicted(&victim_id, "least recently used");
            tracing::info!(
                "[ModelRegistry] Evicted least-recently-used model '{}' to respect max_loaded_models ({})",
                victim_id,
                self.config.server.max_loaded_models
            );
            Some(victim_id)
        } else {
            None
        }
    }

    // --- Delegated Formatting & VRAM Helpers ---

    #[inline]
    pub fn format_bytes(bytes: u64) -> String {
        crate::inference::budget::format_bytes(bytes)
    }

    #[inline]
    pub fn format_params(params: u64) -> String {
        crate::inference::budget::format_params(params)
    }

    #[inline]
    pub fn format_tokens(tokens: u32) -> String {
        crate::inference::budget::format_tokens(tokens)
    }

    #[inline]
    pub fn detect_total_gpu_vram() -> Option<u64> {
        crate::inference::budget::detect_total_gpu_vram()
    }

    #[inline]
    pub fn detect_used_gpu_vram() -> Option<u64> {
        crate::inference::budget::detect_used_gpu_vram()
    }

    /// Start an engine for `model`, trying the configured engines in order
    /// (preferred first) until one starts. Returns the engine and which kind won.
    pub fn create_engine(
        model: &ModelConfig,
        server: &ServerConfig,
        vram_before: Option<u64>,
    ) -> Result<(Arc<dyn InferenceEngine>, EngineKind), Box<dyn std::error::Error>> {
        if !std::path::Path::new(&model.model_path).exists() {
            return Err(format!(
                "Model '{}' file not found at path: {}",
                model.id, model.model_path
            )
            .into());
        }

        Self::start_first_working(model, |kind| {
            Self::build_engine(kind, model, server, vram_before)
        })
    }

    /// Walk `model.engine_plan()`, returning the first engine `build` can start.
    fn start_first_working(
        model: &ModelConfig,
        mut build: impl FnMut(
            EngineKind,
        ) -> Result<Arc<dyn InferenceEngine>, Box<dyn std::error::Error>>,
    ) -> Result<(Arc<dyn InferenceEngine>, EngineKind), Box<dyn std::error::Error>> {
        let plan = model.engine_plan();
        let mut failures: Vec<String> = Vec::new();
        for (attempt, kind) in plan.iter().copied().enumerate() {
            if kind == EngineKind::Encoder {
                failures.push(
                    "encoder: synthetic output is unavailable for inference; configure a real inference engine"
                        .to_string(),
                );
                continue;
            }
            match build(kind) {
                Ok(engine) => {
                    if attempt > 0 {
                        tracing::warn!(
                            "[ModelRegistry] '{}' fell back to engine '{}' ({})",
                            model.id,
                            kind,
                            failures.join("; ")
                        );
                    }
                    return Ok((engine, kind));
                }
                Err(e) => {
                    tracing::warn!(
                        "[ModelRegistry] engine '{}' failed for '{}': {}",
                        kind,
                        model.id,
                        e
                    );
                    failures.push(format!("{kind}: {e}"));
                }
            }
        }
        Err(failures.join("; ").into())
    }

    fn build_engine(
        kind: EngineKind,
        model: &ModelConfig,
        server: &ServerConfig,
        vram_before: Option<u64>,
    ) -> Result<Arc<dyn InferenceEngine>, Box<dyn std::error::Error>> {
        let model_path = std::path::Path::new(&model.model_path);

        let model_vram = std::fs::metadata(&model.model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let param_info = if let Some(p) = model.total_params {
            format!("{} params", Self::format_params(p))
        } else {
            "Weights".to_string()
        };

        let capabilities_str = model.resolved_capabilities().join(", ");
        if kind == EngineKind::Litert {
            return Ok(Arc::new(
                crate::inference::engines::litert::LiteRtEngine::new(model)?,
            ));
        }

        if kind == EngineKind::SdServer {
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

            tracing::info!(
                "[ModelRegistry] ✓ Successfully loaded engine: \"{}\" ({})",
                model.id,
                model.name
            );
            tracing::info!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
            tracing::info!(
                "[ModelRegistry]   • Model:        {} | {}",
                param_info,
                Self::format_bytes(model_vram)
            );
            if text_encoder_vram > 0 {
                tracing::info!(
                    "[ModelRegistry]   • Text Encoder: {}",
                    Self::format_bytes(text_encoder_vram)
                );
            }
            if vae_vram > 0 {
                tracing::info!(
                    "[ModelRegistry]   • VAE:          {}",
                    Self::format_bytes(vae_vram)
                );
            }
            tracing::info!("[ModelRegistry]   • Resolution:   {}", res_str);
            tracing::info!(
                "[ModelRegistry]   • Sampling:     {} steps | {:.1} CFG | {}",
                steps,
                cfg,
                sampler
            );
            tracing::info!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(total_est),
                real_str
            );

            return Ok(Arc::new(engine));
        }

        if kind == EngineKind::AudioCpp {
            let engine = crate::inference::engines::audio::AudioEngine::new(model)?;
            tracing::info!(
                "[ModelRegistry] ✓ Successfully loaded Audio engine (audio.cpp): \"{}\"",
                model.id
            );
            return Ok(Arc::new(engine));
        }

        if kind == EngineKind::NemoSpeech {
            let engine = crate::inference::engines::nemo_speech::NemoSpeechEngine::new(model)?;
            let real_str =
                if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                    let actual_used = after.saturating_sub(before);
                    format!("Real: {}", Self::format_bytes(actual_used))
                } else {
                    "Real: N/A".to_string()
                };

            tracing::info!(
                "[ModelRegistry] ✓ Successfully loaded NeMo-Speech engine: \"{}\" ({})",
                model.id,
                model.name
            );
            tracing::info!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
            tracing::info!(
                "[ModelRegistry]   • Model:        {} | {}",
                param_info,
                Self::format_bytes(model_vram)
            );
            tracing::info!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(model_vram + 1024 * 1024 * 1024),
                real_str
            );
            return Ok(Arc::new(engine));
        }

        if kind == EngineKind::Crispasr {
            let engine = crate::inference::engines::crispasr::CrispAsrEngine::new(model)?;
            let real_str =
                if let (Some(before), Some(after)) = (vram_before, Self::detect_used_gpu_vram()) {
                    let actual_used = after.saturating_sub(before);
                    format!("Real: {}", Self::format_bytes(actual_used))
                } else {
                    "Real: N/A".to_string()
                };

            tracing::info!(
                "[ModelRegistry] ✓ Successfully loaded engine: \"{}\" ({})",
                model.id,
                model.name
            );
            tracing::info!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
            tracing::info!(
                "[ModelRegistry]   • Model:        {} | {}",
                param_info,
                Self::format_bytes(model_vram)
            );
            tracing::info!(
                "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
                Self::format_bytes(model_vram + 1024 * 1024 * 1024),
                real_str
            );
            return Ok(Arc::new(engine));
        }

        if kind == EngineKind::WhisperCpp {
            return Ok(Arc::new(
                crate::inference::engines::whisper::WhisperCppEngine::new(model)?,
            ));
        }

        if kind == EngineKind::Encoder {
            return Err("encoder: synthetic output is unavailable for inference".into());
        }

        let budget_plan = crate::inference::budget::calculate_budget(model, server);
        let n_ctx = budget_plan.n_ctx;
        let context_vram = budget_plan.context_vram_bytes;
        let model_vram_on_gpu = budget_plan.model_vram_on_gpu_bytes;
        let projector_vram = budget_plan.projector_vram_bytes;
        let mtp_vram = budget_plan.mtp_vram_bytes;
        let gpu_layers = model.runtime.as_ref().map(|r| r.gpu_layers);
        let mtp_path = model.mtp_path.as_deref().map(std::path::Path::new);
        let mtp_heads = model.mtp_heads();
        let mut extra_args = model.runtime.as_ref().and_then(|r| r.extra_args.clone());
        if model.modality == crate::config::Modality::Embedding {
            let mut args = extra_args.unwrap_or_default();
            if !args.contains("--embedding") && !args.contains("-embd") {
                args = format!("--embedding {args}").trim().to_string();
            }
            extra_args = Some(args);
        }

        let engine: Arc<dyn InferenceEngine> = match kind {
            EngineKind::Needle => {
                let engine = crate::inference::engines::needle::NeedleEngine::new(
                    model.id.clone(),
                    model_path,
                )?;
                Arc::new(engine)
            }
            EngineKind::LlamaServerPrism => {
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

        let total_est = budget_plan.total_estimated_vram_bytes();
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

        tracing::info!(
            "[ModelRegistry] ✓ Successfully loaded engine: \"{}\" ({})",
            model.id,
            model.name
        );
        tracing::info!("[ModelRegistry]   • Capabilities: {}", capabilities_str);
        tracing::info!("[ModelRegistry]   • Model:        {}", model_mem_str);

        if projector_vram > 0 || model.is_vision() {
            let res_str = model
                .max_resolution
                .as_deref()
                .unwrap_or("4096×4096 (Dynamic 4K)");
            tracing::info!(
                "[ModelRegistry]   • Projector:    {} | {}",
                res_str,
                Self::format_bytes(projector_vram)
            );
        }

        if mtp_vram > 0 || model.has_mtp() {
            let heads_desc = mtp_heads
                .map(|h| format!("{} heads", h))
                .unwrap_or_else(|| "auto".to_string());
            tracing::info!(
                "[ModelRegistry]   • MTP Draft:    {} | {}",
                heads_desc,
                Self::format_bytes(mtp_vram)
            );
        }

        tracing::info!(
            "[ModelRegistry]   • Context:      {} | {}",
            Self::format_tokens(n_ctx),
            Self::format_bytes(context_vram)
        );
        tracing::info!(
            "[ModelRegistry]   • Total VRAM:   Est: {} | {}",
            Self::format_bytes(total_est),
            real_str
        );

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::{InferenceOutput, InferenceTaskRequest};
    use std::error::Error;

    struct Dummy(&'static str);
    impl InferenceEngine for Dummy {
        fn id(&self) -> &str {
            self.0
        }
        fn execute(
            &self,
            _: &InferenceTaskRequest,
            _: Option<&mut dyn FnMut(&str) -> bool>,
        ) -> Result<InferenceOutput, Box<dyn Error>> {
            Err("not used".into())
        }
    }

    fn registry_with(ids: &[&'static str]) -> ModelRegistry {
        let mut r = ModelRegistry::new(ConfigRegistry::default());
        for id in ids {
            r.engines.insert(id.to_string(), Arc::new(Dummy(id)));
            r.lru_keys.push(id.to_string());
            r.last_used.insert(id.to_string(), Instant::now());
        }
        r
    }

    #[test]
    fn idle_engines_are_unloaded_and_fresh_ones_kept() {
        let mut r = registry_with(&["old", "fresh"]);
        r.last_used
            .insert("old".into(), Instant::now() - Duration::from_secs(600));

        let gone = r.evict_idle(Duration::from_secs(300));
        assert_eq!(gone, vec!["old".to_string()]);
        assert_eq!(r.loaded_models(), vec!["fresh".to_string()]);
        assert_eq!(r.lru_keys, vec!["fresh".to_string()]);
        assert!(!r.last_used.contains_key("old"));
    }

    #[test]
    fn engine_in_use_is_never_evicted_and_its_clock_restarts() {
        let mut r = registry_with(&["busy"]);
        r.last_used
            .insert("busy".into(), Instant::now() - Duration::from_secs(600));
        let in_flight = r.engines.get("busy").cloned().unwrap(); // a handler holds it

        assert!(r.evict_idle(Duration::from_secs(300)).is_empty());
        assert!(r.last_used["busy"].elapsed() < Duration::from_secs(5));

        drop(in_flight);
        assert!(
            r.evict_idle(Duration::from_secs(300)).is_empty(),
            "clock restarted"
        );
        assert!(
            !r.evict_idle(Duration::ZERO).is_empty(),
            "idle once released"
        );
    }

    fn model_json(path: &str, engine: &str) -> ModelConfig {
        let entry: crate::schema::ModelEntry = serde_json::from_str(&format!(
            r#"{{"id":"m","name":"M","modality":"Embedding","local":{{"path":{path:?}}}{engine}}}"#
        ))
        .unwrap();
        ModelConfig::try_from(entry).unwrap()
    }

    #[test]
    fn missing_model_file_fails_fast_before_any_engine() {
        let m = model_json(
            "Z:/definitely/missing.safetensors",
            r#","engine":{"supported":["encoder"]}"#,
        );
        let err = ModelRegistry::create_engine(&m, &Default::default(), None)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("file not found"), "{err}");
    }

    fn two_engine_model() -> ModelConfig {
        model_json(
            "x",
            r#","engine":{"supported":["crispasr","nemo-speech"],"preferred":"nemo-speech"}"#,
        )
    }

    #[test]
    fn synthetic_encoder_is_skipped_in_direct_and_fallback_plans() {
        for (plan, real_succeeds) in [
            (r#", "engine":["encoder"]"#, false),
            (r#", "engine":["llama-server","encoder"]"#, false),
            (r#", "engine":["encoder","llama-server"]"#, true),
        ] {
            let model = model_json("Cargo.toml", plan);
            let mut tried = Vec::new();
            let result = ModelRegistry::start_first_working(&model, |kind| {
                tried.push(kind);
                if real_succeeds {
                    Ok(Arc::new(Dummy("real")) as Arc<dyn InferenceEngine>)
                } else {
                    Err("real engine failed".into())
                }
            });
            assert!(!tried.contains(&EngineKind::Encoder), "{plan}");
            if real_succeeds {
                assert_eq!(result.unwrap().1, EngineKind::LlamaServer);
            } else {
                let error = result
                    .err()
                    .expect("synthetic output must be refused")
                    .to_string();
                assert!(
                    error.contains("encoder: synthetic output is unavailable"),
                    "{error}"
                );
                if !tried.is_empty() {
                    assert!(error.contains("real engine failed"), "{error}");
                } else {
                    assert!(
                        ModelRegistry::create_engine(&model, &Default::default(), None).is_err()
                    );
                }
            }
            assert_eq!(tried.len(), usize::from(plan.contains("llama-server")));
        }
    }
    #[test]
    fn falls_back_to_the_next_supported_engine_when_preferred_fails() {
        let mut tried = Vec::new();
        let (engine, kind) = ModelRegistry::start_first_working(&two_engine_model(), |k| {
            tried.push(k);
            if k == EngineKind::NemoSpeech {
                Err("binary missing".into())
            } else {
                Ok(Arc::new(Dummy("ok")) as Arc<dyn InferenceEngine>)
            }
        })
        .unwrap();
        assert_eq!(tried, vec![EngineKind::NemoSpeech, EngineKind::Crispasr]);
        assert_eq!(kind, EngineKind::Crispasr);
        assert_eq!(engine.id(), "ok");
    }

    #[test]
    fn preferred_engine_that_works_is_the_only_one_started() {
        let mut tried = Vec::new();
        let (_, kind) = ModelRegistry::start_first_working(&two_engine_model(), |k| {
            tried.push(k);
            Ok(Arc::new(Dummy("ok")) as Arc<dyn InferenceEngine>)
        })
        .unwrap();
        assert_eq!((kind, tried.len()), (EngineKind::NemoSpeech, 1));
    }

    #[test]
    fn when_every_engine_fails_the_error_names_each_one() {
        let err = ModelRegistry::start_first_working(&two_engine_model(), |k| {
            Err(format!("{k} broke").into())
        })
        .err()
        .expect("all failed")
        .to_string();
        assert_eq!(
            err,
            "nemo-speech: nemo-speech broke; crispasr: crispasr broke"
        );
    }
}
