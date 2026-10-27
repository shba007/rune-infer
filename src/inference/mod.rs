pub mod budget;
pub mod engines;
pub mod process;
pub mod registry;
pub mod traits;
pub mod types;

use std::sync::{Arc, Condvar, Mutex};
use traits::InferenceEngine;
use types::{InferenceOutput, InferenceTaskRequest};

pub struct AppState {
    pub registry: Mutex<registry::ModelRegistry>,
    pub load_slot_condvar: Condvar,
}

impl AppState {
    pub fn new(config: &crate::config::ModelRegistry) -> Self {
        Self {
            registry: Mutex::new(registry::ModelRegistry::new(config.clone())),
            load_slot_condvar: Condvar::new(),
        }
    }

    pub fn get_engine(&self, id: &str) -> Result<Arc<dyn InferenceEngine>, String> {
        loop {
            let mut registry = self
                .registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            // 1. Fast path: model is already loaded and in VRAM/RAM
            if let Some(engine) = registry.get_loaded(id) {
                return Ok(engine);
            }

            // Do not start a model while any engine in its fallback plan is being replaced.
            if registry.config.find(id).is_some_and(|model| {
                model
                    .engine_plan()
                    .iter()
                    .any(|kind| registry.updating_engines.contains(kind))
            }) {
                drop(
                    self.load_slot_condvar
                        .wait(registry)
                        .unwrap_or_else(|p| p.into_inner()),
                );
                continue;
            }

            // 2. Coalesce callers: if another thread is currently loading this exact model, wait on its barrier
            if let Some(barrier) = registry.get_loading(id) {
                drop(registry); // Release global registry lock while waiting on load barrier!
                let mut state = barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                while let registry::LoadState::Loading = *state {
                    state = barrier
                        .condvar
                        .wait(state)
                        .unwrap_or_else(|p| p.into_inner());
                }
                match &*state {
                    registry::LoadState::Ready(res) => return res.clone(),
                    registry::LoadState::Loading => unreachable!(),
                }
            }

            // 3. Serialize heavy GPU/VRAM loads: if a different model is currently loading, wait for the slot
            if !registry.loading.is_empty() {
                registry = self
                    .load_slot_condvar
                    .wait(registry)
                    .unwrap_or_else(|p| p.into_inner());
                continue;
            }

            // 4. We are the loader for this model: prepare load (evict LRU if needed, capture config/VRAM)
            let (model_config, server_config, vram_before, barrier) = registry.prepare_load(id)?;

            let canonical_id = model_config.id.clone();

            // Release global registry lock during the heavy engine creation / subprocess spawn / VRAM allocation!
            drop(registry);

            // RAII guard ensures waiting threads and slot condvar are never deadlocked if create_engine panics
            struct LoadGuard<'a> {
                canonical_id: String,
                barrier: Arc<registry::LoadBarrier>,
                state: &'a AppState,
                completed: bool,
            }
            impl<'a> Drop for LoadGuard<'a> {
                fn drop(&mut self) {
                    if !self.completed {
                        let mut reg = self
                            .state
                            .registry
                            .lock()
                            .unwrap_or_else(|p| p.into_inner());
                        let err_msg = format!(
                            "Model '{}' loading panicked or aborted unexpectedly",
                            self.canonical_id
                        );
                        reg.fail_load(&self.canonical_id, &err_msg);
                        let mut st = self.barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                        *st = registry::LoadState::Ready(Err(err_msg));
                        self.barrier.condvar.notify_all();
                        self.state.load_slot_condvar.notify_all();
                    }
                }
            }

            let mut guard = LoadGuard {
                canonical_id: canonical_id.clone(),
                barrier: barrier.clone(),
                state: self,
                completed: false,
            };

            let load_started = std::time::Instant::now();
            let load_res =
                registry::ModelRegistry::create_engine(&model_config, &server_config, vram_before);

            let mut registry = self
                .registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            guard.completed = true;

            match load_res {
                Ok((engine, engine_kind)) => {
                    registry.finish_load(
                        &canonical_id,
                        engine.clone(),
                        engine_kind,
                        load_started,
                        vram_before,
                    );
                    let mut st = barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                    *st = registry::LoadState::Ready(Ok(engine.clone()));
                    barrier.condvar.notify_all();
                    self.load_slot_condvar.notify_all();
                    return Ok(engine);
                }
                Err(e) => {
                    let err_msg = format!("Failed to load model '{}': {}", canonical_id, e);
                    registry.fail_load(&canonical_id, &err_msg);
                    let mut st = barrier.mutex.lock().unwrap_or_else(|p| p.into_inner());
                    *st = registry::LoadState::Ready(Err(err_msg.clone()));
                    barrier.condvar.notify_all();
                    self.load_slot_condvar.notify_all();
                    return Err(err_msg);
                }
            }
        }
    }

    /// Reserve an idle engine for replacement. A live lease includes evicted models
    /// still referenced by requests, and a load in progress may choose any fallback.
    pub fn try_reserve_engine_update(
        self: &Arc<Self>,
        kind: crate::config::EngineKind,
    ) -> Option<EngineUpdateGuard> {
        let mut registry = self.registry.lock().unwrap_or_else(|p| p.into_inner());
        registry
            .engine_leases
            .retain(|(_, lease)| lease.strong_count() > 0);
        if !registry.loading.is_empty()
            || registry.updating_engines.contains(&kind)
            || registry
                .engine_leases
                .iter()
                .any(|(k, lease)| *k == kind && lease.strong_count() > 0)
        {
            return None;
        }
        registry.updating_engines.insert(kind);
        Some(EngineUpdateGuard {
            state: self.clone(),
            kind,
        })
    }

    /// Unload engines idle for at least `idle`; see [`registry::ModelRegistry::evict_idle`].
    pub fn evict_idle(&self, idle: std::time::Duration) -> Vec<String> {
        let mut registry = self
            .registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let evicted = registry.evict_idle(idle);
        if !evicted.is_empty() {
            self.load_slot_condvar.notify_all();
        }
        evicted
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

/// Holds the model-load fence while an engine installation is replaced.
pub struct EngineUpdateGuard {
    state: Arc<AppState>,
    kind: crate::config::EngineKind,
}
impl Drop for EngineUpdateGuard {
    fn drop(&mut self) {
        let mut registry = self
            .state
            .registry
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        registry.updating_engines.remove(&self.kind);
        self.state.load_slot_condvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct MockEngine(&'static str);
    impl traits::InferenceEngine for MockEngine {
        fn id(&self) -> &str {
            self.0
        }
        fn execute(
            &self,
            _: &types::InferenceTaskRequest,
            _: Option<&mut dyn FnMut(&str) -> bool>,
        ) -> Result<types::InferenceOutput, Box<dyn std::error::Error>> {
            Err("not used in mock".into())
        }
    }

    #[test]
    fn update_reservation_waits_for_live_leases_and_loads() {
        use crate::config::EngineKind;
        let state = Arc::new(AppState::new(&crate::config::ModelRegistry::default()));
        let engine: Arc<dyn traits::InferenceEngine> = Arc::new(MockEngine("loaded"));
        {
            let mut registry = state.registry.lock().unwrap();
            registry
                .engine_leases
                .push((EngineKind::AudioCpp, Arc::downgrade(&engine)));
        }
        assert!(
            state
                .try_reserve_engine_update(EngineKind::AudioCpp)
                .is_none()
        );
        assert!(
            state
                .try_reserve_engine_update(EngineKind::LlamaServer)
                .is_some()
        );
        drop(engine);
        {
            let mut registry = state.registry.lock().unwrap();
            registry
                .loading
                .insert("pending".into(), Arc::new(registry::LoadBarrier::new()));
        }
        assert!(
            state
                .try_reserve_engine_update(EngineKind::AudioCpp)
                .is_none()
        );
        state.registry.lock().unwrap().loading.clear();
        let guard = state
            .try_reserve_engine_update(EngineKind::AudioCpp)
            .unwrap();
        assert!(
            state
                .try_reserve_engine_update(EngineKind::AudioCpp)
                .is_none()
        );
        drop(guard);
        assert!(
            state
                .try_reserve_engine_update(EngineKind::AudioCpp)
                .is_some()
        );
    }

    #[test]
    fn model_load_waits_until_update_reservation_is_released() {
        let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
            "id":"update-fixture","name":"fixture","modality":"TextToSpeech",
            "engine":{"supported":["audio-cpp"]},"local":{"path":"Cargo.toml"},"extra":{}
        }))
        .unwrap();
        let mut config = crate::config::ModelRegistry::default();
        config
            .models
            .push(crate::config::ModelConfig::try_from(entry).unwrap());
        let state = Arc::new(AppState::new(&config));
        let guard = state
            .try_reserve_engine_update(crate::config::EngineKind::AudioCpp)
            .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_state = state.clone();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(
                worker_state
                    .get_engine("update-fixture")
                    .map(|engine| engine.id().to_owned()),
            );
        });
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        {
            let mut registry = state.registry.lock().unwrap();
            registry
                .engines
                .insert("update-fixture".into(), Arc::new(MockEngine("fixture")));
        }
        drop(guard);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap().unwrap(),
            "fixture"
        );
        worker.join().unwrap();
    }

    #[test]
    fn test_get_loaded_fast_path() {
        let state = AppState::new(&crate::config::ModelRegistry::default());
        {
            let mut reg = state.registry.lock().unwrap();
            reg.engines
                .insert("m1".to_string(), Arc::new(MockEngine("m1")));
        }

        let engine = state.get_engine("m1").expect("should return loaded model");
        assert_eq!(engine.id(), "m1");
    }

    #[test]
    fn test_concurrent_waiting_on_loading_barrier() {
        let state = Arc::new(AppState::new(&crate::config::ModelRegistry::default()));

        // Simulate model "m2" starting to load
        let barrier = Arc::new(registry::LoadBarrier::new());
        {
            let mut reg = state.registry.lock().unwrap();
            reg.loading.insert("m2".to_string(), barrier.clone());
        }

        let handles: Vec<_> = (0..5)
            .map(|_| {
                let state_clone = state.clone();
                std::thread::spawn(move || state_clone.get_engine("m2"))
            })
            .collect();

        // Give threads time to hit the barrier and wait on condvar
        std::thread::sleep(Duration::from_millis(50));

        // Complete loading
        let loaded_engine: Arc<dyn traits::InferenceEngine> = Arc::new(MockEngine("m2"));
        {
            let mut reg = state.registry.lock().unwrap();
            reg.engines.insert("m2".to_string(), loaded_engine.clone());
            reg.loading.remove("m2");
            let mut st = barrier.mutex.lock().unwrap();
            *st = registry::LoadState::Ready(Ok(loaded_engine.clone()));
            barrier.condvar.notify_all();
        }

        for h in handles {
            let res = h.join().unwrap();
            assert!(res.is_ok());
            assert_eq!(res.unwrap().id(), "m2");
        }
    }

    #[test]
    fn test_concurrent_waiting_on_loading_barrier_failure() {
        let state = Arc::new(AppState::new(&crate::config::ModelRegistry::default()));

        let barrier = Arc::new(registry::LoadBarrier::new());
        {
            let mut reg = state.registry.lock().unwrap();
            reg.loading.insert("m3".to_string(), barrier.clone());
        }

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let state_clone = state.clone();
                std::thread::spawn(move || state_clone.get_engine("m3"))
            })
            .collect();

        std::thread::sleep(Duration::from_millis(50));

        // Simulate load failure
        {
            let mut reg = state.registry.lock().unwrap();
            reg.loading.remove("m3");
            let mut st = barrier.mutex.lock().unwrap();
            *st = registry::LoadState::Ready(Err("Simulated VRAM OOM error".to_string()));
            barrier.condvar.notify_all();
        }

        for h in handles {
            let res = h.join().unwrap();
            match res {
                Err(err) => assert!(err.contains("Simulated VRAM OOM error")),
                Ok(_) => panic!("Expected error from failed loading barrier"),
            }
        }
    }

    #[test]
    fn test_smart_lru_eviction_prefers_idle_models() {
        let mut reg = registry::ModelRegistry::new(crate::config::ModelRegistry::default());
        let e1: Arc<dyn traits::InferenceEngine> = Arc::new(MockEngine("e1"));
        let e2: Arc<dyn traits::InferenceEngine> = Arc::new(MockEngine("e2"));

        reg.engines.insert("e1".to_string(), e1.clone());
        reg.lru_keys.push("e1".to_string());

        reg.engines.insert("e2".to_string(), e2);
        reg.lru_keys.push("e2".to_string());

        // e1 has an in-flight reference held outside the registry (strong_count = 2)
        assert_eq!(Arc::strong_count(&e1), 2);
        assert_eq!(Arc::strong_count(reg.engines.get("e2").unwrap()), 1);

        // Even though e1 is older (index 0 in lru_keys), evict_lru should evict idle e2
        let evicted = reg.evict_lru();
        assert_eq!(evicted, Some("e2".to_string()));
        assert!(reg.engines.contains_key("e1"));
        assert!(!reg.engines.contains_key("e2"));
    }
}
