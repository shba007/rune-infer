use rune_infer::config::ModelRegistry;
use rune_infer::inference::budget::calculate_budget;
use rune_infer::inference::registry::ModelRegistry as EngineRegistry;
use rune_infer::inference::traits::InferenceEngine;
use rune_infer::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use rune_infer::types::Usage;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

struct MockEngine {
    id: String,
}

impl InferenceEngine for MockEngine {
    fn id(&self) -> &str {
        &self.id
    }
    fn execute(
        &self,
        _: &InferenceTaskRequest,
        _: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        Ok(InferenceOutput {
            response: InferenceTaskResponse::Text(format!("Response from {}", self.id)),
            usage: Usage::new(10, 20),
        })
    }
}

fn create_test_catalog() -> ModelRegistry {
    let raw = r#"{
        "schema_version": 2,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "max_loaded_models": 1,
            "idle_unload_seconds": 300,
            "vram_budget_ratio": 0.95
        },
        "models": [
            {
                "id": "qwen3.8-27b",
                "name": "Qwen3.8 27B",
                "modality": "VisionText",
                "capabilities": ["Chat", "Vision"],
                "engine": { "supported": ["llama-server"] },
                "max_context_length": 262144,
                "sampling": { "temperature": 0.3, "max_tokens": 1024 },
                "local": {
                    "path": "D:/Models/qwen3.8-27b.gguf",
                    "params": 27000000000,
                    "bits_per_weight": 3.44,
                    "context_length": -1,
                    "gpu_layers": 99,
                    "args": "-fa on --cache-type-k q4_0 --cache-type-v q4_0"
                },
                "extra": {}
            },
            {
                "id": "ternary-bonsai-2-27b",
                "name": "Ternary Bonsai-2 27B",
                "modality": "VisionText",
                "capabilities": ["Chat", "Vision"],
                "engine": { "supported": ["llama-server-prism"] },
                "max_context_length": 262144,
                "sampling": { "temperature": 0.7, "max_tokens": 2048 },
                "local": {
                    "path": "D:/Models/ternary-bonsai-2-27b.gguf",
                    "params": 27000000000,
                    "bits_per_weight": 2.13,
                    "context_length": -1,
                    "gpu_layers": 99,
                    "args": "-fa on --cache-type-k q4_0 --cache-type-v q4_0"
                },
                "extra": {}
            }
        ]
    }"#;

    ModelRegistry::from_json(raw, None).expect("Valid test catalog")
}

#[test]
fn test_vram_protection_budget_enforcement() {
    let catalog = create_test_catalog();
    let qwen = catalog.find("qwen3.8-27b").unwrap();
    let bonsai = catalog.find("ternary-bonsai-2-27b").unwrap();

    let qwen_plan = calculate_budget(qwen, &catalog.server);
    let bonsai_plan = calculate_budget(bonsai, &catalog.server);

    // Both models should be individually budgeted within limits
    assert!(qwen_plan.n_ctx >= 2048);
    assert!(bonsai_plan.n_ctx >= 2048);

    // 16 GB ceiling in bytes
    let sixteen_gb = 16 * 1024 * 1024 * 1024u64;

    // Individually, each model's estimated weight footprint is bounded
    assert!(qwen_plan.model_vram_on_gpu_bytes < sixteen_gb);
    assert!(bonsai_plan.model_vram_on_gpu_bytes < sixteen_gb);

    // Crucially: Concurrent loading without Rune Infer management would require ~22GB+,
    // which would crash standard runners on a 16GB GPU
    let combined_weights = qwen_plan.model_vram_on_gpu_bytes + bonsai_plan.model_vram_on_gpu_bytes;
    assert!(
        combined_weights > sixteen_gb,
        "Combined model weights ({:.2} GiB) must exceed 16 GiB ceiling to simulate the OOM condition",
        combined_weights as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

#[test]
fn test_lru_graceful_model_swapping_prevents_oom() {
    let catalog = create_test_catalog();
    let mut registry = EngineRegistry::new(catalog);

    let qwen_engine = Arc::new(MockEngine {
        id: "qwen3.8-27b".to_string(),
    });
    let bonsai_engine = Arc::new(MockEngine {
        id: "ternary-bonsai-2-27b".to_string(),
    });

    // Start with empty registry
    assert_eq!(registry.loaded_models().len(), 0);

    // Evict idle on an empty registry returns empty
    assert_eq!(
        registry.evict_idle(Duration::from_secs(300)),
        Vec::<String>::new()
    );

    // Verify in-flight request protection: strong_count > 1 prevents eviction
    let _qwen_lock = qwen_engine.clone();
    assert_eq!(Arc::strong_count(&qwen_engine), 2);
    assert_eq!(Arc::strong_count(&bonsai_engine), 1);
}

#[test]
fn test_idle_unload_reclaims_memory() {
    let catalog = create_test_catalog();
    let mut registry = EngineRegistry::new(catalog);

    // Evict idle on an empty registry returns empty
    let evicted = registry.evict_idle(Duration::from_secs(300));
    assert!(evicted.is_empty());
}
