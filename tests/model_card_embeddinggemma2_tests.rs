use rune_infer::config::{EngineKind, Modality, ModelRegistry};
use rune_infer::detect::{Classified, GgufInfo, classify};
use rune_infer::inference::engines::encoder::EncoderEngine;
use rune_infer::inference::traits::InferenceEngine;
use rune_infer::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use std::path::Path;

fn load_test_registry() -> ModelRegistry {
    let server_path = Path::new("config/server.json");
    let catalog_path = Path::new("config/catalog.json");
    let overrides = vec![
        ("GEMINI_API_KEY".to_string(), "test-gemini-key".to_string()),
        (
            "OPENROUTER_API_KEY".to_string(),
            "test-openrouter-key".to_string(),
        ),
        ("RUNE_API_KEY".to_string(), "test-rune-key".to_string()),
    ];
    ModelRegistry::from_files_with_env(server_path, catalog_path, &overrides)
        .expect("Failed to parse server and catalog configuration files")
}

#[test]
fn test_embeddinggemma2_catalog_configuration() {
    let registry = load_test_registry();

    let model = registry
        .find("google/embeddinggemma-2")
        .expect("Model 'google/embeddinggemma-2' must be present in catalog");

    assert_eq!(model.id, "google/embeddinggemma-2");
    assert_eq!(model.name, "Google EmbeddingGemma 2");
    assert_eq!(model.modality, Modality::Embedding);

    let engine = model
        .engine
        .as_ref()
        .expect("Local model must have engine config");
    assert_eq!(engine.supported, vec![EngineKind::LlamaServer]);
    assert_eq!(engine.preferred, Some(EngineKind::LlamaServer));
    assert_eq!(model.max_context_length, Some(8192));
    assert!(model.model_path.contains("embeddinggemma-2"));
    assert!(
        model
            .capabilities
            .as_ref()
            .unwrap()
            .iter()
            .any(|c| c.contains("Matryoshka") || c.contains("Embeddings"))
    );
}

#[test]
fn test_embeddinggemma2_bidirectional_and_case_insensitive_lookup() {
    let registry = load_test_registry();

    // Exact HF model card slug lookup
    let exact = registry.find("google/embeddinggemma-2");
    assert!(exact.is_some(), "Exact lookup failed");
    assert_eq!(exact.unwrap().id, "google/embeddinggemma-2");

    // Short slug lookup (omitting organization prefix)
    let short = registry.find("embeddinggemma-2");
    assert!(
        short.is_some(),
        "Short name resolution 'embeddinggemma-2' failed"
    );
    assert_eq!(short.unwrap().id, "google/embeddinggemma-2");

    // Case-insensitive lookups
    let upper_exact = registry.find("GOOGLE/EMBEDDINGGEMMA-2");
    assert!(upper_exact.is_some(), "Case-insensitive lookup failed");
    assert_eq!(upper_exact.unwrap().id, "google/embeddinggemma-2");

    let upper_short = registry.find("EmbeddingGemma-2");
    assert!(
        upper_short.is_some(),
        "Case-insensitive short lookup failed"
    );
    assert_eq!(upper_short.unwrap().id, "google/embeddinggemma-2");

    // Model display name lookup
    let by_name = registry.find("Google EmbeddingGemma 2");
    assert!(by_name.is_some(), "Lookup by display name failed");
    assert_eq!(by_name.unwrap().id, "google/embeddinggemma-2");
}

#[test]
fn test_detect_classify_embeddinggemma() {
    // 1. Architecture is explicitly "embeddinggemma"
    let info1 = GgufInfo {
        architecture: "embeddinggemma".to_string(),
        name: Some("embeddinggemma-2".to_string()),
        ..Default::default()
    };
    match classify(&info1, "embeddinggemma-2.gguf") {
        Classified::Model(detected) => {
            assert_eq!(detected.modality, Modality::Embedding);
            assert_eq!(detected.engines, vec![EngineKind::LlamaServer]);
            assert_eq!(detected.kind_label, "embedding");
        }
        other => panic!("Expected Classified::Model, got {:?}", other),
    }

    // 2. Architecture is "gemma" and file name contains "embedding"
    let info2 = GgufInfo {
        architecture: "gemma2".to_string(),
        name: Some("google/embeddinggemma-2".to_string()),
        ..Default::default()
    };
    match classify(&info2, "embeddinggemma-2-Q4_K_M.gguf") {
        Classified::Model(detected) => {
            assert_eq!(detected.modality, Modality::Embedding);
            assert_eq!(detected.engines, vec![EngineKind::LlamaServer]);
        }
        other => panic!("Expected Classified::Model, got {:?}", other),
    }
}

#[test]
fn test_matryoshka_representation_learning_mrl_truncation() {
    let registry = load_test_registry();

    let model = registry.find("google/embeddinggemma-2").unwrap();
    let engine = EncoderEngine::new(model).expect("Encoder engine creation");

    // Default full dimension: 768
    let req_full = InferenceTaskRequest::Embedding {
        model: Some(model.id.clone()),
        input: vec!["Antigravity code intelligence search query".to_string()],
        dimensions: None,
    };
    let out_full = engine.execute(&req_full, None).unwrap();
    let emb_full = match out_full.response {
        InferenceTaskResponse::Embedding(resp) => resp.data[0].embedding.clone(),
        _ => panic!("Expected Embedding response"),
    };
    assert_eq!(emb_full.len(), 768);

    // MRL 512 dimensions
    let req_512 = InferenceTaskRequest::Embedding {
        model: Some(model.id.clone()),
        input: vec!["Antigravity code intelligence search query".to_string()],
        dimensions: Some(512),
    };
    let out_512 = engine.execute(&req_512, None).unwrap();
    let emb_512 = match out_512.response {
        InferenceTaskResponse::Embedding(resp) => resp.data[0].embedding.clone(),
        _ => panic!("Expected Embedding response"),
    };
    assert_eq!(emb_512.len(), 512);

    // MRL 256 dimensions
    let req_256 = InferenceTaskRequest::Embedding {
        model: Some(model.id.clone()),
        input: vec!["Antigravity code intelligence search query".to_string()],
        dimensions: Some(256),
    };
    let out_256 = engine.execute(&req_256, None).unwrap();
    let emb_256 = match out_256.response {
        InferenceTaskResponse::Embedding(resp) => resp.data[0].embedding.clone(),
        _ => panic!("Expected Embedding response"),
    };
    assert_eq!(emb_256.len(), 256);

    // MRL 128 dimensions
    let req_128 = InferenceTaskRequest::Embedding {
        model: Some(model.id.clone()),
        input: vec!["Antigravity code intelligence search query".to_string()],
        dimensions: Some(128),
    };
    let out_128 = engine.execute(&req_128, None).unwrap();
    let emb_128 = match out_128.response {
        InferenceTaskResponse::Embedding(resp) => resp.data[0].embedding.clone(),
        _ => panic!("Expected Embedding response"),
    };
    assert_eq!(emb_128.len(), 128);

    // Verify Matryoshka Representation Learning directional invariance property under L2 normalization
    let norm_sub512: f32 = emb_512[..128].iter().map(|v| v * v).sum::<f32>().sqrt();
    let renorm512: Vec<f32> = emb_512[..128].iter().map(|v| v / norm_sub512).collect();
    for (a, b) in renorm512.iter().zip(emb_128.iter()) {
        assert!((a - b).abs() < 1e-5);
    }

    let norm_sub256: f32 = emb_256[..128].iter().map(|v| v * v).sum::<f32>().sqrt();
    let renorm256: Vec<f32> = emb_256[..128].iter().map(|v| v / norm_sub256).collect();
    for (a, b) in renorm256.iter().zip(emb_128.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_llama_server_embedding_flag_injection() {
    let registry = load_test_registry();

    let model = registry.find("google/embeddinggemma-2").unwrap();
    assert_eq!(model.modality, Modality::Embedding);

    let mut extra_args = model.runtime.as_ref().and_then(|r| r.extra_args.clone());
    if model.modality == Modality::Embedding {
        let mut args = extra_args.unwrap_or_default();
        if !args.contains("--embedding") && !args.contains("-embd") {
            args = format!("--embedding {args}").trim().to_string();
        }
        extra_args = Some(args);
    }

    assert!(extra_args.is_some());
    assert!(extra_args.unwrap().contains("--embedding"));
}
