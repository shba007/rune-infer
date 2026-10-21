use rune_infer::config::{ModelConfig, SamplingConfig};
use rune_infer::schema::ModelEntry;
use rune_infer::types::{ChatCompletionRequest, ChatMessage, NormalizedThinking};
use serde_json::json;

fn sample_qwen_reasoning_config() -> ModelConfig {
    let raw = r#"{
        "id": "qwen-reasoning",
        "name": "Qwen Reasoning",
        "modality": "Text",
        "capabilities": ["Chat", "Reasoning"],
        "engine": { "supported": ["llama-server"], "preferred": "llama-server" },
        "max_context_length": 32768,
        "sampling": { "temperature": 0.6, "max_tokens": 4096 },
        "local": {
            "path": "Cargo.toml",
            "params": 7000000000,
            "bits_per_weight": 4.0,
            "context_length": 16384,
            "gpu_layers": 99,
            "args": ""
        },
        "extra": {}
    }"#;
    let entry: ModelEntry = serde_json::from_str(raw).expect("valid json");
    ModelConfig::try_from(entry).expect("valid model config")
}

#[test]
fn test_normalized_thinking_parsing() {
    // 1. OpenAI reasoning_effort "none" / "low" / "medium" / "high"
    let req_none = ChatCompletionRequest {
        model: "o3-mini".to_string(),
        stream: false,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some(rune_infer::types::MessageContent::Text("Hi".to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        temperature: None,
        top_p: None,
        max_tokens: None,
        top_k: None,
        min_p: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        reasoning_effort: Some("none".to_string()),
        thinking: None,
        chat_template_kwargs: None,
        stream_options: None,
    };
    assert_eq!(req_none.normalized_thinking(), NormalizedThinking::Disabled);

    let mut req_high = req_none.clone();
    req_high.reasoning_effort = Some("high".to_string());
    assert_eq!(
        req_high.normalized_thinking(),
        NormalizedThinking::Enabled {
            effort: Some("high".to_string()),
            budget_tokens: None
        }
    );

    // 2. Anthropic style object: {"type": "disabled"} / {"type": "enabled", "budget_tokens": 2048}
    let mut req_anthropic_disabled = req_none.clone();
    req_anthropic_disabled.reasoning_effort = None;
    req_anthropic_disabled.thinking = Some(json!({ "type": "disabled" }));
    assert_eq!(
        req_anthropic_disabled.normalized_thinking(),
        NormalizedThinking::Disabled
    );

    let mut req_anthropic_enabled = req_none.clone();
    req_anthropic_enabled.reasoning_effort = None;
    req_anthropic_enabled.thinking = Some(json!({ "type": "enabled", "budget_tokens": 2048 }));
    assert_eq!(
        req_anthropic_enabled.normalized_thinking(),
        NormalizedThinking::Enabled {
            effort: None,
            budget_tokens: Some(2048)
        }
    );

    // 3. Boolean thinking: false / true
    let mut req_bool_false = req_none.clone();
    req_bool_false.reasoning_effort = None;
    req_bool_false.thinking = Some(json!(false));
    assert_eq!(
        req_bool_false.normalized_thinking(),
        NormalizedThinking::Disabled
    );

    // 4. chat_template_kwargs: {"thinking": false}
    let mut req_kwargs = req_none.clone();
    req_kwargs.reasoning_effort = None;
    req_kwargs.chat_template_kwargs = Some(json!({ "thinking": false }));
    assert_eq!(
        req_kwargs.normalized_thinking(),
        NormalizedThinking::Disabled
    );
}

#[test]
fn test_sampling_config_layering_with_reasoning_effort() {
    let defaults = SamplingConfig {
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: None,
        min_p: None,
        max_tokens: Some(2048),
        reasoning_effort: Some("medium".to_string()),
    };

    // Unset request inherits default
    let req_empty = SamplingConfig::default();
    let effective = req_empty.over(&defaults);
    assert_eq!(effective.reasoning_effort, Some("medium".to_string()));

    // Explicit request overrides default
    let req_override = SamplingConfig {
        reasoning_effort: Some("none".to_string()),
        ..Default::default()
    };
    let effective_override = req_override.over(&defaults);
    assert_eq!(
        effective_override.reasoning_effort,
        Some("none".to_string())
    );
}

#[test]
fn test_local_engine_disables_thinking_in_prompt_and_task() {
    let model_config = sample_qwen_reasoning_config();

    let req = ChatCompletionRequest {
        model: "qwen-reasoning".to_string(),
        stream: false,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some(rune_infer::types::MessageContent::Text(
                "Solve 2+2".to_string(),
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        temperature: None,
        top_p: None,
        max_tokens: Some(512),
        top_k: None,
        min_p: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        reasoning_effort: Some("none".to_string()),
        thinking: None,
        chat_template_kwargs: None,
        stream_options: None,
    };

    let prepared = rune_infer::api::chat::prepare_prompt(&req, &model_config)
        .expect("Prompt preparation must succeed");

    match prepared.task {
        rune_infer::inference::types::InferenceTaskRequest::ToolCall {
            prompt,
            reasoning_effort,
            thinking,
            chat_template_kwargs,
            ..
        } => {
            assert_eq!(reasoning_effort, Some("none".to_string()));
            assert!(
                prompt.contains("Do not output any <think> tags"),
                "Prompt must inject directive instructing model not to output <think> tags"
            );
            assert!(thinking.is_none());
            assert!(chat_template_kwargs.is_none());
        }
        _ => panic!("Expected ToolCall task variant"),
    }
}

#[test]
fn test_chat_completion_request_serde_roundtrip() {
    let json_data = json!({
        "model": "claude-3-7-sonnet",
        "messages": [
            { "role": "user", "content": "Hello world" }
        ],
        "reasoning_effort": "medium",
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4096
        },
        "chat_template_kwargs": {
            "thinking": true
        }
    });

    let req: ChatCompletionRequest = serde_json::from_value(json_data).expect("Must deserialize");
    assert_eq!(req.reasoning_effort, Some("medium".to_string()));
    assert_eq!(
        req.thinking,
        Some(json!({ "type": "enabled", "budget_tokens": 4096 }))
    );
    assert_eq!(req.chat_template_kwargs, Some(json!({ "thinking": true })));

    let serialized = serde_json::to_value(&req).expect("Must serialize");
    assert_eq!(serialized["reasoning_effort"], "medium");
    assert_eq!(serialized["thinking"]["budget_tokens"], 4096);
    assert_eq!(serialized["chat_template_kwargs"]["thinking"], true);
}
