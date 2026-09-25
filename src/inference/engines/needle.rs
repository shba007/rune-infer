use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::Usage;
use needle_infer::v3_engine::V3Engine;
use std::error::Error;
use std::path::Path;
use std::sync::Mutex;

pub struct NeedleEngine {
    id: String,
    inner: Mutex<V3Engine>,
}

impl NeedleEngine {
    pub fn new(id: String, model_path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(format!("Model path does not exist: {}", model_path.display()).into());
        }
        let engine = V3Engine::load(model_path)?;
        Ok(Self {
            id,
            inner: Mutex::new(engine),
        })
    }

    fn extract_tool_call_json(raw: &str) -> String {
        if let Some(start) = raw.find("< tool_call>") {
            let content_start = start + "< tool_call>".len();
            if let Some(end) = raw[content_start..].find("< /tool_call>") {
                return raw[content_start..content_start + end].trim().to_string();
            }
            return raw[content_start..].trim().to_string();
        }

        let mut cleaned = raw;
        if let Some(think_end) = raw.find("< /think>") {
            cleaned = &raw[think_end + "< /think>".len()..];
        }

        if let (Some(first_b), Some(last_b)) = (cleaned.find('['), cleaned.rfind(']')) {
            if first_b < last_b {
                return cleaned[first_b..=last_b].trim().to_string();
            }
        }
        if let (Some(first_b), Some(last_b)) = (cleaned.find('{'), cleaned.rfind('}')) {
            if first_b < last_b {
                return cleaned[first_b..=last_b].trim().to_string();
            }
        }

        cleaned.trim().to_string()
    }
}

impl InferenceEngine for NeedleEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::ToolCall { prompt, schema, .. } => {
                let engine = self
                    .inner
                    .lock()
                    .map_err(|e| format!("Lock error: {}", e))?;
                let schema_str = schema.to_string();

                let prompt_tokens = ((prompt.len() + schema_str.len()) / 4).max(1) as u32;
                let raw_output = engine.run(prompt, &schema_str);
                let clean_json = Self::extract_tool_call_json(&raw_output);
                let completion_tokens = (clean_json.len() / 4).max(1) as u32;

                let parsed: serde_json::Value = serde_json::from_str(&clean_json)
                    .unwrap_or_else(|_| serde_json::Value::String(clean_json));

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::ToolCall(parsed),
                    usage: Usage::new(prompt_tokens, completion_tokens),
                })
            }
        }
    }
}
