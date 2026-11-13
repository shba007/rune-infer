//! Reviewed remote cloud metadata. Add entries to registry/models.json; no id
//! prefix or family-name guessing is used and setup never edits the registry.
use super::CloudCapabilities;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegistryStatus {
    Active,
    Deprecated,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub status: RegistryStatus,
    pub capabilities: CloudCapabilities,
    pub modalities_in: Vec<String>,
    pub modalities_out: Vec<String>,
    pub context_tokens: Option<u32>,
    pub max_output_tokens: Option<usize>,
    pub tool_choice_modes: Vec<String>,
    pub notes: String,
    pub verified_on: String,
    pub sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks: Option<crate::schema::BenchmarkMetrics>,
}

impl ModelMetadata {
    pub fn selection_error(&self) -> Option<&'static str> {
        if self.status == RegistryStatus::Unavailable {
            Some("Unavailable in the reviewed registry")
        } else if self.capabilities.chat != Some(true) {
            Some("Chat compatibility is not established in the registry")
        } else {
            None
        }
    }
    pub fn catalog_capabilities(&self) -> Vec<String> {
        [
            ("Chat", self.capabilities.chat),
            ("Vision", self.capabilities.vision),
            ("Tool Call", self.capabilities.tools),
            ("Structured Output", self.capabilities.structured),
        ]
        .into_iter()
        .filter(|(_, supported)| *supported == Some(true))
        .map(|(label, _)| label.to_string())
        .collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema_version: u32,
    models: Vec<ModelMetadata>,
    #[serde(default)]
    local_models: Vec<serde_json::Value>,
}
fn parse_registry(raw: &str) -> Result<Vec<ModelMetadata>, String> {
    let registry: Registry =
        serde_json::from_str(raw).map_err(|e| format!("Invalid model registry: {e}"))?;
    if registry.schema_version != 1 {
        return Err("Unsupported model registry schema.".into());
    }
    let _ = &registry.local_models;
    let mut seen = std::collections::HashSet::new();
    for model in &registry.models {
        if !seen.insert((&model.provider, &model.id)) {
            return Err(format!(
                "Duplicate registry model: {} / {}",
                model.provider, model.id
            ));
        }
        if !matches!(
            model.provider.as_str(),
            "google" | "openai" | "anthropic" | "openrouter"
        ) || model.id.trim().is_empty()
            || model.id.trim() != model.id
            || model.id.chars().any(char::is_control)
            || model.name.trim().is_empty()
        {
            return Err("Registry model needs an explicit provider, id and name.".into());
        }
        if model.sources.is_empty()
            || model
                .sources
                .iter()
                .any(|s| reqwest::Url::parse(s).map_or(true, |url| url.scheme() != "https"))
            || chrono::NaiveDate::parse_from_str(&model.verified_on, "%Y-%m-%d").is_err()
        {
            return Err(format!("{} needs dated HTTPS source references.", model.id));
        }
        if model.capabilities.vision == Some(true)
            && !model.modalities_in.iter().any(|m| m == "image")
        {
            return Err(format!("{} declares vision without image input.", model.id));
        }
        if model.capabilities.chat == Some(true)
            && (!model.modalities_in.iter().any(|m| m == "text")
                || !model.modalities_out.iter().any(|m| m == "text"))
        {
            return Err(format!(
                "{} declares chat without text input/output.",
                model.id
            ));
        }
        if model.context_tokens == Some(0) || model.max_output_tokens == Some(0) {
            return Err(format!("{} has invalid token limits.", model.id));
        }
    }
    Ok(registry.models)
}

pub(crate) struct CloudRegistry(Vec<ModelMetadata>);

#[cfg(test)]
thread_local! {
    static TEST_DOCUMENT: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_fixture_registry<T>(work: impl FnOnce() -> T) -> T {
    struct Restore(Option<Vec<u8>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_DOCUMENT.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }
    let previous = TEST_DOCUMENT
        .with(|slot| slot.replace(Some(include_bytes!("../../registry/models.json").to_vec())));
    let _restore = Restore(previous);
    work()
}

impl CloudRegistry {
    pub(crate) fn fetch() -> Result<Self, String> {
        #[cfg(test)]
        if let Some(raw) = TEST_DOCUMENT.with(|slot| slot.borrow().clone()) {
            return Self::from_fetch(|| Ok(raw));
        }
        Self::from_fetch(|| {
            crate::model_registry::fetch_document().map_err(|_| {
                "Cannot fetch the remote model registry. Check the network and retry.".to_string()
            })
        })
    }
    fn from_fetch(fetch: impl FnOnce() -> Result<Vec<u8>, String>) -> Result<Self, String> {
        let raw = fetch()?;
        let text = std::str::from_utf8(&raw)
            .map_err(|_| "Invalid remote model registry encoding.".to_string())?;
        parse_registry(text).map(Self)
    }
    pub(crate) fn get(&self, provider: &str, id: &str) -> Option<ModelMetadata> {
        self.0
            .iter()
            .find(|m| m.provider == provider && m.id == id)
            .cloned()
    }
}

pub fn registered_model(provider: &str, id: &str) -> Result<Option<ModelMetadata>, String> {
    Ok(CloudRegistry::fetch()?.get(provider, id))
}

#[cfg(test)]
pub(crate) fn fixture_registry() -> CloudRegistry {
    CloudRegistry(parse_registry(include_str!("../../registry/models.json")).unwrap())
}

#[cfg(test)]
pub(crate) fn fixture_model(provider: &str, id: &str) -> Result<Option<ModelMetadata>, String> {
    Ok(fixture_registry().get(provider, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registry_fixture_is_valid_unique_and_exactly_scoped() {
        let models = parse_registry(include_str!("../../registry/models.json")).unwrap();
        assert_eq!(models.len(), 7);
        for m in models {
            assert!(fixture_model(&m.provider, &m.id).unwrap().is_some());
            assert!(fixture_model("other-provider", &m.id).unwrap().is_none());
            assert!(
                fixture_model(&m.provider, &format!("{}-unknown-suffix", m.id))
                    .unwrap()
                    .is_none()
            );
        }
    }
    #[test]
    fn duplicate_entries_and_missing_provenance_are_errors() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../../registry/models.json")).unwrap();
        let first = value["models"][0].clone();
        value["models"].as_array_mut().unwrap().push(first);
        assert!(parse_registry(&value.to_string()).is_err());
        value["models"].as_array_mut().unwrap().pop();
        value["models"][0]["sources"] = serde_json::json!([]);
        assert!(parse_registry(&value.to_string()).is_err());
    }

    #[test]
    fn remote_errors_fail_without_fallback_and_later_fetches_retry() {
        assert!(CloudRegistry::from_fetch(|| Err("offline".into())).is_err());
        assert!(CloudRegistry::from_fetch(|| Ok(b"not json".to_vec())).is_err());
        let registry =
            CloudRegistry::from_fetch(|| Ok(include_bytes!("../../registry/models.json").to_vec()))
                .unwrap();
        assert!(registry.get("google", "gemini-3.8-flash").is_some());
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;
    use crate::setup::{CloudChoice, PROVIDERS};
    #[test]
    fn catalog_uses_reviewed_fields_without_changing_generation_defaults() {
        let metadata = fixture_model("google", "gemini-3.8-flash")
            .unwrap()
            .unwrap();
        let choice = CloudChoice {
            preset: PROVIDERS[0],
            api_key: "fixture-secret".into(),
            model: metadata.clone(),
        };
        let entry = crate::setup::cloud_entry(&choice);
        assert_eq!(entry.max_context_length, metadata.context_tokens);
        assert!(
            entry.sampling.max_tokens.is_none(),
            "provider output limit must not become a sampling default"
        );
        assert_eq!(entry.remote.unwrap().upstream_model, metadata.id);
        let metadata = fixture_model("openrouter", "apodex/apodex-1.1-mini:free")
            .unwrap()
            .unwrap();
        let choice = CloudChoice {
            preset: PROVIDERS[3],
            api_key: "fixture-secret".into(),
            model: metadata,
        };
        let entry = crate::setup::cloud_entry(&choice);
        assert_eq!(entry.modality, crate::config::Modality::Text);
        assert_eq!(entry.capabilities, ["Chat"]);
    }
    #[test]
    fn live_contradictions_and_unknown_ids_cannot_be_selected() {
        let metadata = fixture_model("google", "gemini-3.8-flash")
            .unwrap()
            .unwrap();
        let mut model = crate::setup::CloudModel {
            id: metadata.id.clone(),
            name: metadata.name.clone(),
            context_tokens: None,
            max_output_tokens: None,
            capabilities: crate::setup::CloudCapabilities::default(),
            metadata: Some(metadata),
        };
        assert!(model.selection_error().is_none());
        model.capabilities.vision = Some(false);
        assert!(model.selection_error().is_some());
        model.capabilities.vision = None;
        model.context_tokens = Some(10);
        assert!(model.selection_error().is_some());
        model.context_tokens = None;
        model.metadata = None;
        assert!(model.selection_error().is_some());
        let unavailable = fixture_model("openrouter", "stealth/space-bunny-alpha")
            .unwrap()
            .unwrap();
        assert!(unavailable.selection_error().is_some());
    }
}
