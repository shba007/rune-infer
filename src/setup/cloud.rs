//! Authenticated cloud model discovery. Model ids never imply capabilities.
use super::ProviderPreset;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::time::{Duration, Instant};

/// None means the provider did not establish support, not that support exists.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudCapabilities {
    pub chat: Option<bool>,
    pub vision: Option<bool>,
    pub tools: Option<bool>,
    pub structured: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudModel {
    pub id: String,
    pub name: String,
    pub context_tokens: Option<u64>,
    pub max_output_tokens: Option<usize>,
    pub capabilities: CloudCapabilities,
    pub metadata: Option<super::ModelMetadata>,
}

impl CloudModel {
    pub fn selection_error(&self) -> Option<&'static str> {
        let Some(metadata) = &self.metadata else {
            return Some("Missing entry in registry/models.json");
        };
        if let Some(error) = metadata.selection_error() {
            return Some(error);
        }
        for (observed, registered) in [
            (self.capabilities.chat, metadata.capabilities.chat),
            (self.capabilities.vision, metadata.capabilities.vision),
            (self.capabilities.tools, metadata.capabilities.tools),
            (
                self.capabilities.structured,
                metadata.capabilities.structured,
            ),
        ] {
            if observed == Some(false) && registered == Some(true) {
                return Some("Provider metadata conflicts with registry; review the entry");
            }
        }
        if self
            .context_tokens
            .zip(metadata.context_tokens)
            .is_some_and(|(observed, registered)| observed < u64::from(registered))
            || self
                .max_output_tokens
                .zip(metadata.max_output_tokens)
                .is_some_and(|(observed, registered)| observed < registered)
        {
            return Some("Provider token limits are lower than the registry; review the entry");
        }
        None
    }
}

/// A successful listing authenticates the key; it does not smoke-test inference
/// access, quota, or billing. OpenRouter's public catalog needs a separate key check.
pub fn list_provider_models(preset: &ProviderPreset, key: &str) -> Result<Vec<CloudModel>, String> {
    if key.trim().is_empty() {
        return Err("Enter a provider API key.".into());
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "Cannot initialize the provider client.")?;
    list_with(&client, preset, key)
}

fn get_json(
    client: &reqwest::blocking::Client,
    preset: &ProviderPreset,
    key: &str,
    path: &str,
    query: &[(&str, String)],
) -> Result<Value, String> {
    let mut url = reqwest::Url::parse(&format!("{}{path}", preset.base_url.trim_end_matches('/')))
        .map_err(|_| "Provider endpoint is invalid.")?;
    if !query.is_empty() {
        url.query_pairs_mut()
            .extend_pairs(query.iter().map(|(name, value)| (*name, value.as_str())));
    }
    let request = client.get(url);
    let request = match preset.provider {
        "google" => request.header("x-goog-api-key", key),
        "anthropic" => request
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01"),
        _ => request.bearer_auth(key),
    };
    let response = request.send().map_err(|error| {
        if error.is_timeout() {
            "Provider request timed out. Retry."
        } else {
            "Cannot connect to the provider. Check the network and retry."
        }
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(match status.as_u16() {
            401 | 403 => {
                "The provider rejected this key or its permissions. Re-enter the key and retry."
                    .into()
            }
            429 => "Provider rate limit reached. Wait and retry.".into(),
            code => format!("Provider returned HTTP {code}. Retry."),
        });
    }
    // Never include provider response bodies or credentials in errors/logs.
    serde_json::from_reader(response.take(8 * 1024 * 1024))
        .map_err(|_| "Provider returned an invalid or oversized model response. Retry.".into())
}

fn list_with(
    client: &reqwest::blocking::Client,
    preset: &ProviderPreset,
    key: &str,
) -> Result<Vec<CloudModel>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    if preset.provider == "openrouter" {
        let response = get_json(client, preset, key, "/key", &[])?;
        if !response.get("data").is_some_and(Value::is_object) {
            return Err("Provider returned invalid key validation data. Retry.".into());
        }
    }
    let mut models = BTreeMap::new();
    let mut cursor: Option<String> = None;
    let mut seen = HashSet::new();
    for _ in 0..64 {
        if Instant::now() >= deadline {
            return Err("Model discovery took too long. Retry.".into());
        }
        let mut query = Vec::new();
        match preset.provider {
            "google" => {
                query.push(("pageSize", "1000".into()));
                if let Some(value) = &cursor {
                    query.push(("pageToken", value.clone()));
                }
            }
            "anthropic" => {
                query.push(("limit", "1000".into()));
                if let Some(value) = &cursor {
                    query.push(("after_id", value.clone()));
                }
            }
            _ => {}
        }
        let response = get_json(client, preset, key, "/models", &query)?;
        let (page, next) = parse_page(preset.provider, &response)?;
        for model in page {
            models.entry(model.id.clone()).or_insert(model);
        }
        if models.len() > 20_000 {
            return Err("Provider returned too many models. Retry.".into());
        }
        match next {
            Some(next) if seen.insert(next.clone()) => cursor = Some(next),
            Some(_) => return Err("Provider repeated its pagination cursor. Retry.".into()),
            None => {
                return if models.is_empty() {
                    Err("No models were returned for this provider key. Retry or choose another provider.".into())
                } else {
                    Ok(models.into_values().collect())
                };
            }
        }
    }
    Err("Provider returned too many model pages. Retry.".into())
}

fn parse_page(
    provider: &str,
    response: &Value,
) -> Result<(Vec<CloudModel>, Option<String>), String> {
    let entries = response
        .get(if provider == "google" {
            "models"
        } else {
            "data"
        })
        .and_then(Value::as_array)
        .ok_or("Provider model list has an unexpected format. Retry.")?;
    let mut models = Vec::new();
    for entry in entries {
        let id = entry
            .get(if provider == "google" { "name" } else { "id" })
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or("Provider returned a model without an id. Retry.")?;
        let id = if provider == "google" {
            id.strip_prefix("models/").unwrap_or(id)
        } else {
            id
        };
        if id.is_empty() || id.chars().any(char::is_control) {
            return Err("Provider returned an empty model id. Retry.".into());
        }
        let name = entry
            .get(if provider == "google" {
                "displayName"
            } else if provider == "anthropic" {
                "display_name"
            } else {
                "name"
            })
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(id)
            .to_string();
        let mut capabilities = CloudCapabilities::default();
        let mut context_tokens = None;
        let mut max_output_tokens = None;
        match provider {
            "google" => {
                if let Some(methods) = entry
                    .get("supportedGenerationMethods")
                    .and_then(Value::as_array)
                {
                    capabilities.chat = Some(
                        methods
                            .iter()
                            .any(|m| m.as_str() == Some("generateContent")),
                    );
                }
                context_tokens = entry.get("inputTokenLimit").and_then(Value::as_u64);
                max_output_tokens = entry.get("outputTokenLimit").and_then(Value::as_u64);
            }
            "anthropic" => {
                // All entries are Claude message models; optional capability data
                // may be absent with the existing pinned API version.
                capabilities.chat = Some(true);
                capabilities.vision = entry
                    .pointer("/capabilities/image_input/supported")
                    .and_then(Value::as_bool);
                capabilities.tools = entry
                    .pointer("/capabilities/tool_use/supported")
                    .and_then(Value::as_bool);
                capabilities.structured = entry
                    .pointer("/capabilities/structured_outputs/supported")
                    .and_then(Value::as_bool);
                context_tokens = entry.get("max_input_tokens").and_then(Value::as_u64);
                max_output_tokens = entry.get("max_tokens").and_then(Value::as_u64);
            }
            "openrouter" => {
                let inputs = entry
                    .pointer("/architecture/input_modalities")
                    .and_then(Value::as_array);
                let outputs = entry
                    .pointer("/architecture/output_modalities")
                    .and_then(Value::as_array);
                capabilities.chat = inputs.zip(outputs).map(|(i, o)| {
                    i.iter().any(|v| v.as_str() == Some("text"))
                        && o.iter().any(|v| v.as_str() == Some("text"))
                });
                capabilities.vision = inputs.map(|i| i.iter().any(|v| v.as_str() == Some("image")));
                if let Some(parameters) =
                    entry.get("supported_parameters").and_then(Value::as_array)
                {
                    capabilities.tools =
                        Some(parameters.iter().any(|v| v.as_str() == Some("tools")));
                    capabilities.structured = if parameters
                        .iter()
                        .any(|v| v.as_str() == Some("structured_outputs"))
                    {
                        Some(true)
                    } else if parameters
                        .iter()
                        .any(|v| v.as_str() == Some("response_format"))
                    {
                        None // response_format alone does not establish JSON-schema support
                    } else {
                        Some(false)
                    };
                }
                context_tokens = entry.get("context_length").and_then(Value::as_u64);
                max_output_tokens = entry
                    .pointer("/top_provider/max_completion_tokens")
                    .and_then(Value::as_u64);
            }
            _ => {}
        }
        if entry.get("lifecycle").and_then(Value::as_str) == Some("retired") {
            capabilities.chat = Some(false);
        }
        models.push(CloudModel {
            id: id.to_string(),
            name,
            context_tokens: context_tokens.filter(|n| *n > 0),
            max_output_tokens: max_output_tokens
                .filter(|n| *n > 0)
                .and_then(|n| usize::try_from(n).ok()),
            capabilities,
            metadata: super::registered_model(provider, id)?,
        });
    }
    let next = if provider == "google" {
        response
            .get("nextPageToken")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    } else if provider == "anthropic"
        && response.get("has_more").and_then(Value::as_bool) == Some(true)
    {
        Some(
            response
                .get("last_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or("Provider omitted its next page cursor. Retry.")?
                .to_string(),
        )
    } else {
        None
    };
    Ok((models, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_formats_preserve_ids_limits_and_only_explicit_metadata() {
        let (google,next) = parse_page("google", &json!({"models":[{"name":"models/example-chat","displayName":"Example","supportedGenerationMethods":["generateContent"],"inputTokenLimit":8192,"outputTokenLimit":1024},{"name":"models/example-embedding","supportedGenerationMethods":["embedContent"]}],"nextPageToken":"abc+/="})).unwrap();
        assert_eq!(google[0].id, "example-chat");
        assert_eq!(google[0].context_tokens, Some(8192));
        assert_eq!(google[0].capabilities.chat, Some(true));
        assert_eq!(google[0].capabilities.vision, None);
        assert_eq!(google[1].capabilities.chat, Some(false));
        assert_eq!(next.as_deref(), Some("abc+/="));
        let (openai, _) = parse_page(
            "openai",
            &json!({"data":[{"id":"example-model","owned_by":"provider"}]}),
        )
        .unwrap();
        assert_eq!(openai[0].capabilities, CloudCapabilities::default());
        let (router,_) = parse_page("openrouter",&json!({"data":[{"id":"vendor/example","architecture":{"input_modalities":["text","image"],"output_modalities":["text"]},"supported_parameters":["tools"],"context_length":16000,"top_provider":{"max_completion_tokens":2048}}]})).unwrap();
        assert_eq!(router[0].id, "vendor/example");
        assert_eq!(router[0].capabilities.vision, Some(true));
        assert_eq!(router[0].capabilities.tools, Some(true));
        assert_eq!(router[0].capabilities.structured, Some(false));
        let (claude,next) = parse_page("anthropic",&json!({"data":[{"id":"example","display_name":"Example","max_input_tokens":4096,"capabilities":{"image_input":{"supported":false}}}],"has_more":true,"last_id":"example"})).unwrap();
        assert_eq!(claude[0].capabilities.vision, Some(false));
        assert_eq!(next.as_deref(), Some("example"));
    }

    #[test]
    fn malformed_catalogs_and_missing_pagination_fail_instead_of_guessing() {
        for (provider, response) in [
            ("google", json!({"data":[]})),
            ("openai", json!({"data":[{}]})),
            ("anthropic", json!({"data":[],"has_more":true})),
        ] {
            assert!(parse_page(provider, &response).is_err());
        }
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn server(
        provider: &'static str,
        responses: Vec<(u16, Value)>,
    ) -> (ProviderPreset, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = Box::leak(format!("http://{}", listener.local_addr().unwrap()).into_boxed_str());
        let thread = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                requests.push(String::from_utf8(request).unwrap());
                let body = body.to_string();
                write!(stream,"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            }
            requests
        });
        (
            ProviderPreset {
                provider,
                base_url: base,
                ..super::super::PROVIDERS[0]
            },
            thread,
        )
    }

    #[test]
    fn google_pages_are_authenticated_encoded_sorted_and_deduplicated() {
        let (preset, server) = server(
            "google",
            vec![
                (
                    200,
                    json!({"models":[{"name":"models/z"},{"name":"models/a"}],"nextPageToken":"abc+/="}),
                ),
                (200, json!({"models":[{"name":"models/a"}]})),
            ],
        );
        let models = list_provider_models(&preset, "fixture-secret").unwrap();
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["a", "z"]
        );
        assert!(models.iter().all(|m| m.selection_error().is_some()));
        let requests = server.join().unwrap();
        assert!(
            requests
                .iter()
                .all(|r| r.to_lowercase().contains("x-goog-api-key: fixture-secret"))
        );
        let target = requests[1]
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap();
        let url = reqwest::Url::parse(&format!("http://fixture{target}")).unwrap();
        assert_eq!(
            url.query_pairs()
                .find(|(name, _)| name == "pageToken")
                .unwrap()
                .1,
            "abc+/="
        );
    }

    #[test]
    fn openrouter_checks_key_before_public_catalog_and_redacts_errors() {
        let (preset, server) = server("openrouter", vec![(401, json!({"error":"fixture-secret"}))]);
        let error = list_provider_models(&preset, "fixture-secret").unwrap_err();
        assert!(!error.contains("fixture-secret"));
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("GET /key HTTP/1.1"));
        assert!(
            requests[0]
                .to_lowercase()
                .contains("authorization: bearer fixture-secret")
        );
        let (preset, server) = self::server(
            "openrouter",
            vec![
                (200, json!({"data":{}})),
                (200, json!({"data":[{"id":"vendor/example"}]})),
            ],
        );
        assert_eq!(
            list_provider_models(&preset, "fixture-secret")
                .unwrap()
                .len(),
            1
        );
        let requests = server.join().unwrap();
        assert!(requests[1].starts_with("GET /models HTTP/1.1"));
    }

    #[test]
    fn repeated_cursors_and_auth_failures_do_not_accept_partial_lists() {
        let (preset, server) = server(
            "google",
            vec![
                (
                    200,
                    json!({"models":[{"name":"models/a"}],"nextPageToken":"same"}),
                ),
                (
                    200,
                    json!({"models":[{"name":"models/b"}],"nextPageToken":"same"}),
                ),
            ],
        );
        assert!(
            list_provider_models(&preset, "fixture-secret")
                .unwrap_err()
                .contains("repeated")
        );
        server.join().unwrap();
        let (preset, server) =
            self::server("anthropic", vec![(403, json!({"error":"fixture-secret"}))]);
        assert!(list_provider_models(&preset, "fixture-secret").is_err());
        let requests = server.join().unwrap();
        assert!(
            requests[0]
                .to_lowercase()
                .contains("anthropic-version: 2023-06-01")
        );
    }
}
