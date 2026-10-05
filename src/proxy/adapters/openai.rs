use axum::Json;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt;

use crate::config::ModelConfig;
use crate::proxy::adapters::{BoxEventStream, parse_sse_stream};
use crate::types::{ApiError, ChatCompletionRequest, ChatCompletionResponse};

pub async fn execute_chat(
    client: &reqwest::Client,
    config: &ModelConfig,
    request: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let base_url = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");
    let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let api_key = config.api_key.as_deref().unwrap_or("");
    let upstream_model = config
        .upstream_model
        .clone()
        .unwrap_or_else(|| request.model.clone());

    let mut body = serde_json::to_value(request).map_err(|e| {
        ApiError::new(format!("Failed to serialize request: {e}"))
            .with_type("invalid_request_error")
    })?;

    if let Some(obj) = body.as_object_mut() {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(upstream_model),
        );
    }

    let mut req_builder = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json");

    if config.provider == crate::config::ProviderType::OpenRouter
        || endpoint.contains("openrouter.ai")
    {
        req_builder = req_builder
            .header("HTTP-Referer", "https://github.com/shba007/rune-infer")
            .header("X-Title", "Rune Infer");
    }

    let resp = req_builder.json(&body).send().await.map_err(|e| {
        ApiError::new(format!("Failed to connect to OpenAI endpoint: {e}")).with_type("api_error")
    })?;

    let status = resp.status();
    if !status.is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(
            ApiError::new(format!("OpenAI returned HTTP {status}: {err_text}"))
                .with_type("upstream_error")
                .with_code(status.as_u16().to_string()),
        );
    }

    if request.stream {
        let stream = resp.bytes_stream();
        let sse_stream: BoxEventStream = Box::pin(async_stream::stream! {
            let data_stream = parse_sse_stream(Box::pin(stream));
            tokio::pin!(data_stream);

            while let Some(data) = data_stream.next().await {
                yield Ok(Event::default().data(data));
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
            .into_response())
    } else {
        let parsed: ChatCompletionResponse = resp.json().await.map_err(|e| {
            ApiError::new(format!("Failed to parse OpenAI JSON response: {e}"))
                .with_type("api_error")
        })?;
        Ok(Json(parsed).into_response())
    }
}
