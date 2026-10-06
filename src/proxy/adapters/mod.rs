pub mod anthropic;
pub mod gemini;
pub mod openai;

use axum::body::Bytes;
use axum::response::Response;
use std::pin::Pin;
use tokio_stream::Stream;

use crate::config::{ModelConfig, ProviderType};
use crate::types::{ApiError, ChatCompletionRequest};

pub type BoxEventStream = Pin<
    Box<dyn Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> + Send>,
>;

/// Dispatches a chat completion request to the corresponding cloud provider adapter.
pub async fn dispatch_chat(
    client: &reqwest::Client,
    config: &ModelConfig,
    request: &ChatCompletionRequest,
) -> Result<Response, ApiError> {
    match config.provider {
        ProviderType::OpenAi | ProviderType::OpenRouter | ProviderType::Custom => {
            openai::execute_chat(client, config, request).await
        }
        ProviderType::Anthropic => anthropic::execute_chat(client, config, request).await,
        ProviderType::Google => gemini::execute_chat(client, config, request).await,
        ProviderType::Local => Err(
            ApiError::new("Local models must use internal engine registry")
                .with_type("invalid_request_error"),
        ),
    }
}

/// Helper function to parse an incoming reqwest byte stream into individual SSE data lines.
/// Normalizes CRLF (\r\n) to LF (\n) to universally match any provider's SSE framing.
pub fn parse_sse_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
) -> impl Stream<Item = String> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        use tokio_stream::StreamExt;

        while let Some(item) = stream.next().await {
            if let Ok(bytes) = item {
                let chunk = String::from_utf8_lossy(&bytes);
                buffer.push_str(&chunk);

                // Universal newline normalization: converts \r\n and \r into \n
                buffer = buffer.replace("\r\n", "\n").replace('\r', "\n");

                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    let mut event_data = String::new();
                    for line in event_block.lines() {
                        let trimmed = line.trim();
                        if let Some(data) = trimmed.strip_prefix("data:") {
                            let clean_data = data.trim();
                            if !clean_data.is_empty() {
                                if !event_data.is_empty() {
                                    event_data.push('\n');
                                }
                                event_data.push_str(clean_data);
                            }
                        }
                    }

                    if !event_data.is_empty() {
                        yield event_data;
                    }
                }
            }
        }

        // Flush any trailing buffer data
        buffer = buffer.replace("\r\n", "\n").replace('\r', "\n");
        let mut event_data = String::new();
        for line in buffer.lines() {
            let trimmed = line.trim();
            if let Some(data) = trimmed.strip_prefix("data:") {
                let clean_data = data.trim();
                if !clean_data.is_empty() {
                    if !event_data.is_empty() {
                        event_data.push('\n');
                    }
                    event_data.push_str(clean_data);
                }
            }
        }
        if !event_data.is_empty() {
            yield event_data;
        }
    }
}

/// Helper function to parse data URI into MIME type and base64 string.
pub fn parse_data_uri(uri: &str) -> Option<(String, String)> {
    if let Some(rest) = uri.strip_prefix("data:") {
        if let Some((header, b64)) = rest.split_once(',') {
            let mime = header.split(';').next().unwrap_or("image/jpeg");
            return Some((mime.to_string(), b64.trim().to_string()));
        }
    }
    None
}
