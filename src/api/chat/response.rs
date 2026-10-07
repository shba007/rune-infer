use crate::types::{ChatCompletionResponse, Choice, ChoiceDelta, ResponseMessage, ToolCall, Usage};

/// Constructs an SSE chunk for chat completions.
pub fn make_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: ChoiceDelta,
    finish_reason: Option<String>,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: Some("fp_rune_rust_v1".to_string()),
        choices: vec![Choice {
            index: 0,
            message: None,
            delta: Some(delta),
            logprobs: None,
            finish_reason,
        }],
        usage,
    }
}

/// Constructs a full, non-streaming ChatCompletionResponse.
/// If the model spent all tokens in reasoning (<think>), it falls back to
/// providing think_part so desktop clients don't see an empty response.
pub fn chat_completion_response(
    model: &str,
    raw_content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    finish_reason: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionResponse {
    let (reasoning_content, content) = match raw_content {
        Some(text) => {
            if let Some(start) = text.find("<think>") {
                if let Some(end) = text.find("</think>") {
                    let think_part = text[start + "<think>".len()..end].trim().to_string();
                    let rem = format!("{}{}", &text[..start], &text[end + "</think>".len()..])
                        .trim()
                        .to_string();
                    (
                        if think_part.is_empty() {
                            None
                        } else {
                            Some(think_part.clone())
                        },
                        if rem.is_empty() {
                            if think_part.is_empty() {
                                None
                            } else {
                                Some(think_part)
                            }
                        } else {
                            Some(rem)
                        },
                    )
                } else {
                    (None, Some(text))
                }
            } else {
                (None, Some(text))
            }
        }
        None => (None, None),
    };

    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        system_fingerprint: Some("fp_rune_rust_v1".to_string()),
        choices: vec![Choice {
            index: 0,
            message: Some(ResponseMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content,
                tool_calls,
                refusal: None,
            }),
            delta: None,
            logprobs: None,
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage,
    }
}
