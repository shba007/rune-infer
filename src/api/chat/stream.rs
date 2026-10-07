use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::response::make_chunk;
use super::tools::convert_to_tool_calls;
use crate::config::ModelConfig;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ChoiceDelta, ToolCallChunk, Usage};

pub fn handle_chat_stream(
    task: InferenceTaskRequest,
    engine: Arc<dyn InferenceEngine>,
    model_config: ModelConfig,
    created: u64,
    start_time: std::time::Instant,
    image_count: usize,
    video_count: usize,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let model_id = model_config.id.clone();

    let initial_chunk = make_chunk(
        &format!("chatcmpl-{created}"),
        created,
        &model_id,
        ChoiceDelta {
            content: Some(String::new()),
            reasoning_content: None,
            role: Some("assistant".to_string()),
            tool_calls: None,
        },
        None,
        Usage::default(),
    );
    let _ =
        tx.send(Event::default().data(serde_json::to_string(&initial_chunk).unwrap_or_default()));

    tokio::task::spawn_blocking(move || {
        let tx_clone = tx.clone();
        let model_id_clone = model_id.clone();
        let mut completion_tokens = 0u32;
        let has_streamed_content = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let has_streamed_clone = has_streamed_content.clone();
        let mut is_thinking = false;
        let mut tool_call_buffering = false;

        let mut on_token = move |piece: &str| -> bool {
            completion_tokens += 1;

            if piece.contains("<think>") {
                is_thinking = true;
                let clean = piece
                    .replace("<think>", "")
                    .trim_start_matches('\n')
                    .to_string();
                if !clean.is_empty() {
                    has_streamed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                    let chunk = make_chunk(
                        &format!("chatcmpl-{created}"),
                        created,
                        &model_id_clone,
                        ChoiceDelta {
                            content: None,
                            reasoning_content: Some(clean),
                            role: None,
                            tool_calls: None,
                        },
                        None,
                        Usage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                        },
                    );
                    let _ = tx_clone.send(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                    );
                }
                return true;
            }

            if piece.contains("</think>") {
                is_thinking = false;
                let clean = piece.replace("</think>", "").trim_matches('\n').to_string();
                if !clean.is_empty() {
                    has_streamed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                    let chunk = make_chunk(
                        &format!("chatcmpl-{created}"),
                        created,
                        &model_id_clone,
                        ChoiceDelta {
                            content: Some(clean),
                            reasoning_content: None,
                            role: None,
                            tool_calls: None,
                        },
                        None,
                        Usage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                        },
                    );
                    let _ = tx_clone.send(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                    );
                }
                return true;
            }

            if is_thinking {
                has_streamed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                let chunk = make_chunk(
                    &format!("chatcmpl-{created}"),
                    created,
                    &model_id_clone,
                    ChoiceDelta {
                        content: None,
                        reasoning_content: Some(piece.to_string()),
                        role: None,
                        tool_calls: None,
                    },
                    None,
                    Usage {
                        prompt_tokens: 0,
                        completion_tokens,
                        total_tokens: completion_tokens,
                    },
                );
                return tx_clone
                    .send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
                    .is_ok();
            }

            if piece.contains("<tool_call>") || tool_call_buffering {
                tool_call_buffering = true;
                return true;
            }

            has_streamed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
            let chunk = make_chunk(
                &format!("chatcmpl-{created}"),
                created,
                &model_id_clone,
                ChoiceDelta {
                    content: Some(piece.to_string()),
                    reasoning_content: None,
                    role: None,
                    tool_calls: None,
                },
                None,
                Usage {
                    prompt_tokens: 0,
                    completion_tokens,
                    total_tokens: completion_tokens,
                },
            );
            tx_clone
                .send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
                .is_ok()
        };

        match engine.execute(&task, Some(&mut on_token)) {
            Ok(output) => match output.response {
                InferenceTaskResponse::ToolCall {
                    content,
                    tool_calls,
                } => {
                    if let Some(parsed_calls) = convert_to_tool_calls(&tool_calls, created) {
                        if !has_streamed_content.load(std::sync::atomic::Ordering::Relaxed) {
                            if let Some(ref text) = content {
                                if !text.is_empty() {
                                    let text_chunk = make_chunk(
                                        &format!("chatcmpl-{created}"),
                                        created,
                                        &model_id,
                                        ChoiceDelta {
                                            content: Some(text.clone()),
                                            reasoning_content: None,
                                            role: None,
                                            tool_calls: None,
                                        },
                                        None,
                                        output.usage.clone(),
                                    );
                                    let _ = tx.send(Event::default().data(
                                        serde_json::to_string(&text_chunk).unwrap_or_default(),
                                    ));
                                }
                            }
                        }

                        let tool_chunks = parsed_calls
                            .into_iter()
                            .enumerate()
                            .map(|(idx, tc)| ToolCallChunk {
                                index: idx,
                                id: Some(tc.id),
                                r#type: Some(tc.r#type),
                                function: Some(crate::types::FunctionCallChunk {
                                    name: Some(tc.function.name),
                                    arguments: Some(tc.function.arguments),
                                }),
                            })
                            .collect();

                        let tool_chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id,
                            ChoiceDelta {
                                content: None,
                                reasoning_content: None,
                                role: None,
                                tool_calls: Some(tool_chunks),
                            },
                            None,
                            output.usage.clone(),
                        );
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&tool_chunk).unwrap_or_default()),
                        );

                        let final_chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id,
                            ChoiceDelta {
                                content: None,
                                reasoning_content: None,
                                role: None,
                                tool_calls: None,
                            },
                            Some("tool_calls".to_string()),
                            output.usage,
                        );
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                        );
                    } else {
                        let chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id,
                            ChoiceDelta {
                                content: Some(tool_calls.to_string()),
                                reasoning_content: None,
                                role: None,
                                tool_calls: None,
                            },
                            Some("stop".to_string()),
                            output.usage,
                        );
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                    }
                }
                InferenceTaskResponse::Text(ref text) => {
                    if !has_streamed_content.load(std::sync::atomic::Ordering::Relaxed)
                        && !text.is_empty()
                    {
                        let text_chunk = make_chunk(
                            &format!("chatcmpl-{created}"),
                            created,
                            &model_id,
                            ChoiceDelta {
                                content: Some(text.clone()),
                                reasoning_content: None,
                                role: None,
                                tool_calls: None,
                            },
                            None,
                            output.usage.clone(),
                        );
                        let _ = tx.send(
                            Event::default()
                                .data(serde_json::to_string(&text_chunk).unwrap_or_default()),
                        );
                    }

                    let final_chunk = make_chunk(
                        &format!("chatcmpl-{created}"),
                        created,
                        &model_id,
                        ChoiceDelta {
                            content: None,
                            reasoning_content: None,
                            role: None,
                            tool_calls: None,
                        },
                        Some("stop".to_string()),
                        output.usage,
                    );
                    let _ = tx.send(
                        Event::default()
                            .data(serde_json::to_string(&final_chunk).unwrap_or_default()),
                    );
                }
                InferenceTaskResponse::Error(other) => {
                    let err_chunk = serde_json::json!({
                        "error": {
                            "message": other,
                            "type": "server_error",
                            "code": "inference_error"
                        }
                    });
                    let _ = tx.send(Event::default().event("error").data(err_chunk.to_string()));
                }
                _ => {}
            },
            Err(e) => {
                let err_chunk = serde_json::json!({
                    "error": {
                        "message": e.to_string(),
                        "type": "server_error",
                        "code": "engine_execution_failed"
                    }
                });
                let _ = tx.send(Event::default().event("error").data(err_chunk.to_string()));
            }
        }
    });

    tracing::info!(
        target: "audit",
        status = 200,
        latency_ms = start_time.elapsed().as_millis(),
        model = %model_config.id,
        media = %format!("images: {image_count}, videos: {video_count}"),
        "SSE stream started"
    );

    let stream = UnboundedReceiverStream::new(rx);
    let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));
    let sse_stream = stream.map(Ok::<_, Infallible>).chain(done_stream);
    let mut resp = Sse::new(sse_stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response();

    if let Some(heads) = model_config.mtp_heads() {
        if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
            resp.headers_mut().insert("x-mtp-heads", val);
        }
    }
    resp
}
