use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::response::make_chunk;
use super::tools::convert_to_tool_calls;
use crate::config::ModelConfig;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{ChoiceDelta, ToolCallChunk};

pub fn handle_chat_stream(
    task: InferenceTaskRequest,
    engine: Arc<dyn InferenceEngine>,
    model_config: ModelConfig,
    created: u64,
    start_time: std::time::Instant,
    image_count: usize,
    video_count: usize,
    include_usage: bool,
    yield_notify: Option<(Arc<crate::api::RequestConcurrencyState>, u64)>,
    concurrency_guard: crate::api::gate::ConcurrencyGuard,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    let model_id = model_config.id.clone();
    let yielded = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let execution_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (execution_done_tx, execution_done_rx) = tokio::sync::oneshot::channel::<()>();

    let yield_task = if let Some((concurrency, epoch)) = yield_notify {
        let engine_for_yield = engine.clone();
        let tx_yield = tx.clone();
        let yielded = yielded.clone();
        let execution_finished = execution_finished.clone();
        Some(tokio::spawn(async move {
            tokio::select! {
                _ = tx_yield.closed() => return,
                _ = execution_done_rx => return,
                _ = concurrency.wait_for_foreground(epoch) => {},
            }
            if execution_finished.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            yielded.store(true, std::sync::atomic::Ordering::SeqCst);
            engine_for_yield.cancel();
            let err_chunk = serde_json::json!({
                "error": {
                    "message": "Background request yielded to foreground request",
                    "type": "server_busy",
                    "code": "yielded"
                }
            });
            let _ = tx_yield
                .send(Event::default().event("error").data(err_chunk.to_string()))
                .await;
        }))
    } else {
        None
    };

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
        None,
    );
    let _ = tx
        .try_send(Event::default().data(serde_json::to_string(&initial_chunk).unwrap_or_default()));

    let engine_for_task = engine.clone();
    let yielded_for_task = yielded.clone();
    let execution_finished_for_task = execution_finished.clone();
    tokio::task::spawn_blocking(move || {
        let _execution_done = execution_done_tx;
        struct ExecutionFinished(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for ExecutionFinished {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _execution_finished = ExecutionFinished(execution_finished_for_task);
        let tx_sender = tx.clone();
        let tx_status = tx.clone();
        let model_id_clone = model_id.clone();
        let has_streamed_content = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let has_streamed_clone = has_streamed_content.clone();
        let mut is_thinking = false;
        let mut tool_call_buffering = false;
        let engine_for_cancel = engine_for_task.clone();
        let yielded_for_tokens = yielded_for_task.clone();

        let mut on_token = move |piece: &str| -> bool {
            if yielded_for_tokens.load(std::sync::atomic::Ordering::SeqCst) {
                return false;
            }
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
                        None,
                    );
                    if tx_sender
                        .blocking_send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_err()
                    {
                        engine_for_cancel.cancel();
                        return false;
                    }
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
                        None,
                    );
                    if tx_sender
                        .blocking_send(
                            Event::default()
                                .data(serde_json::to_string(&chunk).unwrap_or_default()),
                        )
                        .is_err()
                    {
                        engine_for_cancel.cancel();
                        return false;
                    }
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
                    None,
                );
                if tx_sender
                    .blocking_send(
                        Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                    )
                    .is_err()
                {
                    engine_for_cancel.cancel();
                    return false;
                }
                return true;
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
                None,
            );
            if tx_sender
                .blocking_send(
                    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                )
                .is_err()
            {
                engine_for_cancel.cancel();
                return false;
            }
            true
        };

        let gen_started = std::time::Instant::now();
        let exec_result = engine_for_task.execute(&task, Some(&mut on_token));
        _execution_finished
            .0
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let gen_elapsed = gen_started.elapsed();
        if yielded_for_task.load(std::sync::atomic::Ordering::SeqCst) {
            crate::telemetry::record_failure(
                &model_id,
                503,
                start_time.elapsed().as_millis(),
                "background stream yielded",
            );
            return;
        }
        if tx_status.is_closed() {
            crate::telemetry::record_failure(
                &model_id,
                499,
                start_time.elapsed().as_millis(),
                "client disconnected during streaming",
            );
        } else {
            match &exec_result {
                Ok(out) => crate::telemetry::record_completion(
                    &model_id,
                    start_time.elapsed().as_millis(),
                    out.usage.prompt_tokens,
                    out.usage.completion_tokens,
                    gen_elapsed.as_millis(),
                ),
                Err(e) => crate::telemetry::record_failure(
                    &model_id,
                    500,
                    start_time.elapsed().as_millis(),
                    &e.to_string(),
                ),
            }
        }
        match exec_result {
            Ok(output) => {
                let usage_to_send = if include_usage {
                    Some(output.usage)
                } else {
                    None
                };
                match output.response {
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
                                            None,
                                        );
                                        let _ = tx.blocking_send(Event::default().data(
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
                                None,
                            );
                            let _ = tx.blocking_send(
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
                                usage_to_send,
                            );
                            let _ = tx.blocking_send(
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
                                usage_to_send,
                            );
                            let _ = tx.blocking_send(
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
                                None,
                            );
                            let _ = tx.blocking_send(
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
                            usage_to_send,
                        );
                        let _ = tx.blocking_send(
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
                        let _ = tx.blocking_send(
                            Event::default().event("error").data(err_chunk.to_string()),
                        );
                    }
                    _ => {}
                }
            }
            Err(e) => {
                let err_chunk = serde_json::json!({
                    "error": {
                        "message": e.to_string(),
                        "type": "server_error",
                        "code": "engine_execution_failed"
                    }
                });
                let _ =
                    tx.blocking_send(Event::default().event("error").data(err_chunk.to_string()));
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

    struct StreamCancelGuard<S> {
        inner: S,
        _concurrency_guard: crate::api::gate::ConcurrencyGuard,
        yield_task: Option<tokio::task::JoinHandle<()>>,
        yielded: Arc<std::sync::atomic::AtomicBool>,
        execution_finished: Arc<std::sync::atomic::AtomicBool>,
        engine: Arc<dyn InferenceEngine>,
        is_done: Arc<std::sync::atomic::AtomicBool>,
    }

    impl<S: tokio_stream::Stream + Unpin> tokio_stream::Stream for StreamCancelGuard<S> {
        type Item = S::Item;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            let res = std::pin::Pin::new(&mut self.inner).poll_next(cx);
            if let std::task::Poll::Ready(None) = &res {
                self.is_done
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            res
        }
    }

    impl<S> Drop for StreamCancelGuard<S> {
        fn drop(&mut self) {
            if let Some(task) = &self.yield_task {
                task.abort();
            }
            if !self.is_done.load(std::sync::atomic::Ordering::Relaxed)
                && !self.yielded.load(std::sync::atomic::Ordering::SeqCst)
                && !self
                    .execution_finished
                    .load(std::sync::atomic::Ordering::SeqCst)
            {
                self.engine.cancel();
            }
        }
    }

    let stream = ReceiverStream::new(rx);
    let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));
    let sse_stream = stream.map(Ok::<_, Infallible>).chain(done_stream);
    let is_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancel_guarded = StreamCancelGuard {
        inner: sse_stream,
        _concurrency_guard: concurrency_guard,
        yield_task,
        yielded,
        execution_finished,
        engine: engine.clone(),
        is_done,
    };
    let mut resp = Sse::new(cancel_guarded)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response();

    if let Ok(val) = axum::http::HeaderValue::from_str(&model_config.id) {
        resp.headers_mut().insert("x-model-served", val);
    }
    resp.headers_mut().insert(
        "x-model-locality",
        axum::http::HeaderValue::from_static(model_config.locality()),
    );
    resp.headers_mut().insert(
        "x-model-resident",
        axum::http::HeaderValue::from_static("true"),
    );

    if let Some(heads) = model_config.mtp_heads() {
        if let Ok(val) = axum::http::HeaderValue::from_str(&heads.to_string()) {
            resp.headers_mut().insert("x-mtp-heads", val);
        }
    }
    resp
}
