//! HTTP request lifecycle tracking, including streamed bodies and early refusals.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use super::{RequestPhase, RequestRecord};
use axum::body::Body;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tokio_stream::StreamExt;

type Context = Arc<Mutex<RequestRecord>>;
tokio::task_local! { static CURRENT: Context; }
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn update(f: impl FnOnce(&mut RequestRecord)) -> u64 {
    CURRENT
        .try_with(|context| {
            let mut record = context.lock().unwrap_or_else(|e| e.into_inner());
            f(&mut record);
            record.id
        })
        .unwrap_or(0)
}

pub(crate) fn set_request_model(model: &str) {
    update(|r| {
        r.model = model.to_string();
        emit(r);
    });
}

pub(super) fn completion(
    model: &str,
    latency: u128,
    prompt: u32,
    output: u32,
    generation: u128,
) -> u64 {
    update(|r| {
        r.model = model.to_string();
        r.latency_ms = latency as u64;
        r.prompt_tokens = prompt;
        r.completion_tokens = output;
        r.generation_ms = generation as u64;
    })
}

pub(super) fn failure(model: &str, status: u16, latency: u128, error: &str) -> u64 {
    update(|r| {
        r.model = model.to_string();
        r.status = status;
        r.latency_ms = latency as u64;
        r.error = Some(error.to_string());
    })
}

/// Retain request correlation for audit events produced by blocking engine workers.
pub(crate) fn spawn_blocking<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let context = CURRENT.try_with(Clone::clone).ok();
    let dispatcher = tracing::dispatcher::get_default(Clone::clone);
    tokio::task::spawn_blocking(move || {
        tracing::dispatcher::with_default(&dispatcher, || match context {
            Some(context) => CURRENT.sync_scope(context, f),
            None => f(),
        })
    })
}

fn emit(r: &RequestRecord) {
    let phase = match r.phase {
        RequestPhase::Incoming => "incoming",
        RequestPhase::Processing => "processing",
        RequestPhase::Succeeded => "succeeded",
        RequestPhase::Failed => "failed",
    };
    // Lifecycle events contain metadata only; bodies use the separate payload log.
    tracing::info!(target: "audit", event = "request_state", request_id = r.id,
        phase, model = %r.model, status = r.status, latency_ms = r.latency_ms,
        prompt_tokens = r.prompt_tokens, completion_tokens = r.completion_tokens,
        generation_ms = r.generation_ms, error = r.error.as_deref().unwrap_or(""),
        "Request state changed");
}

struct Lifecycle {
    context: Context,
    started: Instant,
    status: u16,
    finished: bool,
}

impl Lifecycle {
    fn finish(&mut self, status: u16) {
        if self.finished {
            return;
        }
        self.finished = true;
        let mut r = self.context.lock().unwrap_or_else(|e| e.into_inner());
        if r.status < 400 {
            r.status = status;
        }
        r.latency_ms = self.started.elapsed().as_millis() as u64;
        if r.status >= 400 && r.error.is_none() {
            r.error = Some(if r.status == 499 {
                "Client disconnected".to_string()
            } else {
                axum::http::StatusCode::from_u16(r.status)
                    .ok()
                    .and_then(|s| s.canonical_reason())
                    .unwrap_or("Request failed")
                    .to_string()
            });
        }
        r.phase = if r.status >= 400 || r.error.is_some() {
            RequestPhase::Failed
        } else {
            RequestPhase::Succeeded
        };
        emit(&r);
    }
}

impl Drop for Lifecycle {
    fn drop(&mut self) {
        self.finish(499);
    }
}

pub(crate) async fn request_lifecycle(req: Request, next: Next) -> Response {
    if req.method() != axum::http::Method::POST {
        return next.run(req).await;
    }
    let context = Arc::new(Mutex::new(RequestRecord {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        phase: RequestPhase::Incoming,
        ts: super::now_ts(),
        model: req.uri().path().to_string(),
        status: 0,
        latency_ms: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        generation_ms: 0,
        error: None,
    }));
    let mut lifecycle = Lifecycle {
        context: context.clone(),
        started: Instant::now(),
        status: 200,
        finished: false,
    };
    {
        let mut r = context.lock().unwrap_or_else(|e| e.into_inner());
        emit(&r);
        r.phase = RequestPhase::Processing;
        emit(&r);
    }
    let response = CURRENT.scope(context.clone(), next.run(req)).await;
    lifecycle.status = response.status().as_u16();
    if let Some(model) = response
        .headers()
        .get("x-model-served")
        .and_then(|v| v.to_str().ok())
    {
        let mut record = context.lock().unwrap_or_else(|e| e.into_inner());
        record.model = model.to_string();
        emit(&record);
    }
    let (parts, body) = response.into_parts();
    let is_sse = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .is_some_and(|s| s.starts_with("text/event-stream"));
    let stream = async_stream::stream! {
        let mut lifecycle = lifecycle;
        let mut body = body.into_data_stream();
        let mut line = Vec::new();
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => {
                    if is_sse {
                        for byte in bytes.iter().copied() {
                            if byte == b'\n' {
                                if let Some(data) = line.strip_prefix(b"data:") {
                                    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) {
                                        if value.get("error").is_some() {
                                            let mut r = context.lock().unwrap_or_else(|e| e.into_inner());
                                            if r.status < 400 { r.status = 500; }
                                            r.error = Some("Stream reported an error".into());
                                        }
                                    }
                                }
                                line.clear();
                            } else if line.len() < 65_536 { line.push(byte); }
                        }
                    }
                    yield Ok::<_, axum::Error>(bytes);
                }
                Err(error) => { lifecycle.finish(500); yield Err(error); return; }
            }
        }
        lifecycle.finish(lifecycle.status);
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{EventLayer, UiEvent};
    use axum::{
        Router,
        body::{Bytes, to_bytes},
        http::StatusCode,
        routing::post,
    };
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::layer::SubscriberExt;

    fn requests(rx: &mut tokio::sync::mpsc::UnboundedReceiver<UiEvent>) -> Vec<RequestRecord> {
        let mut records = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let UiEvent::Request(r) = event {
                records.push(r);
            }
        }
        records
    }

    fn req(path: &str) -> Request {
        Request::builder()
            .method("POST")
            .uri(path)
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn incoming_is_emitted_before_handler_finishes_and_worker_stats_stay_correlated() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = tracing_subscriber::registry().with(EventLayer::new(tx));
        let notify = Arc::new(tokio::sync::Notify::new());
        let gate = notify.clone();
        let app = Router::new()
            .route(
                "/ok",
                post(move || async move {
                    set_request_model("same-model");
                    gate.notified().await;
                    spawn_blocking(|| super::super::record_completion("same-model", 20, 17, 23, 5))
                        .await
                        .unwrap();
                    "ok"
                }),
            )
            .layer(axum::middleware::from_fn(request_lifecycle));
        let task = tokio::spawn(
            async move {
                let response = app.oneshot(req("/ok")).await.unwrap();
                to_bytes(response.into_body(), 1024).await.unwrap()
            }
            .with_subscriber(subscriber),
        );
        let mut phases = Vec::new();
        let mut id = None;
        while phases.len() < 2 {
            if let UiEvent::Request(record) = rx.recv().await.unwrap() {
                assert_eq!(*id.get_or_insert(record.id), record.id);
                phases.push(record.phase);
            }
        }
        assert_eq!(phases, [RequestPhase::Incoming, RequestPhase::Processing]);
        assert!(!task.is_finished());
        notify.notify_one();
        assert_eq!(task.await.unwrap(), "ok");
        let records = requests(&mut rx);
        let final_records: Vec<_> = records.iter().filter(|r| r.phase.is_finished()).collect();
        assert_eq!(final_records.len(), 1);
        let r = final_records[0];
        assert_eq!(Some(r.id), id);
        assert_eq!(r.phase, RequestPhase::Succeeded);
        assert_eq!(r.model, "same-model");
        assert_eq!(
            (r.prompt_tokens, r.completion_tokens, r.generation_ms),
            (17, 23, 5)
        );
    }

    #[tokio::test]
    async fn early_refusals_and_concurrent_requests_have_distinct_ids() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = tracing_subscriber::registry().with(EventLayer::new(tx));
        async {
            let app = Router::new()
                .route("/reject", post(|| async { StatusCode::BAD_REQUEST }))
                .layer(axum::middleware::from_fn(request_lifecycle));
            let run = |app: Router| async move {
                let response = app.oneshot(req("/reject")).await.unwrap();
                to_bytes(response.into_body(), 1024).await.unwrap();
            };
            tokio::join!(run(app.clone()), run(app));
        }
        .with_subscriber(subscriber)
        .await;
        let records = requests(&mut rx);
        let final_records: Vec<_> = records.iter().filter(|r| r.phase.is_finished()).collect();
        assert_eq!(final_records.len(), 2);
        assert_ne!(final_records[0].id, final_records[1].id);
        assert!(
            final_records
                .iter()
                .all(|r| r.phase == RequestPhase::Failed && r.status == 400)
        );
    }

    #[tokio::test]
    async fn stream_headers_do_not_finish_and_dropping_body_reports_disconnect() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = tracing_subscriber::registry().with(EventLayer::new(tx));
        async {
            let app = Router::new()
                .route(
                    "/stream",
                    post(|| async {
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(tokio_stream::pending::<
                                Result<Bytes, std::convert::Infallible>,
                            >()))
                            .unwrap()
                    }),
                )
                .layer(axum::middleware::from_fn(request_lifecycle));
            let response = app.oneshot(req("/stream")).await.unwrap();
            let initial = requests(&mut rx);
            assert_eq!(initial.len(), 2);
            assert!(!initial.iter().any(|r| r.phase.is_finished()));
            drop(response);
        }
        .with_subscriber(subscriber)
        .await;
        let final_records = requests(&mut rx);
        assert_eq!(final_records.len(), 1);
        assert_eq!(final_records[0].phase, RequestPhase::Failed);
        assert_eq!(final_records[0].status, 499);
    }

    #[tokio::test]
    async fn split_sse_error_is_failed_even_after_http_200() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let subscriber = tracing_subscriber::registry().with(EventLayer::new(tx));
        async {
            let app = Router::new()
                .route(
                    "/stream",
                    post(|| async {
                        let chunks = [
                            "data: {\"err",
                            "or\":{\"message\":\"failed\"}}\n\n",
                            "data: [DONE]\n\n",
                        ];
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from_stream(tokio_stream::iter(
                                chunks.map(|s| Ok::<_, std::convert::Infallible>(Bytes::from(s))),
                            )))
                            .unwrap()
                    }),
                )
                .layer(axum::middleware::from_fn(request_lifecycle));
            let response = app.oneshot(req("/stream")).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            to_bytes(response.into_body(), 1024).await.unwrap();
        }
        .with_subscriber(subscriber)
        .await;
        let records = requests(&mut rx);
        let final_records: Vec<_> = records.iter().filter(|r| r.phase.is_finished()).collect();
        assert_eq!(final_records.len(), 1);
        assert_eq!(final_records[0].phase, RequestPhase::Failed);
        assert_eq!(final_records[0].status, 500);
    }
}
