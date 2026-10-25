//! Structured server events for UIs (the TUI dashboard, or any embedding
//! application) built on top of `tracing`.
//!
//! Producers emit ordinary `tracing` events; [`EventLayer`] converts them into
//! typed [`UiEvent`]s and forwards them over a channel. Nothing here depends on
//! a terminal, so library users can install the layer for their own display.

pub mod payload;
mod request;
pub(crate) use request::{request_lifecycle, set_request_model, spawn_blocking};

use std::collections::HashMap;

use tokio::sync::mpsc::UnboundedSender;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// Target used for model load/evict lifecycle events.
pub const LIFECYCLE_TARGET: &str = "rune_infer::lifecycle";
/// Target used for per-request audit events.
pub const AUDIT_TARGET: &str = "audit";

#[derive(Debug, Clone)]
pub struct LogLine {
    pub ts: String,
    pub level: Level,
    pub target: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    Incoming,
    Processing,
    Succeeded,
    Failed,
}

impl RequestPhase {
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }
}

#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// Stable identity across lifecycle updates; zero for standalone audit records.
    pub id: u64,
    pub phase: RequestPhase,
    pub ts: String,
    pub model: String,
    pub status: u16,
    pub latency_ms: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Time spent generating (excludes model load); 0 when unknown.
    pub generation_ms: u64,
    pub error: Option<String>,
}

impl RequestRecord {
    /// Generation throughput in tokens/second, when measurable.
    pub fn tokens_per_second(&self) -> Option<f64> {
        (self.generation_ms > 0 && self.completion_tokens > 0)
            .then(|| self.completion_tokens as f64 / (self.generation_ms as f64 / 1000.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelEventKind {
    Loading,
    Loaded,
    Evicted,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ModelEvent {
    pub ts: String,
    pub model: String,
    pub kind: ModelEventKind,
    pub duration_ms: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub enum UiEvent {
    Log(LogLine),
    Request(RequestRecord),
    Model(ModelEvent),
}

fn now_ts() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

// ---------------------------------------------------------------------------
// Producer helpers
// ---------------------------------------------------------------------------

/// Record a finished (successful) inference request.
pub fn record_completion(
    model: &str,
    latency_ms: u128,
    prompt_tokens: u32,
    completion_tokens: u32,
    generation_ms: u128,
) {
    let request_id = request::completion(
        model,
        latency_ms,
        prompt_tokens,
        completion_tokens,
        generation_ms,
    );
    tracing::info!(
        target: "audit",
        request_id,
        event = "completed",
        status = 200,
        model = %model,
        latency_ms = latency_ms as u64,
        prompt_tokens,
        completion_tokens,
        generation_ms = generation_ms as u64,
        "Request completed"
    );
}

/// Record a failed inference request.
pub fn record_failure(model: &str, status: u16, latency_ms: u128, error: &str) {
    let request_id = request::failure(model, status, latency_ms, error);
    tracing::warn!(
        target: "audit",
        event = "failed",
        request_id,
        status,
        model = %model,
        latency_ms = latency_ms as u64,
        error = %error,
        "Request failed"
    );
}

pub fn model_loading(model: &str) {
    tracing::info!(target: "rune_infer::lifecycle", event = "loading", model = %model, "Loading model");
}

pub fn model_loaded(model: &str, duration_ms: u128, detail: &str) {
    tracing::info!(
        target: "rune_infer::lifecycle",
        event = "loaded",
        model = %model,
        duration_ms = duration_ms as u64,
        detail = %detail,
        "Model loaded"
    );
}

pub fn model_evicted(model: &str, reason: &str) {
    tracing::info!(
        target: "rune_infer::lifecycle",
        event = "evicted",
        model = %model,
        detail = %reason,
        "Model evicted"
    );
}

pub fn model_failed(model: &str, error: &str) {
    tracing::error!(
        target: "rune_infer::lifecycle",
        event = "failed",
        model = %model,
        detail = %error,
        "Model failed to load"
    );
}

// ---------------------------------------------------------------------------
// Tracing layer
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Fields {
    message: String,
    map: HashMap<&'static str, String>,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.map.insert(field.name(), format!("{value:?}"));
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.map.insert(field.name(), value.to_string());
        }
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.map.insert(field.name(), value.to_string());
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.map.insert(field.name(), value.to_string());
    }
}

impl Fields {
    fn num<T: std::str::FromStr + Default>(&self, key: &str) -> T {
        self.map
            .get(key)
            .and_then(|v| v.trim_matches('"').parse().ok())
            .unwrap_or_default()
    }
    fn text(&self, key: &str) -> String {
        self.map
            .get(key)
            .map(|v| v.trim_matches('"').to_string())
            .unwrap_or_default()
    }
}

/// A `tracing` layer that forwards every event as a [`UiEvent`].
pub struct EventLayer {
    tx: UnboundedSender<UiEvent>,
}

impl EventLayer {
    pub fn new(tx: UnboundedSender<UiEvent>) -> Self {
        Self { tx }
    }
}

impl<S: Subscriber> Layer<S> for EventLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let mut f = Fields::default();
        event.record(&mut f);
        let ts = now_ts();

        match meta.target() {
            LIFECYCLE_TARGET => {
                let kind = match f.text("event").as_str() {
                    "loading" => ModelEventKind::Loading,
                    "loaded" => ModelEventKind::Loaded,
                    "evicted" => ModelEventKind::Evicted,
                    _ => ModelEventKind::Failed,
                };
                let duration = f
                    .map
                    .contains_key("duration_ms")
                    .then(|| f.num("duration_ms"));
                let _ = self.tx.send(UiEvent::Model(ModelEvent {
                    ts: ts.clone(),
                    model: f.text("model"),
                    kind,
                    duration_ms: duration,
                    detail: f.text("detail"),
                }));
            }
            AUDIT_TARGET => match f.text("event").as_str() {
                "completed" | "failed" | "request_state"
                    if f.num::<u64>("request_id") == 0 || f.text("event") == "request_state" =>
                {
                    let error = f
                        .map
                        .contains_key("error")
                        .then(|| f.text("error"))
                        .filter(|e| !e.is_empty());
                    let _ = self.tx.send(UiEvent::Request(RequestRecord {
                        id: f.num("request_id"),
                        phase: match f.text("phase").as_str() {
                            "incoming" => RequestPhase::Incoming,
                            "processing" => RequestPhase::Processing,
                            _ if f.num::<u16>("status") >= 400 || error.is_some() => {
                                RequestPhase::Failed
                            }
                            _ => RequestPhase::Succeeded,
                        },
                        ts: ts.clone(),
                        model: f.text("model"),
                        status: f.num("status"),
                        latency_ms: f.num("latency_ms"),
                        prompt_tokens: f.num("prompt_tokens"),
                        completion_tokens: f.num("completion_tokens"),
                        generation_ms: f.num("generation_ms"),
                        error,
                    }));
                }
                _ => {}
            },
            _ => {}
        }

        let mut message = f.message.clone();
        if let Some(err) = f.map.get("error").filter(|e| !e.is_empty()) {
            message.push_str(&format!(" — {}", err.trim_matches('"')));
        }
        if let Some(model) = f.map.get("model") {
            if meta.target() == AUDIT_TARGET {
                message = format!("[{}] {message}", model.trim_matches('"'));
            }
        }

        let _ = self.tx.send(UiEvent::Log(LogLine {
            ts,
            level: *meta.level(),
            target: meta.target().to_string(),
            message,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    fn capture(f: impl FnOnce()) -> Vec<UiEvent> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sub = tracing_subscriber::registry().with(EventLayer::new(tx));
        tracing::subscriber::with_default(sub, f);
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    #[test]
    fn completion_becomes_a_typed_request_with_throughput() {
        let evs = capture(|| record_completion("qwen", 2500, 100, 50, 2000));
        let req = evs
            .iter()
            .find_map(|e| {
                if let UiEvent::Request(r) = e {
                    Some(r)
                } else {
                    None
                }
            })
            .expect("request event");
        assert_eq!(req.model, "qwen");
        assert_eq!(req.status, 200);
        assert_eq!(req.latency_ms, 2500);
        assert_eq!((req.prompt_tokens, req.completion_tokens), (100, 50));
        assert!((req.tokens_per_second().unwrap() - 25.0).abs() < 1e-9);
        assert!(
            evs.iter().any(|e| matches!(e, UiEvent::Log(_))),
            "also logged"
        );
    }

    #[test]
    fn failures_carry_status_and_error() {
        let evs = capture(|| record_failure("m", 502, 10, "upstream down"));
        let r = evs
            .iter()
            .find_map(|e| {
                if let UiEvent::Request(r) = e {
                    Some(r)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(r.status, 502);
        assert_eq!(r.error.as_deref(), Some("upstream down"));
        assert!(r.tokens_per_second().is_none());
    }

    #[test]
    fn model_lifecycle_events_roundtrip() {
        let evs = capture(|| {
            model_loading("a");
            model_loaded("a", 4200, "+5.1 GiB VRAM");
            model_evicted("a", "lru");
            model_failed("b", "file not found");
        });
        let kinds: Vec<_> = evs
            .iter()
            .filter_map(|e| {
                if let UiEvent::Model(m) = e {
                    Some((m.model.clone(), m.kind, m.duration_ms))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(kinds[0], ("a".into(), ModelEventKind::Loading, None));
        assert_eq!(kinds[1], ("a".into(), ModelEventKind::Loaded, Some(4200)));
        assert_eq!(kinds[2].1, ModelEventKind::Evicted);
        assert_eq!(kinds[3], ("b".into(), ModelEventKind::Failed, None));
    }

    #[test]
    fn plain_audit_lines_do_not_count_as_requests() {
        let evs = capture(|| tracing::info!(target: "audit", status = 200, "SSE stream started"));
        assert!(!evs.iter().any(|e| matches!(e, UiEvent::Request(_))));
    }
}
