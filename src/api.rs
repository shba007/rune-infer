// pub mod anthropic;
pub mod audio;
pub mod chat;
pub mod decisions;
pub mod embeddings;
pub mod gate;
pub mod images;
mod local_execution;
pub mod moderation;
pub mod nlu;
pub mod ocr;
mod task;
// pub mod responses;

pub use gate::ModelExecutionGate;

#[cfg(test)]
mod contract_tests;

use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

use crate::config::ModelRegistry;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::proxy::ProxyService;
use crate::types::{
    ApiError, ErrorResponse, HealthResponse, LivezResponse, ModelInfo, ModelsResponse,
    ReadyzResponse,
};

#[derive(Default)]
pub struct RequestConcurrencyState {
    pub active_foreground: std::sync::atomic::AtomicUsize,
    pub active_background: std::sync::atomic::AtomicUsize,
    pub yield_notify: tokio::sync::Notify,
    pub foreground_epoch: std::sync::atomic::AtomicU64,
}

impl RequestConcurrencyState {
    /// Register before checking the epoch so a foreground arrival cannot be lost.
    pub async fn wait_for_foreground(&self, epoch: u64) {
        let notified = self.yield_notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self
            .foreground_epoch
            .load(std::sync::atomic::Ordering::SeqCst)
            != epoch
            || self
                .active_foreground
                .load(std::sync::atomic::Ordering::SeqCst)
                > 0
        {
            return;
        }
        notified.await;
    }
}

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub proxy: Arc<ProxyService>,
    pub config: Arc<std::sync::RwLock<ModelRegistry>>,
    pub concurrency: Arc<RequestConcurrencyState>,
}

impl AppState {
    pub fn new(
        inference: Arc<crate::inference::AppState>,
        proxy: Arc<ProxyService>,
        config: ModelRegistry,
    ) -> Self {
        Self {
            inference,
            proxy,
            config: Arc::new(std::sync::RwLock::new(config)),
            concurrency: Arc::new(RequestConcurrencyState::default()),
        }
    }

    pub fn config(&self) -> std::sync::RwLockReadGuard<'_, ModelRegistry> {
        self.config.read().unwrap_or_else(|p| p.into_inner())
    }

    pub fn reload_catalog(&self, new_config: ModelRegistry) {
        if let Ok(mut reg) = self.inference.registry.lock() {
            reg.reload_catalog(new_config.clone());
        }
        if let Ok(mut cfg) = self.config.write() {
            *cfg = new_config;
        }
    }
}

pub fn create_router(state: AppState) -> Router {
    let shared_state = Arc::new(state);

    let v1_routes = Router::new()
        // Chat & Responses
        .route("/chat/completions", post(chat::chat_completions_handler))
        // .route("/messages", post(anthropic::anthropic_messages_handler))
        // Images
        .route(
            "/images/generations",
            post(images::image_generation_handler),
        )
        .route("/images/edits", post(images::image_edit_handler))
        .route("/images/variations", post(images::image_variation_handler))
        .route("/images/captions", post(images::image_caption_handler))
        .route("/images/detections", post(images::detect_objects_handler))
        .route("/images/embeddings", post(images::image_embeddings_handler))
        .route("/images/recognize", post(images::recognize_objects_handler))
        .route("/images/upscales", post(images::image_upscale_handler))
        .route(
            "/images/restorations",
            post(images::image_restoration_handler),
        )
        .route(
            "/images/style-transfers",
            post(images::image_style_transfer_handler),
        )
        .route("/images/grounding", post(images::image_grounding_handler))
        // Audio
        .route(
            "/audio/transcriptions",
            post(audio::audio_transcriptions_handler),
        )
        .route(
            "/audio/translations",
            post(audio::audio_translations_handler),
        )
        .route(
            "/audio/diarizations",
            post(audio::audio_diarization_handler),
        )
        .route("/audio/speech", post(audio::audio_speech_handler))
        .route(
            "/audio/speech-to-speech",
            post(audio::speech_to_speech_handler),
        )
        // Text & NLU & OCR
        .route("/inference", post(inference_handler))
        .route("/models", get(models_handler))
        .route("/capabilities", get(local_execution::capabilities))
        .route("/models/{model}", get(retrieve_model_handler))
        .route("/embeddings", post(embeddings::embeddings_handler))
        .route("/moderations", post(moderation::moderations_handler))
        .route("/nlu/intents", post(nlu::nlu_intents_handler))
        .route("/ocr", post(ocr::document_ocr_handler))
        .route("/decisions", post(decisions::decisions_handler))
        .layer(from_fn_with_state(shared_state.clone(), auth_middleware))
        .layer(axum::middleware::from_fn(
            crate::telemetry::request_lifecycle,
        ));

    Router::new()
        .nest("/v1", v1_routes)
        // .route(
        //     "/anthropic/v1/messages",
        //     post(anthropic::anthropic_messages_handler),
        // )
        // .route(
        //     "/anthropic/messages",
        //     post(anthropic::anthropic_messages_handler),
        // )
        .route("/health", get(health_handler))
        .route("/livez", get(livez_handler))
        .route("/readyz", get(readyz_handler))
        .layer(axum::middleware::from_fn(local_execution::enforce))
        .layer(axum::middleware::from_fn(
            crate::telemetry::payload::middleware,
        ))
        .layer(DefaultBodyLimit::max(128 * 1024 * 1024))
        .with_state(shared_state)
}

fn constant_time_eq(provided: &str, expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    let h1 = Sha256::digest(provided.as_bytes());
    let h2 = Sha256::digest(expected.as_bytes());
    let mut diff = 0u8;
    for (b1, b2) in h1.iter().zip(h2.iter()) {
        diff |= b1 ^ b2;
    }
    diff == 0 && provided.len() == expected.len()
}

async fn auth_middleware(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let configured_key = state
        .config()
        .server
        .api_key
        .clone()
        .or_else(|| std::env::var("RUNE_API_KEY").ok())
        .or_else(|| std::env::var("API_KEY").ok());

    if let Some(expected) = configured_key {
        if !expected.trim().is_empty() {
            let auth_header = req.headers().get(header::AUTHORIZATION);
            let x_api_key = req.headers().get("x-api-key");

            let is_valid = match auth_header.and_then(|h| h.to_str().ok()) {
                Some(header_val) => {
                    if let Some(token) = header_val.strip_prefix("Bearer ") {
                        constant_time_eq(token.trim(), expected.trim())
                    } else {
                        constant_time_eq(header_val.trim(), expected.trim())
                    }
                }
                None => match x_api_key.and_then(|h| h.to_str().ok()) {
                    Some(key_val) => constant_time_eq(key_val.trim(), expected.trim()),
                    None => false,
                },
            };

            if !is_valid {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: ApiError::new(
                            "Incorrect API key provided or authorization header is missing.",
                        )
                        .with_type("authentication_error")
                        .with_code("invalid_api_key"),
                    }),
                )
                    .into_response();
            }
        }
    }

    next.run(req).await
}

#[derive(Deserialize, Default)]
struct HealthProbeQuery {
    probe: Option<String>,
    #[serde(rename = "type")]
    probe_type: Option<String>,
    check: Option<String>,
}

#[axum::debug_handler]
async fn livez_handler() -> Response {
    Json(LivezResponse::new()).into_response()
}

#[axum::debug_handler]
async fn readyz_handler(State(state): State<Arc<AppState>>) -> Response {
    let (busy, active_requests) = {
        let fg = state
            .concurrency
            .active_foreground
            .load(std::sync::atomic::Ordering::Relaxed);
        let bg = state
            .concurrency
            .active_background
            .load(std::sync::atomic::Ordering::Relaxed);
        (fg > 0 || bg > 0, fg + bg)
    };

    match state.inference.registry.try_lock() {
        Ok(reg) => {
            let configured = reg.config.models.len();
            let loaded = reg.loaded_models();
            let loading = reg.loading_models();
            let resp = ReadyzResponse::new(configured, loaded, loading)
                .with_concurrency(busy, active_requests);
            Json(resp).into_response()
        }
        Err(std::sync::TryLockError::WouldBlock) => {
            let configured = state.config().models.len();
            let resp = ReadyzResponse::new(configured, Vec::new(), Vec::new())
                .with_concurrency(busy, active_requests);
            Json(resp).into_response()
        }
        Err(std::sync::TryLockError::Poisoned(_)) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ApiError::new("Model execution runtime registry is poisoned.")
                    .with_type("server_error")
                    .with_code("runtime_poisoned"),
            }),
        )
            .into_response(),
    }
}

#[axum::debug_handler]
async fn health_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<HealthProbeQuery>,
) -> Response {
    let (busy, active_requests) = {
        let fg = state
            .concurrency
            .active_foreground
            .load(std::sync::atomic::Ordering::Relaxed);
        let bg = state
            .concurrency
            .active_background
            .load(std::sync::atomic::Ordering::Relaxed);
        (fg > 0 || bg > 0, fg + bg)
    };

    let mode = query
        .probe
        .as_deref()
        .or(query.probe_type.as_deref())
        .or(query.check.as_deref())
        .unwrap_or("")
        .trim()
        .to_lowercase();

    match mode.as_str() {
        "live" | "livez" | "liveness" => Json(LivezResponse::new()).into_response(),
        "ready" | "readyz" | "readiness" => match state.inference.registry.try_lock() {
            Ok(reg) => {
                let configured = reg.config.models.len();
                let loaded = reg.loaded_models();
                let loading = reg.loading_models();
                let resp = ReadyzResponse::new(configured, loaded, loading)
                    .with_concurrency(busy, active_requests);
                Json(resp).into_response()
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                let configured = state.config().models.len();
                let resp = ReadyzResponse::new(configured, Vec::new(), Vec::new())
                    .with_concurrency(busy, active_requests);
                Json(resp).into_response()
            }
            Err(std::sync::TryLockError::Poisoned(_)) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: ApiError::new("Model execution runtime registry is poisoned.")
                        .with_type("server_error")
                        .with_code("runtime_poisoned"),
                }),
            )
                .into_response(),
        },
        _ => match state.inference.registry.try_lock() {
            Ok(reg) => {
                let loaded = reg.loaded_models();
                let mut res = HealthResponse::new().with_concurrency(busy, active_requests);
                if !loaded.is_empty() {
                    res = res.with_loaded_models(loaded);
                }
                Json(res).into_response()
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                let res = HealthResponse::new().with_concurrency(busy, active_requests);
                Json(res).into_response()
            }
            Err(std::sync::TryLockError::Poisoned(_)) => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "Model execution runtime is currently unready or poisoned.",
                    )
                    .with_type("server_error")
                    .with_code("runtime_unhealthy"),
                }),
            )
                .into_response(),
        },
    }
}

fn model_config_to_info(
    m: &crate::config::ModelConfig,
    loaded_models: &[String],
    loading_models: &[String],
    load_duration_ms: Option<u64>,
    selected_engine: Option<crate::config::EngineKind>,
) -> ModelInfo {
    let heads = m.mtp_heads();
    let is_resident = loaded_models.iter().any(|id| id == &m.id);
    let is_loading = loading_models.iter().any(|id| id == &m.id);
    let status = if !m.is_local() {
        "remote"
    } else if is_resident {
        "loaded"
    } else if is_loading {
        "loading"
    } else {
        "available"
    };

    let (max_w, max_h) = m.max_dimensions();
    let max_edge = if max_w > 0 || max_h > 0 {
        max_w.max(max_h)
    } else {
        4096
    };

    let engine = selected_engine.or_else(|| {
        m.engine
            .as_ref()
            .and_then(|e| e.preferred)
            .or_else(|| m.engine.as_ref().and_then(|e| e.supported.first().cloned()))
    });

    let mut info = ModelInfo::new(&m.id);
    info.tasks = m.tasks.clone();
    info.audio_operations = m.resolved_audio_operations(engine);
    info.with_mtp_heads(heads)
        .with_capabilities(m.resolved_capabilities_with_engine(engine))
        .with_locality(m.locality())
        .with_modalities_in(m.modalities_in())
        .with_modalities_out(m.modalities_out())
        .with_tool_calls(m.supports_tool_calls_with_engine(engine))
        .with_thinking_control(m.thinking_control())
        .with_max_image_edge_px(max_edge)
        .with_context_tokens(m.max_context_length.map(|l| l as usize).unwrap_or(4096))
        .with_max_output_tokens(m.sampling.as_ref().and_then(|s| s.max_tokens))
        .with_engine(engine.map(|e| e.to_string()))
        .with_quantization(m.quantization())
        .with_vram_mb_estimate(m.vram_mb_estimate())
        .with_load_ms_estimate(load_duration_ms)
        .with_resident(m.is_local() && is_resident)
        .with_status(status)
        .with_roles(m.roles.clone())
        .with_aliases(m.aliases.clone())
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelFilters {
    censorship: Option<String>,
    modality: Option<String>,
    min_params: Option<u64>,
    max_params: Option<u64>,
    quantization: Option<String>,
}

impl ModelFilters {
    fn matches(&self, model: &crate::config::ModelConfig) -> bool {
        let caps = model.capabilities.as_deref().unwrap_or(&[]);
        let uncensored = caps
            .iter()
            .any(|cap| cap.eq_ignore_ascii_case("uncensored"));
        if let Some(value) = &self.censorship {
            if (value == "uncensored") != uncensored {
                return false;
            }
        }
        if let Some(value) = &self.modality {
            let normalize = |text: &str| text.to_ascii_lowercase().replace([' ', '_'], "-");
            if !model
                .tasks
                .iter()
                .any(|task| normalize(task) == normalize(value))
            {
                return false;
            }
        }
        if self.min_params.is_some() || self.max_params.is_some() {
            let Some(params) = model.total_params else {
                return false;
            };
            if self.min_params.is_some_and(|min| params < min)
                || self.max_params.is_some_and(|max| params > max)
            {
                return false;
            }
        }
        if let Some(value) = &self.quantization {
            if !model
                .quantization()
                .is_some_and(|q| q.eq_ignore_ascii_case(value))
            {
                return false;
            }
        }
        true
    }
}

#[axum::debug_handler]
async fn models_handler(
    State(state): State<Arc<AppState>>,
    Query(filters): Query<ModelFilters>,
) -> Response {
    if filters.modality.as_deref().is_some_and(|task| {
        !crate::schema::valid_task(&task.to_ascii_lowercase().replace([' ', '_'], "-"))
    }) {
        return (StatusCode::BAD_REQUEST, Json(ErrorResponse {
            error: ApiError::new("modality must be an explicit input-to-output task, for example text-image-to-text".to_owned()).with_type("invalid_request_error"),
        })).into_response();
    }
    if filters
        .censorship
        .as_deref()
        .is_some_and(|v| v != "censored" && v != "uncensored")
        || filters
            .min_params
            .zip(filters.max_params)
            .is_some_and(|(min, max)| min > max)
    {
        return (StatusCode::BAD_REQUEST, Json(ErrorResponse {
            error: ApiError::new("Invalid model filters: use censorship=censored or uncensored and min_params <= max_params".to_string()).with_type("invalid_request_error"),
        })).into_response();
    }
    let (vram_total_bytes, vram_used_bytes, vram_free_bytes) = match (
        crate::inference::budget::detect_total_gpu_vram(),
        crate::inference::budget::detect_used_gpu_vram(),
    ) {
        (Some(total), Some(used)) => (Some(total), Some(used), Some(total.saturating_sub(used))),
        (Some(total), None) => (Some(total), None, None),
        _ => (None, None, None),
    };

    let (loaded_models, loading_models) = state
        .inference
        .registry
        .lock()
        .map(|r| (r.loaded_models(), r.loading_models()))
        .unwrap_or_default();

    let mut mtp_models = Vec::new();
    let data = state
        .config()
        .models
        .iter()
        .filter(|m| filters.matches(m))
        .map(|m| {
            if m.has_mtp() {
                mtp_models.push(m.id.clone());
            }
            let load_ms = state
                .inference
                .registry
                .lock()
                .ok()
                .and_then(|r| r.get_load_duration_ms(&m.id));
            let selected_engine = state
                .inference
                .registry
                .lock()
                .ok()
                .and_then(|r| r.selected_engine_kind(&m.id));
            model_config_to_info(m, &loaded_models, &loading_models, load_ms, selected_engine)
        })
        .collect();

    let mut headers = HeaderMap::new();
    if !mtp_models.is_empty() {
        headers.insert(
            "x-mtp-available",
            axum::http::HeaderValue::from_static("true"),
        );
        if let Ok(val) = axum::http::HeaderValue::from_str(&mtp_models.join(", ")) {
            headers.insert("x-mtp-models", val);
        }
    }

    (
        headers,
        Json(ModelsResponse {
            object: "list".to_string(),
            data,
            vram_total_bytes,
            vram_free_bytes,
            vram_used_bytes,
        }),
    )
        .into_response()
}

#[axum::debug_handler]
async fn retrieve_model_handler(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    if let Some(m) = state.config().find(&model_id) {
        let (loaded_models, loading_models) = state
            .inference
            .registry
            .lock()
            .map(|r| (r.loaded_models(), r.loading_models()))
            .unwrap_or_default();
        let load_ms = state
            .inference
            .registry
            .lock()
            .ok()
            .and_then(|r| r.get_load_duration_ms(&m.id));
        let selected_engine = state
            .inference
            .registry
            .lock()
            .ok()
            .and_then(|r| r.selected_engine_kind(&m.id));
        let model_info =
            model_config_to_info(m, &loaded_models, &loading_models, load_ms, selected_engine);
        Json(model_info).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: ApiError::new(format!(
                    "The model '{}' does not exist or you do not have access to it.",
                    model_id
                ))
                .with_type("invalid_request_error")
                .with_param("model")
                .with_code("model_not_found"),
            }),
        )
            .into_response()
    }
}

#[derive(Deserialize, Default)]
struct InferenceQuery {
    engine: Option<String>,
}

#[axum::debug_handler]
async fn inference_handler(
    State(state): State<Arc<AppState>>,
    Query(q): Query<InferenceQuery>,
    headers: HeaderMap,
    Json(task): Json<InferenceTaskRequest>,
) -> Response {
    let engine_id = match q.engine.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(id) => id.to_string(),
        None => {
            tracing::warn!(target: "audit", status = 400, "Inference call missing 'engine' param");
            return (
                StatusCode::BAD_REQUEST,
                Json(InferenceTaskResponse::Error(
                    "Missing required query parameter: 'engine'".into(),
                )),
            )
                .into_response();
        }
    };

    let gate = match ModelExecutionGate::enter(&state, &headers, &engine_id) {
        Ok(gate) => gate,
        Err(resp) => return resp,
    };
    let engine = match gate.load_task_engine(&state, &task).await {
        Ok(engine) => engine,
        Err(resp) => return resp,
    };
    match gate.execute_task(&state, engine, task).await {
        Ok(output) => gate.attach_headers(&state, Json(output).into_response()),
        Err(resp) => resp,
    }
}
