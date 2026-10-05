pub mod audio;
pub mod chat;
pub mod embeddings;
pub mod images;
pub mod moderation;
pub mod nlu;
pub mod ocr;
pub mod responses;

use axum::extract::{Path, Query, Request, State};
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
use crate::types::{ApiError, ErrorResponse, HealthResponse, ModelInfo, ModelsResponse};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub proxy: Arc<ProxyService>,
    pub config: ModelRegistry,
}

pub fn create_router(state: AppState) -> Router {
    let shared_state = Arc::new(state);

    let v1_routes = Router::new()
        // Chat & Responses
        .route("/chat/completions", post(chat::chat_completions_handler))
        .route("/responses", post(responses::responses_handler))
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
        // Audio
        .route(
            "/audio/transcriptions",
            post(audio::audio_transcriptions_handler),
        )
        .route(
            "/audio/translations",
            post(audio::audio_translations_handler),
        )
        .route("/audio/speech", post(audio::audio_speech_handler))
        .route(
            "/audio/speech-to-speech",
            post(audio::speech_to_speech_handler),
        )
        // Text & NLU & OCR
        .route("/inference", post(inference_handler))
        .route("/models", get(models_handler))
        .route("/models/{model}", get(retrieve_model_handler))
        .route("/embeddings", post(embeddings::embeddings_handler))
        .route("/moderations", post(moderation::moderations_handler))
        .route("/nlu/intents", post(nlu::nlu_intents_handler))
        .route("/ocr", post(ocr::document_ocr_handler))
        .layer(from_fn_with_state(shared_state.clone(), auth_middleware));

    Router::new()
        .nest("/v1", v1_routes)
        .route("/health", get(health_handler))
        .with_state(shared_state)
}

async fn auth_middleware(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let configured_key = state
        .config
        .server
        .api_key
        .clone()
        .or_else(|| std::env::var("RUNE_API_KEY").ok())
        .or_else(|| std::env::var("API_KEY").ok());

    if let Some(expected) = configured_key {
        if !expected.trim().is_empty() {
            let auth_header = req.headers().get(header::AUTHORIZATION);
            let is_valid = match auth_header.and_then(|h| h.to_str().ok()) {
                Some(header_val) => {
                    if let Some(token) = header_val.strip_prefix("Bearer ") {
                        token.trim() == expected.trim()
                    } else {
                        false
                    }
                }
                None => false,
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

#[axum::debug_handler]
async fn health_handler(State(state): State<Arc<AppState>>) -> Response {
    let is_healthy = state.inference.registry.lock().is_ok();
    if is_healthy {
        Json(HealthResponse::new()).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Model execution runtime is currently unready or out of memory.",
                )
                .with_type("server_error")
                .with_code("runtime_unhealthy"),
            }),
        )
            .into_response()
    }
}

#[axum::debug_handler]
async fn models_handler(State(state): State<Arc<AppState>>) -> Response {
    let mut mtp_models = Vec::new();
    let data = state
        .config
        .models
        .iter()
        .map(|m| {
            let heads = m.mtp_heads();
            if m.has_mtp() {
                mtp_models.push(m.id.clone());
            }
            ModelInfo::new(&m.id)
                .with_mtp_heads(heads)
                .with_capabilities(m.resolved_capabilities())
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
        }),
    )
        .into_response()
}

#[axum::debug_handler]
async fn retrieve_model_handler(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    if let Some(m) = state.config.find(&model_id) {
        let heads = m.mtp_heads();
        let model_info = ModelInfo::new(&m.id)
            .with_mtp_heads(heads)
            .with_capabilities(m.resolved_capabilities());
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
    Json(task): Json<InferenceTaskRequest>,
) -> Response {
    let start_time = std::time::Instant::now();
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

    let inference = state.inference.clone();
    let res =
        tokio::task::spawn_blocking(move || inference.execute_task(&engine_id, &task, None)).await;

    match res {
        Ok(Ok(output)) => {
            tracing::info!(
                target: "audit",
                status = 200,
                latency_ms = start_time.elapsed().as_millis(),
                tokens = output.usage.total_tokens,
                "Inference task completed"
            );
            Json(output).into_response()
        }
        Ok(Err(e)) => {
            tracing::warn!(
                target: "audit",
                status = 404,
                latency_ms = start_time.elapsed().as_millis(),
                error = %e,
                "Inference task failed: engine not found or execution error"
            );
            (StatusCode::NOT_FOUND, Json(InferenceTaskResponse::Error(e))).into_response()
        }
        Err(join_err) => {
            tracing::error!(
                target: "audit",
                status = 500,
                latency_ms = start_time.elapsed().as_millis(),
                error = %join_err,
                "Inference task thread join error"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(InferenceTaskResponse::Error(format!(
                    "Task failed: {join_err}"
                ))),
            )
                .into_response()
        }
    }
}
