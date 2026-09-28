pub mod audio;
pub mod chat;
pub mod images;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

use crate::config::ModelRegistry;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{HealthResponse, ModelInfo, ModelsResponse};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub config: ModelRegistry,
}

pub fn create_router(state: AppState) -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions_handler))
        .route(
            "/v1/images/generations",
            post(images::image_generation_handler),
        )
        .route(
            "/v1/audio/transcriptions",
            post(audio::audio_transcriptions_handler),
        )
        .route("/v1/audio/speech", post(audio::audio_speech_handler))
        .route("/v1/inference", post(inference_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::new(state))
}

#[axum::debug_handler]
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded_models = state.inference.loaded_models();
    Json(HealthResponse::new().with_loaded_models(loaded_models))
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
                .with_ownership("Rune Infer".to_string())
                .with_mtp_heads(heads)
                .with_capabilities(m.resolved_capabilities())
        })
        .collect();

    let mut headers = axum::http::HeaderMap::new();
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
