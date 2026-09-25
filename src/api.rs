use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::{
    Json, Router,
    extract::{Query, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::config::ModelRegistry;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    ApiError, ChatCompletionRequest, ChatCompletionResponse, Choice, ChoiceDelta, ErrorResponse,
    HealthResponse, ModelInfo, ModelsResponse, Usage,
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<crate::inference::AppState>,
    pub config: ModelRegistry,
}

pub fn create_router(state: AppState) -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/inference", post(inference_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::new(state))
}

fn decode_image(url: &str) -> Result<Vec<u8>, ErrorResponse> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, b64_data) = rest.split_once(',').ok_or_else(|| ErrorResponse {
            error: ApiError::new("Invalid image data URL: missing comma separator")
                .with_type("invalid_request_error")
                .with_code("invalid_image"),
        })?;

        let header_lower = header.to_lowercase();
        if !header_lower.starts_with("image/") || !header_lower.contains(";base64") {
            return Err(ErrorResponse {
                error: ApiError::new("Invalid image data URL (expected `image/*;base64,<data>`)")
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            });
        }

        let clean_b64 = b64_data.trim();

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(clean_b64)
            .map_err(|e| ErrorResponse {
                error: ApiError::new(&format!("Invalid base64 payload: {e}"))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            })?;

        Ok(bytes)
    } else {
        match std::fs::read(url) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(ErrorResponse {
                error: ApiError::new(&format!("Could not read image file '{}': {e}", url))
                    .with_type("invalid_request_error")
                    .with_code("invalid_image"),
            }),
        }
    }
}

#[axum::debug_handler]
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let loaded_models = state.inference.loaded_models();
    Json(HealthResponse::new().with_loaded_models(loaded_models))
}

#[axum::debug_handler]
async fn models_handler(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let data = state
        .config
        .models
        .iter()
        .map(|m| ModelInfo::new(&m.id).with_ownership("Rune Infer".to_string()))
        .collect();
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
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
    let engine_id = match q.engine.as_deref().filter(|s| !s.trim().is_empty()) {
        Some(id) => id.to_string(),
        None => {
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
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(e)) => {
            (StatusCode::NOT_FOUND, Json(InferenceTaskResponse::Error(e))).into_response()
        }
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(InferenceTaskResponse::Error(format!(
                "Task failed: {join_err}"
            ))),
        )
            .into_response(),
    }
}

#[axum::debug_handler]
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.model.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("Missing required field: 'model'")
                    .with_type("invalid_request_error")
                    .with_code("model_missing"),
            }),
        )
            .into_response();
    }

    let model_id = request.model.clone();
    let inference = state.inference.clone();
    let engine = match tokio::task::spawn_blocking(move || inference.get_engine(&model_id)).await {
        Ok(Ok(e)) => e,
        Ok(Err(err)) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(&format!(
                        "Model '{}' not found or failed to load: {err}",
                        request.model
                    ))
                    .with_type("invalid_request_error")
                    .with_code("model_not_found"),
                }),
            )
                .into_response();
        }
        Err(join_err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(&format!("Thread execution error: {join_err}")),
                }),
            )
                .into_response();
        }
    };

    let (prompt, images) = {
        let mut prompt = String::new();
        let mut images = Vec::new();
        for msg in &request.messages {
            let (text, img_urls) = msg.split_text_and_images();
            prompt.push_str(&text);
            for url in img_urls {
                prompt.push_str("<__media__>");
                match decode_image(&url) {
                    Ok(bytes) => images.push(bytes),
                    Err(e) => return (StatusCode::BAD_REQUEST, Json(e)).into_response(),
                }
            }
            prompt.push_str("\n\n");
        }
        (prompt, images)
    };

    let task = InferenceTaskRequest::ToolCall {
        prompt,
        schema: serde_json::Value::Object(Default::default()),
        images,
    };

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if request.stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let task_clone = task.clone();
        let engine_clone = engine.clone();

        tokio::task::spawn_blocking(move || {
            let mut on_token = |piece: &str| -> bool { tx.send(piece.to_string()).is_ok() };
            let _ = engine_clone.execute(&task_clone, Some(&mut on_token));
        });

        let model_id = request.model.clone();
        let stream = UnboundedReceiverStream::new(rx).map(move |token_piece| {
            let chunk = ChatCompletionResponse {
                id: format!("chatcmpl-{created}"),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_id.clone(),
                choices: vec![Choice {
                    index: 0,
                    delta: ChoiceDelta {
                        content: Some(token_piece),
                        role: Some("assistant".to_string()),
                    },
                    finish_reason: None,
                }],
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, Infallible>(Event::default().data(json))
        });

        let done_stream = tokio_stream::once(Ok::<_, Infallible>(Event::default().data("[DONE]")));

        let sse_stream = stream.chain(done_stream);
        return Sse::new(sse_stream).into_response();
    }

    let task_clone = task.clone();
    let content = match tokio::task::spawn_blocking(move || {
        engine.execute(&task_clone, None).map_err(|e| e.to_string())
    })
    .await
    {
        Ok(Ok(InferenceTaskResponse::Text(text))) => text,
        Ok(Ok(InferenceTaskResponse::ToolCall(value))) => value.to_string(),
        Ok(Ok(InferenceTaskResponse::Error(e))) => e,
        Ok(Ok(other)) => format!("{other:?}"),
        Ok(Err(e)) => e,
        Err(join_err) => format!("Thread execution error: {join_err}"),
    };

    Json(chat_completion(&request.model, &content, created)).into_response()
}

fn chat_completion(model: &str, content: &str, created: u64) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            delta: ChoiceDelta {
                content: Some(content.to_string()),
                role: Some("assistant".to_string()),
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}
