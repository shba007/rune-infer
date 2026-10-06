use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use std::sync::Arc;

use super::AppState;
use crate::inference::types::{InferenceTaskRequest, InferenceTaskResponse};
use crate::media::decode_media;
use crate::types::{
    ApiError, DetectObjectsRequest, ErrorResponse, ImageCaptionRequest, ImageData,
    ImageEmbeddingsRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageRestorationRequest, ImageStyleTransferRequest, ImageUpscaleRequest,
    RecognizeObjectsRequest,
};

pub fn resolve_image_dimensions(
    size: Option<&str>,
    aspect_ratio: Option<&str>,
    default_res: Option<&str>,
) -> String {
    if let Some(s) = size {
        if s.contains('x') || s.contains('×') || s.contains('*') {
            return s.to_string();
        }
    }

    if let Some(ar) = aspect_ratio {
        let clean = ar.to_lowercase();
        if clean.contains("16:9") {
            return "1024x576".to_string();
        } else if clean.contains("9:16") {
            return "576x1024".to_string();
        } else if clean.contains("1:1") {
            return "1024x1024".to_string();
        } else if clean.contains("4:3") {
            return "1024x768".to_string();
        } else if clean.contains("3:4") {
            return "768x1024".to_string();
        } else if clean.contains("21:9") {
            return "1280x544".to_string();
        } else if clean.contains("9:21") {
            return "544x1280".to_string();
        }
    }

    if let Some(d) = default_res {
        let clean = d.split('(').next().unwrap_or(d).trim();
        if clean.contains('x') || clean.contains('×') || clean.contains('*') {
            return clean.replace(['×', '*'], "x");
        }
    }

    "1024x1024".to_string()
}

#[axum::debug_handler]
pub async fn image_generation_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageGenerationRequest>,
) -> Response {
    if request.prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'prompt' is a required string field for image generation.")
                    .with_type("invalid_request_error")
                    .with_param("prompt")
                    .with_code("missing_required_parameter"),
            }),
        )
            .into_response();
    }

    if let Some(ref s) = request.size {
        let allowed = ["256x256", "512x512", "1024x1024"];
        if !allowed.contains(&s.as_str()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(format!(
                        "'{}' is not one of ['256x256', '512x512', '1024x1024'] - 'size'",
                        s
                    ))
                    .with_type("invalid_request_error")
                    .with_param("size")
                    .with_code("invalid_parameter_value"),
                }),
            )
                .into_response();
        }
    }

    let model_id = match &request.model {
        Some(m) if !m.trim().is_empty() => m.clone(),
        _ => {
            match state
                .config
                .models
                .iter()
                .find(|m| m.modality == crate::config::Modality::ImageGeneration)
            {
                Some(m) => m.id.clone(),
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: ApiError::new(
                                "No image generation model specified and none found in config",
                            )
                            .with_type("invalid_request_error"),
                        }),
                    )
                        .into_response();
                }
            }
        }
    };

    let model_config = match state.config.find(&model_id) {
        Some(c) => c.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ApiError::new(format!("Model '{}' not found", model_id))
                        .with_type("invalid_request_error"),
                }),
            )
                .into_response();
        }
    };

    let inference = state.inference.clone();
    let model_id_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_id_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model failed to load: {err}"))
                            .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Thread join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let resolved_size = resolve_image_dimensions(
        request.size.as_deref(),
        request.aspect_ratio.as_deref(),
        model_config.max_resolution.as_deref(),
    );

    let task = InferenceTaskRequest::ImageGeneration {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone(),
        size: Some(resolved_size),
        response_format: request.response_format.clone(),
        steps: request.steps.or(model_config.default_steps),
        cfg_scale: request.cfg_scale.or(model_config.default_cfg_scale),
        seed: request.seed,
        sample_method: request
            .sample_method
            .clone()
            .or_else(|| model_config.default_sample_method.clone()),
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Image(img_resp) => Json(img_resp).into_response(),
            InferenceTaskResponse::Error(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(e).with_type("server_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response type from engine")
                        .with_type("server_error"),
                }),
            )
                .into_response(),
        },
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e).with_type("server_error"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(e.to_string()).with_type("server_error"),
            }),
        )
            .into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_caption_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageCaptionRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'image' is a required string field containing a URL or base64 data URI.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("image")
                    .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    let model_id = match request.model {
        Some(ref m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => match state.config.models.iter().find(|m| {
            m.vision
                || m.modality == crate::config::Modality::VisionText
                || m.id.contains("vision")
                || m.id.contains("ornith")
        }) {
            Some(m) => m.id.clone(),
            None => "ornith-1.5-9b".to_string(),
        },
    };

    let model_config = state.config.find(&model_id).cloned();
    let max_dims = model_config
        .as_ref()
        .map(|c| c.max_dimensions())
        .unwrap_or((2048, 2048));

    let decoded_frames = match decode_media(&image_uri, max_dims) {
        Ok(frames) if !frames.is_empty() => frames,
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("Failed to parse image from input.")
                        .with_type("invalid_request_error")
                        .with_param("image")
                        .with_code("invalid_image"),
                }),
            )
                .into_response();
        }
        Err(err_resp) => return (StatusCode::BAD_REQUEST, Json(err_resp)).into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();

    let inference = state.inference.clone();
    let model_id_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_id_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Vision model failed to load: {err}"))
                            .with_type("invalid_request_error")
                            .with_param("model")
                            .with_code("model_not_found"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Execution thread error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let detail = request.detail.unwrap_or_else(|| "detailed".to_string());
    let max_tokens = request.max_tokens.unwrap_or(128);

    let active_model = model_id.clone();
    let task = InferenceTaskRequest::ImageCaption {
        model: Some(active_model.clone()),
        image_bytes,
        detail,
        max_tokens,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;

    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Caption(cap_resp) => Json(cap_resp).into_response(),
            InferenceTaskResponse::Text(caption_str) => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let cap_resp = crate::types::ImageCaptionResponse {
                    id: format!("cap_{created}"),
                    object: "image.caption".to_string(),
                    created,
                    model: active_model,
                    caption: caption_str,
                    usage: output.usage,
                };
                Json(cap_resp).into_response()
            }
            InferenceTaskResponse::Error(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new(err).with_type("server_error"),
                }),
            )
                .into_response(),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ApiError::new("Unexpected response variant from caption engine"),
                }),
            )
                .into_response(),
        },
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err).with_type("server_error"),
            }),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: ApiError::new(err.to_string()).with_type("server_error"),
            }),
        )
            .into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_edit_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut prompt: Option<String> = None;
    let mut image_bytes: Vec<u8> = Vec::new();
    let mut mask_bytes: Vec<u8> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "prompt" => {
                if let Ok(text) = field.text().await {
                    prompt = Some(text);
                }
            }
            "image" => {
                if let Ok(bytes) = field.bytes().await {
                    image_bytes = bytes.to_vec();
                }
            }
            "mask" => {
                if let Ok(bytes) = field.bytes().await {
                    mask_bytes = bytes.to_vec();
                }
            }
            _ => {}
        }
    }

    if prompt.as_ref().is_none_or(|p| p.trim().is_empty()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new("'prompt' is a required string field for image edits.")
                    .with_type("invalid_request_error")
                    .with_param("prompt")
                    .with_code("missing_required_parameter"),
            }),
        )
            .into_response();
    }

    let mask_mismatch = match (
        image::load_from_memory(&image_bytes),
        image::load_from_memory(&mask_bytes),
    ) {
        (Ok(img), Ok(mask)) => (img.width(), img.height()) != (mask.width(), mask.height()),
        _ => false,
    };

    if mask_mismatch {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Uploaded mask dimensions must match the source image dimensions exactly.",
                )
                .with_type("invalid_request_error")
                .with_param("mask")
                .with_code("mask_dimension_mismatch"),
            }),
        )
            .into_response();
    }

    let model_id = match state
        .config
        .models
        .iter()
        .find(|m| m.modality == crate::config::Modality::ImageGeneration)
    {
        Some(m) => m.id.clone(),
        None => "z-image".to_string(),
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            Ok(Err(err)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Model failed to load: {err}"))
                            .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ApiError::new(format!("Thread join error: {e}")),
                    }),
                )
                    .into_response();
            }
        };

    let task = InferenceTaskRequest::ImageGeneration {
        prompt: prompt.unwrap_or_default(),
        negative_prompt: None,
        size: Some("1024x1024".to_string()),
        response_format: Some("b64_json".to_string()),
        steps: None,
        cfg_scale: None,
        seed: None,
        sample_method: None,
    };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Image(img) => Json(img).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_variation_handler(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut image_bytes: Vec<u8> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "image" {
            if let Ok(bytes) = field.bytes().await {
                image_bytes = bytes.to_vec();
            }
        }
    }

    if image_bytes.is_empty() || image_bytes.len() > 4 * 1024 * 1024 {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(ErrorResponse {
                error: ApiError::new(
                    "Uploaded image file must be a valid PNG, JPEG, or WEBP image format under 4 MB.",
                )
                .with_type("invalid_request_error")
                .with_param("image")
                .with_code("invalid_image_type"),
            }),
        )
            .into_response();
    }

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);
    Json(ImageGenerationResponse {
        created,
        data: vec![ImageData {
            b64_json: Some(b64),
            url: None,
        }],
    })
    .into_response()
}

#[axum::debug_handler]
pub async fn detect_objects_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetectObjectsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new(
                        "'image' is a required string field containing a URL or base64 data URI.",
                    )
                    .with_type("invalid_request_error")
                    .with_param("image")
                    .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "yolo-11x-detection".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("Failed to parse image from input.")
                        .with_type("invalid_request_error")
                        .with_param("image")
                        .with_code("invalid_image"),
                }),
            )
                .into_response();
        }
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();

    let task = InferenceTaskRequest::DetectObjects {
        model: Some(model_id.clone()),
        image_bytes,
        prompt: request.prompt,
        categories: request.categories,
        confidence_threshold: request.confidence_threshold,
        iou_threshold: request.iou_threshold,
        features: request.features.unwrap_or_default(),
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: ApiError::new("Detection model failed to load.")
                            .with_type("invalid_request_error"),
                    }),
                )
                    .into_response();
            }
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Detection(det) => Json(det).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageEmbeddingsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ApiError::new("'image' is a required string field.")
                        .with_type("invalid_request_error")
                        .with_param("image")
                        .with_code("missing_required_image"),
                }),
            )
                .into_response();
        }
    };

    if let Some(ref regs) = request.regions {
        for (idx, reg) in regs.iter().enumerate() {
            for &coord in &reg.box_ {
                if coord < 0.0 || coord > 1.0 {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            error: ApiError::new(
                                "Region bounding box coordinates must be normalized between 0.0 and 1.0.",
                            )
                            .with_type("invalid_request_error")
                            .with_param(format!("regions[{}].box", idx))
                            .with_code("invalid_box_coordinates"),
                        }),
                    )
                        .into_response();
                }
            }
        }
    }

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "clip-vit-large-patch14".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageEmbedding {
        model: Some(model_id.clone()),
        image_bytes,
        regions: request.regions,
        encoding_format: request.encoding_format,
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::ImageEmbedding(emb) => Json(emb).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn recognize_objects_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RecognizeObjectsRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "clip-vit-large-patch14".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::RecognizeObjects {
        model: Some(model_id.clone()),
        image_bytes,
        top_k: request.top_k.unwrap_or(3),
        targets: request.targets,
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Recognition(rec) => Json(rec).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_upscale_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageUpscaleRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "real-esrgan-x4plus".to_string());

    let decoded_frames = match decode_media(&image_uri, (4096, 4096)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageUpscale {
        model: Some(model_id.clone()),
        image_bytes,
        scale: request.scale.unwrap_or(4),
        response_format: request.response_format,
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Upscale(up) => Json(up).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_restoration_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageRestorationRequest>,
) -> Response {
    let image_uri = match request.image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "codeformer-v0.1".to_string());

    let decoded_frames = match decode_media(&image_uri, (2048, 2048)) {
        Ok(frames) if !frames.is_empty() => frames,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (image_bytes, _, _) = decoded_frames.into_iter().next().unwrap();
    let task = InferenceTaskRequest::ImageRestoration {
        model: Some(model_id.clone()),
        image_bytes,
        fidelity_weight: request.fidelity_weight.unwrap_or(0.7),
        face_upsample: request.face_upsample.unwrap_or(true),
        response_format: request.response_format,
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::Restoration(rest) => Json(rest).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[axum::debug_handler]
pub async fn image_style_transfer_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageStyleTransferRequest>,
) -> Response {
    let content_uri = match request.content_image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    let style_uri = match request.style_image {
        Some(ref img) if !img.trim().is_empty() => img.trim().to_string(),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model_id = request
        .model
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "neural-style-v2".to_string());

    let c_frames = match decode_media(&content_uri, (2048, 2048)) {
        Ok(f) if !f.is_empty() => f,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    let s_frames = match decode_media(&style_uri, (2048, 2048)) {
        Ok(f) if !f.is_empty() => f,
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };

    let (content_bytes, _, _) = c_frames.into_iter().next().unwrap();
    let (style_bytes, _, _) = s_frames.into_iter().next().unwrap();

    let task = InferenceTaskRequest::ImageStyleTransfer {
        model: Some(model_id.clone()),
        content_bytes,
        style_bytes,
        strength: request.strength.unwrap_or(0.85),
        response_format: request.response_format,
    };

    let inference = state.inference.clone();
    let model_to_fetch = model_id.clone();
    let engine =
        match tokio::task::spawn_blocking(move || inference.get_engine(&model_to_fetch)).await {
            Ok(Ok(e)) => e,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };

    let res =
        tokio::task::spawn_blocking(move || engine.execute(&task, None).map_err(|e| e.to_string()))
            .await;
    match res {
        Ok(Ok(output)) => match output.response {
            InferenceTaskResponse::StyleTransfer(st) => Json(st).into_response(),
            _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
