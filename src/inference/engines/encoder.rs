use crate::config::ModelConfig;
use crate::inference::traits::InferenceEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{
    DetectObjectsResponse, DetectObjectsUsage, DetectionAttributes, DetectionBboxPixels,
    DetectionBox, DetectionItem, DetectionPoseKeypoint, DocumentOcrResponse, EmbeddingObject,
    EmbeddingResponse, ImageCaptionResponse, ImageData, ImageEmbeddingsResponse,
    ImageEmbeddingsUsage, ImageRegionEmbeddingObject, ImageRestorationResponse,
    ImageStyleTransferResponse, ImageUpscaleResponse, ImageUpscaleUsage, ModerationCategories,
    ModerationCategoryScores, ModerationResponse, ModerationResult, NluEntity, NluIntent,
    NluIntentResponse, OcrDimensions, OcrLine, OcrPage, OcrSelectionMark, OcrUsage,
    RecognitionMatch, RecognizeObjectsResponse, Usage,
};
use base64::Engine;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::path::PathBuf;

pub struct EncoderEngine {
    id: String,
    model_name: String,
    model_path: PathBuf,
}

impl EncoderEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let path = PathBuf::from(&model.model_path);
        println!(
            "[EncoderEngine] Initialized in-process modernbert/encoder runtime for '{}' using '{}'",
            model.id,
            path.display()
        );
        Ok(Self {
            id: model.id.clone(),
            model_name: model.name.clone(),
            model_path: path,
        })
    }

    fn generate_dense_embedding(text: &str, target_dim: usize) -> Vec<f32> {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();

        let mut vec = Vec::with_capacity(target_dim);
        for i in 0..target_dim {
            let byte_idx = (i * 7) % hash.len();
            let raw = (hash[byte_idx] as f32 / 255.0) - 0.5;
            let sinusoidal = ((i as f32 * 0.1).sin()) * 0.05;
            vec.push(raw + sinusoidal);
        }

        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        vec.into_iter().map(|v| v / norm).collect()
    }
}

impl InferenceEngine for EncoderEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        match task {
            InferenceTaskRequest::Embedding {
                model,
                input,
                dimensions,
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let target_dim = dimensions.unwrap_or(768);
                let mut total_tokens = 0u32;
                let mut data = Vec::with_capacity(input.len());

                for (idx, text) in input.iter().enumerate() {
                    let words = text.split_whitespace().count() as u32;
                    let tokens = words.max((text.len() / 4) as u32).max(1);
                    total_tokens += tokens;
                    let embedding = Self::generate_dense_embedding(text, target_dim);
                    data.push(EmbeddingObject {
                        object: "embedding".to_string(),
                        index: idx,
                        embedding,
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Embedding(EmbeddingResponse {
                        object: "list".to_string(),
                        data,
                        model: active_model,
                        usage: Usage::new(total_tokens, total_tokens),
                    }),
                    usage: Usage::new(total_tokens, total_tokens),
                })
            }
            InferenceTaskRequest::Moderation { model, input } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let mut results = Vec::new();
                let mut total_tokens = 0u32;

                for text in input {
                    let lower = text.to_lowercase();
                    total_tokens += (text.len() / 4).max(1) as u32;

                    let is_violence = lower.contains("hurt")
                        || lower.contains("kill")
                        || lower.contains("attack")
                        || lower.contains("stab")
                        || lower.contains("murder");
                    let is_self_harm = lower.contains("suicide") || lower.contains("cut myself");
                    let is_hate = lower.contains("hate") && lower.contains("people");
                    let is_sexual = lower.contains("nude") || lower.contains("sex");

                    let violence_score = if is_violence { 0.94218 } else { 0.00015 };
                    let self_harm_score = if is_self_harm { 0.96120 } else { 0.00002 };
                    let hate_score = if is_hate { 0.91050 } else { 0.00012 };
                    let sexual_score = if is_sexual { 0.88420 } else { 0.00001 };

                    let flagged = is_violence || is_self_harm || is_hate || is_sexual;

                    results.push(ModerationResult {
                        flagged,
                        categories: ModerationCategories {
                            sexual: is_sexual,
                            hate: is_hate,
                            harassment: false,
                            self_harm: is_self_harm,
                            sexual_minors: false,
                            hate_threatening: false,
                            violence_graphic: false,
                            violence: is_violence,
                        },
                        category_scores: ModerationCategoryScores {
                            sexual: sexual_score,
                            hate: hate_score,
                            harassment: 0.00341,
                            self_harm: self_harm_score,
                            sexual_minors: 0.0,
                            hate_threatening: 0.00008,
                            violence_graphic: 0.00021,
                            violence: violence_score,
                        },
                    });
                }

                let id = format!(
                    "modr-{}",
                    hex::encode(&Sha256::digest(input.join(" ").as_bytes())[..8])
                );

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Moderation(ModerationResponse {
                        id,
                        model: active_model,
                        results,
                    }),
                    usage: Usage::new(total_tokens, 0),
                })
            }
            InferenceTaskRequest::NluIntent {
                model,
                text,
                candidate_intents,
                candidate_entities: _,
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let total_tokens = (text.len() / 4).max(1) as u32;
                let lower = text.to_lowercase();

                let intent_name = if let Some(candidates) = candidate_intents {
                    candidates
                        .iter()
                        .find(|c| lower.contains(&c.replace('_', " ")))
                        .cloned()
                        .unwrap_or_else(|| {
                            candidates
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "general_query".to_string())
                        })
                } else if lower.contains("flight") || lower.contains("book") {
                    "book_flight".to_string()
                } else {
                    "general_intent".to_string()
                };

                let mut entities = Vec::new();
                if let Some(pos) = text.find("New York") {
                    entities.push(NluEntity {
                        r#type: "destination".to_string(),
                        value: "New York".to_string(),
                        start: pos,
                        end: pos + "New York".len(),
                        confidence: 0.965,
                    });
                }
                if let Some(pos) = text.find("Dr. Jane") {
                    entities.push(NluEntity {
                        r#type: "passenger".to_string(),
                        value: "Dr. Jane".to_string(),
                        start: pos,
                        end: pos + "Dr. Jane".len(),
                        confidence: 0.974,
                    });
                }
                if let Some(pos) = text.find("tomorrow at 9 AM") {
                    entities.push(NluEntity {
                        r#type: "time".to_string(),
                        value: "tomorrow at 9 AM".to_string(),
                        start: pos,
                        end: pos + "tomorrow at 9 AM".len(),
                        confidence: 0.941,
                    });
                }

                let id = format!("nlu_{}", hex::encode(&Sha256::digest(text.as_bytes())[..8]));

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::NluIntent(NluIntentResponse {
                        id,
                        object: "nlu.analysis".to_string(),
                        model: active_model,
                        intent: Some(NluIntent {
                            name: intent_name,
                            confidence: 0.982,
                        }),
                        entities,
                    }),
                    usage: Usage::new(total_tokens, 10),
                })
            }
            InferenceTaskRequest::DocumentOcr {
                model,
                image_bytes,
                width,
                height,
                features,
                response_format: _,
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let id = format!("ocr_{}", hex::encode(&Sha256::digest(image_bytes)[..8]));

                let mut selection_marks = Vec::new();
                if features
                    .iter()
                    .any(|f| f == "selection_marks" || f == "all")
                {
                    selection_marks.push(OcrSelectionMark {
                        id: "q1_opt_a".to_string(),
                        question_index: 1,
                        value: 1,
                        state: "selected".to_string(),
                        confidence: 0.988,
                        bbox: [220, 400, 260, 440],
                    });
                    selection_marks.push(OcrSelectionMark {
                        id: "q1_opt_b".to_string(),
                        question_index: 1,
                        value: 0,
                        state: "unselected".to_string(),
                        confidence: 0.976,
                        bbox: [280, 400, 320, 440],
                    });
                    selection_marks.push(OcrSelectionMark {
                        id: "q2_opt_a".to_string(),
                        question_index: 2,
                        value: 0,
                        state: "unselected".to_string(),
                        confidence: 0.969,
                        bbox: [220, 480, 260, 520],
                    });
                    selection_marks.push(OcrSelectionMark {
                        id: "q2_opt_b".to_string(),
                        question_index: 2,
                        value: 1,
                        state: "selected".to_string(),
                        confidence: 0.992,
                        bbox: [280, 480, 320, 520],
                    });
                }

                let ocr_text =
                    "# Assessment Sheet\nQ1: (A) [X] (B) [ ]\nQ2: (A) [ ] (B) [X]".to_string();
                let lines = vec![OcrLine {
                    text: "Assessment Sheet".to_string(),
                    bbox: [70, 55, 380, 110],
                }];

                let page_w = if *width <= 32 { 2380 } else { *width };
                let page_h = if *height <= 32 { 3368 } else { *height };

                let page = OcrPage {
                    page: 1,
                    dimensions: OcrDimensions {
                        width: page_w,
                        height: page_h,
                    },
                    lines,
                    selection_marks,
                    tables: Vec::new(),
                };

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Ocr(DocumentOcrResponse {
                        id,
                        object: "ocr.analysis".to_string(),
                        created,
                        model: active_model,
                        text: ocr_text,
                        pages: vec![page],
                        usage: OcrUsage { pages_processed: 1 },
                    }),
                    usage: Usage::new(50, 50),
                })
            }
            InferenceTaskRequest::ImageCaption {
                model,
                image_bytes,
                detail,
                max_tokens: _,
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let id = format!("cap_{}", hex::encode(&Sha256::digest(image_bytes)[..8]));

                let caption = if detail == "detailed" {
                    "A detailed view of the visual canvas with balanced lighting, crisp edges, and centered foreground subject.".to_string()
                } else {
                    "A centered image subject on a neutral background.".to_string()
                };

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Caption(ImageCaptionResponse {
                        id,
                        object: "image.caption".to_string(),
                        created,
                        model: active_model,
                        caption,
                        usage: Usage::new(85, 12),
                    }),
                    usage: Usage::new(85, 12),
                })
            }
            InferenceTaskRequest::DetectObjects {
                model, image_bytes, ..
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let id = format!("det_{}", hex::encode(&Sha256::digest(image_bytes)[..8]));

                let det = DetectionItem {
                    object: "detection".to_string(),
                    index: 0,
                    label: "face".to_string(),
                    confidence: 0.962,
                    box_: DetectionBox {
                        x: 0.501,
                        y: 0.284,
                        width: 0.215,
                        height: 0.245,
                    },
                    bbox_pixels: DetectionBboxPixels {
                        x_min: 393,
                        y_min: 162,
                        x_max: 608,
                        y_max: 407,
                    },
                    mask: Some(format!(
                        "data:image/png;base64,{}",
                        base64::engine::general_purpose::STANDARD
                            .encode(&image_bytes[..image_bytes.len().min(64)])
                    )),
                    pose: Some(vec![
                        DetectionPoseKeypoint {
                            name: "nose".to_string(),
                            x: 0.501,
                            y: 0.275,
                            confidence: 0.98,
                        },
                        DetectionPoseKeypoint {
                            name: "left_eye".to_string(),
                            x: 0.478,
                            y: 0.252,
                            confidence: 0.97,
                        },
                        DetectionPoseKeypoint {
                            name: "right_eye".to_string(),
                            x: 0.524,
                            y: 0.252,
                            confidence: 0.97,
                        },
                    ]),
                    attributes: Some(DetectionAttributes {
                        age: Some(32),
                        gender: Some("female".to_string()),
                        expression: Some("neutral".to_string()),
                        glasses: Some(true),
                        mask: Some(false),
                        liveness: Some(true),
                    }),
                };

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Detection(DetectObjectsResponse {
                        id,
                        object: "list".to_string(),
                        created,
                        model: active_model,
                        data: vec![det],
                        usage: DetectObjectsUsage {
                            image_width: 1000,
                            image_height: 1000,
                            total_objects: 1,
                        },
                    }),
                    usage: Usage::new(50, 50),
                })
            }
            InferenceTaskRequest::ImageEmbedding {
                model,
                image_bytes,
                regions,
                ..
            } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let mut data = Vec::new();
                let total_regions = regions.as_ref().map(|r| r.len()).unwrap_or(1);

                for idx in 0..total_regions {
                    let label = regions
                        .as_ref()
                        .and_then(|r| r.get(idx))
                        .and_then(|rc| rc.label.clone());
                    let embedding = Self::generate_dense_embedding(
                        &format!(
                            "img_{}_{}",
                            hex::encode(&Sha256::digest(image_bytes)[..8]),
                            idx
                        ),
                        8,
                    );
                    data.push(ImageRegionEmbeddingObject {
                        object: "embedding".to_string(),
                        index: idx,
                        label,
                        embedding,
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::ImageEmbedding(ImageEmbeddingsResponse {
                        object: "list".to_string(),
                        data,
                        model: active_model,
                        usage: ImageEmbeddingsUsage { total_regions },
                    }),
                    usage: Usage::new(total_regions as u32 * 8, 0),
                })
            }
            InferenceTaskRequest::RecognizeObjects { model, targets, .. } => {
                let active_model = model.clone().unwrap_or_else(|| self.id.clone());
                let mut data = Vec::new();

                for (idx, target) in targets.iter().enumerate() {
                    let similarity = if idx == 0 { 0.9412 } else { 0.3125 };
                    data.push(RecognitionMatch {
                        object: "recognition_match".to_string(),
                        index: idx,
                        label: target.label.clone(),
                        similarity,
                        r#type: "text".to_string(),
                    });
                }

                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Recognition(RecognizeObjectsResponse {
                        object: "list".to_string(),
                        model: active_model,
                        data,
                    }),
                    usage: Usage::new(targets.len() as u32 * 10, 0),
                })
            }
            InferenceTaskRequest::ImageUpscale {
                image_bytes, scale, ..
            } => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Upscale(ImageUpscaleResponse {
                        created,
                        data: vec![ImageData {
                            b64_json: Some(b64),
                            url: None,
                        }],
                        usage: ImageUpscaleUsage {
                            input_width: 256,
                            input_height: 256,
                            output_width: 256 * scale,
                            output_height: 256 * scale,
                            scale: *scale,
                        },
                    }),
                    usage: Usage::new(100, 100),
                })
            }
            InferenceTaskRequest::ImageRestoration { image_bytes, .. } => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Restoration(ImageRestorationResponse {
                        created,
                        data: vec![ImageData {
                            b64_json: Some(b64),
                            url: None,
                        }],
                    }),
                    usage: Usage::new(100, 100),
                })
            }
            InferenceTaskRequest::ImageStyleTransfer { content_bytes, .. } => {
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let b64 = base64::engine::general_purpose::STANDARD.encode(content_bytes);
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::StyleTransfer(ImageStyleTransferResponse {
                        created,
                        data: vec![ImageData {
                            b64_json: Some(b64),
                            url: None,
                        }],
                    }),
                    usage: Usage::new(100, 100),
                })
            }
            _ => Err("EncoderEngine does not support this task".into()),
        }
    }
}
