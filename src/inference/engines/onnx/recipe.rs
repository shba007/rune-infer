//! Catalog-declared ONNX models.
//!
//! The reviewed YOLO/BiRefNet exports are recognised by hash. Every other
//! single-input image model is described by `extra.onnx_recipe` in the catalog:
//! how to resize and normalise the picture, and how to read the outputs. The
//! recipes here cover image embeddings, classifiers and semantic segmentation.
//! Callers compose models (detect faces, crop, classify); Infer never chains them.
use crate::types::{
    ClassificationHead, ClassificationScore, DetectionBboxPixels, DetectionBox, DetectionItem,
    DetectionPoseKeypoint, Measurement, SemanticClass,
};
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Embedding,
    Classification,
    SemanticSegmentation,
    /// SCRFD-style anchor-free face detector with five landmarks.
    FaceDetection,
    /// 6DRepNet-style rotation matrix to pitch/yaw/roll.
    HeadPose,
    /// L2CS-style binned yaw/pitch.
    Gaze,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// Stretch to `input_size`.
    #[default]
    Stretch,
    /// Keep the aspect ratio and pad bottom/right with black.
    Letterbox,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChannelOrder {
    #[default]
    Rgb,
    Bgr,
}

fn zeros() -> [f32; 3] {
    [0.0; 3]
}
fn ones() -> [f32; 3] {
    [1.0; 3]
}
fn byte_scale() -> f32 {
    1.0 / 255.0
}
fn default_strides() -> Vec<u32> {
    vec![8, 16, 32]
}
fn default_anchors() -> u32 {
    2
}
fn default_landmarks() -> Vec<String> {
    ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
        .map(String::from)
        .to_vec()
}
fn default_bin_degrees() -> f32 {
    4.0
}
fn default_bin_offset() -> f32 {
    180.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub kind: Kind,
    /// `[width, height]` the picture is stretched to before inference.
    pub input_size: [u32; 2],
    /// Per-channel mean, subtracted after `pixel_scale` (in `channel_order`).
    #[serde(default = "zeros")]
    pub mean: [f32; 3],
    #[serde(default = "ones")]
    pub std: [f32; 3],
    /// Multiplier applied to 0..=255 pixels first (default 1/255).
    #[serde(default = "byte_scale")]
    pub pixel_scale: f32,
    #[serde(default)]
    pub channel_order: ChannelOrder,
    /// Classification: one label list per model output, in output order.
    #[serde(default)]
    pub heads: Vec<Vec<String>>,
    /// Semantic segmentation: class labels indexed by class id.
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub fit: Fit,
    /// Face detection: feature-map strides, in output order.
    #[serde(default = "default_strides")]
    pub strides: Vec<u32>,
    /// Face detection: anchors per feature-map cell.
    #[serde(default = "default_anchors")]
    pub anchors: u32,
    /// Face detection: landmark names, in output order.
    #[serde(default = "default_landmarks")]
    pub landmarks: Vec<String>,
    /// Gaze: degrees covered by one bin and the angle of bin zero.
    #[serde(default = "default_bin_degrees")]
    pub bin_degrees: f32,
    #[serde(default = "default_bin_offset")]
    pub bin_offset: f32,
}

impl Recipe {
    pub fn parse(value: &serde_json::Value) -> Result<Self, Box<dyn Error>> {
        let recipe: Self = serde_json::from_value(value.clone())
            .map_err(|e| format!("Invalid extra.onnx_recipe: {e}"))?;
        let [w, h] = recipe.input_size;
        if !(8..=2048).contains(&w) || !(8..=2048).contains(&h) {
            return Err("extra.onnx_recipe.input_size must be between 8 and 2048".into());
        }
        if recipe.std.iter().any(|v| !v.is_finite() || *v == 0.0)
            || recipe.mean.iter().any(|v| !v.is_finite())
            || !recipe.pixel_scale.is_finite()
        {
            return Err(
                "extra.onnx_recipe mean, std and pixel_scale must be finite and std non-zero"
                    .into(),
            );
        }
        match recipe.kind {
            Kind::Classification
                if recipe.heads.is_empty() || recipe.heads.iter().any(Vec::is_empty) =>
            {
                return Err("A classification recipe needs a non-empty label list per head".into());
            }
            Kind::SemanticSegmentation if recipe.labels.is_empty() || recipe.labels.len() > 256 => {
                return Err("A semantic-segmentation recipe needs 1 to 256 labels".into());
            }
            Kind::FaceDetection
                if recipe.strides.is_empty()
                    || recipe.anchors == 0
                    || recipe
                        .strides
                        .iter()
                        .any(|s| *s == 0 || w % s != 0 || h % s != 0) =>
            {
                return Err(
                    "A face-detection recipe needs strides that divide input_size and at least one anchor"
                        .into(),
                );
            }
            Kind::Gaze if !recipe.bin_degrees.is_finite() || recipe.bin_degrees <= 0.0 => {
                return Err("A gaze recipe needs a positive bin_degrees".into());
            }
            _ => {}
        }
        Ok(recipe)
    }

    pub fn input_bytes(&self) -> usize {
        (self.input_size[0] * self.input_size[1] * 3 * 4) as usize
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Decode the worker's framed reply: `u32 outputs`, then per output
/// `u32 rank`, `rank * u32 dims`, `u32 count`, `count * f32`.
pub fn parse_tensors(bytes: &[u8]) -> Result<Vec<Tensor>, Box<dyn Error>> {
    let mut at = 0usize;
    let word = |at: &mut usize| -> Result<u32, Box<dyn Error>> {
        let end = *at + 4;
        let chunk = bytes.get(*at..end).ok_or("Truncated ONNX worker reply")?;
        *at = end;
        Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    };
    let outputs = word(&mut at)? as usize;
    let mut tensors = Vec::with_capacity(outputs.min(16));
    for _ in 0..outputs {
        let rank = word(&mut at)? as usize;
        if rank > 8 {
            return Err("ONNX worker reply has an invalid rank".into());
        }
        let shape = (0..rank)
            .map(|_| word(&mut at).map(|v| v as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let count = word(&mut at)? as usize;
        let end = at + count * 4;
        let raw = bytes.get(at..end).ok_or("Truncated ONNX worker reply")?;
        at = end;
        let data: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if !data.iter().all(|v| v.is_finite()) {
            return Err("ONNX returned non-finite values".into());
        }
        tensors.push(Tensor { shape, data });
    }
    Ok(tensors)
}

pub struct Prepared {
    /// NCHW f32 little-endian model input.
    pub bytes: Vec<u8>,
    /// Source picture size.
    pub width: u32,
    pub height: u32,
    /// Source-to-input scale (1 for stretched inputs on each axis is not used).
    pub scale: f32,
}

/// Decode the picture, fit it to `input_size`, and write the model input.
pub fn preprocess(bytes: &[u8], recipe: &Recipe) -> Result<Prepared, Box<dyn Error>> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    reader.limits(limits);
    let source = reader.decode()?.to_rgb8();
    let (w, h) = source.dimensions();
    if w == 0 || h == 0 {
        return Err("empty image".into());
    }
    let [tw, th] = recipe.input_size;
    let (resized, scale) = match recipe.fit {
        Fit::Stretch => (
            image::imageops::resize(&source, tw, th, image::imageops::FilterType::Triangle),
            1.0,
        ),
        Fit::Letterbox => {
            let scale = (tw as f32 / w as f32).min(th as f32 / h as f32);
            let nw = ((w as f32 * scale).round() as u32).clamp(1, tw);
            let nh = ((h as f32 * scale).round() as u32).clamp(1, th);
            let fitted =
                image::imageops::resize(&source, nw, nh, image::imageops::FilterType::Triangle);
            let mut canvas = image::RgbImage::new(tw, th);
            image::imageops::replace(&mut canvas, &fitted, 0, 0);
            (canvas, scale)
        }
    };
    let order: [usize; 3] = match recipe.channel_order {
        ChannelOrder::Rgb => [0, 1, 2],
        ChannelOrder::Bgr => [2, 1, 0],
    };
    let mut input = Vec::with_capacity(recipe.input_bytes());
    for (plane, channel) in order.into_iter().enumerate() {
        for pixel in resized.pixels() {
            let value = (pixel[channel] as f32 * recipe.pixel_scale - recipe.mean[plane])
                / recipe.std[plane];
            input.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(Prepared {
        bytes: input,
        width: w,
        height: h,
        scale,
    })
}

/// First output, flattened and scaled to unit length.
pub fn decode_embedding(tensors: &[Tensor]) -> Result<Vec<f32>, Box<dyn Error>> {
    let first = tensors.first().ok_or("ONNX returned no outputs")?;
    if first.data.is_empty() {
        return Err("ONNX returned an empty embedding".into());
    }
    let norm = first.data.iter().map(|v| v * v).sum::<f32>().sqrt();
    Ok(if norm > 0.0 {
        first.data.iter().map(|v| v / norm).collect()
    } else {
        first.data.clone()
    })
}

fn label(labels: &[String], index: usize) -> String {
    labels
        .get(index)
        .cloned()
        .unwrap_or_else(|| format!("class_{index}"))
}

pub fn decode_classification(
    tensors: &[Tensor],
    recipe: &Recipe,
    top_k: usize,
) -> Result<Vec<ClassificationHead>, Box<dyn Error>> {
    if tensors.len() < recipe.heads.len() {
        return Err(format!(
            "ONNX returned {} outputs but the recipe declares {} heads",
            tensors.len(),
            recipe.heads.len()
        )
        .into());
    }
    let mut heads = Vec::new();
    for (index, labels) in recipe.heads.iter().enumerate() {
        let data = &tensors[index].data;
        let scores = if data.len() == 1 {
            // An already-decided class id.
            let id = data[0];
            if id < 0.0 || id.fract() != 0.0 {
                return Err("ONNX returned an invalid class id".into());
            }
            let id = id as usize;
            vec![ClassificationScore {
                index: id,
                label: label(labels, id),
                score: 1.0,
            }]
        } else {
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = data.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let mut ranked: Vec<(usize, f32)> = exp.iter().map(|v| v / sum).enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked
                .into_iter()
                .take(top_k.max(1))
                .map(|(id, score)| ClassificationScore {
                    index: id,
                    label: label(labels, id),
                    score,
                })
                .collect()
        };
        heads.push(ClassificationHead { index, scores });
    }
    Ok(heads)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// A 3x3 rotation matrix to pitch, yaw and roll in degrees (6DRepNet convention).
pub fn decode_head_pose(tensors: &[Tensor]) -> Result<Vec<Measurement>, Box<dyn Error>> {
    let r = &tensors.first().ok_or("ONNX returned no outputs")?.data;
    if r.len() != 9 {
        return Err("Head-pose output must be a 3x3 rotation matrix".into());
    }
    let sy = (r[0] * r[0] + r[3] * r[3]).sqrt();
    let (pitch, yaw, roll) = if sy >= 1e-6 {
        (r[7].atan2(r[8]), (-r[6]).atan2(sy), r[3].atan2(r[0]))
    } else {
        ((-r[5]).atan2(r[4]), (-r[6]).atan2(sy), 0.0)
    };
    Ok([("pitch", pitch), ("yaw", yaw), ("roll", roll)]
        .into_iter()
        .map(|(name, radians)| Measurement {
            name: name.into(),
            value: radians.to_degrees() as f64,
            unit: "degrees".into(),
        })
        .collect())
}

/// Two binned outputs (yaw then pitch): softmax expectation, scaled to degrees.
pub fn decode_gaze(
    tensors: &[Tensor],
    recipe: &Recipe,
) -> Result<Vec<Measurement>, Box<dyn Error>> {
    if tensors.len() < 2 {
        return Err("Gaze models return yaw and pitch bins as two outputs".into());
    }
    ["yaw", "pitch"]
        .into_iter()
        .zip(tensors)
        .map(|(name, tensor)| {
            if tensor.data.len() < 2 {
                return Err("Gaze output has too few bins".into());
            }
            let expected: f32 = softmax(&tensor.data)
                .iter()
                .enumerate()
                .map(|(bin, p)| bin as f32 * p)
                .sum();
            Ok(Measurement {
                name: name.into(),
                value: (expected * recipe.bin_degrees - recipe.bin_offset) as f64,
                unit: "degrees".into(),
            })
        })
        .collect()
}

/// Anchor-free face detector outputs: per stride a score map (`N x 1`), box
/// distances (`N x 4`, in stride units) and, optionally, landmarks (`N x 2k`).
pub fn decode_faces(
    tensors: &[Tensor],
    recipe: &Recipe,
    source: (u32, u32, f32),
    threshold: f32,
    iou_threshold: f32,
    include_landmarks: bool,
) -> Result<Vec<DetectionItem>, Box<dyn Error>> {
    let (w, h, scale) = source;
    let strides = recipe.strides.len();
    let with_landmarks = match tensors.len() {
        n if n == strides * 3 => true,
        n if n == strides * 2 => false,
        n => {
            return Err(format!("Face detector returned {n} outputs for {strides} strides").into());
        }
    };
    let (iw, ih) = (recipe.input_size[0], recipe.input_size[1]);
    let mut candidates: Vec<(f32, [f32; 4], Vec<[f32; 2]>)> = Vec::new();
    for (level, stride) in recipe.strides.iter().enumerate() {
        let (cols, rows) = ((iw / stride) as usize, (ih / stride) as usize);
        let cells = cols * rows * recipe.anchors as usize;
        let scores = &tensors[level].data;
        let boxes = &tensors[level + strides].data;
        if scores.len() != cells || boxes.len() != cells * 4 {
            return Err("Face detector output does not match the declared strides".into());
        }
        let points = with_landmarks.then(|| &tensors[level + 2 * strides].data);
        let landmark_values = points.map(|p| p.len() / cells).unwrap_or(0);
        for index in 0..cells {
            if scores[index] < threshold {
                continue;
            }
            let cell = index / recipe.anchors as usize;
            let (cx, cy) = (
                (cell % cols) as f32 * *stride as f32,
                (cell / cols) as f32 * *stride as f32,
            );
            let d = &boxes[index * 4..index * 4 + 4];
            let s = *stride as f32;
            let rect = [
                (cx - d[0] * s) / scale,
                (cy - d[1] * s) / scale,
                (cx + d[2] * s) / scale,
                (cy + d[3] * s) / scale,
            ];
            let marks = points
                .map(|p| {
                    p[index * landmark_values..(index + 1) * landmark_values]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|xy| [(cx + xy[0] * s) / scale, (cy + xy[1] * s) / scale])
                        .collect()
                })
                .unwrap_or_default();
            candidates.push((scores[index], rect, marks));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let iou = |a: &[f32; 4], b: &[f32; 4]| {
        let inter =
            (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    };
    let mut kept: Vec<(f32, [f32; 4], Vec<[f32; 2]>)> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|other| iou(&candidate.1, &other.1) <= iou_threshold)
        {
            kept.push(candidate);
            if kept.len() == 300 {
                break;
            }
        }
    }
    let (wf, hf) = (w as f32, h as f32);
    let mut items = Vec::new();
    for (score, rect, marks) in kept {
        let (x1, y1) = (rect[0].clamp(0.0, wf), rect[1].clamp(0.0, hf));
        let (x2, y2) = (rect[2].clamp(0.0, wf), rect[3].clamp(0.0, hf));
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let pose = (include_landmarks && !marks.is_empty()).then(|| {
            marks
                .iter()
                .enumerate()
                .map(|(i, xy)| DetectionPoseKeypoint {
                    name: recipe
                        .landmarks
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("landmark_{i}")),
                    x: (xy[0].clamp(0.0, wf) / wf) as f64,
                    y: (xy[1].clamp(0.0, hf) / hf) as f64,
                    confidence: score as f64,
                })
                .collect()
        });
        items.push(DetectionItem {
            object: "detection".into(),
            index: items.len(),
            label: "face".into(),
            confidence: score as f64,
            box_: DetectionBox {
                x: ((x1 + x2) / 2.0 / wf) as f64,
                y: ((y1 + y2) / 2.0 / hf) as f64,
                width: ((x2 - x1) / wf) as f64,
                height: ((y2 - y1) / hf) as f64,
            },
            bbox_pixels: DetectionBboxPixels {
                x_min: x1.round() as u32,
                y_min: y1.round() as u32,
                x_max: x2.round() as u32,
                y_max: y2.round() as u32,
            },
            oriented_box: None,
            mask: None,
            pose,
            attributes: None,
        });
    }
    Ok(items)
}

/// `[1, classes, h, w]` logits to a grayscale class-id PNG at the source size.
pub fn decode_segmentation(
    tensors: &[Tensor],
    recipe: &Recipe,
    width: u32,
    height: u32,
) -> Result<(String, Vec<SemanticClass>), Box<dyn Error>> {
    use base64::Engine as _;
    let logits = tensors.first().ok_or("ONNX returned no outputs")?;
    let [1, classes, h, w] = logits.shape[..] else {
        return Err(format!("Unsupported segmentation output shape {:?}", logits.shape).into());
    };
    if classes == 0 || classes > 256 || logits.data.len() != classes * h * w {
        return Err("ONNX returned an invalid segmentation tensor".into());
    }
    let mut map = image::GrayImage::new(w as u32, h as u32);
    for (position, pixel) in map.pixels_mut().enumerate() {
        let mut best = (0usize, f32::NEG_INFINITY);
        for class in 0..classes {
            let value = logits.data[class * h * w + position];
            if value > best.1 {
                best = (class, value);
            }
        }
        pixel.0[0] = best.0 as u8;
    }
    let output = image::imageops::resize(&map, width, height, image::imageops::FilterType::Nearest);
    let mut seen = [false; 256];
    for pixel in output.pixels() {
        seen[pixel.0[0] as usize] = true;
    }
    let classes = seen
        .iter()
        .enumerate()
        .filter(|(_, present)| **present)
        .map(|(id, _)| SemanticClass {
            id: id as u8,
            label: label(&recipe.labels, id),
        })
        .collect();
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(output).write_to(&mut encoded, image::ImageFormat::Png)?;
    Ok((
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())
        ),
        classes,
    ))
}

/// Worker for catalog-declared models: reads `width * height * 3` f32 values
/// per request and writes every model output as one framed reply.
#[cfg(windows)]
pub fn worker(
    runtime: &std::path::Path,
    model: &std::path::Path,
    cpu: bool,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    use ort::{session::Session, value::Tensor};
    use std::io::{Read, Write};
    super::verify_runtime(runtime, !cpu).map_err(|e| anyhow::anyhow!("{e}"))?;
    let environment = ort::init_from(runtime)?.build()?;
    let mut builder = Session::builder(&environment)?;
    if !cpu {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let mut session = builder.commit_from_file(model)?;
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut bytes = vec![0u8; (width * height * 3 * 4) as usize];
    loop {
        if stdin.read(&mut bytes[..1])? == 0 {
            break;
        }
        stdin.read_exact(&mut bytes[1..])?;
        let input: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect();
        let outputs = session.run(ort::inputs![Tensor::from_array((
            [1usize, 3, height as usize, width as usize],
            input
        ))?])?;
        let mut reply = Vec::new();
        reply.extend_from_slice(&(outputs.len() as u32).to_le_bytes());
        for index in 0..outputs.len() {
            let (shape, values): (Vec<i64>, Vec<f32>) =
                if let Ok((shape, values)) = outputs[index].try_extract_tensor::<f32>() {
                    (shape.to_vec(), values.to_vec())
                } else if let Ok((shape, values)) = outputs[index].try_extract_tensor::<i64>() {
                    (shape.to_vec(), values.iter().map(|v| *v as f32).collect())
                } else {
                    let (shape, values) = outputs[index].try_extract_tensor::<i32>()?;
                    (shape.to_vec(), values.iter().map(|v| *v as f32).collect())
                };
            reply.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in &shape {
                reply.extend_from_slice(&(*dim as u32).to_le_bytes());
            }
            reply.extend_from_slice(&(values.len() as u32).to_le_bytes());
            for value in &values {
                reply.extend_from_slice(&value.to_le_bytes());
            }
        }
        stdout.write_all(&reply)?;
        stdout.flush()?;
    }
    Ok(())
}
#[cfg(not(windows))]
pub fn worker(
    _runtime: &std::path::Path,
    _model: &std::path::Path,
    _cpu: bool,
    _width: u32,
    _height: u32,
) -> anyhow::Result<()> {
    anyhow::bail!("ONNX models currently support Windows x64 only")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(kind: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": kind, "input_size": [8, 8],
            "heads": [["a", "b", "c"]], "labels": ["bg", "skin"]
        })
    }

    #[test]
    fn recipe_validation_rejects_unusable_declarations() {
        assert!(Recipe::parse(&recipe("classification")).is_ok());
        assert!(
            Recipe::parse(&serde_json::json!({"kind": "embedding", "input_size": [4, 4]})).is_err()
        );
        assert!(Recipe::parse(&serde_json::json!({"kind": "embedding", "input_size": [112, 112], "std": [0, 1, 1]})).is_err());
        assert!(
            Recipe::parse(&serde_json::json!({"kind": "classification", "input_size": [64, 64]}))
                .is_err()
        );
        assert!(
            Recipe::parse(
                &serde_json::json!({"kind": "embedding", "input_size": [64, 64], "bogus": 1})
            )
            .is_err()
        );
    }

    #[test]
    fn preprocessing_normalises_and_orders_channels() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 51])))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let mut recipe = Recipe::parse(&serde_json::json!({
            "kind": "embedding", "input_size": [8, 8], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]
        }))
        .unwrap();
        let prepared = preprocess(png.get_ref(), &recipe).unwrap();
        let bytes = prepared.bytes;
        assert_eq!(
            (prepared.width, prepared.height, bytes.len()),
            (2, 2, 8 * 8 * 3 * 4)
        );
        let at = |plane: usize| {
            f32::from_le_bytes(
                bytes[plane * 64 * 4..plane * 64 * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        assert!(
            (at(0) - 1.0).abs() < 1e-5 && (at(1) + 1.0).abs() < 1e-5 && (at(2) + 0.6).abs() < 1e-5
        );
        recipe.channel_order = ChannelOrder::Bgr;
        let bytes = preprocess(png.get_ref(), &recipe).unwrap().bytes;
        assert!((f32::from_le_bytes(bytes[..4].try_into().unwrap()) + 0.6).abs() < 1e-5);
    }

    fn framed(outputs: &[(&[u32], &[f32])]) -> Vec<u8> {
        let mut bytes = (outputs.len() as u32).to_le_bytes().to_vec();
        for (shape, values) in outputs {
            bytes.extend((shape.len() as u32).to_le_bytes());
            shape.iter().for_each(|d| bytes.extend(d.to_le_bytes()));
            bytes.extend((values.len() as u32).to_le_bytes());
            values.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
        }
        bytes
    }

    #[test]
    fn framed_replies_round_trip_and_reject_truncation() {
        let bytes = framed(&[(&[1, 3], &[1.0, 2.0, 3.0]), (&[1], &[4.0])]);
        let tensors = parse_tensors(&bytes).unwrap();
        assert_eq!(tensors[0].shape, [1, 3]);
        assert_eq!(tensors[1].data, [4.0]);
        assert!(parse_tensors(&bytes[..bytes.len() - 2]).is_err());
        assert!(parse_tensors(&framed(&[(&[1], &[f32::NAN])])).is_err());
    }

    #[test]
    fn embeddings_are_unit_length() {
        let embedding =
            decode_embedding(&parse_tensors(&framed(&[(&[1, 2], &[3.0, 4.0])])).unwrap()).unwrap();
        assert!((embedding[0] - 0.6).abs() < 1e-6 && (embedding[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn classification_ranks_logits_and_passes_ids_through() {
        let recipe = Recipe::parse(&serde_json::json!({
            "kind": "classification", "input_size": [8, 8], "heads": [["a", "b", "c"], ["x", "y"]]
        }))
        .unwrap();
        let tensors =
            parse_tensors(&framed(&[(&[1, 3], &[0.1, 2.0, 0.5]), (&[1], &[1.0])])).unwrap();
        let heads = decode_classification(&tensors, &recipe, 2).unwrap();
        assert_eq!(heads[0].scores.len(), 2);
        assert_eq!(heads[0].scores[0].label, "b");
        assert!(heads[0].scores[0].score > heads[0].scores[1].score);
        assert_eq!(
            (heads[1].scores[0].label.as_str(), heads[1].scores[0].score),
            ("y", 1.0)
        );
        assert!(decode_classification(&tensors[..1], &recipe, 2).is_err());
    }

    #[test]
    fn segmentation_takes_the_arg_max_class_at_source_size() {
        let recipe = Recipe::parse(&recipe("semantic-segmentation")).unwrap();
        // Two classes over a 1x2 map: class 1 wins the second pixel.
        let tensors = parse_tensors(&framed(&[(&[1, 2, 1, 2], &[1.0, 0.0, 0.0, 1.0])])).unwrap();
        let (uri, classes) = decode_segmentation(&tensors, &recipe, 4, 2).unwrap();
        assert!(uri.starts_with("data:image/png;base64,"));
        assert_eq!(
            classes.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(),
            ["bg", "skin"]
        );
        assert!(decode_segmentation(&tensors[..0], &recipe, 4, 2).is_err());
    }

    #[test]
    fn head_pose_reads_euler_angles_from_a_rotation_matrix() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let tensors = parse_tensors(&framed(&[(&[1, 3, 3], &identity)])).unwrap();
        let pose = decode_head_pose(&tensors).unwrap();
        assert_eq!(
            pose.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
            ["pitch", "yaw", "roll"]
        );
        assert!(
            pose.iter()
                .all(|m| m.value.abs() < 1e-6 && m.unit == "degrees")
        );
        // Rotation of 90 degrees about the x axis is a pitch of 90.
        let about_x = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let tensors = parse_tensors(&framed(&[(&[1, 3, 3], &about_x)])).unwrap();
        assert!((decode_head_pose(&tensors).unwrap()[0].value - 90.0).abs() < 1e-4);
        assert!(
            decode_head_pose(&parse_tensors(&framed(&[(&[2], &[0.0, 1.0])])).unwrap()).is_err()
        );
    }

    #[test]
    fn gaze_bins_become_degrees() {
        let recipe =
            Recipe::parse(&serde_json::json!({"kind": "gaze", "input_size": [8, 8]})).unwrap();
        let flat = vec![0.0f32; 90];
        let mut peaked = vec![-50.0f32; 90];
        peaked[45] = 50.0; // bin 45 -> 45 * 4 - 180 = 0 degrees
        let tensors = parse_tensors(&framed(&[(&[1, 90], &peaked), (&[1, 90], &flat)])).unwrap();
        let gaze = decode_gaze(&tensors, &recipe).unwrap();
        assert_eq!(gaze[0].name, "yaw");
        assert!(gaze[0].value.abs() < 1e-3);
        assert!((gaze[1].value - (44.5 * 4.0 - 180.0)).abs() < 1e-3);
        assert!(decode_gaze(&tensors[..1], &recipe).is_err());
    }

    #[test]
    fn faces_decode_boxes_landmarks_and_suppress_overlaps() {
        let recipe = Recipe::parse(&serde_json::json!({
            "kind": "face-detection", "input_size": [16, 16], "strides": [8], "anchors": 1,
            "fit": "letterbox"
        }))
        .unwrap();
        // 2x2 grid; cell 3 sits at (8, 8) and reaches 8 px in every direction.
        let mut scores = [0.0f32; 4];
        scores[3] = 0.9;
        scores[2] = 0.8; // overlaps the first box heavily at the same extent
        let mut boxes = [0.0f32; 16];
        boxes[12..16].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        boxes[8..12].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let mut marks = [0.0f32; 40];
        marks[30..40].copy_from_slice(&[0.0, 0.0, 0.5, 0.5, 0.0, 0.5, -0.5, 0.0, 0.0, -0.5]);
        let tensors = parse_tensors(&framed(&[
            (&[4, 1], &scores),
            (&[4, 4], &boxes),
            (&[4, 10], &marks),
        ]))
        .unwrap();
        // The neighbouring cell overlaps by a third, so a 0.3 limit merges them.
        let faces = decode_faces(&tensors, &recipe, (16, 16, 1.0), 0.5, 0.3, true).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].label, "face");
        assert_eq!(faces[0].bbox_pixels.x_max, 16);
        let pose = faces[0].pose.as_ref().unwrap();
        assert_eq!((pose.len(), pose[0].name.as_str()), (5, "left_eye"));
        assert!((pose[0].x - 0.5).abs() < 1e-6);
        assert!(decode_faces(&tensors[..2], &recipe, (16, 16, 1.0), 0.5, 0.4, true).is_ok());
        assert!(decode_faces(&tensors[..1], &recipe, (16, 16, 1.0), 0.5, 0.4, true).is_err());
        assert!(
            decode_faces(&tensors, &recipe, (16, 16, 1.0), 0.95, 0.4, false)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn letterbox_keeps_the_aspect_ratio_and_reports_the_scale() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            32,
            16,
            image::Rgb([255, 255, 255]),
        ))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
        let recipe = Recipe::parse(&serde_json::json!({
            "kind": "face-detection", "input_size": [16, 16], "strides": [8], "fit": "letterbox"
        }))
        .unwrap();
        let prepared = preprocess(png.get_ref(), &recipe).unwrap();
        assert_eq!(
            (prepared.width, prepared.height, prepared.scale),
            (32, 16, 0.5)
        );
        // The padded bottom half is black, the picture sits at the top.
        let first = f32::from_le_bytes(prepared.bytes[..4].try_into().unwrap());
        let last_row = f32::from_le_bytes(
            prepared.bytes[(15 * 16) * 4..(15 * 16) * 4 + 4]
                .try_into()
                .unwrap(),
        );
        assert!(first > 0.99 && last_row.abs() < 1e-6);
    }
}
