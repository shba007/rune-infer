//! Work out what a GGUF file actually is.
//!
//! A `.gguf` can hold a chat model, an image diffusion model, a speech model,
//! a vision projector, an MTP draft head, a LoRA adapter and more. Guessing
//! from the file name gets this wrong, so the setup wizard reads the header:
//! the architecture, a few distinctive keys and the tensor names.
//!
//! [`discover`] scans a file or folder and returns the models it can set up
//! (with modality, engine and engine-specific `extra` filled in) plus the
//! files it deliberately skipped and why.

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::config::{EngineKind, Modality};

// ---------------------------------------------------------------------------
// GGUF header
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct GgufInfo {
    pub architecture: String,
    /// `general.name`
    pub name: Option<String>,
    /// `general.finetune`
    pub finetune: Option<String>,
    /// `general.type` (`model`, `mmproj`, `adapter`, ...)
    pub general_type: Option<String>,
    pub context_length: Option<u32>,
    /// `<arch>.embedding_length`: a vision projector must output exactly this size.
    pub embedding_length: Option<u32>,
    /// `clip.vision.projection_dim` (projectors only).
    pub projection_dim: Option<u32>,
    pub has_tokenizer: bool,
    /// Any `prism.*` key (Prism ternary quantisation needs the Prism llama.cpp build).
    pub has_prism_keys: bool,
    /// Any `*.pooling_type` key (embedding models).
    pub has_pooling: bool,
    /// `audiocpp.model_spec.family`
    pub audiocpp_family: Option<String>,
    pub tensor_count: u64,
    pub has_token_embd: bool,
    pub has_first_block: bool,
    /// Tensor names typical of diffusion transformers / UNets (some GGUFs carry no metadata at all).
    pub diffusion_tensors: bool,
    /// `audio_tower.*` tensors: a speech-recognition audio encoder.
    pub audio_tower: bool,
}

fn rd<const N: usize>(r: &mut impl Read) -> Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn rd_u32(r: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(rd(r)?))
}
fn rd_u64(r: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(rd(r)?))
}
fn rd_string(r: &mut impl Read) -> Result<String> {
    let len = rd_u64(r)? as usize;
    anyhow::ensure!(len <= 1 << 20, "implausible string length in GGUF header");
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_bytes(r: &mut impl Read, n: u64) -> Result<()> {
    let copied = std::io::copy(&mut r.take(n), &mut std::io::sink())?;
    anyhow::ensure!(copied == n, "GGUF header is truncated");
    Ok(())
}

fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn skip_value(r: &mut impl Read, ty: u32) -> Result<()> {
    if let Some(n) = fixed_size(ty) {
        return skip_bytes(r, n);
    }
    match ty {
        8 => {
            let len = rd_u64(r)?;
            skip_bytes(r, len)
        }
        9 => {
            let elem = rd_u32(r)?;
            let n = rd_u64(r)?;
            match fixed_size(elem) {
                Some(sz) => skip_bytes(r, n.saturating_mul(sz)),
                None => {
                    for _ in 0..n {
                        skip_value(r, elem)?;
                    }
                    Ok(())
                }
            }
        }
        _ => anyhow::bail!("unknown GGUF value type {ty}"),
    }
}

/// Read the header (key/values and tensor names) from `r`.
pub fn parse_gguf(r: &mut impl Read) -> Result<GgufInfo> {
    anyhow::ensure!(&rd::<4>(r)? == b"GGUF", "not a GGUF file");
    let version = rd_u32(r)?;
    anyhow::ensure!(
        (2..=3).contains(&version),
        "unsupported GGUF version {version}"
    );
    let tensor_count = rd_u64(r)?;
    let kv_count = rd_u64(r)?;
    anyhow::ensure!(kv_count <= 65_536, "implausible GGUF key count");

    let mut info = GgufInfo {
        tensor_count,
        ..Default::default()
    };
    for _ in 0..kv_count {
        let key = rd_string(r)?;
        let ty = rd_u32(r)?;
        match (key.as_str(), ty) {
            ("general.architecture", 8) => info.architecture = rd_string(r)?,
            ("general.name", 8) => info.name = Some(rd_string(r)?),
            ("general.type", 8) => info.general_type = Some(rd_string(r)?),
            ("general.finetune", 8) => info.finetune = Some(rd_string(r)?),
            ("audiocpp.model_spec.family", 8) => info.audiocpp_family = Some(rd_string(r)?),
            (k, 4) if k.ends_with(".context_length") && info.context_length.is_none() => {
                info.context_length = Some(rd_u32(r)?)
            }
            (k, 4)
                if k.ends_with(".embedding_length")
                    && !k.starts_with("clip.")
                    && info.embedding_length.is_none() =>
            {
                info.embedding_length = Some(rd_u32(r)?)
            }
            ("clip.vision.projection_dim" | "clip.projection_dim", 4) => {
                info.projection_dim = Some(rd_u32(r)?)
            }
            (k, _) => {
                if k.starts_with("tokenizer.") {
                    info.has_tokenizer = true;
                } else if k.starts_with("prism.") {
                    info.has_prism_keys = true;
                } else if k.ends_with(".pooling_type") {
                    info.has_pooling = true;
                }
                skip_value(r, ty)?;
            }
        }
    }
    for _ in 0..tensor_count.min(500_000) {
        let name = rd_string(r)?;
        let dims = rd_u32(r)?;
        anyhow::ensure!(dims <= 8, "implausible tensor rank");
        skip_bytes(r, u64::from(dims) * 8 + 4 + 8)?;
        if name == "token_embd.weight" {
            info.has_token_embd = true;
        } else if name.starts_with("blk.0.") {
            info.has_first_block = true;
        } else if name.starts_with("audio_tower.") {
            info.audio_tower = true;
        } else if DIFFUSION_TENSOR_MARKERS.iter().any(|m| name.contains(m)) {
            info.diffusion_tensors = true;
        }
    }
    Ok(info)
}

pub fn read_gguf(path: &Path) -> Result<GgufInfo> {
    parse_gguf(&mut BufReader::new(std::fs::File::open(path)?))
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Tensor-name fragments of diffusion models (used when the file has no architecture metadata).
const DIFFUSION_TENSOR_MARKERS: &[&str] = &[
    "model.diffusion_model.",
    "transformer_blocks.",
    "time_text_embed",
    "double_blocks.",
    "joint_blocks.",
    "input_blocks.",
    "down_blocks.",
];

/// Diffusion architectures served by stable-diffusion.cpp's `sd-server`.
const IMAGE_ARCHS: &[&str] = &[
    "qwen_image",
    "lumina2",
    "flux",
    "chroma",
    "sd1",
    "sd2",
    "sdxl",
    "sd3",
    "hidream",
];
/// Language-model architectures that are vision-language models (they need an `mmproj` file).
const VISION_LM_ARCHS: &[&str] = &[
    "qwen2vl",
    "qwen25vl",
    "qwen3vl",
    "qwen3vlmoe",
    "llava",
    "mllama",
    "gemma3",
    "internvl",
    "minicpmv",
    "glm4v",
    "pixtral",
    "smolvlm",
    "idefics3",
];
const EMBEDDING_ARCHS: &[&str] = &[
    "bert",
    "nomic-bert",
    "modern-bert",
    "jina-bert-v2",
    "jina-bert-v3",
    "neo-bert",
];
const PRISM_ARCHS: &[&str] = &["ternary-bonsai", "bonsai", "bonsai-vl", "ternary-bonsai-vl"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Llm,
    Other,
}

/// What the wizard learned about one model file.
#[derive(Debug, Clone)]
pub struct Detected {
    pub modality: Modality,
    /// Engines that can serve it, preferred first.
    pub engines: Vec<EngineKind>,
    /// Human label: "chat model", "image generation", "text-to-speech", ...
    pub kind_label: &'static str,
    pub capabilities: Vec<&'static str>,
    /// `extra` keys the user still has to fill in for the model to run.
    pub needs: Vec<&'static str>,
    pub notes: Vec<String>,
    class: Class,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    Model,
    /// Vision projector (`mmproj`): attached to a vision model, never listed alone.
    Projector,
    /// MTP draft head: attached to its base model as `extra.mtp_path`.
    Draft,
}

#[derive(Debug, Clone)]
pub enum Classified {
    Model(Detected),
    Projector,
    Draft,
    Skip(String),
}

fn audio_modality(family: &str) -> (Modality, Option<String>) {
    let f = family.to_lowercase();
    let known_tts = ["fish_audio", "voxcpm", "vevo", "tts", "speech", "voice"];
    if ["asr", "whisper", "stt", "parakeet", "conformer", "transcri"]
        .iter()
        .any(|w| f.contains(w))
    {
        return (Modality::SpeechToText, None);
    }
    if known_tts.iter().any(|w| f.contains(w)) {
        let note = f.contains("vevo").then(|| {
            "Vevo2 also does voice conversion; it is set up as text-to-speech".to_string()
        });
        return (Modality::TextToSpeech, note);
    }
    (
        Modality::TextToSpeech,
        Some(format!(
            "audio family '{family}' is not recognised; assumed text-to-speech — change `modality` if that is wrong"
        )),
    )
}

/// Lower-case words from the model's name, fine-tune label and file name.
fn name_tokens(info: &GgufInfo, file_name: &str) -> Vec<String> {
    [
        info.name.as_deref(),
        info.finetune.as_deref(),
        Some(file_name),
    ]
    .into_iter()
    .flatten()
    .flat_map(|s| {
        s.to_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .map(String::from)
            .collect::<Vec<_>>()
    })
    .filter(|w| !w.is_empty())
    .collect()
}

pub fn classify(info: &GgufInfo, file_name: &str) -> Classified {
    let arch = info.architecture.to_lowercase();
    let lower_name = file_name.to_lowercase();
    let det = |modality, engines: Vec<EngineKind>, kind_label, capabilities: Vec<&'static str>| {
        Detected {
            modality,
            engines,
            kind_label,
            capabilities,
            needs: Vec::new(),
            notes: Vec::new(),
            class: Class::Other,
        }
    };

    if arch == "clip" || info.general_type.as_deref() == Some("mmproj") {
        return Classified::Projector;
    }
    if info.general_type.as_deref() == Some("adapter") || arch.ends_with("adapter") {
        return Classified::Skip(
            "LoRA adapter — list it under extra.lora of the model it belongs to".into(),
        );
    }

    if arch == "audiocpp" {
        let family = info.audiocpp_family.clone().unwrap_or_default();
        let (modality, note) = audio_modality(&family);
        let (label, caps) = if modality == Modality::SpeechToText {
            ("speech-to-text", vec!["Speech-to-Text"])
        } else {
            ("text-to-speech", vec!["Text-to-Speech"])
        };
        let mut d = det(modality, vec![EngineKind::AudioCpp], label, caps);
        d.notes.extend(note);
        return Classified::Model(d);
    }
    if arch == "audio8" {
        // Audio8 models: an `audio_tower` encoder means speech recognition.
        let stt = info.audio_tower;
        let mut d = det(
            if stt {
                Modality::SpeechToText
            } else {
                Modality::TextToSpeech
            },
            vec![EngineKind::AudioCpp],
            if stt {
                "speech-to-text"
            } else {
                "text-to-speech"
            },
            vec![if stt {
                "Speech-to-Text"
            } else {
                "Text-to-Speech"
            }],
        );
        if !stt {
            d.notes.push("Audio8 model without a speech-recognition encoder; assumed text-to-speech — change `modality` if that is wrong".into());
        }
        return Classified::Model(d);
    }
    if arch == "asr" || arch == "parakeet" || arch.contains("fastconformer") {
        return Classified::Model(det(
            Modality::SpeechToText,
            vec![EngineKind::NemoSpeech],
            "speech-to-text",
            vec!["Speech-to-Text"],
        ));
    }
    if arch == "whisper" || arch.starts_with("voxtral") {
        return Classified::Model(det(
            Modality::SpeechToText,
            vec![EngineKind::Crispasr],
            "speech-to-text",
            vec!["Speech-to-Text"],
        ));
    }
    let metadata_less_diffusion = arch.is_empty() && info.diffusion_tensors;
    if IMAGE_ARCHS.contains(&arch.as_str()) || metadata_less_diffusion {
        let editing = lower_name.contains("edit");
        let mut d = det(
            Modality::ImageGeneration,
            vec![EngineKind::SdServer],
            if editing {
                "image editing"
            } else {
                "image generation"
            },
            if editing {
                vec!["Text-to-Image", "Image Editing"]
            } else {
                vec!["Text-to-Image"]
            },
        );
        d.needs = vec!["vae_path", "text_encoder_path"];
        if editing {
            d.needs.push("llm_vision_path");
        }
        d.notes.push(
            "diffusion models need companion files (VAE, text encoder) that are separate downloads"
                .into(),
        );
        if metadata_less_diffusion {
            d.notes.push(
                "this file has no metadata; recognised as a diffusion model from its tensor names"
                    .into(),
            );
        }
        return Classified::Model(d);
    }
    if EMBEDDING_ARCHS.contains(&arch.as_str()) || info.has_pooling {
        return Classified::Skip(
            "embedding model in GGUF form — use a safetensors embedding model with the encoder engine"
                .into(),
        );
    }

    if info.has_tokenizer && info.has_token_embd {
        if !info.has_first_block {
            return Classified::Draft;
        }
        // A language-model body fine-tuned for speech is a plain LM structurally; only
        // its name gives it away (e.g. "Qwen3 1.7B Multilingual TTS").
        let words = name_tokens(info, file_name);
        let speech = [
            (
                "tts",
                Modality::TextToSpeech,
                "text-to-speech",
                "Text-to-Speech",
            ),
            (
                "asr",
                Modality::SpeechToText,
                "speech-to-text",
                "Speech-to-Text",
            ),
        ]
        .into_iter()
        .find(|(w, ..)| words.iter().any(|x| x == w));
        if let Some((word, modality, label, cap)) = speech {
            let mut d = det(modality, vec![EngineKind::Crispasr], label, vec![cap]);
            d.notes.push(format!(
                "a language-model fine-tune recognised as '{word}' from its name; assumed to run on crispasr — verify"
            ));
            return Classified::Model(d);
        }
        let prism = info.has_prism_keys || PRISM_ARCHS.contains(&arch.as_str());
        let mut d = det(
            Modality::Text,
            vec![if prism {
                EngineKind::LlamaServerPrism
            } else {
                EngineKind::LlamaServer
            }],
            "chat model",
            vec!["Chat", "Tool Call", "Structured Output"],
        );
        d.class = Class::Llm;
        if prism {
            d.notes
                .push("ternary Prism quantisation: needs the Prism llama.cpp build".into());
        }
        return Classified::Model(d);
    }

    Classified::Skip(if arch.is_empty() {
        "GGUF has no architecture metadata and no recognisable tensor layout".to_string()
    } else {
        format!(
            "unrecognised GGUF architecture '{}' (not a language, image or speech model rune-infer knows)",
            info.architecture
        )
    })
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LocalCandidate {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub architecture: String,
    pub display_name: String,
    pub context_length: Option<u32>,
    pub modality: Modality,
    pub engines: Vec<EngineKind>,
    pub kind_label: &'static str,
    pub capabilities: Vec<String>,
    /// Engine-specific settings found next to the model (`mmproj_path`, `mtp_path`).
    pub extra: Map<String, Value>,
    /// `extra` keys the user still has to provide before it will run.
    pub needs: Vec<&'static str>,
    pub notes: Vec<String>,
    class: Class,
    embedding_length: Option<u32>,
}

impl LocalCandidate {
    pub fn engine(&self) -> EngineKind {
        self.engines[0]
    }
    pub fn is_vision(&self) -> bool {
        self.modality == Modality::VisionText
    }
    /// Ready to run without any manual follow-up.
    pub fn is_ready(&self) -> bool {
        self.needs.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Default)]
pub struct Discovery {
    pub models: Vec<LocalCandidate>,
    pub skipped: Vec<Skipped>,
}

/// Hugging Face caches nest models `hub/models--org--name/snapshots/<hash>/<subfolder>/file.gguf`,
/// so a models root such as `D:/Models` is 6-7 levels above the files.
const MAX_DEPTH: usize = 9;
const MAX_FILES: usize = 2000;

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH || out.len() >= MAX_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, depth + 1, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

fn file_stem(p: &Path) -> String {
    p.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// `general.name` is sometimes junk ("Hf") or a whole sentence; fall back to the file name then.
fn sensible_name(name: &Option<String>) -> Option<String> {
    name.as_ref()
        .map(|n| n.trim().to_string())
        .filter(|n| (4..=48).contains(&n.chars().count()))
}

/// Prefer the f16 projector over bf16/quantised ones.
fn projector_rank(p: &Path) -> (bool, String) {
    let n = p
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    (!n.contains("f16") || n.contains("bf16"), n)
}

fn normalized_stem(p: &Path) -> String {
    file_stem(p)
        .to_lowercase()
        .replace("mmproj", "")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Give each vision model the projector that belongs to it.
///
/// A projector's output size (`clip.vision.projection_dim`) must equal its
/// model's `embedding_length`, so that is the primary match. Several models of
/// the same size in one folder are told apart by file-name similarity; if that
/// still cannot decide, nothing is attached and the model gets a note.
fn pair_projectors(
    candidates: &mut [LocalCandidate],
    projectors: Vec<(PathBuf, GgufInfo)>,
    skipped: &mut Vec<Skipped>,
) {
    let mut chosen: Vec<Vec<PathBuf>> = vec![Vec::new(); candidates.len()];
    let mut ambiguous: Vec<Vec<usize>> = vec![Vec::new(); candidates.len()];

    for (pi, (path, info)) in projectors.iter().enumerate() {
        let dir = path.parent();
        let in_dir: Vec<usize> = (0..candidates.len())
            .filter(|&i| candidates[i].class == Class::Llm && candidates[i].path.parent() == dir)
            .collect();
        let mut matches: Vec<usize> = in_dir
            .iter()
            .copied()
            .filter(|&i| {
                matches!((info.projection_dim, candidates[i].embedding_length), (Some(p), Some(e)) if p == e)
            })
            .collect();
        let dims_known =
            |i: usize| info.projection_dim.is_some() && candidates[i].embedding_length.is_some();
        if matches.is_empty() && in_dir.len() == 1 && !dims_known(in_dir[0]) {
            matches = in_dir.clone();
        }
        if matches.len() > 1 {
            let stem = normalized_stem(path);
            let close: Vec<usize> = matches
                .iter()
                .copied()
                .filter(|&i| common_prefix_len(&stem, &normalized_stem(&candidates[i].path)) >= 6)
                .collect();
            if close.len() == 1 {
                matches = close;
            } else {
                for &i in &matches {
                    ambiguous[i].push(pi);
                }
                matches.clear();
            }
        }
        match matches.first() {
            Some(&i) => chosen[i].push(path.clone()),
            None if ambiguous.iter().any(|a| a.contains(&pi)) => {}
            None => skipped.push(Skipped {
                path: path.clone(),
                reason: match info.projection_dim {
                    Some(d) => {
                        format!("vision projector (output size {d}) matches no model in its folder")
                    }
                    None => "vision projector with no matching model in its folder".to_string(),
                },
            }),
        }
    }

    for (i, c) in candidates.iter_mut().enumerate() {
        let mut options = std::mem::take(&mut chosen[i]);
        options.sort_by_key(|p| projector_rank(p));
        if let Some(best) = options.first() {
            c.extra
                .insert("mmproj_path".into(), json!(best.to_string_lossy()));
            c.modality = Modality::VisionText;
            c.capabilities.push("Vision".into());
            c.kind_label = "vision chat model";
            c.notes
                .retain(|n| !n.starts_with("vision-language model, but no matching"));
        } else if !ambiguous[i].is_empty() {
            c.notes.push(format!(
                "{} vision projector(s) in this folder could belong to it; set extra.mmproj_path",
                ambiguous[i].len()
            ));
        }
    }
}

/// Scan `input` (a file, or a folder searched up to 9 levels) and classify
/// every GGUF found.
pub fn discover(input: &Path) -> Result<Discovery> {
    if !input.exists() {
        anyhow::bail!("'{}' does not exist", input.display());
    }
    let mut files = Vec::new();
    if input.is_file() {
        files.push(input.to_path_buf());
    } else {
        walk(input, 0, &mut files);
    }
    files.sort();

    let mut out = Discovery::default();
    let mut projectors: Vec<(PathBuf, GgufInfo)> = Vec::new();
    let mut drafts: Vec<(PathBuf, GgufInfo)> = Vec::new();
    let mut found: Vec<(PathBuf, GgufInfo, Detected)> = Vec::new();

    for path in files {
        let file_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let info = match read_gguf(&path) {
            Ok(i) => i,
            Err(e) => {
                out.skipped.push(Skipped {
                    path,
                    reason: format!("not a readable GGUF ({e})"),
                });
                continue;
            }
        };
        match classify(&info, &file_name) {
            Classified::Model(d) => found.push((path, info, d)),
            Classified::Projector => projectors.push((path, info)),
            Classified::Draft => drafts.push((path, info)),
            Classified::Skip(reason) => out.skipped.push(Skipped { path, reason }),
        }
    }

    let mut candidates: Vec<LocalCandidate> = found
        .into_iter()
        .map(|(path, info, d)| {
            let extra = Map::new();
            let modality = d.modality.clone();
            let caps: Vec<String> = d.capabilities.iter().map(|c| c.to_string()).collect();
            let label = d.kind_label;
            let mut notes = d.notes.clone();
            if d.class == Class::Llm
                && VISION_LM_ARCHS.contains(&info.architecture.to_lowercase().as_str())
            {
                notes.push(
                    "vision-language model, but no matching mmproj file was found — \
                     add extra.mmproj_path to enable image input (it may also serve as a text encoder for image models)"
                        .into(),
                );
            }
            LocalCandidate {
                size_bytes: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                architecture: if info.architecture.is_empty() {
                    "(no metadata)".to_string()
                } else {
                    info.architecture.clone()
                },
                display_name: sensible_name(&info.name).unwrap_or_else(|| file_stem(&path)),
                context_length: info.context_length,
                modality,
                engines: d.engines,
                kind_label: label,
                capabilities: caps,
                extra,
                needs: d.needs,
                notes,
                class: d.class,
                embedding_length: info.embedding_length,
                path,
            }
        })
        .collect();

    pair_projectors(&mut candidates, projectors, &mut out.skipped);

    // Attach MTP draft heads to their base model (same folder, same name or architecture).
    for (path, info) in drafts {
        let dir = path.parent();
        let in_dir = |c: &LocalCandidate| c.class == Class::Llm && c.path.parent() == dir;
        let by_name = candidates.iter().position(|c| {
            in_dir(c) && sensible_name(&info.name).is_some_and(|n| c.display_name == n)
        });
        let base = by_name.or_else(|| {
            candidates
                .iter()
                .position(|c| in_dir(c) && c.architecture == info.architecture)
        });
        match base {
            Some(i) => {
                let base = &mut candidates[i];
                base.extra
                    .insert("mtp_path".into(), json!(path.to_string_lossy()));
                base.capabilities.push("MTP".into());
                base.notes.push(format!(
                    "MTP draft head attached ({})",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
            None => out.skipped.push(Skipped {
                path,
                reason: "MTP draft head with no matching base model in the same folder".into(),
            }),
        }
    }
    // Two files sharing one `general.name` (different quantisations): tell them apart by file name.
    let mut name_count: HashMap<String, usize> = HashMap::new();
    for c in &candidates {
        *name_count.entry(c.display_name.clone()).or_default() += 1;
    }
    for c in &mut candidates {
        if name_count[&c.display_name] > 1 {
            c.display_name = file_stem(&c.path);
        }
    }

    candidates.sort_by(|a, b| a.path.cmp(&b.path));
    out.models = candidates;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Test helpers (also used by the wizard tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod testing {
    use std::path::Path;

    /// Build a minimal GGUF byte stream with string keys and named tensors.
    #[derive(Default)]
    pub struct Gguf {
        kvs: Vec<(String, u32, Vec<u8>)>,
        tensors: Vec<String>,
    }

    fn s(b: &mut Vec<u8>, v: &str) {
        b.extend((v.len() as u64).to_le_bytes());
        b.extend(v.as_bytes());
    }

    impl Gguf {
        pub fn new(arch: &str) -> Self {
            Gguf::default().string("general.architecture", arch)
        }
        pub fn string(mut self, k: &str, v: &str) -> Self {
            let mut b = Vec::new();
            s(&mut b, v);
            self.kvs.push((k.into(), 8, b));
            self
        }
        pub fn u32(mut self, k: &str, v: u32) -> Self {
            self.kvs.push((k.into(), 4, v.to_le_bytes().to_vec()));
            self
        }
        /// A string-array key such as `tokenizer.ggml.tokens`.
        pub fn string_array(mut self, k: &str, items: &[&str]) -> Self {
            let mut b = Vec::new();
            b.extend(8u32.to_le_bytes());
            b.extend((items.len() as u64).to_le_bytes());
            for i in items {
                s(&mut b, i);
            }
            self.kvs.push((k.into(), 9, b));
            self
        }
        pub fn tensor(mut self, name: &str) -> Self {
            self.tensors.push(name.into());
            self
        }
        /// A language model: tokenizer plus embedding and first-block tensors.
        pub fn llm(arch: &str, name: &str) -> Self {
            Self::llm_with_dim(arch, name, 4096)
        }
        /// Like [`Gguf::llm`] with a chosen embedding size (projectors must match it).
        pub fn llm_with_dim(arch: &str, name: &str, dim: u32) -> Self {
            Gguf::new(arch)
                .string("general.name", name)
                .u32(&format!("{arch}.context_length"), 32768)
                .u32(&format!("{arch}.embedding_length"), dim)
                .string_array("tokenizer.ggml.tokens", &["a", "b"])
                .tensor("token_embd.weight")
                .tensor("blk.0.attn_q.weight")
        }
        pub fn bytes(&self) -> Vec<u8> {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend((self.tensors.len() as u64).to_le_bytes());
            b.extend((self.kvs.len() as u64).to_le_bytes());
            for (k, ty, v) in &self.kvs {
                s(&mut b, k);
                b.extend(ty.to_le_bytes());
                b.extend(v);
            }
            for t in &self.tensors {
                s(&mut b, t);
                b.extend(1u32.to_le_bytes()); // 1 dim
                b.extend(8u64.to_le_bytes());
                b.extend(0u32.to_le_bytes()); // type
                b.extend(0u64.to_le_bytes()); // offset
            }
            b
        }
        pub fn write(&self, path: &Path) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, self.bytes()).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Gguf;
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rune-det-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cls(g: &Gguf, name: &str) -> Classified {
        classify(&parse_gguf(&mut g.bytes().as_slice()).unwrap(), name)
    }

    fn model(c: Classified) -> Detected {
        match c {
            Classified::Model(d) => d,
            other => panic!("expected a model, got {other:?}"),
        }
    }

    #[test]
    fn parses_keys_and_tensors_and_skips_big_arrays() {
        let g = Gguf::llm("qwen35", "My Model").string("prism.hadamard.axis", "x");
        let i = parse_gguf(&mut g.bytes().as_slice()).unwrap();
        assert_eq!(i.architecture, "qwen35");
        assert_eq!(i.name.as_deref(), Some("My Model"));
        assert_eq!(i.context_length, Some(32768));
        assert!(i.has_tokenizer && i.has_prism_keys && i.has_token_embd && i.has_first_block);
        assert_eq!(i.tensor_count, 2);
        assert!(parse_gguf(&mut b"NOPE....".as_slice()).is_err());
    }

    #[test]
    fn language_models_get_llama_server_or_prism() {
        let d = model(cls(&Gguf::llm("qwen35", "Q"), "q.gguf"));
        assert_eq!(
            (d.modality, d.engines.clone(), d.kind_label),
            (Modality::Text, vec![EngineKind::LlamaServer], "chat model")
        );
        let prism = model(cls(
            &Gguf::llm("qwen35", "B").string("prism.hadamard.axis", "x"),
            "b.gguf",
        ));
        assert_eq!(prism.engines, vec![EngineKind::LlamaServerPrism]);
        let by_arch = model(cls(&Gguf::llm("ternary-bonsai-vl", "B"), "b.gguf"));
        assert_eq!(by_arch.engines, vec![EngineKind::LlamaServerPrism]);
    }

    #[test]
    fn voice_models_are_text_to_speech_on_audio_cpp() {
        for family in ["fish_audio", "voxcpm2", "vevo2"] {
            let g = Gguf::new("audiocpp")
                .string("general.name", "x")
                .string("audiocpp.model_spec.family", family);
            let d = model(cls(&g, "v.gguf"));
            assert_eq!(
                (d.modality, d.engines, d.kind_label),
                (
                    Modality::TextToSpeech,
                    vec![EngineKind::AudioCpp],
                    "text-to-speech"
                ),
                "{family}"
            );
        }
        let asr = Gguf::new("audiocpp").string("audiocpp.model_spec.family", "whisper_large");
        assert_eq!(model(cls(&asr, "w.gguf")).modality, Modality::SpeechToText);
        let unknown = model(cls(
            &Gguf::new("audiocpp").string("audiocpp.model_spec.family", "zzz"),
            "z.gguf",
        ));
        assert!(unknown.notes[0].contains("not recognised"));
    }

    #[test]
    fn metadata_less_diffusion_files_are_recognised_by_tensor_names() {
        // e.g. qwen-image-2.1-Q8_0.gguf: zero key/values, only tensors.
        for tensor in [
            "transformer_blocks.0.attn.norm_k.weight",
            "model.diffusion_model.img_in.weight",
        ] {
            let g = Gguf::default().tensor("img_in.weight").tensor(tensor);
            let d = model(cls(&g, "qwen-image-2.1-Q8_0.gguf"));
            assert_eq!(
                (d.modality, d.engines),
                (Modality::ImageGeneration, vec![EngineKind::SdServer]),
                "{tensor}"
            );
            assert!(d.notes.iter().any(|n| n.contains("no metadata")));
        }
        let junk = Gguf::default().tensor("some.random.weight");
        assert!(
            matches!(cls(&junk, "x.gguf"), Classified::Skip(r) if r.contains("no architecture metadata"))
        );
    }

    #[test]
    fn language_model_fine_tunes_for_speech_are_found_by_name() {
        // Real example: arch qwen3, general.finetune "Multilingual-TTS".
        let tts = Gguf::llm("qwen3", "Qwen3 1.7B Multilingual TTS")
            .string("general.finetune", "Multilingual-TTS");
        let d = model(cls(&tts, "qwen3-1.7B-Multilingual-TTS-f16.gguf"));
        assert_eq!(
            (d.modality, d.engines, d.kind_label),
            (
                Modality::TextToSpeech,
                vec![EngineKind::Crispasr],
                "text-to-speech"
            )
        );
        assert!(d.notes[0].contains("verify"));
        // Whole words only: ordinary models with look-alike substrings stay chat models.
        for (name, file) in [
            ("Settings Helper", "settings-8b.gguf"),
            ("OxCoder 9B", "oxcoder.gguf"),
            ("Fasr Model", "fasrtest.gguf"),
        ] {
            let d = model(cls(&Gguf::llm("qwen35", name), file));
            assert_eq!(d.modality, Modality::Text, "{name}");
        }
    }

    #[test]
    fn audio8_with_an_audio_tower_is_speech_recognition() {
        let asr = model(cls(
            &Gguf::new("audio8").tensor("audio_tower.embedder.conv1.weight"),
            "a.gguf",
        ));
        assert_eq!(
            (asr.modality, asr.engines),
            (Modality::SpeechToText, vec![EngineKind::AudioCpp])
        );
        let other = model(cls(&Gguf::new("audio8").tensor("decoder.weight"), "a.gguf"));
        assert_eq!(other.modality, Modality::TextToSpeech);
        assert!(other.notes[0].contains("assumed text-to-speech"));
    }

    #[test]
    fn speech_recognition_models() {
        let d = model(cls(
            &Gguf::new("asr").tensor("encoder.pos_enc.pe"),
            "p.gguf",
        ));
        assert_eq!(
            (d.modality, d.engines),
            (Modality::SpeechToText, vec![EngineKind::NemoSpeech])
        );
        let w = model(cls(&Gguf::new("whisper"), "w.gguf"));
        assert_eq!(w.engines, vec![EngineKind::Crispasr]);
    }

    #[test]
    fn diffusion_models_are_image_generation_and_flag_missing_companions() {
        for arch in ["qwen_image", "lumina2", "flux"] {
            let d = model(cls(&Gguf::new(arch).tensor("img_in.weight"), "m.gguf"));
            assert_eq!(
                (d.modality, d.engines.clone()),
                (Modality::ImageGeneration, vec![EngineKind::SdServer]),
                "{arch}"
            );
            assert_eq!(d.needs, vec!["vae_path", "text_encoder_path"]);
        }
        let edit = model(cls(
            &Gguf::new("qwen_image"),
            "qwen-image-edit-2511-Q3.gguf",
        ));
        assert_eq!(edit.kind_label, "image editing");
        assert!(edit.needs.contains(&"llm_vision_path"));
    }

    #[test]
    fn projectors_drafts_adapters_embeddings_and_unknowns_are_not_models() {
        assert!(matches!(
            cls(&Gguf::new("clip"), "mm.gguf"),
            Classified::Projector
        ));
        assert!(matches!(
            cls(&Gguf::new("x").string("general.type", "mmproj"), "mm.gguf"),
            Classified::Projector
        ));
        let draft = Gguf::new("qwen35")
            .string_array("tokenizer.ggml.tokens", &["a"])
            .tensor("token_embd.weight")
            .tensor("blk.64.attn_k.weight");
        assert!(matches!(cls(&draft, "mtp-x.gguf"), Classified::Draft));
        assert!(
            matches!(cls(&Gguf::new("llama").string("general.type", "adapter"), "l.gguf"), Classified::Skip(r) if r.contains("LoRA"))
        );
        assert!(
            matches!(cls(&Gguf::new("nomic-bert"), "e.gguf"), Classified::Skip(r) if r.contains("embedding"))
        );
        assert!(
            matches!(cls(&Gguf::new("mystery"), "m.gguf"), Classified::Skip(r) if r.contains("unrecognised"))
        );
    }

    fn clip(dim: u32) -> Gguf {
        Gguf::new("clip").u32("clip.vision.projection_dim", dim)
    }

    #[test]
    fn folder_scan_pairs_projectors_and_drafts_and_names_models_sensibly() {
        let d = tmp("pair");
        Gguf::llm_with_dim("qwen35", "Qwen3.8-27B", 5120)
            .write(&d.join("Qwen3.8-27B-UD-IQ3_S.gguf"));
        clip(5120).write(&d.join("mmproj-BF16.gguf"));
        clip(5120).write(&d.join("mmproj-F16.gguf"));
        Gguf::new("qwen35")
            .string("general.name", "Qwen3.8-27B")
            .string_array("tokenizer.ggml.tokens", &["a"])
            .tensor("token_embd.weight")
            .tensor("blk.64.attn_k.weight")
            .write(&d.join("mtp-Qwen3.8-27B-Q4_0.gguf"));
        Gguf::llm("qwen35", "Hf")
            .string("prism.x", "1")
            .write(&d.join("sub/Ternary-Bonsai-PQ2_0.gguf"));
        Gguf::new("audiocpp")
            .string("audiocpp.model_spec.family", "voxcpm2")
            .string("general.name", "VoxCPM2")
            .write(&d.join("voice/voxcpm2-q8.gguf"));
        Gguf::new("mystery").write(&d.join("odd.gguf"));
        std::fs::write(d.join("broken.gguf"), b"not gguf").unwrap();

        let found = discover(&d).unwrap();
        let by = |n: &str| {
            found
                .models
                .iter()
                .find(|m| m.path.ends_with(n))
                .unwrap_or_else(|| {
                    panic!(
                        "{n}: {:?}",
                        found.models.iter().map(|m| &m.path).collect::<Vec<_>>()
                    )
                })
        };

        let q = by("Qwen3.8-27B-UD-IQ3_S.gguf");
        assert_eq!(q.modality, Modality::VisionText);
        assert!(
            q.extra["mmproj_path"]
                .as_str()
                .unwrap()
                .ends_with("mmproj-F16.gguf"),
            "prefers f16"
        );
        assert!(
            q.extra["mtp_path"]
                .as_str()
                .unwrap()
                .ends_with("mtp-Qwen3.8-27B-Q4_0.gguf")
        );
        assert!(
            q.capabilities.contains(&"Vision".to_string())
                && q.capabilities.contains(&"MTP".to_string())
        );

        let b = by("Ternary-Bonsai-PQ2_0.gguf");
        assert_eq!(b.engine(), EngineKind::LlamaServerPrism);
        assert_eq!(
            b.display_name, "Ternary-Bonsai-PQ2_0",
            "junk general.name falls back to the file name"
        );
        assert_eq!(b.modality, Modality::Text, "its folder has no projector");

        assert_eq!(by("voxcpm2-q8.gguf").modality, Modality::TextToSpeech);
        assert_eq!(
            found.models.len(),
            3,
            "draft and projectors are not listed alone"
        );
        let reasons: Vec<&str> = found.skipped.iter().map(|s| s.reason.as_str()).collect();
        assert!(
            reasons.iter().any(|r| r.contains("unrecognised"))
                && reasons.iter().any(|r| r.contains("not a readable GGUF"))
        );
        assert_eq!(
            found.skipped.len(),
            2,
            "both projector variants were used: {:?}",
            found.skipped
        );
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn a_projector_only_attaches_to_the_model_whose_size_it_matches() {
        // A flat folder: three unrelated chat models and two projectors.
        let d = tmp("flat");
        Gguf::llm_with_dim("qwen35", "Big 27B", 5120).write(&d.join("big-27b.gguf"));
        Gguf::llm_with_dim("qwen35", "Mid 9B", 4096).write(&d.join("mid-9b.gguf"));
        Gguf::llm_with_dim("llama", "Small 3B", 2048).write(&d.join("small-3b.gguf"));
        clip(5120).write(&d.join("mmproj-big.gguf"));
        clip(4096).write(&d.join("mmproj-mid.gguf"));

        let f = discover(&d).unwrap();
        let get = |n: &str| f.models.iter().find(|m| m.path.ends_with(n)).unwrap();
        assert!(
            get("big-27b.gguf").extra["mmproj_path"]
                .as_str()
                .unwrap()
                .ends_with("mmproj-big.gguf")
        );
        assert!(
            get("mid-9b.gguf").extra["mmproj_path"]
                .as_str()
                .unwrap()
                .ends_with("mmproj-mid.gguf")
        );
        let small = get("small-3b.gguf");
        assert!(
            !small.extra.contains_key("mmproj_path") && small.modality == Modality::Text,
            "no projector of its size"
        );
        assert!(f.skipped.is_empty());
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn same_size_models_are_told_apart_by_name_and_ambiguity_is_reported() {
        let d = tmp("ambig");
        Gguf::llm("llama", "Alpha").write(&d.join("alphamodel-7b.gguf"));
        Gguf::llm("llama", "Zeta").write(&d.join("zetamodel-7b.gguf"));
        clip(4096).write(&d.join("mmproj-alphamodel-7b-F16.gguf"));
        let f = discover(&d).unwrap();
        let alpha = f.models.iter().find(|m| m.display_name == "Alpha").unwrap();
        let zeta = f.models.iter().find(|m| m.display_name == "Zeta").unwrap();
        assert!(
            alpha.is_vision() && !zeta.is_vision(),
            "file-name affinity decides"
        );

        // A projector whose name matches neither: genuinely ambiguous, so attach nothing.
        clip(4096).write(&d.join("mmproj-mystery.gguf"));
        std::fs::remove_file(d.join("mmproj-alphamodel-7b-F16.gguf")).unwrap();
        let f = discover(&d).unwrap();
        assert!(f.models.iter().all(|m| !m.is_vision()));
        assert!(
            f.models
                .iter()
                .all(|m| m.notes.iter().any(|n| n.contains("could belong to it"))),
            "{:?}",
            f.models.iter().map(|m| &m.notes).collect::<Vec<_>>()
        );
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn finds_models_deep_inside_a_huggingface_cache_from_the_models_root() {
        let d = tmp("deep");
        // root/huggingface/hub/models--org--name/snapshots/<hash>/Sub-GGUF/file.gguf  (depth 6)
        let deep = d.join("huggingface/hub/models--org--name/snapshots/abc123/Voice-GGUF/v.gguf");
        Gguf::new("audiocpp")
            .string("audiocpp.model_spec.family", "voxcpm2")
            .write(&deep);
        let f = discover(&d).unwrap();
        assert_eq!(f.models.len(), 1, "{:?}", f.skipped);
        assert_eq!(f.models[0].modality, Modality::TextToSpeech);
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn orphans_and_duplicate_names_are_handled() {
        let d = tmp("orphan");
        clip(999).write(&d.join("lonely-mmproj.gguf"));
        Gguf::new("qwen35")
            .string_array("tokenizer.ggml.tokens", &["a"])
            .tensor("token_embd.weight")
            .write(&d.join("draft.gguf"));
        Gguf::llm("llama", "Same Name").write(&d.join("same-Q4_K_M.gguf"));
        Gguf::llm("llama", "Same Name").write(&d.join("same-Q8_0.gguf"));
        Gguf::llm("qwen2vl", "VL Model").write(&d.join("vl.gguf"));

        let f = discover(&d).unwrap();
        assert_eq!(
            f.skipped.len(),
            2,
            "orphan projector and orphan draft: {:?}",
            f.skipped
        );
        assert!(
            f.skipped
                .iter()
                .any(|s| s.reason.contains("output size 999"))
        );
        let names: Vec<&str> = f.models.iter().map(|m| m.display_name.as_str()).collect();
        assert!(
            names.contains(&"same-Q4_K_M") && names.contains(&"same-Q8_0"),
            "{names:?}"
        );
        let vl = f
            .models
            .iter()
            .find(|m| m.architecture == "qwen2vl")
            .unwrap();
        assert!(
            vl.notes.iter().any(|n| n.contains("no matching mmproj")),
            "vision-language arch without projector is flagged"
        );
        assert_eq!(vl.modality, Modality::Text);
        std::fs::remove_dir_all(d).ok();
    }

    /// Compare detection with the hand-written catalog (the best ground truth available).
    /// `cargo test catalog_agrees_with_detection -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn catalog_agrees_with_detection() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/models.json");
        let reg = crate::config::ModelRegistry::from_file_with_env(
            &path,
            &[
                ("GEMINI_API_KEY".into(), "x".into()),
                ("OPENROUTER_API_KEY".into(), "x".into()),
            ],
        )
        .unwrap();
        let (mut checked, mut bad) = (0, 0);
        for m in reg.models.iter().filter(|m| m.is_local()) {
            let p = Path::new(&m.model_path);
            if !p.is_file() || !m.model_path.to_lowercase().ends_with(".gguf") {
                println!(
                    "SKIP  {:<36} (not an existing .gguf file: {})",
                    m.id,
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                );
                continue;
            }
            let info = match read_gguf(p) {
                Ok(i) => i,
                Err(e) => {
                    println!("ERR   {:<36} {e}", m.id);
                    bad += 1;
                    continue;
                }
            };
            checked += 1;
            let want_engine = m.engine_plan()[0];
            match classify(&info, &p.file_name().unwrap().to_string_lossy()) {
                Classified::Model(det) => {
                    let same_modality = det.modality == m.modality
                        || (det.modality == Modality::Text && m.modality == Modality::VisionText);
                    let ok = same_modality && det.engines[0] == want_engine;
                    println!(
                        "{} {:<36} catalog={:<16}/{:<18} detected={:<16}/{:<18}",
                        if ok { "OK   " } else { "DIFF " },
                        m.id,
                        format!("{:?}", m.modality),
                        want_engine,
                        format!("{:?}", det.modality),
                        det.engines[0]
                    );
                    if !ok {
                        bad += 1;
                    }
                }
                other => {
                    println!(
                        "DIFF  {:<36} catalog={:?}/{} but detected {:?}",
                        m.id, m.modality, want_engine, other
                    );
                    bad += 1;
                }
            }
        }
        println!("checked {checked} files, {bad} disagreement(s)");
        assert_eq!(bad, 0);
    }

    /// Run with `RUNE_PROBE_DIR=D:/Models cargo test probe_real_models -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_real_models() {
        let Ok(dir) = std::env::var("RUNE_PROBE_DIR") else {
            return;
        };
        let f = discover(Path::new(&dir)).unwrap();
        for m in &f.models {
            println!(
                "MODEL {:<44} {:<17} {:<19} {:<20} needs={:?} extra={:?}",
                m.display_name,
                format!("{:?}", m.modality),
                m.kind_label,
                m.engine(),
                m.needs,
                m.extra.keys().collect::<Vec<_>>()
            );
        }
        for s in &f.skipped {
            println!("SKIP  {} — {}", s.path.display(), s.reason);
        }
    }
}
