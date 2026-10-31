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

use std::collections::{BTreeSet, HashMap};
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
    /// `clip.vision.projection_dim` (projectors only). A value of 0 (MiniCPM's resampler)
    /// carries no information and is stored as `None`.
    pub projection_dim: Option<u32>,
    /// `clip.has_vision_encoder` / `clip.has_audio_encoder` (projectors only).
    pub has_vision_encoder: Option<bool>,
    pub has_audio_encoder: Option<bool>,
    pub has_tokenizer: bool,
    /// Any `prism.*` key (Prism ternary quantisation needs the Prism llama.cpp build).
    pub has_prism_keys: bool,
    /// Any `*.pooling_type` key (embedding models).
    pub has_pooling: bool,
    /// `audiocpp.model_spec.family`
    pub audiocpp_family: Option<String>,
    pub tensor_count: u64,
    /// Distinct GGML tensor storage types, read without loading weights.
    pub tensor_types: BTreeSet<u32>,
    pub has_token_embd: bool,
    pub has_first_block: bool,
    /// Tensor names typical of diffusion transformers / UNets (some GGUFs carry no metadata at all).
    pub diffusion_tensors: bool,
    /// `audio_tower.*` tensors: a speech-recognition audio encoder.
    pub audio_tower: bool,
    /// `audio_projector.*` tensors: an audio encoder that feeds a language model.
    pub audio_projector: bool,
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
                info.projection_dim = Some(rd_u32(r)?).filter(|d| *d > 0)
            }
            ("clip.has_vision_encoder", 7) => info.has_vision_encoder = Some(rd::<1>(r)?[0] != 0),
            ("clip.has_audio_encoder", 7) => info.has_audio_encoder = Some(rd::<1>(r)?[0] != 0),
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
        skip_bytes(r, u64::from(dims) * 8)?;
        let tensor_type = rd_u32(r)?;
        info.tensor_types.insert(tensor_type);
        skip_bytes(r, 8)?;
        if name == "token_embd.weight" {
            info.has_token_embd = true;
        } else if name.starts_with("blk.0.") {
            info.has_first_block = true;
        } else if name.starts_with("audio_tower.") {
            info.audio_tower = true;
        } else if name.starts_with("audio_projector.") {
            info.audio_projector = true;
        } else if DIFFUSION_TENSOR_MARKERS.iter().any(|m| name.contains(m)) {
            info.diffusion_tensors = true;
        }
    }
    // An audio encoder shipped as its own file (MiniCPM-o's `audio/` GGUF is arch `whisper`).
    if info.audio_projector && info.has_audio_encoder.is_none() {
        info.has_audio_encoder = Some(true);
        info.has_vision_encoder.get_or_insert(false);
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
    "gemma4",
    "gemma4v",
    "gemma4vl",
    "internvl",
    "minicpmv",
    "glm4v",
    "pixtral",
    "smolvlm",
    "idefics3",
    "lfm",
    "lfm2",
    "lfm2.5",
    "lfm2_5",
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
    /// The model belongs to an omni family (Gemma-4, MiniCPM-o, ...). Only a hint: what it
    /// can actually do is decided by the encoder files found next to it.
    omni_family: bool,
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

/// Architectures that are omni (image + audio + video in) by construction.
const OMNI_ARCHS: &[&str] = &[
    "gemma4",
    "gemma_4",
    "gemma-4",
    "minicpmo",
    "qwen2_5_omni",
    "qwen25omni",
    "qwen3_omni",
    "qwen3omni",
];

/// Whether this language model belongs to an omni family.
///
/// Several omni models are plain language-model architectures structurally —
/// MiniCPM-o's body is `qwen3` and its `general.name` is just "Llm" — so the
/// architecture alone cannot tell, and the file name is the only evidence.
/// Matching is on the squashed file name (`MiniCPM-o-4_5` → `minicpmo45`) or a
/// whole `omni` word, so `MiniCPM-V` and look-alikes are not caught.
fn is_omni_family(info: &GgufInfo, arch: &str, file_name: &str) -> bool {
    if OMNI_ARCHS.contains(&arch) {
        return true;
    }
    let squashed: String = file_name
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    squashed.contains("gemma4")
        || squashed.contains("minicpmo")
        || name_tokens(info, file_name).iter().any(|w| w == "omni")
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
            omni_family: false,
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
    if arch == "whisper" && info.audio_projector {
        return Classified::Projector;
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
    if arch == "embeddinggemma"
        || arch == "embedding_gemma"
        || (arch.starts_with("gemma")
            && (lower_name.contains("embedding")
                || info
                    .name
                    .as_deref()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains("embedding")))
    {
        return Classified::Model(det(
            Modality::Embedding,
            vec![EngineKind::LlamaServer],
            "embedding",
            vec!["Embedding"],
        ));
    }
    // A draft head (MTP / "assistant") is not a standalone model. Qwen's has a tokenizer and
    // embeddings but no first block; Gemma's (`gemma4-assistant`) does have blocks, so its
    // architecture name is the signal. Check this before the family branches below, or a
    // drafter is listed as a model of its own.
    if info.has_tokenizer
        && info.has_token_embd
        && (!info.has_first_block || arch.ends_with("-assistant") || arch.contains("mtp"))
    {
        return Classified::Draft;
    }
    if info.has_tokenizer && info.has_token_embd && is_omni_family(info, &arch, file_name) {
        // What an omni model can do depends on its encoder files, which are separate downloads
        // (MiniCPM-o's body is a plain `qwen3`). Start as a chat model; pairing the projector
        // upgrades it to exactly what that projector provides.
        let mut d = det(
            Modality::Text,
            vec![EngineKind::LlamaServer],
            "chat model",
            vec!["Chat", "Tool Call", "Structured Output"],
        );
        d.class = Class::Llm;
        d.omni_family = true;
        return Classified::Model(d);
    }
    if arch == "lfm"
        || arch == "lfm2"
        || arch == "lfm2.5"
        || arch == "lfm2_5"
        || arch.contains("liquid")
        || lower_name.contains("d1-3b")
        || lower_name.contains("liquid-d1")
    {
        let mut d = det(
            Modality::Decision,
            vec![EngineKind::LlamaServer],
            "decision model",
            vec!["Decision", "System1", "Classification", "Scoring", "Vision"],
        );
        d.class = Class::Llm;
        return Classified::Model(d);
    }
    if EMBEDDING_ARCHS.contains(&arch.as_str()) || info.has_pooling {
        return Classified::Skip(
            "embedding model in GGUF form — use a safetensors embedding model with the encoder engine"
                .into(),
        );
    }

    if info.has_tokenizer && info.has_token_embd {
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

// Reviewed 2026-10-25 against the Prism architecture table and multimodal guide:
// https://github.com/PrismML-Eng/llama.cpp/blob/prism/src/llama-arch.cpp
// https://github.com/PrismML-Eng/llama.cpp/blob/prism/docs/multimodal.md
// These are alternatives for stock GGUFs, not an assertion that every installed
// binary supports every model. Setup tests only the owner's chosen default.
const PRISM_TEXT_ALTERNATIVES: &[&str] = &[
    "llama",
    "qwen2",
    "qwen2moe",
    "qwen3",
    "qwen3moe",
    "qwen35",
    "qwen35moe",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "qwen2vl",
];
const PRISM_VISION_ALTERNATIVES: &[&str] = &["gemma3", "gemma4", "qwen2vl"];

fn reviewed_stock_tensors(info: &GgufInfo) -> bool {
    // Conservative shared F32/F16, Q4/Q5/Q8, K-quant, IQ and BF16 encodings.
    // Removed, newer and fork-private ids are deliberately excluded.
    // https://github.com/PrismML-Eng/llama.cpp/blob/prism/ggml/include/ggml.h
    info.tensor_count > 0
        && info.tensor_count <= 500_000
        && !info.tensor_types.is_empty()
        && info
            .tensor_types
            .iter()
            .all(|ty| matches!(ty, 0..=3 | 6..=30))
}

fn with_reviewed_alternatives(info: &GgufInfo, detected: &Detected) -> Vec<EngineKind> {
    let mut engines = detected.engines.clone();
    if engines == [EngineKind::LlamaServer]
        && detected.class == Class::Llm
        && detected.modality == Modality::Text
        && !info.has_prism_keys
        && PRISM_TEXT_ALTERNATIVES.contains(&info.architecture.to_lowercase().as_str())
        && reviewed_stock_tensors(info)
    {
        engines.push(EngineKind::LlamaServerPrism);
    }
    engines
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
    omni_family: bool,
}

impl LocalCandidate {
    pub fn engine(&self) -> EngineKind {
        self.engines[0]
    }
    pub fn is_vision(&self) -> bool {
        self.modality == Modality::VisionText
    }
    /// For an omni-family model: what is still missing for full omni input, if anything.
    pub fn omni_gap(&self) -> Option<&'static str> {
        if !self.omni_family || self.modality == Modality::Omni {
            return None;
        }
        Some(if self.extra.contains_key("mmproj_path") {
            "Omni family, but only text and image input are wired up (audio needs an audio-capable mmproj)"
        } else {
            "Omni family: image, audio and video input need extra.mmproj_path"
        })
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
        } else if path.extension().is_some_and(|e| {
            e.eq_ignore_ascii_case("gguf")
                || e.eq_ignore_ascii_case("cact")
                || e.eq_ignore_ascii_case("tflite")
                || (e.eq_ignore_ascii_case("bin") && has_ggml_magic(&path))
        }) {
            out.push(path);
        }
    }
}

fn has_ggml_magic(path: &Path) -> bool {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    let mut magic = [0; 4];
    file.read_exact(&mut magic).is_ok() && &magic == b"lmgg"
}

fn litert_candidate(path: &Path) -> Result<LocalCandidate> {
    anyhow::ensure!(
        cfg!(all(windows, target_arch = "x86_64")),
        "LiteRT depth currently requires Windows x64 CPU"
    );
    let profile = crate::inference::engines::litert::reviewed_model(path)?;
    Ok(LocalCandidate {
        path: path.to_path_buf(),
        size_bytes: std::fs::metadata(path)?.len(),
        architecture: profile.architecture.into(),
        display_name: profile.name.into(),
        context_length: None,
        modality: Modality::DepthEstimation,
        engines: vec![EngineKind::Litert],
        kind_label: "relative depth estimation",
        capabilities: Vec::new(),
        extra: Map::new(),
        needs: Vec::new(),
        notes: vec![
            "Reviewed FP32 TFLite export; LiteRT 2.3.0 Windows x64 CPU only. Relative depth, not meters. Task support is saved only after the live setup check passes.".into(),
            format!("Source: {}", profile.source),
        ],
        class: Class::Other,
        embedding_length: None,
        omni_family: false,
    })
}

fn whisper_candidate(path: &Path) -> Result<LocalCandidate> {
    let size = std::fs::metadata(path)?.len();
    anyhow::ensure!(size > 1024 * 1024, "Whisper GGML model file is too small");
    let mut file = std::fs::File::open(path)?;
    let mut bytes = [0u8; 48];
    file.read_exact(&mut bytes)?;
    anyhow::ensure!(&bytes[..4] == b"lmgg", "not a Whisper GGML model");
    let mut h = [0u32; 11];
    for (index, value) in h.iter_mut().enumerate() {
        *value = u32::from_le_bytes(bytes[4 + index * 4..8 + index * 4].try_into()?);
    }
    anyhow::ensure!(
        (50_000..=100_000).contains(&h[0])
            && (1..=4096).contains(&h[1])
            && (64..=8192).contains(&h[2])
            && (1..=128).contains(&h[3])
            && [4, 6, 12, 24, 32].contains(&h[4])
            && (1..=4096).contains(&h[5])
            && h[6] == h[2]
            && (1..=128).contains(&h[7])
            && [4, 6, 12, 24, 32].contains(&h[8])
            && [80, 128].contains(&h[9]),
        "invalid Whisper GGML model header"
    );
    Ok(LocalCandidate {
        path: path.to_path_buf(),
        size_bytes: size,
        architecture: "whisper-ggml".into(),
        display_name: file_stem(path),
        context_length: None,
        modality: Modality::SpeechToText,
        engines: vec![EngineKind::WhisperCpp, EngineKind::Crispasr],
        kind_label: "speech-to-text",
        capabilities: vec!["Speech-to-Text".into()],
        extra: Map::new(),
        needs: Vec::new(),
        notes: vec!["Whisper GGML format; audio.cpp GGUF packages are a different format.".into()],
        class: Class::Other,
        embedding_length: None,
        omni_family: false,
    })
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

/// What a projector file provides. Legacy `mmproj` files predate the flags and are vision-only.
fn projector_abilities(info: &GgufInfo) -> (bool, bool) {
    (
        info.has_vision_encoder.unwrap_or(true),
        info.has_audio_encoder.unwrap_or(false),
    )
}

fn projector_label(info: &GgufInfo) -> &'static str {
    match projector_abilities(info) {
        (true, true) => "vision+audio projector",
        (false, true) => "audio encoder",
        _ => "vision projector",
    }
}

/// Give each model the projector that belongs to it, and set what it can do from that file.
///
/// * A projector belongs to the models in the *nearest enclosing folder that holds a model*,
///   so `snapshot/vision/x.gguf` pairs with `snapshot/model.gguf`.
/// * Its output size (`clip.vision.projection_dim`) must equal the model's `embedding_length`
///   (a size of 0, which some projectors record, says nothing and is ignored). Models of the
///   same size are told apart by file-name similarity; if that cannot decide, nothing is
///   attached and the model gets a note.
/// * `llama-server` loads a single `--mmproj`, so when several projectors match, the most
///   capable one is attached (vision+audio, then vision, then audio) and the rest are noted.
/// * Capabilities come from the projector's own `has_vision_encoder` / `has_audio_encoder`
///   flags: both → omni, vision → vision chat, audio → audio chat. A model family's name never
///   grants an ability the files do not provide.
fn pair_projectors(
    candidates: &mut [LocalCandidate],
    projectors: Vec<(PathBuf, GgufInfo)>,
    skipped: &mut Vec<Skipped>,
) {
    let llm = |c: &LocalCandidate| c.class == Class::Llm;
    let mut model_dirs: Vec<PathBuf> = candidates
        .iter()
        .filter(|c| llm(c))
        .filter_map(|c| c.path.parent().map(Path::to_path_buf))
        .collect();
    model_dirs.sort();
    model_dirs.dedup();

    let mut chosen: Vec<Vec<usize>> = vec![Vec::new(); candidates.len()];
    let mut ambiguous: Vec<Vec<usize>> = vec![Vec::new(); candidates.len()];

    for (pi, (path, info)) in projectors.iter().enumerate() {
        let owner = path.parent().and_then(|proj_dir| {
            model_dirs
                .iter()
                .filter(|d| proj_dir.starts_with(d))
                .max_by_key(|d| d.components().count())
        });
        let in_dir: Vec<usize> = (0..candidates.len())
            .filter(|&i| {
                llm(&candidates[i]) && candidates[i].path.parent() == owner.map(|d| d.as_path())
            })
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
            Some(&i) => chosen[i].push(pi),
            None if ambiguous.iter().any(|a| a.contains(&pi)) => {}
            None => skipped.push(Skipped {
                path: path.clone(),
                reason: match info.projection_dim {
                    Some(d) => format!(
                        "{} (output size {d}) matches no model in its folder",
                        projector_label(info)
                    ),
                    None => format!(
                        "{} with no matching model in its folder",
                        projector_label(info)
                    ),
                },
            }),
        }
    }

    for (i, c) in candidates.iter_mut().enumerate() {
        let mut options = std::mem::take(&mut chosen[i]);
        // Most capable first; on a tie vision beats audio (image input is what llama-server
        // supports); then the f16-preferring name rank.
        options.sort_by_key(|&pi| {
            let (v, a) = projector_abilities(&projectors[pi].1);
            (
                std::cmp::Reverse(u8::from(v) + u8::from(a)),
                std::cmp::Reverse(v),
                projector_rank(&projectors[pi].0),
            )
        });
        if let Some(&best) = options.first() {
            let (path, info) = &projectors[best];
            let (vision, audio) = projector_abilities(info);
            if c.engine() == EngineKind::LlamaServer
                && (audio
                    || !vision
                    || !PRISM_VISION_ALTERNATIVES.contains(&c.architecture.to_lowercase().as_str())
                    || !reviewed_stock_tensors(info))
            {
                c.engines
                    .retain(|engine| *engine != EngineKind::LlamaServerPrism);
            }
            c.extra
                .insert("mmproj_path".into(), json!(path.to_string_lossy()));
            let cap = |c: &mut LocalCandidate, name: &str| {
                if !c.capabilities.iter().any(|x| x == name) {
                    c.capabilities.push(name.into());
                }
            };
            if vision {
                cap(c, "Vision");
            }
            if audio {
                cap(c, "Audio");
            }
            if c.modality != Modality::Decision {
                match (vision, audio) {
                    (true, true) => {
                        c.modality = Modality::Omni;
                        c.kind_label = "omni multimodal model";
                        cap(c, "Video");
                        cap(c, "Omni");
                    }
                    (true, false) => {
                        c.modality = Modality::VisionText;
                        c.kind_label = "vision chat model";
                    }
                    (false, true) => c.kind_label = "audio chat model",
                    (false, false) => {}
                }
            }
            c.notes
                .retain(|n| !n.starts_with("vision-language model, but no matching"));
            // The rest could not be attached: llama-server takes one projector.
            for &pi in options.iter().skip(1) {
                let (other_path, other_info) = &projectors[pi];
                let (ov, oa) = projector_abilities(other_info);
                if (oa && !audio) || (ov && !vision) {
                    c.notes.push(format!(
                        "{} {} found but not attached: llama-server loads a single mmproj",
                        projector_label(other_info),
                        other_path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
            }
        } else if !ambiguous[i].is_empty() {
            c.notes.push(format!(
                "{} projector(s) in this folder could belong to it; set extra.mmproj_path",
                ambiguous[i].len()
            ));
        }
        if c.omni_family && c.modality != Modality::Omni {
            c.notes.push(if c.extra.contains_key("mmproj_path") {
                OMNI_PARTIAL_NOTE.into()
            } else {
                OMNI_NO_ENCODER_NOTE.into()
            });
        }
    }
}

/// Shown for an omni-family model whose projector provides vision but not audio.
pub const OMNI_PARTIAL_NOTE: &str = "omni family, but its projector only provides image input — \
     audio and video need an audio-capable mmproj (llama-server loads a single file)";

/// Shown for an omni model found without its vision/audio encoder file.
pub const OMNI_NO_ENCODER_NOTE: &str = "omni family, but only the language model was found — \
     add extra.mmproj_path (its vision/audio encoder GGUF) for image, audio and video input";

/// Read the documented Needle 3 container without loading weights. Speech
/// attachments are excluded: the existing Needle adapter only handles tool calls.
fn cactus_candidate(path: &Path) -> Result<LocalCandidate> {
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let header = rd::<196>(&mut file)?;
    let field =
        |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().expect("header field"));
    anyhow::ensure!(
        field(0) == 0x05E12A84,
        "unsupported Cactus generation; the installed adapter requires Needle 3"
    );
    let count = field(4);
    let codebook = field(8);
    anyhow::ensure!(
        count > 0 && count <= 100_000 && codebook <= 1024,
        "invalid Cactus directory"
    );
    let directory = 196 + u64::from(codebook) * 4;
    let data_start = directory + u64::from(count) * 44;
    anyhow::ensure!(data_start <= size, "truncated Cactus directory");
    file.seek(SeekFrom::Start(directory))?;
    let mut raw = 0;
    let mut weights = 0;
    for _ in 0..count {
        let entry = rd::<44>(&mut file)?;
        let offset = u64::from_le_bytes(entry[20..28].try_into()?);
        let length = u64::from_le_bytes(entry[28..36].try_into()?);
        anyhow::ensure!(
            (1..=4).contains(&entry[0])
                && entry[1] <= 4
                && offset >= data_start
                && length > 0
                && offset.checked_add(length).is_some_and(|end| end <= size),
            "invalid Cactus tensor bounds"
        );
        raw += usize::from(entry[0] == 4);
        weights += usize::from(entry[0] != 4);
    }
    anyhow::ensure!(
        raw == 1 && weights > 0,
        "Cactus archive has additional or missing attachments; speech containers are not supported by the Needle tool adapter"
    );
    Ok(LocalCandidate {
        path: path.to_path_buf(),
        size_bytes: size,
        architecture: "needle3".into(),
        display_name: file_stem(path),
        context_length: None,
        modality: Modality::Text,
        engines: vec![EngineKind::Needle],
        kind_label: "tool-calling model",
        capabilities: vec!["Tool Call".into()],
        extra: Map::new(),
        needs: Vec::new(),
        notes: vec!["Needle 3 Cactus container; requires the compatible Needle runtime.".into()],
        class: Class::Other,
        embedding_length: None,
        omni_family: false,
    })
}

/// Scan `input` (a file, or a folder searched up to 9 levels) and classify
/// supported GGUF, GGML, Cactus and reviewed TFLite artifacts.
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
    let mut cactus = Vec::new();
    let mut whisper = Vec::new();
    let mut litert = Vec::new();

    for path in files {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("tflite"))
        {
            match litert_candidate(&path) {
                Ok(model) => litert.push(model),
                Err(error) => out.skipped.push(Skipped {
                    path,
                    reason: format!("not a supported LiteRT model ({error})"),
                }),
            }
            continue;
        }
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        {
            match whisper_candidate(&path) {
                Ok(model) => whisper.push(model),
                Err(error) => out.skipped.push(Skipped {
                    path,
                    reason: format!("not a supported Whisper GGML model ({error})"),
                }),
            }
            continue;
        }
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("cact"))
        {
            match cactus_candidate(&path) {
                Ok(model) => cactus.push(model),
                Err(error) => out.skipped.push(Skipped {
                    path,
                    reason: format!("not a supported Cactus model ({error})"),
                }),
            }
            continue;
        }
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
                engines: with_reviewed_alternatives(&info, &d),
                kind_label: label,
                capabilities: caps,
                extra,
                needs: d.needs,
                notes,
                class: d.class,
                embedding_length: info.embedding_length,
                omni_family: d.omni_family,
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
                // Draft compatibility is separate from base-model compatibility.
                if base.engine() == EngineKind::LlamaServer {
                    base.engines
                        .retain(|engine| *engine != EngineKind::LlamaServerPrism);
                }
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

    candidates.extend(cactus);
    candidates.extend(whisper);
    candidates.extend(litert);
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
        pub fn bool_kv(mut self, k: &str, v: bool) -> Self {
            self.kvs.push((k.into(), 7, vec![u8::from(v)]));
            self
        }
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
    fn tflite_discovery_reports_unreviewed_or_malformed_artifacts() {
        let root = tmp("tflite");
        for (name, bytes) in [
            ("depth_anything_v2_small.tflite", b"\0\0\0\0TFL3".as_slice()),
            ("broken.TFLITE", b"not a flatbuffer".as_slice()),
        ] {
            std::fs::write(root.join(name), bytes).unwrap();
            let result = discover(&root.join(name)).unwrap();
            assert!(result.models.is_empty());
            assert_eq!(result.skipped.len(), 1);
            assert!(
                result.skipped[0]
                    .reason
                    .contains("not a supported LiteRT model")
            );
            if cfg!(all(windows, target_arch = "x86_64")) {
                assert!(
                    result.skipped[0]
                        .reason
                        .contains(if name.starts_with("broken") {
                            "not a TFLite model"
                        } else {
                            "Unreviewed LiteRT model artifact"
                        })
                );
            }
        }
        let result = discover(&root).unwrap();
        assert!(result.models.is_empty());
        assert_eq!(result.skipped.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
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
    fn cactus_discovery_checks_format_and_excludes_speech_attachments() {
        let root = tmp("cactus");
        let fixture = |raw_count: usize| {
            let count = 1 + raw_count;
            let start = 196 + count * 44;
            let mut bytes = vec![0u8; start + count * 64];
            bytes[0..4].copy_from_slice(&0x05E12A84u32.to_le_bytes());
            bytes[4..8].copy_from_slice(&(count as u32).to_le_bytes());
            for i in 0..count {
                let entry = 196 + i * 44;
                bytes[entry] = if i == 0 { 1 } else { 4 };
                bytes[entry + 1] = 1;
                bytes[entry + 20..entry + 28]
                    .copy_from_slice(&((start + i * 64) as u64).to_le_bytes());
                bytes[entry + 28..entry + 36].copy_from_slice(&64u64.to_le_bytes());
            }
            bytes
        };
        std::fs::write(root.join("model.cact"), fixture(1)).unwrap();
        std::fs::write(root.join("speech.cact"), fixture(2)).unwrap();
        std::fs::write(root.join("invalid.cact"), b"invalid").unwrap();
        let found = discover(&root).unwrap();
        assert_eq!(found.models.len(), 1);
        assert_eq!(found.skipped.len(), 2);
        assert_eq!(found.models[0].engines, [EngineKind::Needle]);
        assert_eq!(found.models[0].capabilities, ["Tool Call"]);
        assert_eq!(discover(&root.join("model.cact")).unwrap().models.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn whisper_ggml_discovery_prefers_compatible_engines_and_rejects_invalid_headers() {
        let root = tmp("whisper-ggml");
        let path = root.join("whisper.bin");
        let mut bytes = vec![0u8; 1024 * 1024 + 1];
        bytes[..4].copy_from_slice(b"lmgg");
        for (i, n) in [51866u32, 1500, 1280, 20, 32, 448, 1280, 20, 4, 128, 1]
            .iter()
            .enumerate()
        {
            bytes[4 + i * 4..8 + i * 4].copy_from_slice(&n.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let candidate = whisper_candidate(&path).unwrap();
        assert_eq!(
            candidate.engines,
            [EngineKind::WhisperCpp, EngineKind::Crispasr]
        );
        assert_eq!(discover(&root).unwrap().models.len(), 1);
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(whisper_candidate(&path).is_err());
        std::fs::remove_dir_all(root).unwrap();
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
    fn discovery_offers_only_reviewed_stock_engine_alternatives() {
        let root = tmp("engine-alternatives");
        for arch in PRISM_TEXT_ALTERNATIVES {
            Gguf::llm(arch, "Stock").write(&root.join(format!("{arch}.gguf")));
        }
        Gguf::llm("future-unreviewed", "Qwen3").write(&root.join("unknown.gguf"));
        Gguf::llm("llama", "Private")
            .string("prism.x", "1")
            .write(&root.join("private.gguf"));
        for ty in [4, 39, 142, 999] {
            let mut bytes = Gguf::llm("llama", "New tensor format").bytes();
            let offset = bytes.len() - 12;
            bytes[offset..offset + 4].copy_from_slice(&(ty as u32).to_le_bytes());
            std::fs::write(root.join(format!("type-{ty}.gguf")), bytes).unwrap();
        }
        for candidate in discover(&root).unwrap().models {
            let file = candidate.path.file_stem().unwrap().to_string_lossy();
            if file == "private" {
                assert_eq!(candidate.engines, [EngineKind::LlamaServerPrism]);
            } else if file == "unknown" || file.starts_with("type-") {
                assert_eq!(candidate.engines, [EngineKind::LlamaServer]);
            } else {
                assert_eq!(
                    candidate.engines,
                    [EngineKind::LlamaServer, EngineKind::LlamaServerPrism],
                    "{}",
                    candidate.architecture
                );
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn projector_and_draft_require_their_own_engine_compatibility() {
        let root = tmp("companion-alternatives");
        for (folder, arch, vision, audio) in [
            ("vision", "gemma3", true, false),
            ("omni", "gemma4", true, true),
            ("unknown-projector", "qwen3", true, false),
        ] {
            let dir = root.join(folder);
            Gguf::llm(arch, "Base model").write(&dir.join("model.gguf"));
            Gguf::new("clip")
                .u32("clip.vision.projection_dim", 4096)
                .bool_kv("clip.has_vision_encoder", vision)
                .bool_kv("clip.has_audio_encoder", audio)
                .tensor("v.patch_embd.weight")
                .write(&dir.join("mmproj.gguf"));
        }
        let dir = root.join("draft");
        Gguf::llm("llama", "Base model").write(&dir.join("model.gguf"));
        Gguf::new("llama")
            .string("general.name", "Base model")
            .string_array("tokenizer.ggml.tokens", &["a"])
            .tensor("token_embd.weight")
            .write(&dir.join("draft.gguf"));
        let found = discover(&root).unwrap();
        assert_eq!(found.models.len(), 4);
        for candidate in found.models {
            let folder = candidate
                .path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            if folder == "vision" {
                assert_eq!(candidate.modality, Modality::VisionText);
                assert_eq!(
                    candidate.engines,
                    [EngineKind::LlamaServer, EngineKind::LlamaServerPrism]
                );
            } else {
                assert_eq!(candidate.engines, [EngineKind::LlamaServer], "{folder}");
            }
        }
        std::fs::remove_dir_all(root).unwrap();
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
    fn decision_models_are_recognized() {
        let d = model(cls(&Gguf::new("lfm2.5"), "d1-3b-q4_k_m.gguf"));
        assert_eq!(d.modality, Modality::Decision);
        assert_eq!(d.engines, vec![EngineKind::LlamaServer]);
        assert_eq!(d.kind_label, "decision model");
        assert!(d.capabilities.contains(&"Decision"));
        assert!(d.capabilities.contains(&"System1"));
    }

    #[test]
    fn gemma_4_is_an_omni_family_but_only_a_chat_model_until_its_projector_is_found() {
        let g = Gguf::llm("gemma4", "Gemma 4 12B IT");
        let d = model(cls(&g, "gemma-4-12b-it-Q4_K_M.gguf"));
        assert!(d.omni_family);
        assert_eq!(
            d.modality,
            Modality::Text,
            "the family name alone claims nothing"
        );
        assert_eq!(d.engines, vec![EngineKind::LlamaServer]);
        for cap in ["Vision", "Audio", "Video", "Omni"] {
            assert!(!d.capabilities.contains(&cap), "{cap}");
        }
    }
    #[test]
    fn minicpm_o_is_recognised_as_an_omni_family_despite_its_plain_qwen3_body() {
        // Real file: arch qwen3, general.name "Llm", only the file name says MiniCPM-o.
        let d = model(cls(&Gguf::llm("qwen3", "Llm"), "MiniCPM-o-4_5-Q8_0.gguf"));
        assert!(d.omni_family);
        assert_eq!((d.modality, d.kind_label), (Modality::Text, "chat model"));
        // A whole "omni" word marks other families too.
        let q = model(cls(
            &Gguf::llm("qwen2", "Qwen2.5-Omni-7B"),
            "Qwen2.5-Omni-7B-Q4_K_M.gguf",
        ));
        assert!(q.omni_family);
    }
    #[test]
    fn omni_detection_does_not_catch_look_alikes() {
        // MiniCPM-V is vision-only; "omnibus" merely contains the letters.
        for file in [
            "MiniCPM-V-4_5-Q8_0.gguf",
            "omnibus-7b.gguf",
            "OxCoder-9B.gguf",
            "minicpm-4b.gguf",
        ] {
            let d = model(cls(&Gguf::llm("qwen3", "Llm"), file));
            assert!(!d.omni_family, "{file}");
            assert_eq!(d.modality, Modality::Text, "{file}");
        }
    }
    #[test]
    fn draft_heads_of_omni_families_are_not_listed_as_models() {
        // Same shape as `mtp-gemma-4-12b-it.gguf`: tokenizer + embeddings, no first block.
        for (arch, file) in [
            ("gemma4", "mtp-gemma-4-12b-it.gguf"),
            ("qwen3", "MiniCPM-o-4_5-mtp.gguf"),
        ] {
            let drafter = Gguf::new(arch)
                .string_array("tokenizer.ggml.tokens", &["a"])
                .tensor("token_embd.weight")
                .tensor("blk.40.attn_k.weight");
            assert!(matches!(cls(&drafter, file), Classified::Draft), "{file}");
        }
        // Gemma's real drafter has blocks (`blk.0.*`) — arch `gemma4-assistant` identifies it.
        let assistant = Gguf::llm("gemma4-assistant", "Gemma 4 12B It Assistant");
        assert!(matches!(
            cls(&assistant, "mtp-gemma-4-12b-it.gguf"),
            Classified::Draft
        ));
        // The ordinary Gemma model next to it is still an omni chat model.
        let base = model(cls(
            &Gguf::llm("gemma4", "Gemma-4-12B-It"),
            "gemma-4-12b-it-UD-Q6_K_XL.gguf",
        ));
        assert!(base.omni_family && base.modality == Modality::Text);
    }

    /// A projector with explicit encoder flags (`dim` None = records no projection size).
    fn projector(dim: Option<u32>, vision: bool, audio: bool) -> Gguf {
        let g = Gguf::new("clip")
            .bool_kv("clip.has_vision_encoder", vision)
            .bool_kv("clip.has_audio_encoder", audio);
        match dim {
            Some(d) => g.u32("clip.vision.projection_dim", d),
            None => g.u32("clip.vision.projection_dim", 0),
        }
    }

    /// MiniCPM-o's separate audio encoder: arch `whisper` with an `audio_projector`.
    fn audio_encoder() -> Gguf {
        Gguf::new("whisper")
            .tensor("encoder.layers.0.fc1.weight")
            .tensor("audio_projector.linear1.weight")
    }

    fn minicpm(d: &Path) {
        Gguf::llm_with_dim("qwen3", "Llm", 4096).write(&d.join("MiniCPM-o-4_5-Q8_0.gguf"));
    }

    #[test]
    fn an_omni_family_model_without_encoders_is_a_chat_model_and_says_so() {
        let d = tmp("omni-alone");
        minicpm(&d);
        let m = &discover(&d).unwrap().models[0];
        assert_eq!(m.modality, Modality::Text);
        assert!(
            !m.capabilities.iter().any(|c| c == "Vision" || c == "Audio"),
            "{:?}",
            m.capabilities
        );
        assert!(
            m.notes.iter().any(|n| n == OMNI_NO_ENCODER_NOTE),
            "{:?}",
            m.notes
        );
        assert!(m.omni_gap().unwrap().contains("need extra.mmproj_path"));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn minicpm_o_with_its_real_layout_is_vision_only() {
        // The real snapshot: model at the root, encoders in vision/ and audio/ subfolders; the
        // vision file records projection_dim 0 and the audio file is arch whisper.
        let d = tmp("omni-real");
        minicpm(&d);
        projector(None, true, false).write(&d.join("vision/MiniCPM-o-4_5-vision-F16.gguf"));
        audio_encoder().write(&d.join("audio/MiniCPM-o-4_5-audio-F16.gguf"));

        let f = discover(&d).unwrap();
        assert_eq!(
            f.models.len(),
            1,
            "the audio encoder is not a speech model: {:?}",
            f.models.iter().map(|m| &m.path).collect::<Vec<_>>()
        );
        assert!(f.skipped.is_empty(), "{:?}", f.skipped);
        let m = &f.models[0];
        assert_eq!(
            m.modality,
            Modality::VisionText,
            "only what the attached projector provides"
        );
        assert!(
            m.extra["mmproj_path"]
                .as_str()
                .unwrap()
                .ends_with("MiniCPM-o-4_5-vision-F16.gguf")
        );
        assert!(
            m.capabilities.iter().any(|c| c == "Vision")
                && !m.capabilities.iter().any(|c| c == "Audio")
        );
        assert!(
            m.notes
                .iter()
                .any(|n| n.contains("audio encoder") && n.contains("not attached")),
            "{:?}",
            m.notes
        );
        assert!(
            m.notes.iter().any(|n| n == OMNI_PARTIAL_NOTE),
            "{:?}",
            m.notes
        );
        assert!(m.omni_gap().unwrap().contains("only text and image"));
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn a_projector_with_both_encoders_makes_the_model_omni() {
        // Gemma-4's single mmproj has vision and audio encoders.
        let d = tmp("omni-both");
        Gguf::llm_with_dim("gemma4", "Gemma-4-12B-It", 3840)
            .write(&d.join("gemma-4-12b-it-UD-Q6_K_XL.gguf"));
        projector(Some(3840), true, true).write(&d.join("mmproj-F16.gguf"));
        let m = &discover(&d).unwrap().models[0];
        assert_eq!(m.modality, Modality::Omni);
        for cap in ["Vision", "Audio", "Video", "Omni"] {
            assert!(
                m.capabilities.iter().any(|c| c == cap),
                "{cap}: {:?}",
                m.capabilities
            );
        }
        assert!(m.omni_gap().is_none());
        assert!(
            !m.notes
                .iter()
                .any(|n| n == OMNI_PARTIAL_NOTE || n == OMNI_NO_ENCODER_NOTE)
        );
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn the_most_capable_matching_projector_is_attached() {
        let d = tmp("omni-pick");
        Gguf::llm_with_dim("qwen35", "Model", 4096).write(&d.join("model.gguf"));
        projector(Some(4096), true, false).write(&d.join("mmproj-vision.gguf"));
        projector(Some(4096), true, true).write(&d.join("mmproj-both.gguf"));
        let m = &discover(&d).unwrap().models[0];
        assert!(
            m.extra["mmproj_path"]
                .as_str()
                .unwrap()
                .ends_with("mmproj-both.gguf")
        );
        assert_eq!(m.modality, Modality::Omni);
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn projectors_belong_to_the_nearest_folder_that_holds_a_model() {
        // Two models, each with its own vision/ subfolder, plus a model one level above.
        let d = tmp("omni-nearest");
        Gguf::llm_with_dim("qwen3", "Alpha", 4096).write(&d.join("a/alpha.gguf"));
        projector(Some(4096), true, false).write(&d.join("a/vision/alpha-vision.gguf"));
        Gguf::llm_with_dim("qwen3", "Bravo", 4096).write(&d.join("b/bravo.gguf"));
        projector(Some(4096), true, false).write(&d.join("b/vision/bravo-vision.gguf"));
        Gguf::llm_with_dim("qwen3", "Topmost", 4096).write(&d.join("top.gguf"));

        let f = discover(&d).unwrap();
        let mm = |n: &str| {
            f.models
                .iter()
                .find(|m| m.display_name == n)
                .unwrap()
                .extra
                .get("mmproj_path")
                .and_then(|v| v.as_str().map(String::from))
        };
        assert!(mm("Alpha").unwrap().ends_with("alpha-vision.gguf"));
        assert!(mm("Bravo").unwrap().ends_with("bravo-vision.gguf"));
        assert!(
            mm("Topmost").is_none(),
            "a model above the folders must not take their projectors"
        );
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn a_projection_size_of_zero_is_unknown_not_a_mismatch() {
        let g = Gguf::new("clip")
            .u32("clip.vision.projection_dim", 0)
            .bool_kv("clip.has_vision_encoder", true);
        let i = parse_gguf(&mut g.bytes().as_slice()).unwrap();
        assert_eq!(i.projection_dim, None);
        assert_eq!(
            (i.has_vision_encoder, i.has_audio_encoder),
            (Some(true), None)
        );
    }

    #[test]
    fn audio_encoder_files_are_companions_not_speech_models() {
        let i = parse_gguf(&mut audio_encoder().bytes().as_slice()).unwrap();
        assert!(i.audio_projector);
        assert_eq!(
            (i.has_audio_encoder, i.has_vision_encoder),
            (Some(true), Some(false))
        );
        assert!(matches!(classify(&i, "audio.gguf"), Classified::Projector));
        // A real whisper speech model (no audio_projector) is unaffected.
        let w = Gguf::new("whisper").tensor("encoder.layers.0.fc1.weight");
        assert_eq!(model(cls(&w, "w.gguf")).modality, Modality::SpeechToText);
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
        let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("config");
        let reg = crate::config::ModelRegistry::from_files_with_env(
            config.join("server.json"),
            config.join("catalog.json"),
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
                    // Pairing a projector (a separate file) later upgrades a chat model to vision/omni.
                    let same_modality = det.modality == m.modality
                        || (det.modality == Modality::Text
                            && matches!(m.modality, Modality::VisionText | Modality::Omni));
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
