use super::{NLU_LABEL_COUNT, OnnxEngine};
use crate::config::ModelConfig;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{NluIntent, NluIntentResponse, Usage};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    io::{Read, Write},
    path::Path,
};
use tokenizers::Tokenizer;

const TOKENIZER_SHA: &str = "17f7d8b9518c403d7429ad9eeeabb6eed49c8d3311de8ef4ed5ad811381a2ced";
const MAX_INPUT_TOKENS: usize = 512;
// Label order from the verified export's category_mapping.json.
const LABELS: [&str; NLU_LABEL_COUNT] = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
];

pub(super) fn load_tokenizer(model: &ModelConfig) -> Result<Tokenizer, Box<dyn Error>> {
    let path = Path::new(
        model
            .tokenizer_path
            .as_deref()
            .ok_or("NLU model requires a tokenizer")?,
    );
    if crate::inference::engines::download::file_sha256(path)? != TOKENIZER_SHA {
        return Err("NLU tokenizer SHA-256 does not match the reviewed tokenizer".into());
    }
    let mut tokenizer = Tokenizer::from_file(path).map_err(|error| error.to_string())?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| error.to_string())?;
    Ok(tokenizer)
}

pub(super) fn execute(
    engine: &OnnxEngine,
    tokenizer: &Tokenizer,
    task: &InferenceTaskRequest,
) -> Result<InferenceOutput, Box<dyn Error>> {
    let InferenceTaskRequest::NluIntent {
        text,
        candidate_intents,
        ..
    } = task
    else {
        return Err("This ONNX model supports intent classification only".into());
    };
    let encoded = tokenizer
        .encode(text.as_str(), true)
        .map_err(|error| error.to_string())?;
    let ids = encoded.get_ids();
    if ids.is_empty() || ids.len() > MAX_INPUT_TOKENS {
        return Err(
            format!("NLU input must contain between 1 and {MAX_INPUT_TOKENS} tokens").into(),
        );
    }
    let length = ids.len() as u32;
    let mut input = Vec::with_capacity(4 + ids.len() * 4);
    input.extend_from_slice(&length.to_le_bytes());
    for id in ids {
        input.extend_from_slice(&id.to_le_bytes());
    }
    let mut worker = engine
        .worker
        .lock()
        .map_err(|_| "ONNX inference lock poisoned")?;
    let logits = engine.run_worker(&mut worker, &input, true)?;
    let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let probabilities: Vec<_> = logits
        .iter()
        .map(|&logit| (f64::from(logit) - max).exp())
        .collect();
    let sum: f64 = probabilities.iter().sum();
    let intent = LABELS
        .into_iter()
        .zip(probabilities)
        .filter(|(label, _)| {
            candidate_intents.as_ref().is_none_or(|candidates| {
                candidates.is_empty()
                    || candidates
                        .iter()
                        .any(|candidate| candidate.eq_ignore_ascii_case(label))
            })
        })
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(label, score)| NluIntent::new(label, score / sum));
    Ok(InferenceOutput {
        response: InferenceTaskResponse::NluIntent(NluIntentResponse {
            id: format!("nlu_{}", hex::encode(&Sha256::digest(text.as_bytes())[..8])),
            object: "nlu.analysis".into(),
            model: engine.id.clone(),
            intent,
            entities: Vec::new(),
        }),
        usage: Usage::new(length, 0),
    })
}

pub(super) fn worker(
    session: &mut Session,
    stdin: &mut impl Read,
    stdout: &mut impl Write,
) -> anyhow::Result<()> {
    loop {
        let mut count = [0; 4];
        if stdin.read(&mut count[..1])? == 0 {
            return Ok(());
        }
        stdin.read_exact(&mut count[1..])?;
        let count = u32::from_le_bytes(count) as usize;
        anyhow::ensure!(
            (1..=MAX_INPUT_TOKENS).contains(&count),
            "Invalid NLU input length"
        );
        let mut bytes = vec![0; count * 4];
        stdin.read_exact(&mut bytes)?;
        let ids: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&id| i64::from(u32::from_le_bytes(id)))
            .collect();
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, count], ids))?,
            "attention_mask" => Tensor::from_array(([1, count], vec![1i64; count]))?,
        ])?;
        let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.as_ref() == [1, NLU_LABEL_COUNT as i64],
            "Invalid NLU output shape: {shape:?}"
        );
        for logit in logits {
            stdout.write_all(&logit.to_le_bytes())?;
        }
        stdout.flush()?;
    }
}
