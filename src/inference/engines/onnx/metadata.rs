use super::Profile;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::HashMap, sync::LazyLock};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelMetadata<'a> {
    #[serde(borrow)]
    profiles: HashMap<&'a str, Profile>,
    #[serde(borrow)]
    pub(super) coco_labels: Vec<&'a str>,
    #[serde(borrow)]
    pub(super) coco_keypoints: Vec<&'a str>,
    #[serde(borrow)]
    pub(super) dota_labels: Vec<&'a str>,
    #[serde(borrow)]
    pub(super) ade20k_labels: Vec<&'a str>,
    #[serde(borrow)]
    pub(super) nlu_labels: Vec<&'a str>,
}

#[derive(Deserialize)]
struct BundledRegistry<'a> {
    #[serde(borrow)]
    extensions: Extensions<'a>,
}

#[derive(Deserialize)]
struct Extensions<'a> {
    #[serde(borrow)]
    onnx: ModelMetadata<'a>,
}

pub(super) fn data() -> &'static ModelMetadata<'static> {
    static DATA: LazyLock<ModelMetadata<'static>> = LazyLock::new(|| {
        let registry: BundledRegistry<'_> = serde_json::from_str(crate::registry_document::MODELS)
            .expect("invalid bundled ONNX model metadata");
        registry.extensions.onnx
    });
    &DATA
}

pub(super) struct ReviewedModels {
    pub(super) profiles_by_hash: HashMap<String, Profile>,
    pub(super) nlu_tokenizer_sha: String,
    pub(super) nlu_max_input_tokens: usize,
}

pub(super) fn reviewed() -> &'static ReviewedModels {
    static REVIEWED: LazyLock<ReviewedModels> = LazyLock::new(|| {
        load_reviewed().expect("ONNX profiles must reference reviewed bundled model artifacts")
    });
    &REVIEWED
}

fn load_reviewed() -> Result<ReviewedModels> {
    let listing = crate::model_registry::bundled_listing()?;
    let metadata = data();
    let mut profiles_by_hash = HashMap::new();
    let mut nlu = None;
    for (&id, &profile) in &metadata.profiles {
        let model = listing
            .iter()
            .find_map(|entry| entry.resolve_variant(id))
            .with_context(|| format!("missing ONNX registry model {id}"))?;
        ensure!(
            model.engine == "onnx" && model.format == "onnx",
            "{id} must be an ONNX model"
        );
        let files = &model
            .versions
            .get(&model.latest)
            .with_context(|| format!("missing ONNX release for {id}"))?
            .files;
        let hash = |role| -> Result<String> {
            let mut matches = files.iter().filter(|file| file.id == role);
            let sha = &matches
                .next()
                .with_context(|| format!("missing {role} for {id}"))?
                .sha256;
            ensure!(
                matches.next().is_none()
                    && sha.len() == 64
                    && sha.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid {role} checksum for {id}"
            );
            Ok(sha.to_ascii_lowercase())
        };
        ensure!(
            profile != Profile::Recipe
                && profiles_by_hash.insert(hash("model")?, profile).is_none(),
            "duplicate or invalid ONNX profile for {id}"
        );
        if profile == Profile::Nlu {
            let limit = model
                .catalog
                .and_then(|catalog| catalog.max_context_length)
                .filter(|&value| value > 0)
                .context("missing NLU context limit")?;
            ensure!(
                nlu.replace((hash("tokenizer")?, limit as usize)).is_none(),
                "duplicate reviewed NLU model"
            );
        }
    }
    let (nlu_tokenizer_sha, nlu_max_input_tokens) = nlu.context("missing reviewed NLU profile")?;
    Ok(ReviewedModels {
        profiles_by_hash,
        nlu_tokenizer_sha,
        nlu_max_input_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_metadata_matches_supported_tensor_contracts() {
        let expected = [
            Profile::Detection,
            Profile::Pose,
            Profile::Segmentation,
            Profile::Oriented,
            Profile::Semantic,
            Profile::Matting,
            Profile::Nlu,
        ];
        let profiles = &reviewed().profiles_by_hash;
        assert_eq!(profiles.len(), expected.len());
        for profile in expected {
            assert!(profiles.values().any(|value| *value == profile));
        }
        let metadata = data();
        assert_eq!(metadata.coco_labels.len(), 80);
        assert_eq!(metadata.coco_keypoints.len(), 17);
        assert_eq!(metadata.dota_labels.len(), 15);
        assert_eq!(metadata.ade20k_labels.len(), 150);
        assert_eq!(metadata.nlu_labels.len(), 14);
    }
}
