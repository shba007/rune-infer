//! Verified operations are independent of primitive input/output tasks.
use crate::config::{EngineKind, ModelConfig};
use serde::{Deserialize, Serialize};

pub(crate) fn checked_engine_version(path: &std::path::Path) -> Result<String, String> {
    let output = std::process::Command::new(path)
        .arg("--version")
        .output()
        .map_err(|error| format!("Cannot read checked engine version: {error}"))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let version = text.lines().map(str::trim).find(|line| {
        let lower = line.to_ascii_lowercase();
        (lower.starts_with("version")
            || lower.starts_with("audio.cpp ")
            || lower.starts_with("nemo-speech ")
            || lower.starts_with("whisper.cpp ")
            || lower.as_bytes().first().is_some_and(u8::is_ascii_digit))
            && line.bytes().any(|byte| byte.is_ascii_digit())
    });
    if !output.status.success() || version.is_none() {
        return Err("The checked engine did not report a usable version".into());
    }
    let mut version = version.unwrap().to_string();
    if let Some(commit) = text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("git:") || line.starts_with("git sha"))
    {
        version.push_str("; ");
        version.push_str(commit);
    }
    Ok(version)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SupportStatus {
    Supported,
    Unsupported,
    #[default]
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineVerification {
    pub engine: EngineKind,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct OperationSupport {
    pub status: SupportStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub verified_with: Vec<EngineVerification>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioOperation {
    Transcription,
    SpeechTranslation,
    SpeechGeneration,
}

impl AudioOperation {
    pub const ALL: [Self; 3] = [
        Self::Transcription,
        Self::SpeechTranslation,
        Self::SpeechGeneration,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transcription => "transcription",
            Self::SpeechTranslation => "speech_translation",
            Self::SpeechGeneration => "speech_generation",
        }
    }

    pub fn implemented_by(self, engine: EngineKind) -> bool {
        use EngineKind::*;
        match self {
            Self::Transcription => matches!(
                engine,
                LlamaServer | LlamaServerPrism | AudioCpp | NemoSpeech | WhisperCpp | Crispasr
            ),
            Self::SpeechTranslation => matches!(engine, NemoSpeech | WhisperCpp | Crispasr),
            Self::SpeechGeneration => {
                matches!(engine, LlamaServer | LlamaServerPrism | AudioCpp | Crispasr)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct AudioOperations {
    pub transcription: OperationSupport,
    pub speech_translation: OperationSupport,
    pub speech_generation: OperationSupport,
}

impl AudioOperations {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub fn get(&self, operation: AudioOperation) -> &OperationSupport {
        match operation {
            AudioOperation::Transcription => &self.transcription,
            AudioOperation::SpeechTranslation => &self.speech_translation,
            AudioOperation::SpeechGeneration => &self.speech_generation,
        }
    }

    pub fn get_mut(&mut self, operation: AudioOperation) -> &mut OperationSupport {
        match operation {
            AudioOperation::Transcription => &mut self.transcription,
            AudioOperation::SpeechTranslation => &mut self.speech_translation,
            AudioOperation::SpeechGeneration => &mut self.speech_generation,
        }
    }

    pub fn validate(&self, engines: &[EngineKind]) -> Result<(), String> {
        for operation in AudioOperation::ALL {
            let value = self.get(operation);
            if value.status == SupportStatus::Supported && value.verified_with.is_empty() {
                return Err(format!(
                    "{} needs live engine verification",
                    operation.as_str()
                ));
            }
            if value.status != SupportStatus::Supported && !value.verified_with.is_empty() {
                return Err(format!(
                    "{} cannot have successful checks without supported status",
                    operation.as_str()
                ));
            }
            if value.status == SupportStatus::Unsupported
                && value.reason.as_deref().is_none_or(|s| s.trim().is_empty())
            {
                return Err(format!(
                    "{} needs an unsupported reason",
                    operation.as_str()
                ));
            }
            for check in &value.verified_with {
                if check.version.trim().is_empty()
                    || !engines.contains(&check.engine)
                    || !operation.implemented_by(check.engine)
                {
                    return Err(format!(
                        "{} has invalid engine verification",
                        operation.as_str()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Merge successful setup checks without replacing an explicit unsupported policy.
    pub fn merge_verified(&mut self, incoming: &Self) -> Result<bool, String> {
        let mut result = self.clone();
        for operation in AudioOperation::ALL {
            let new = incoming.get(operation);
            if new.status != SupportStatus::Supported {
                continue;
            }
            let old = result.get_mut(operation);
            if old.status == SupportStatus::Unsupported {
                return Err(format!(
                    "{} conflicts with an existing unsupported declaration",
                    operation.as_str()
                ));
            }
            old.status = SupportStatus::Supported;
            old.reason = None;
            for check in &new.verified_with {
                old.verified_with
                    .retain(|previous| previous.engine != check.engine);
                old.verified_with.push(check.clone());
            }
        }
        let changed = result != *self;
        *self = result;
        Ok(changed)
    }
}

impl ModelConfig {
    pub fn resolved_audio_operation(
        &self,
        operation: AudioOperation,
        engine: Option<EngineKind>,
    ) -> OperationSupport {
        let mut value = self.audio_operations.get(operation).clone();
        let resolve = |status, reason: String| OperationSupport {
            status,
            reason: Some(reason),
            verified_with: value.verified_with.clone(),
        };
        if value.status == SupportStatus::Unsupported {
            return value;
        }
        if !self.is_local() {
            return resolve(
                SupportStatus::Unverified,
                "Cloud audio operations have not been live-verified".into(),
            );
        }
        let Some(engine) = engine.or_else(|| self.engine_plan().first().copied()) else {
            return resolve(
                SupportStatus::Unverified,
                "No execution engine is configured".into(),
            );
        };
        if !operation.implemented_by(engine) {
            return resolve(
                SupportStatus::Unsupported,
                format!(
                    "Engine '{engine}' does not implement {}",
                    operation.as_str()
                ),
            );
        }
        let (input, output) = if operation == AudioOperation::SpeechGeneration {
            ("text", "audio")
        } else {
            ("audio", "text")
        };
        if !self.tasks.iter().any(|task| {
            task.split_once("-to-").is_some_and(|(inputs, outputs)| {
                outputs == output && inputs.split('-').any(|v| v == input)
            })
        }) {
            return resolve(
                SupportStatus::Unverified,
                format!("No verified {input}-to-{output} task establishes this operation"),
            );
        }
        if value.status != SupportStatus::Supported
            || !value
                .verified_with
                .iter()
                .any(|check| check.engine == engine)
        {
            return resolve(
                SupportStatus::Unverified,
                format!(
                    "{} has not passed a live check with '{engine}'",
                    operation.as_str()
                ),
            );
        }
        if !std::path::Path::new(&self.model_path).exists() {
            return resolve(
                SupportStatus::Unsupported,
                "The configured model artifact is missing".into(),
            );
        }
        if matches!(
            engine,
            EngineKind::LlamaServer | EngineKind::LlamaServerPrism
        ) {
            let projector = self
                .mmproj_path
                .as_deref()
                .and_then(|p| crate::detect::read_gguf(std::path::Path::new(p)).ok());
            let available = projector.is_some_and(|info| {
                if operation == AudioOperation::SpeechGeneration {
                    info.has_gen_audio_encoder == Some(true)
                } else {
                    info.has_audio_encoder == Some(true)
                }
            });
            if !available {
                return resolve(
                    SupportStatus::Unsupported,
                    "The configured projector does not establish support for this audio operation"
                        .into(),
                );
            }
        }
        value.reason = Some(format!(
            "Live-verified {} with '{engine}'",
            operation.as_str()
        ));
        value
    }

    pub fn resolved_audio_operations(&self, engine: Option<EngineKind>) -> AudioOperations {
        AudioOperations {
            transcription: self.resolved_audio_operation(AudioOperation::Transcription, engine),
            speech_translation: self
                .resolved_audio_operation(AudioOperation::SpeechTranslation, engine),
            speech_generation: self
                .resolved_audio_operation(AudioOperation::SpeechGeneration, engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_schema_defaults_to_unverified_and_preserves_explicit_evidence() {
        let legacy = json!({"id":"speech","name":"Speech","engine":["crispasr"],"tasks":["audio-to-text"],"local":{"path":"Cargo.toml"}});
        let mut entry: crate::schema::ModelEntry = serde_json::from_value(legacy.clone()).unwrap();
        assert!(entry.audio_operations.is_empty());
        assert!(
            serde_json::to_value(&entry)
                .unwrap()
                .get("audio_operations")
                .is_none()
        );
        entry.audio_operations.transcription = OperationSupport {
            status: SupportStatus::Supported,
            reason: None,
            verified_with: vec![EngineVerification {
                engine: EngineKind::Crispasr,
                version: "fixture-1".into(),
            }],
        };
        let saved = serde_json::to_value(&entry).unwrap();
        let loaded: crate::schema::ModelEntry = serde_json::from_value(saved.clone()).unwrap();
        assert_eq!(loaded.audio_operations, entry.audio_operations);
        let model = ModelConfig::try_from(loaded).unwrap();
        assert_eq!(
            model
                .resolved_audio_operation(AudioOperation::Transcription, None)
                .status,
            SupportStatus::Supported
        );
        assert_eq!(
            model
                .resolved_audio_operation(AudioOperation::SpeechTranslation, None)
                .status,
            SupportStatus::Unverified
        );
        assert_eq!(
            model
                .resolved_audio_operation(AudioOperation::Transcription, Some(EngineKind::AudioCpp))
                .status,
            SupportStatus::Unverified
        );
        for bad in [
            json!({"status":"supported"}),
            json!({"status":"unsupported"}),
            json!({"status":"supported","verified_with":[{"engine":"audio-cpp","version":"fixture"}]}),
            json!({"status":"supported","verified_with":[{"engine":"crispasr","version":""}]}),
        ] {
            let mut value = saved.clone();
            value["audio_operations"]["transcription"] = bad;
            assert!(
                ModelConfig::try_from(
                    serde_json::from_value::<crate::schema::ModelEntry>(value).unwrap()
                )
                .is_err()
            );
        }
        let mut unsupported = AudioOperations::default();
        unsupported.transcription.status = SupportStatus::Unsupported;
        unsupported.transcription.reason = Some("Owner policy".into());
        let previous = unsupported.clone();
        assert!(unsupported.merge_verified(&entry.audio_operations).is_err());
        assert_eq!(unsupported, previous);
        let mut merged = AudioOperations::default();
        assert!(merged.merge_verified(&entry.audio_operations).unwrap());
        assert!(!merged.merge_verified(&entry.audio_operations).unwrap());
    }
}
