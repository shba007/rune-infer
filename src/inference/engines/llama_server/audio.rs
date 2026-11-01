use super::{LlamaServerEngine, upstream_usage};
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{AudioTranscriptionResponse, Usage};
use base64::Engine;
use std::error::Error;

pub(crate) fn validate_task(task: &InferenceTaskRequest) -> Result<(), &'static str> {
    match task {
        InferenceTaskRequest::AudioTranscription {
            response_format, ..
        } => {
            if !matches!(response_format.as_deref(), None | Some("json" | "text")) {
                return Err(
                    "llama-server transcription supports json and text; timestamp formats are not implemented",
                );
            }
        }
        InferenceTaskRequest::AudioSpeech {
            voice,
            response_format,
            speed,
            ..
        } => {
            if !matches!(response_format.as_deref(), None | Some("wav")) {
                return Err("llama-server speech currently supports WAV only");
            }
            if !matches!(voice.as_deref(), None | Some("female" | "male")) {
                return Err("llama-server speech currently supports female and male voices only");
            }
            if speed.is_some_and(|value| value != 1.0) {
                return Err("llama-server speech currently supports normal speed (1.0) only");
            }
        }
        _ => {}
    }
    Ok(())
}

impl LlamaServerEngine {
    pub(super) fn execute_audio(
        &self,
        task: &InferenceTaskRequest,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        validate_task(task)?;
        match task {
            InferenceTaskRequest::AudioTranscription {
                audio_bytes,
                language,
                prompt,
                temperature,
                response_format,
                ..
            } => {
                let mut instruction = match language.as_deref().map(str::trim) {
                    Some(value) if !value.is_empty() && value != "auto" => {
                        format!("Perform ASR in {value}.")
                    }
                    _ => "Perform ASR in the original language.".to_string(),
                };
                if let Some(context) = prompt.as_deref().filter(|value| !value.trim().is_empty()) {
                    instruction.push_str("\nContext: ");
                    instruction.push_str(context);
                }
                let body = serde_json::json!({
                    "messages": [
                        {"role":"system", "content":instruction},
                        {"role":"user", "content":[{"type":"input_audio", "input_audio":{
                            "data":base64::engine::general_purpose::STANDARD.encode(audio_bytes)
                        }}]}
                    ],
                    "temperature":temperature.unwrap_or(0.0), "max_tokens":1024, "stream":false
                });
                let response = self
                    .client
                    .post(format!(
                        "http://127.0.0.1:{}/v1/chat/completions",
                        self.port
                    ))
                    .json(&body)
                    .send()?;
                if !response.status().is_success() {
                    return Err(
                        format!("llama-server transcription failed: {}", response.text()?).into(),
                    );
                }
                let value: serde_json::Value = response.json()?;
                let choice = &value["choices"][0];
                if choice["finish_reason"] != "stop" {
                    return Err(
                        "llama-server transcription did not complete; try a shorter audio clip"
                            .into(),
                    );
                }
                let text = choice["message"]["content"]
                    .as_str()
                    .filter(|text| !text.trim().is_empty())
                    .ok_or("llama-server returned an empty transcript")?
                    .to_string();
                let usage = upstream_usage(&value).unwrap_or_default();
                let response = if response_format.as_deref() == Some("text") {
                    InferenceTaskResponse::Text(text)
                } else {
                    InferenceTaskResponse::Transcription(AudioTranscriptionResponse {
                        text,
                        language: None,
                        duration: None,
                        segments: None,
                        words: None,
                        speakers: None,
                        speaker_activity: None,
                    })
                };
                Ok(InferenceOutput { response, usage })
            }
            InferenceTaskRequest::AudioSpeech { input, voice, .. } => {
                let mut body = serde_json::json!({"input":input, "response_format":"wav"});
                if let Some(voice) = voice {
                    body["voice"] = voice.clone().into();
                }
                let response = self
                    .client
                    .post(format!("http://127.0.0.1:{}/tts", self.port))
                    .json(&body)
                    .send()?;
                if !response.status().is_success() {
                    return Err(format!("llama-server speech failed: {}", response.text()?).into());
                }
                let bytes = response.bytes()?.to_vec();
                if bytes.len() <= 44 || !bytes.starts_with(b"RIFF") || &bytes[8..12] != b"WAVE" {
                    return Err("llama-server did not return nonempty WAV audio".into());
                }
                Ok(InferenceOutput {
                    response: InferenceTaskResponse::Audio(bytes),
                    usage: Usage::default(),
                })
            }
            _ => Err("Unsupported llama-server audio task".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lfm_audio_options_accept_only_implemented_controls() {
        for (options, accepted) in [
            (json!({}), true),
            (
                json!({"voice":"female","speed":1.0,"response_format":"wav"}),
                true,
            ),
            (json!({"voice":"male"}), true),
            (json!({"voice":"alloy"}), false),
            (json!({"speed":1.1}), false),
            (json!({"response_format":"pcm"}), false),
            (json!({"response_format":"mp3"}), false),
        ] {
            let mut task = json!({"type":"audio_speech","input":"Hello"});
            task.as_object_mut()
                .unwrap()
                .extend(options.as_object().unwrap().clone());
            let task: InferenceTaskRequest = serde_json::from_value(task).unwrap();
            assert_eq!(validate_task(&task).is_ok(), accepted);
        }
        for (format, accepted) in [
            (None, true),
            (Some("json"), true),
            (Some("text"), true),
            (Some("verbose_json"), false),
            (Some("srt"), false),
            (Some("vtt"), false),
        ] {
            let task = InferenceTaskRequest::AudioTranscription {
                audio_bytes: vec![1],
                filename: "fixture.wav".into(),
                language: None,
                prompt: None,
                temperature: None,
                response_format: format.map(String::from),
            };
            assert_eq!(validate_task(&task).is_ok(), accepted);
        }
    }
}
