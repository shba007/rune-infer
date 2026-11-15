//! Text-to-video through sd-server's native asynchronous `/sdcpp/v1/vid_gen` API.
use super::SdServerEngine;
use crate::inference::types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse};
use crate::types::{Usage, VideoData, VideoGenerationResponse};
use std::error::Error;
use std::time::{Duration, Instant};

const DEFAULT_SIZE: (u32, u32) = (832, 480);
const DEFAULT_FRAMES: u32 = 33;
const DEFAULT_FPS: u32 = 24;
/// Upper bound for one clip, including queueing, sampling and encoding.
const JOB_DEADLINE: Duration = Duration::from_secs(60 * 60);

fn parse_size(size: Option<&str>) -> (u32, u32) {
    size.and_then(|value| {
        let (w, h) = value.split_once(['x', 'X', '×', '*'])?;
        Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
    })
    .unwrap_or(DEFAULT_SIZE)
}

impl SdServerEngine {
    pub(super) fn execute_video(
        &self,
        task: &InferenceTaskRequest,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        let InferenceTaskRequest::VideoGeneration {
            prompt,
            negative_prompt,
            size,
            frames,
            fps,
            steps,
            cfg_scale,
            seed,
            sample_method,
            output_format,
        } = task
        else {
            return Err("Not a video generation task".into());
        };
        let (width, height) = parse_size(size.as_deref());
        let frames = frames.unwrap_or(DEFAULT_FRAMES);
        let fps = fps.unwrap_or(DEFAULT_FPS);
        let format = output_format.as_deref().unwrap_or("webm");
        let steps = steps.unwrap_or(self.default_steps);
        let cfg = cfg_scale.unwrap_or(self.default_cfg);

        let mut body = serde_json::json!({
            "prompt": prompt.trim(),
            "width": width,
            "height": height,
            "video_frames": frames,
            "fps": fps,
            "output_format": format,
            "sample_params": super::native_sampling_args(steps, cfg, sample_method.as_deref()),
        });
        if let Some(seed) = seed.filter(|seed| *seed >= 0) {
            body["seed"] = serde_json::json!(seed);
        }
        if let Some(negative) = negative_prompt.as_deref().map(str::trim)
            && !negative.is_empty()
        {
            body["negative_prompt"] = serde_json::json!(negative);
        }

        tracing::info!(
            "[SdServerEngine] Generating video ({width}x{height} | {frames} frames @ {fps} fps | {steps} steps | CFG {cfg:.1})..."
        );
        let base = format!("http://127.0.0.1:{}", self.port);
        let submit = self
            .client
            .post(format!("{base}/sdcpp/v1/vid_gen"))
            .json(&body)
            .send()?;
        if !submit.status().is_success() {
            let text = submit.text().unwrap_or_default();
            return Err(format!("sd-server rejected the video request: {text}").into());
        }
        let job: serde_json::Value = submit.json()?;
        let poll_url = job["poll_url"]
            .as_str()
            .map(|path| format!("{base}{path}"))
            .or_else(|| {
                job["id"]
                    .as_str()
                    .map(|id| format!("{base}/sdcpp/v1/jobs/{id}"))
            })
            .ok_or("sd-server did not return a job to poll")?;

        let started = Instant::now();
        let result = loop {
            if started.elapsed() > JOB_DEADLINE {
                return Err("Timed out waiting for the video job".into());
            }
            std::thread::sleep(Duration::from_secs(2));
            let response = self.client.get(&poll_url).send()?;
            if !response.status().is_success() {
                let text = response.text().unwrap_or_default();
                return Err(format!("sd-server job poll failed: {text}").into());
            }
            let state: serde_json::Value = response.json()?;
            match state["status"].as_str() {
                Some("completed") => break state["result"].clone(),
                Some("failed") | Some("cancelled") => {
                    let message = state["error"]["message"]
                        .as_str()
                        .unwrap_or("video generation failed");
                    return Err(format!("sd-server video job failed: {message}").into());
                }
                _ => {}
            }
        };

        let b64_json = result["b64_json"]
            .as_str()
            .ok_or("sd-server returned no video data")?
            .to_string();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(InferenceOutput {
            response: InferenceTaskResponse::Video(VideoGenerationResponse {
                created,
                data: vec![VideoData {
                    b64_json,
                    mime_type: result["mime_type"].as_str().unwrap_or("video/webm").into(),
                    output_format: result["output_format"].as_str().unwrap_or(format).into(),
                    fps: result["fps"].as_u64().map_or(fps, |v| v as u32),
                    frame_count: result["frame_count"].as_u64().map_or(frames, |v| v as u32),
                }],
            }),
            usage: Usage::new(prompt.len() as u32 / 4, 1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_parsing_falls_back_to_the_default() {
        assert_eq!(parse_size(Some("704x1280")), (704, 1280));
        assert_eq!(parse_size(Some("bad")), DEFAULT_SIZE);
        assert_eq!(parse_size(None), DEFAULT_SIZE);
    }
}
