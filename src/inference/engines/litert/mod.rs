//! CPU-only Depth Anything V2 Small adapter; native code runs in a child process.
pub mod native;
use crate::{
    config::ModelConfig,
    inference::{
        traits::InferenceEngine,
        types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse},
    },
    types::Usage,
};
use std::{
    error::Error,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};
pub const WIDTH: u32 = 686;
pub const HEIGHT: u32 = 518;
pub const PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const RUNTIME_SHA: &str = "6fe42cfa518879dd7bae4500244cf371d949bf0c8559cb60be207d81112da16d";
const RUNTIME_URL: &str =
    "https://storage.googleapis.com/litert/binaries/2.3.0/windows_x86_64/libLiteRt.dll";
const MODEL_SHAS: [&str; 1] = ["651415f1774e02d06de946eccf2e6dbbc5f499a9e932d9de7ada3d59637d2e8e"];

pub fn verify_artifacts(runtime: &Path, model: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        cfg!(all(windows, target_arch = "x86_64")),
        "LiteRT depth currently requires Windows x64 CPU"
    );
    validate_runtime(runtime).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    verify_model(model)
}

/// Identify the reviewed export by content, independently of its filename.
pub(crate) fn verify_model(model: &Path) -> anyhow::Result<()> {
    let mut header = [0; 8];
    std::fs::File::open(model)?.read_exact(&mut header)?;
    anyhow::ensure!(&header[4..] == b"TFL3", "not a TFLite model");
    let hash = super::download::file_sha256(model).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    anyhow::ensure!(
        MODEL_SHAS.contains(&hash.as_str()),
        "Unreviewed LiteRT model artifact; this cycle supports only the pinned Depth Anything V2 Small exports"
    );
    Ok(())
}

pub fn runtime_path() -> anyhow::Result<PathBuf> {
    let path = std::env::var_os("LITERT_LIBRARY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(managed_runtime_path);
    anyhow::ensure!(
        path.is_file(),
        "LiteRT runtime is missing; install it from Engines (e, then u), or configure LITERT_LIBRARY_PATH with the reviewed Windows x64 2.3.0 libLiteRt.dll"
    );
    Ok(std::fs::canonicalize(path)?)
}

fn managed_runtime_path() -> PathBuf {
    let directory = super::download::engine_bin().join("litert");
    let direct = directory.join("libLiteRt.dll");
    if direct.is_file() {
        direct
    } else {
        super::download::find_executable_recursive(&directory, "libLiteRt.dll").unwrap_or(direct)
    }
}

pub(crate) fn validate_runtime(path: &Path) -> Result<(), Box<dyn Error>> {
    if super::download::file_sha256(path)? != RUNTIME_SHA {
        return Err(
            "LiteRT requires the pinned official Windows x64 2.3.0 runtime; checksum mismatch"
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn ensure_runtime_installed() -> Result<PathBuf, Box<dyn Error>> {
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("LiteRT installation currently supports Windows x64 CPU only".into());
    }
    if let Some(path) = std::env::var_os("LITERT_LIBRARY_PATH") {
        let path = PathBuf::from(path);
        validate_runtime(&path).map_err(|e| format!(
            "LITERT_LIBRARY_PATH must point to the pinned runtime; fix or remove this override before installing: {e}"
        ))?;
        return Ok(std::fs::canonicalize(path)?);
    }
    let path = managed_runtime_path();
    if path.exists() {
        validate_runtime(&path)?;
        return Ok(std::fs::canonicalize(path)?);
    }
    std::fs::create_dir_all(path.parent().ok_or("Missing LiteRT directory")?)?;
    let client = super::download::create_download_client()?;
    super::download::download_verified(&client, RUNTIME_URL, &path, "[LiteRT]", RUNTIME_SHA)?;
    validate_runtime(&path)?;
    Ok(std::fs::canonicalize(path)?)
}

#[cfg(feature = "tui")]
pub(crate) fn runtime_version(path: &Path) -> Option<String> {
    super::download::file_sha256(path)
        .ok()
        .filter(|hash| hash == RUNTIME_SHA)
        .map(|_| "LiteRT 2.3.0 CPU".into())
}

pub(crate) fn helper_path() -> Result<PathBuf, Box<dyn Error>> {
    let helper = if let Some(path) = std::env::var_os("LITERT_HELPER_PATH") {
        PathBuf::from(path)
    } else {
        let path = std::env::current_exe()?;
        if !path.file_stem().is_some_and(|name| name == "rune-infer") {
            return Err(
                "Embedding hosts must set LITERT_HELPER_PATH to the rune-infer executable".into(),
            );
        }
        path
    };
    Ok(std::fs::canonicalize(helper)?)
}

pub struct LiteRtEngine {
    id: String,
    runtime: PathBuf,
    model: PathBuf,
    helper: PathBuf,
    serial: Mutex<()>,
}
impl LiteRtEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let runtime = runtime_path()?;
        let path = std::fs::canonicalize(&model.model_path)?;
        verify_artifacts(&runtime, &path)?;
        let helper = helper_path()?;
        Ok(Self {
            id: model.id.clone(),
            runtime,
            model: path,
            helper: std::fs::canonicalize(helper)?,
            serial: Mutex::new(()),
        })
    }
    fn infer(&self, input: Vec<u8>) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut command = Command::new(&self.helper);
        super::super::process::configure_death_signal(&mut command);
        command
            .arg("litert-worker")
            .arg("--runtime")
            .arg(&self.runtime)
            .arg("--model")
            .arg(&self.model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn()?;
        let mut stdin = child.stdin.take().ok_or("Missing worker input pipe")?;
        let stdout = child.stdout.take().ok_or("Missing worker output pipe")?;
        let stderr = child
            .stderr
            .take()
            .ok_or("Missing worker diagnostic pipe")?;
        let guard = super::super::process::ProcessGuard::new(child, format!("litert-{}", self.id));
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let output = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout
                .take((PIXELS * 4 + 1) as u64)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        });
        let diagnostics = std::thread::spawn(move || {
            let mut reader = stderr;
            let mut saved = Vec::new();
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    break;
                }
                if saved.len() < 8192 {
                    saved.extend_from_slice(&buf[..n.min(8192 - saved.len())]);
                }
            }
            saved
        });
        let started = Instant::now();
        let status = loop {
            if let Some(status) = guard
                .lock()
                .map_err(|_| "Worker lock poisoned")?
                .try_wait()?
            {
                break status;
            }
            if started.elapsed() > Duration::from_secs(120) {
                return Err("LiteRT CPU inference timed out; worker stopped".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let diagnostics = diagnostics
            .join()
            .map_err(|_| "Worker diagnostic reader failed")?;
        if !status.success() {
            return Err(format!(
                "LiteRT worker failed ({status}): {}",
                String::from_utf8_lossy(&diagnostics)
            )
            .into());
        }
        writer.join().map_err(|_| "Worker input writer failed")??;
        let bytes = output.join().map_err(|_| "Worker output reader failed")??;
        decode_depth(&bytes)
    }
}

fn decode_depth(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    if bytes.len() != PIXELS * 4 {
        return Err("LiteRT returned an incorrect depth map size".into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect();
    if !values.iter().all(|value| value.is_finite()) {
        return Err("LiteRT returned non-finite depth values".into());
    }
    Ok(values)
}

fn preprocess(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), Box<dyn Error>> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    reader.limits(limits);
    let image = reader.decode()?;
    let (width, height) = (image.width(), image.height());
    let rgb = image::imageops::resize(
        &image.to_rgb8(),
        WIDTH,
        HEIGHT,
        image::imageops::FilterType::CatmullRom,
    );
    let mut input = Vec::with_capacity(PIXELS * 3 * 4);
    for channel in 0..3 {
        for pixel in rgb.pixels() {
            let value = (pixel[channel] as f32 / 255.0 - [0.485, 0.456, 0.406][channel])
                / [0.229, 0.224, 0.225][channel];
            input.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok((input, width, height))
}

impl InferenceEngine for LiteRtEngine {
    fn id(&self) -> &str {
        &self.id
    }
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        let InferenceTaskRequest::ImageDepth {
            image_bytes,
            preview,
            ..
        } = task
        else {
            return Err("LiteRT implements only image depth estimation in this cycle".into());
        };
        let _lock = self
            .serial
            .lock()
            .map_err(|_| "LiteRT inference lock poisoned")?;
        let (input, source_width, source_height) = preprocess(image_bytes)?;
        let values = self.infer(input)?;
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let preview_png_base64 = if *preview {
            use base64::Engine;
            let normalized: Vec<u16> = values
                .iter()
                .map(|value| {
                    if max > min {
                        ((value - min) / (max - min) * 65535.0).clamp(0.0, 65535.0) as u16
                    } else {
                        0
                    }
                })
                .collect();
            let image =
                image::ImageBuffer::<image::Luma<u16>, _>::from_raw(WIDTH, HEIGHT, normalized)
                    .ok_or("Invalid preview dimensions")?;
            let mut bytes = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageLuma16(image)
                .write_to(&mut bytes, image::ImageFormat::Png)?;
            Some(base64::engine::general_purpose::STANDARD.encode(bytes.into_inner()))
        } else {
            None
        };
        Ok(InferenceOutput {
            response: InferenceTaskResponse::Depth(crate::types::ImageDepthResponse {
                width: WIDTH,
                height: HEIGHT,
                source_width,
                source_height,
                units: "relative".into(),
                dtype: "float32".into(),
                layout: "row-major".into(),
                values,
                min,
                max,
                preview_png_base64,
            }),
            usage: Usage::new(0, 0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preprocessing_is_rgb_nchw_float32_and_depth_output_is_bounded_and_finite() {
        let image = image::RgbImage::from_pixel(2, 3, image::Rgb([255, 0, 128]));
        let mut png = std::io::Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let (input, width, height) = preprocess(png.get_ref()).unwrap();
        assert_eq!((width, height), (2, 3));
        assert_eq!(input.len(), PIXELS * 3 * 4);
        for (channel, expected) in [
            (0, (1.0 - 0.485) / 0.229),
            (1, (0.0 - 0.456) / 0.224),
            (2, (128.0 / 255.0 - 0.406) / 0.225),
        ] {
            let actual = f32::from_le_bytes(
                input[channel * PIXELS * 4..channel * PIXELS * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(preprocess(b"not an image").is_err());
        assert!(decode_depth(&[0; 4]).is_err());
        let mut output = vec![0u8; PIXELS * 4];
        assert!(decode_depth(&output).is_ok());
        output[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_depth(&output).is_err());
    }
}
