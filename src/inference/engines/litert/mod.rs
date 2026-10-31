//! Reviewed depth adapters; native CPU/GPU code runs in isolated child processes.
mod gpu_runtime;
pub mod native;
pub mod native_gpu;
mod resize;
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
pub(crate) struct ModelProfile {
    pub sha256: &'static str,
    pub architecture: &'static str,
    pub name: &'static str,
    pub source: &'static str,
    reference_resize: bool,
    gpu_verified: bool,
}
const MODEL_PROFILES: [ModelProfile; 3] = [
    ModelProfile {
        sha256: "651415f1774e02d06de946eccf2e6dbbc5f499a9e932d9de7ada3d59637d2e8e",
        architecture: "depth-anything-v2-small",
        name: "Depth Anything V2 Small FP32",
        source: "https://huggingface.co/litert-community/depth-anything-v2-small/tree/178427e448dbf4da93b1e7b1b2abc103ad329bd6/tflite",
        reference_resize: false,
        gpu_verified: true,
    },
    ModelProfile {
        sha256: "6f03564f13abfae18185581a809cb5c0ee5c0a425dc5f85102194b2b321eba3d",
        architecture: "depth-anything-v2-large",
        name: "Depth Anything V2 Large FP32",
        source: "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/tree/7581137eff8d4e94f6e796d3baea0e9fa79b22d2 (local export using tools/model-converter; CC-BY-NC-4.0)",
        reference_resize: true,
        gpu_verified: false,
    },
    ModelProfile {
        sha256: "1a4bf05de63bb60a450ad8b8de834a612d6de748df5095785976c58605a7d705",
        architecture: "depth-anything-v2-large",
        name: "Depth Anything V2 Large FP32",
        source: "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/tree/7581137eff8d4e94f6e796d3baea0e9fa79b22d2 (local LayerScale-folded export; CC-BY-NC-4.0)",
        reference_resize: true,
        gpu_verified: true,
    },
];

pub fn verify_gpu_artifacts(runtime: &Path, model: &Path) -> anyhow::Result<()> {
    let profile = checked_profile(runtime, model)?;
    anyhow::ensure!(
        profile.gpu_verified,
        "Model export has not passed GPU parity checks"
    );
    gpu_runtime::validate_bundle(
        runtime
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Missing GPU runtime directory"))?,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

pub fn verify_artifacts(runtime: &Path, model: &Path) -> anyhow::Result<()> {
    checked_profile(runtime, model).map(|_| ())
}

fn checked_profile(runtime: &Path, model: &Path) -> anyhow::Result<&'static ModelProfile> {
    anyhow::ensure!(
        cfg!(all(windows, target_arch = "x86_64")),
        "LiteRT depth currently requires Windows x64"
    );
    validate_runtime(runtime).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    reviewed_model(model)
}

/// Identify the reviewed export by content, independently of its filename.
pub(crate) fn reviewed_model(model: &Path) -> anyhow::Result<&'static ModelProfile> {
    let mut header = [0; 8];
    std::fs::File::open(model)?.read_exact(&mut header)?;
    anyhow::ensure!(&header[4..] == b"TFL3", "not a TFLite model");
    let hash = super::download::file_sha256(model).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    MODEL_PROFILES.iter().find(|profile| profile.sha256 == hash).ok_or_else(||
        anyhow::anyhow!("Unreviewed LiteRT model artifact; only checksum-pinned Depth Anything V2 FP32 exports are supported"))
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
        return Err("LiteRT installation currently supports Windows x64 only".into());
    }
    if let Some(path) = std::env::var_os("LITERT_LIBRARY_PATH") {
        let path = PathBuf::from(path);
        validate_runtime(&path).map_err(|e| format!(
            "LITERT_LIBRARY_PATH must point to the pinned runtime; fix or remove this override before installing: {e}"
        ))?;
        gpu_runtime::ensure_installed(&path)?;
        return Ok(std::fs::canonicalize(path)?);
    }
    let path = managed_runtime_path();
    if path.exists() {
        validate_runtime(&path)?;
        gpu_runtime::ensure_installed(&path)?;
        return Ok(std::fs::canonicalize(path)?);
    }
    std::fs::create_dir_all(path.parent().ok_or("Missing LiteRT directory")?)?;
    let client = super::download::create_download_client()?;
    super::download::download_verified(&client, RUNTIME_URL, &path, "[LiteRT]", RUNTIME_SHA)?;
    validate_runtime(&path)?;
    gpu_runtime::ensure_installed(&path)?;
    Ok(std::fs::canonicalize(path)?)
}

#[cfg(feature = "tui")]
pub(crate) fn runtime_version(path: &Path) -> Option<String> {
    super::download::file_sha256(path)
        .ok()
        .filter(|hash| hash == RUNTIME_SHA)
        .map(|_| {
            if gpu_runtime::validate_bundle(&gpu_runtime::bundle_path()).is_ok() {
                "LiteRT 2.3.0 GPU + CPU".into()
            } else {
                "LiteRT 2.3.0 CPU".into()
            }
        })
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

pub(crate) fn verification_files() -> Vec<PathBuf> {
    let mut files = runtime_path().into_iter().collect::<Vec<_>>();
    let bundle = gpu_runtime::bundle_path();
    if gpu_runtime::validate_bundle(&bundle).is_ok() {
        for name in [
            "libLiteRt.dll",
            "libLiteRtWebGpuAccelerator.dll",
            "dxcompiler.dll",
            "dxil.dll",
        ] {
            files.push(bundle.join(name));
        }
    }
    files
}

pub struct LiteRtEngine {
    id: String,
    runtime: PathBuf,
    model: PathBuf,
    helper: PathBuf,
    serial: Mutex<()>,
    profile: &'static ModelProfile,
    gpu_runtime: Option<PathBuf>,
}
impl LiteRtEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        let runtime = runtime_path()?;
        let path = std::fs::canonicalize(&model.model_path)?;
        let profile = checked_profile(&runtime, &path)?;
        let helper = helper_path()?;
        let gpu_runtime = if profile.gpu_verified
            && model
                .runtime
                .as_ref()
                .is_none_or(|policy| policy.gpu_layers > 0)
        {
            match gpu_runtime::ensure_installed(&runtime) {
                Ok(path) => Some(path),
                Err(error) => {
                    tracing::warn!("LiteRT GPU runtime unavailable; using CPU: {error}");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            id: model.id.clone(),
            runtime,
            model: path,
            helper: std::fs::canonicalize(helper)?,
            serial: Mutex::new(()),
            profile,
            gpu_runtime,
        })
    }
    fn infer(&self, input: Vec<u8>) -> Result<Vec<f32>, Box<dyn Error>> {
        if let Some(runtime) = &self.gpu_runtime {
            match self.infer_worker(input.clone(), runtime, true) {
                Ok(values) => return Ok(values),
                Err(error) => tracing::warn!("LiteRT GPU attempt failed; retrying on CPU: {error}"),
            }
        }
        self.infer_worker(input, &self.runtime, false)
    }
    fn infer_worker(
        &self,
        input: Vec<u8>,
        runtime: &Path,
        gpu: bool,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut command = Command::new(&self.helper);
        super::super::process::configure_death_signal(&mut command);
        command
            .arg(if gpu {
                "litert-gpu-worker"
            } else {
                "litert-worker"
            })
            .arg("--runtime")
            .arg(runtime)
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
                if saved.len() < 65536 {
                    saved.extend_from_slice(&buf[..n.min(65536 - saved.len())]);
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
                return Err("LiteRT inference timed out; worker stopped".into());
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
        if gpu {
            let diagnostic = String::from_utf8_lossy(&diagnostics);
            if !diagnostic.contains("adapterType=Discrete GPU")
                && !diagnostic.contains("adapterType=Integrated GPU")
            {
                return Err("LiteRT did not select a physical GPU".into());
            }
            tracing::info!("LiteRT depth completed on GPU (CPU operations permitted)");
        } else {
            tracing::info!("LiteRT depth completed on CPU");
        }
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

fn preprocess(bytes: &[u8], profile: &ModelProfile) -> Result<(Vec<u8>, u32, u32), Box<dyn Error>> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    reader.limits(limits);
    let image = reader.decode()?;
    let (width, height) = (image.width(), image.height());
    let rgb = if profile.reference_resize {
        resize::reference_bicubic(&image.to_rgb8(), WIDTH, HEIGHT)
    } else {
        image::imageops::resize(
            &image.to_rgb8(),
            WIDTH,
            HEIGHT,
            image::imageops::FilterType::CatmullRom,
        )
    };
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
        let (input, source_width, source_height) = preprocess(image_bytes, self.profile)?;
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
    #[ignore = "requires real model/helper and images; exports CPU references for GPU parity checks"]
    fn live_export_cpu_parity_fixtures() {
        let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
            "id":"depth-parity-reference", "name":"Depth parity reference", "engine":["litert"],
            "tasks":["image-to-depth"], "local":{"path":std::env::var("LITERT_TEST_MODEL").unwrap(),"gpu_layers":0}
        })).unwrap();
        let engine = LiteRtEngine::new(&ModelConfig::try_from(entry).unwrap()).unwrap();
        assert!(
            engine.gpu_runtime.is_none(),
            "Reference generation must explicitly use CPU"
        );
        let images: Vec<PathBuf> =
            serde_json::from_str(&std::env::var("LITERT_TEST_IMAGES").unwrap()).unwrap();
        assert!((1..=16).contains(&images.len()));
        let root = PathBuf::from(std::env::var("LITERT_TEST_OUTPUT_DIR").unwrap());
        std::fs::create_dir(&root).unwrap();
        let mut fixtures = Vec::new();
        for (index, image) in images.iter().enumerate() {
            let (input, _, _) = preprocess(&std::fs::read(image).unwrap(), engine.profile).unwrap();
            let values = engine.infer(input.clone()).unwrap();
            let output: Vec<_> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(root.join(format!("fixture-{index}.input.f32")), input).unwrap();
            std::fs::write(root.join(format!("fixture-{index}.reference.f32")), output).unwrap();
            fixtures.push(serde_json::json!({"image":image,"values":values.len()}));
        }
        std::fs::write(
            root.join("cpu-reference-report.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "model_sha256":engine.profile.sha256,
                "runtime_sha256":super::super::download::file_sha256(&engine.runtime).unwrap(),
                "helper_sha256":super::super::download::file_sha256(&engine.helper).unwrap(),
                "fixtures":fixtures
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    #[ignore = "requires a GPU-reviewed model, pinned runtimes, real helper and CPU reference; checks GPU and forced CPU fallback"]
    fn live_gpu_and_forced_cpu_fallback() {
        gpu_runtime::ensure_installed(&runtime_path().unwrap())
            .expect("Managed GPU installation must succeed");
        let entry: crate::schema::ModelEntry = serde_json::from_value(serde_json::json!({
            "id":"depth-gpu-proof", "name":"Depth GPU proof", "engine":["litert"],
            "tasks":["image-to-depth"], "local":{"path":std::env::var("LITERT_TEST_MODEL").unwrap(),"gpu_layers":99}
        })).unwrap();
        let config = ModelConfig::try_from(entry).unwrap();
        let mut engine = LiteRtEngine::new(&config).unwrap();
        let input = std::fs::read(std::env::var("LITERT_TEST_REFERENCE_INPUT").unwrap()).unwrap();
        let reference =
            std::fs::read(std::env::var("LITERT_TEST_REFERENCE_OUTPUT").unwrap()).unwrap();
        let gpu = engine
            .gpu_runtime
            .as_ref()
            .expect("Managed GPU bundle must be installed");
        let accelerated = engine
            .infer_worker(input.clone(), gpu, true)
            .expect("Real GPU path must succeed before testing fallback");
        // A missing GPU runtime forces the child to fail before dispatch. The
        // gateway must then retry the same input through its verified CPU DLL.
        engine.gpu_runtime = Some(gpu.with_file_name("missing-gpu-runtime.dll"));
        let fallback = engine.infer(input).expect("CPU retry must succeed");
        for values in [&accelerated, &fallback] {
            assert_eq!(values.len() * 4, reference.len());
            for (actual, expected) in values.iter().zip(reference.as_chunks::<4>().0) {
                let expected = f32::from_le_bytes(*expected);
                assert!((actual - expected).abs() <= 0.001 + 0.001 * expected.abs());
            }
        }
    }

    #[test]
    fn gpu_support_is_limited_to_parity_verified_profiles() {
        assert!(MODEL_PROFILES[0].gpu_verified);
        assert!(!MODEL_PROFILES[1].gpu_verified);
        assert!(MODEL_PROFILES[2].gpu_verified);
    }
    #[test]
    #[ignore = "requires a real image and its converter-generated input tensor"]
    fn live_preprocessing_matches_reference_input() {
        let image = std::fs::read(std::env::var("LITERT_TEST_IMAGE").unwrap()).unwrap();
        let expected =
            std::fs::read(std::env::var("LITERT_TEST_REFERENCE_INPUT").unwrap()).unwrap();
        let (actual, _, _) = preprocess(&image, &MODEL_PROFILES[1]).unwrap();
        assert_eq!(actual.len(), expected.len());
        let mut maximum = 0_f32;
        let mut sum = 0_f64;
        for (actual, expected) in actual
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_chunks::<4>().0)
        {
            let error = (f32::from_le_bytes(*actual) - f32::from_le_bytes(*expected)).abs();
            maximum = maximum.max(error);
            sum += f64::from(error);
        }
        println!(
            "Preprocessing maximum error={maximum}, mean error={}",
            sum / (PIXELS * 3) as f64
        );
        assert!(
            maximum <= 0.00001,
            "Preprocessing differs from the reference input"
        );
    }

    #[test]
    fn preprocessing_is_rgb_nchw_float32_and_depth_output_is_bounded_and_finite() {
        let image = image::RgbImage::from_pixel(2, 3, image::Rgb([255, 0, 128]));
        let mut png = std::io::Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let (input, width, height) = preprocess(png.get_ref(), &MODEL_PROFILES[0]).unwrap();
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
        assert!(preprocess(b"not an image", &MODEL_PROFILES[0]).is_err());
        assert!(decode_depth(&[0; 4]).is_err());
        let mut output = vec![0u8; PIXELS * 4];
        assert!(decode_depth(&output).is_ok());
        output[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_depth(&output).is_err());
    }
}
