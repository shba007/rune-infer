//! Isolated ONNX Runtime adapter for reviewed YOLO26x vision exports.
mod ade20k;
pub(crate) mod runtime;
use crate::{
    config::ModelConfig,
    inference::{
        traits::InferenceEngine,
        types::{InferenceOutput, InferenceTaskRequest, InferenceTaskResponse},
    },
    types::{
        DetectObjectsResponse, DetectObjectsUsage, DetectionBboxPixels, DetectionBox,
        DetectionItem, DetectionOrientedBox, DetectionPoseKeypoint, Usage,
    },
};
use std::{
    error::Error,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

const MODEL_SHA: &str = "5267bf1883b43df3aca8791721909679ee83de8a7d5546a7ecdd75b99877bce4";
const POSE_MODEL_SHA: &str = "f9fa35b3181b72328414649cad937bf752c15ad0bf5131d91dea12f1bf4090e7";
const SEG_MODEL_SHA: &str = "c815dd6377a436a3c453c435a8da6b88c5136135148304750aa9d8c2300eb25d";
const OBB_MODEL_SHA: &str = "565c3c7f1611027b509d6df2b81a8abbb2d3f3b004f84c17607545394458bfec";
const SEM_MODEL_SHA: &str = "3740e49cd9150fac97a6b6fdb610b66e6d522b816c37c3fbb1a7642fe51efef0";
const MATTING_MODEL_SHA: &str = "6065d27c615ea27308f5b88598dd8db116eb07436c7a323ca40d13b2866c309e";
pub(crate) const MATTING_SIDE: u32 = 1024;
pub(crate) const MATTING_MAX_EDGE: u32 = 8192;
pub(crate) const MATTING_MAX_PIXELS: u64 = 40_000_000;
const MATTING_GPU_ARENA_BYTES: usize = 10 * 1024 * 1024 * 1024;

#[cfg(windows)]
fn available_system_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is zero initialized and its length field is set as required.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        None
    } else {
        Some(status.ullAvailPhys)
    }
}
#[cfg(not(windows))]
fn available_system_memory() -> Option<u64> {
    None
}
const RUNTIME_SHA: &str = "51c002601f22ecb7089931565b8908cd7c155d7bf8f8cffe21336835da861d07";
const PROVIDER_SHARED_SHA: &str =
    "05c836c1f75858ffa034823cf227335014f02a7297fd273dbd5d487f5262c87b";
const PROVIDER_CUDA_SHA: &str = "8629b290bc1a2c8877f0ae794baef861c0f7b964a982cdfd5e90ee1bc01a33ee";
const SIDE: u32 = 640;
const POSE_HEIGHT: u32 = 480;
const POSE_ANCHORS: usize = 6300;
const POSE_CHANNELS: usize = 56;
const KEYPOINTS: [&str; 17] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Detection,
    Pose,
    Segmentation,
    Oriented,
    Semantic,
    Matting,
}
impl Profile {
    fn from_model(path: &Path) -> Result<Self, Box<dyn Error>> {
        match super::download::file_sha256(path)?.as_str() {
            MODEL_SHA => Ok(Self::Detection),
            POSE_MODEL_SHA => Ok(Self::Pose),
            SEG_MODEL_SHA => Ok(Self::Segmentation),
            OBB_MODEL_SHA => Ok(Self::Oriented),
            SEM_MODEL_SHA => Ok(Self::Semantic),
            MATTING_MODEL_SHA => Ok(Self::Matting),
            _ => Err(
                "ONNX model is not a reviewed YOLO26x detection, pose, segmentation or oriented-box export"
                    .into(),
            ),
        }
    }
    fn height(self) -> u32 {
        if self == Self::Matting {
            MATTING_SIDE
        } else if matches!(self, Self::Pose | Self::Segmentation | Self::Oriented) {
            POSE_HEIGHT
        } else {
            SIDE
        }
    }
    fn output_len(self) -> usize {
        match self {
            Self::Pose => POSE_CHANNELS * POSE_ANCHORS,
            Self::Detection => 300 * 6,
            Self::Segmentation => 116 * POSE_ANCHORS + 32 * 120 * 160,
            Self::Oriented => 20 * POSE_ANCHORS,
            Self::Semantic => (SIDE * SIDE) as usize,
            Self::Matting => (MATTING_SIDE * MATTING_SIDE) as usize,
        }
    }
}

fn declares_matting(model: &ModelConfig) -> bool {
    model
        .capabilities
        .as_ref()
        .is_some_and(|v| v.iter().any(|c| c.eq_ignore_ascii_case("Portrait Matting")))
}

pub(crate) fn declares_pose(model: &ModelConfig) -> bool {
    model
        .capabilities
        .as_ref()
        .is_some_and(|v| v.iter().any(|c| c.eq_ignore_ascii_case("Pose Estimation")))
}
pub(crate) fn declares_segmentation(model: &ModelConfig) -> bool {
    model.capabilities.as_ref().is_some_and(|v| {
        v.iter()
            .any(|c| c.eq_ignore_ascii_case("Instance Segmentation"))
    })
}
pub(crate) fn declares_oriented(model: &ModelConfig) -> bool {
    model.capabilities.as_ref().is_some_and(|v| {
        v.iter()
            .any(|c| c.eq_ignore_ascii_case("Oriented Bounding Boxes"))
    })
}
fn declares_semantic(model: &ModelConfig) -> bool {
    model.capabilities.as_ref().is_some_and(|v| {
        v.iter()
            .any(|c| c.eq_ignore_ascii_case("Semantic Segmentation"))
    })
}
const LABELS: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

pub struct OnnxEngine {
    id: String,
    model: PathBuf,
    profile: Profile,
    runtime: PathBuf,
    helper: PathBuf,
    cpu_only: bool,
    cuda_directory: Option<PathBuf>,
    cuda_bundle_invalid: bool,
    worker: Mutex<Option<ResidentWorker>>,
}

struct ResidentWorker {
    _guard: Arc<Mutex<super::super::process::ProcessGuard>>,
    request: mpsc::Sender<Vec<u8>>,
    response: mpsc::Receiver<Result<Vec<u8>, String>>,
    diagnostics: Arc<Mutex<Vec<u8>>>,
    cpu: bool,
}
impl OnnxEngine {
    pub fn new(model: &ModelConfig) -> Result<Self, Box<dyn Error>> {
        if !cfg!(all(windows, target_arch = "x86_64")) {
            return Err("ONNX detection currently supports Windows x64 only".into());
        }
        let model_path = std::fs::canonicalize(&model.model_path)?;
        let profile = Profile::from_model(&model_path)?;
        if (profile == Profile::Matting) != declares_matting(model) {
            return Err(
                "ONNX model profile conflicts with its declared portrait matting capability".into(),
            );
        }
        if (profile == Profile::Pose) != declares_pose(model) {
            return Err("ONNX model profile conflicts with its declared pose capability".into());
        }
        if (profile == Profile::Segmentation) != declares_segmentation(model) {
            return Err(
                "ONNX model profile conflicts with its declared segmentation capability".into(),
            );
        }
        if (profile == Profile::Oriented) != declares_oriented(model) {
            return Err(
                "ONNX model profile conflicts with its declared oriented-box capability".into(),
            );
        }
        if (profile == Profile::Semantic) != declares_semantic(model) {
            return Err(
                "ONNX model profile conflicts with its declared semantic segmentation capability"
                    .into(),
            );
        }
        let runtime = std::env::var_os("ONNX_RUNTIME_LIBRARY_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| super::download::engine_bin().join("onnxruntime/onnxruntime.dll"));
        let runtime = std::fs::canonicalize(runtime).map_err(|_| "ONNX Runtime is missing; set ONNX_RUNTIME_LIBRARY_PATH to the verified runtime DLL")?;
        verify_runtime(&runtime, false)?;
        let (cuda_directory, cuda_bundle_invalid) = match runtime::managed_gpu_directory(&runtime) {
            Ok(directory) => (directory, false),
            Err(error) => {
                tracing::warn!("ONNX CUDA bundle is unverified; using CPU: {error}");
                (None, true)
            }
        };
        let helper = std::env::current_exe()?;
        if !helper.file_stem().is_some_and(|v| v == "rune-infer") {
            return Err("ONNX detection requires the rune-infer worker executable".into());
        }
        Ok(Self {
            id: model.id.clone(),
            model: model_path,
            profile,
            runtime,
            helper,
            cpu_only: model.runtime.as_ref().is_some_and(|v| v.gpu_layers == 0),
            cuda_directory,
            cuda_bundle_invalid,
            worker: Mutex::new(None),
        })
    }
    fn start_worker(&self, cpu: bool) -> Result<ResidentWorker, Box<dyn Error>> {
        let mut cmd = Command::new(&self.helper);
        super::super::process::configure_death_signal(&mut cmd);
        cmd.arg("onnx-worker")
            .arg("--runtime")
            .arg(&self.runtime)
            .arg("--model")
            .arg(&self.model);
        if cpu {
            cmd.arg("--cpu");
        } else if let Some(directory) = &self.cuda_directory {
            runtime::managed_gpu_directory(&self.runtime)?
                .filter(|verified| verified == directory)
                .ok_or("ONNX CUDA bundle is no longer available")?;
            let mut paths = vec![directory.clone()];
            if let Some(existing) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&existing));
            }
            cmd.env("PATH", std::env::join_paths(paths)?);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let mut child = cmd.spawn()?;
        let mut stdin = child.stdin.take().ok_or("missing worker input")?;
        let stdout = child.stdout.take().ok_or("missing worker output")?;
        let stderr = child.stderr.take().ok_or("missing worker diagnostics")?;
        let guard =
            super::super::process::ProcessGuard::try_new(child, format!("onnx-{}", self.id))?;
        let output_bytes = self.profile.output_len() * 4;
        let (request, requests) = mpsc::channel::<Vec<u8>>();
        let (responses, response) = mpsc::channel();
        std::thread::spawn(move || {
            let mut stdout = stdout;
            while let Ok(payload) = requests.recv() {
                let mut bytes = vec![0; output_bytes];
                let result = stdin
                    .write_all(&payload)
                    .and_then(|()| stdout.read_exact(&mut bytes))
                    .map(|()| bytes)
                    .map_err(|e| e.to_string());
                let failed = result.is_err();
                if responses.send(result).is_err() || failed {
                    break;
                }
            }
        });
        let diagnostics = Arc::new(Mutex::new(Vec::new()));
        let diagnostic_writer = Arc::clone(&diagnostics);
        std::thread::spawn(move || {
            let mut reader = stderr;
            let mut buf = [0; 4096];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let Ok(mut saved) = diagnostic_writer.lock() else {
                    break;
                };
                if saved.len() < 65536 {
                    let available = 65536 - saved.len();
                    saved.extend_from_slice(&buf[..n.min(available)]);
                }
            }
        });
        Ok(ResidentWorker {
            _guard: guard,
            request,
            response,
            diagnostics,
            cpu,
        })
    }

    fn run_worker(
        &self,
        worker: &mut Option<ResidentWorker>,
        input: &[u8],
        cpu: bool,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if worker.as_ref().is_some_and(|w| w.cpu != cpu) {
            *worker = None;
        }
        if worker.is_none() {
            *worker = Some(self.start_worker(cpu)?);
        }
        let active = worker.as_ref().ok_or("ONNX worker missing")?;
        let result = active
            .request
            .send(input.to_vec())
            .map_err(|e| e.to_string())
            .and_then(|()| {
                active
                    .response
                    .recv_timeout(Duration::from_secs(if self.profile == Profile::Matting {
                        600
                    } else {
                        120
                    }))
                    .map_err(|e| e.to_string())
            })
            .and_then(|v| v);
        let bytes = match result {
            Ok(bytes) => bytes,
            Err(error) => {
                let diagnostics = active
                    .diagnostics
                    .lock()
                    .map(|v| String::from_utf8_lossy(&v).into_owned())
                    .unwrap_or_default();
                *worker = None;
                return Err(format!("ONNX worker failed: {error} {diagnostics}").into());
            }
        };
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
            .collect();
        if !values.iter().all(|x| x.is_finite()) {
            *worker = None;
            return Err("ONNX returned non-finite values".into());
        }
        tracing::info!(
            "ONNX inference completed using {}",
            if cpu {
                "CPU"
            } else {
                "CUDA with CPU operator fallback"
            }
        );
        Ok(values)
    }

    fn infer_preprocessed(&self, input: &[u8]) -> Result<(Vec<f32>, &'static str), Box<dyn Error>> {
        if self.profile == Profile::Matting
            && (self.cpu_only || self.cuda_bundle_invalid || self.cuda_directory.is_none())
        {
            return Err("Portrait matting requires the verified ONNX CUDA runtime; CPU execution is not enabled for this model".into());
        }
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| "ONNX inference lock poisoned")?;
        if self.profile == Profile::Matting {
            let available = available_system_memory()
                .ok_or("Cannot determine available system memory for portrait matting")?;
            if available < 12 * 1024 * 1024 * 1024 {
                return Err(
                    "Portrait matting requires at least 12 GiB of available system memory".into(),
                );
            }
            if worker.is_none() {
                let total = super::super::budget::detect_total_gpu_vram()
                    .ok_or("Cannot determine GPU memory for portrait matting")?;
                let used = super::super::budget::detect_used_gpu_vram()
                    .ok_or("Cannot determine free GPU memory for portrait matting")?;
                if total.saturating_sub(used) < 12 * 1024 * 1024 * 1024 {
                    return Err(
                        "Portrait matting requires at least 12 GiB of free GPU memory".into(),
                    );
                }
            }
        }
        if self.cpu_only
            || self.cuda_bundle_invalid
            || self.cuda_directory.is_none()
            || worker.as_ref().is_some_and(|w| w.cpu)
        {
            if !self.cpu_only && self.cuda_directory.is_none() {
                tracing::warn!("Verified ONNX CUDA dependencies are unavailable; using CPU");
            }
            Ok((self.run_worker(&mut worker, input, true)?, "cpu"))
        } else {
            match self.run_worker(&mut worker, input, false) {
                Ok(values) => Ok((values, "cuda_registered_cpu_fallback")),
                Err(error) => {
                    if self.profile == Profile::Matting {
                        return Err(format!("Portrait matting failed on CUDA; automatic CPU retry is disabled: {error}").into());
                    }
                    tracing::warn!("ONNX CUDA failed; retrying on CPU: {error}");
                    Ok((self.run_worker(&mut worker, input, true)?, "cpu"))
                }
            }
        }
    }
}

fn verify_runtime(runtime: &Path, cuda: bool) -> Result<(), Box<dyn Error>> {
    if super::download::file_sha256(runtime)? != RUNTIME_SHA {
        return Err(
            "ONNX Runtime DLL does not match the reviewed Windows x64 1.31.1 CUDA 12 bundle".into(),
        );
    }
    if cuda {
        let directory = runtime
            .parent()
            .ok_or("ONNX Runtime has no parent directory")?;
        for (name, expected) in [
            ("onnxruntime_providers_shared.dll", PROVIDER_SHARED_SHA),
            ("onnxruntime_providers_cuda.dll", PROVIDER_CUDA_SHA),
        ] {
            if super::download::file_sha256(&directory.join(name))? != expected {
                return Err(format!("ONNX Runtime CUDA provider checksum mismatch: {name}").into());
            }
        }
    }
    Ok(())
}

type Preprocessed = (Vec<u8>, u32, u32, f32, u32, u32);
type PreprocessedMatting = (Vec<u8>, u32, u32, bool);

pub(crate) fn validate_matting_image(bytes: &[u8]) -> Result<(u32, u32), Box<dyn Error>> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let (w, h) = reader.into_dimensions()?;
    if w == 0
        || h == 0
        || w > MATTING_MAX_EDGE
        || h > MATTING_MAX_EDGE
        || u64::from(w) * u64::from(h) > MATTING_MAX_PIXELS
    {
        return Err("Portrait image exceeds the 8192-pixel edge or 40-million-pixel limit".into());
    }
    Ok((w, h))
}

fn preprocess_matting(bytes: &[u8]) -> Result<PreprocessedMatting, Box<dyn Error>> {
    use image::ImageDecoder;
    validate_matting_image(bytes)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MATTING_MAX_EDGE);
    limits.max_image_height = Some(MATTING_MAX_EDGE);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let applied = orientation != image::metadata::Orientation::NoTransforms;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let image = image.to_rgb8();
    let (w, h) = image.dimensions();
    let resized = image::imageops::resize(
        &image,
        MATTING_SIDE,
        MATTING_SIDE,
        image::imageops::FilterType::Triangle,
    );
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut input = Vec::with_capacity((MATTING_SIDE * MATTING_SIDE * 3 * 4) as usize);
    for c in 0..3 {
        for pixel in resized.pixels() {
            input.extend_from_slice(&((pixel[c] as f32 / 255.0 - mean[c]) / std[c]).to_le_bytes());
        }
    }
    Ok((input, w, h, applied))
}

fn decode_matte(values: &[f32], w: u32, h: u32) -> Result<String, Box<dyn Error>> {
    use base64::Engine as _;
    if values.len() != (MATTING_SIDE * MATTING_SIDE) as usize
        || !values.iter().all(|v| v.is_finite())
    {
        return Err("Matting model returned an invalid alpha tensor".into());
    }
    let mut matte = image::GrayImage::new(MATTING_SIDE, MATTING_SIDE);
    let mut max_alpha = 0u8;
    let mut min_alpha = u8::MAX;
    for (pixel, value) in matte.pixels_mut().zip(values) {
        let alpha = (255.0 / (1.0 + (-value).exp())).round() as u8;
        pixel.0[0] = alpha;
        max_alpha = max_alpha.max(alpha);
        min_alpha = min_alpha.min(alpha);
    }
    if max_alpha == min_alpha {
        return Err("Matting model produced a uniform alpha matte without a usable subject".into());
    }
    let matte = image::imageops::resize(&matte, w, h, image::imageops::FilterType::CatmullRom);
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageLuma8(matte).write_to(&mut png, image::ImageFormat::Png)?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    ))
}

pub(crate) fn validate_task(task: &InferenceTaskRequest, pose: bool) -> Result<(), &'static str> {
    let InferenceTaskRequest::DetectObjects {
        prompt,
        categories,
        confidence_threshold,
        iou_threshold,
        features,
        ..
    } = task
    else {
        return Ok(());
    };
    if prompt.as_ref().is_some_and(|v| !v.trim().is_empty())
        || categories.as_ref().is_some_and(|v| !v.is_empty())
        || (!pose && iou_threshold.is_some())
        || features
            .iter()
            .any(|v| v != "boxes" && !(pose && v == "keypoints"))
    {
        return Err(if pose {
            "This YOLO26x pose export supports boxes, keypoints, confidence and IoU thresholds only"
        } else {
            "This YOLO26x detection export supports boxes and confidence threshold only; NMS is embedded"
        });
    }
    if confidence_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("confidence_threshold must be between 0 and 1");
    }
    if iou_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("iou_threshold must be between 0 and 1");
    }
    Ok(())
}

pub(crate) fn validate_segmentation_task(task: &InferenceTaskRequest) -> Result<(), &'static str> {
    let InferenceTaskRequest::DetectObjects {
        prompt,
        categories,
        confidence_threshold,
        iou_threshold,
        features,
        ..
    } = task
    else {
        return Ok(());
    };
    if prompt.as_ref().is_some_and(|v| !v.trim().is_empty())
        || categories.as_ref().is_some_and(|v| !v.is_empty())
        || features.iter().any(|v| v != "boxes" && v != "masks")
    {
        return Err(
            "This YOLO26x segmentation export supports boxes, masks, confidence and IoU thresholds only",
        );
    }
    if confidence_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("confidence_threshold must be between 0 and 1");
    }
    if iou_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("iou_threshold must be between 0 and 1");
    }
    Ok(())
}

pub(crate) fn validate_oriented_task(task: &InferenceTaskRequest) -> Result<(), &'static str> {
    let InferenceTaskRequest::DetectObjects {
        prompt,
        categories,
        confidence_threshold,
        iou_threshold,
        features,
        ..
    } = task
    else {
        return Ok(());
    };
    if prompt.as_ref().is_some_and(|v| !v.trim().is_empty())
        || categories.as_ref().is_some_and(|v| !v.is_empty())
        || features
            .iter()
            .any(|v| v != "boxes" && v != "oriented_boxes")
    {
        return Err(
            "This YOLO26x oriented-box export supports boxes, oriented_boxes, confidence and IoU thresholds only",
        );
    }
    if confidence_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("confidence_threshold must be between 0 and 1");
    }
    if iou_threshold.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err("iou_threshold must be between 0 and 1");
    }
    Ok(())
}

fn preprocess(bytes: &[u8], profile: Profile) -> Result<Preprocessed, Box<dyn Error>> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(64 * 1024 * 1024);
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    reader.limits(limits);
    let image = reader.decode()?.to_rgb8();
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return Err("empty image".into());
    }
    let target_height = profile.height();
    let scale = (SIDE as f32 / w as f32).min(target_height as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, SIDE);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, target_height);
    let left = (SIDE - nw) / 2;
    let top = (target_height - nh) / 2;
    let resized = image::imageops::resize(&image, nw, nh, image::imageops::FilterType::Triangle);
    let mut canvas = image::RgbImage::from_pixel(SIDE, target_height, image::Rgb([114, 114, 114]));
    image::imageops::replace(&mut canvas, &resized, left.into(), top.into());
    let mut input = Vec::with_capacity((SIDE * target_height * 3 * 4) as usize);
    for c in 0..3 {
        for pixel in canvas.pixels() {
            input.extend_from_slice(&(pixel[c] as f32 / 255.0).to_le_bytes());
        }
    }
    Ok((input, w, h, scale, left, top))
}

fn decode_detections(
    values: &[f32],
    w: u32,
    h: u32,
    scale: f32,
    left: u32,
    top: u32,
    threshold: f32,
) -> Result<Vec<DetectionItem>, Box<dyn Error>> {
    if !values.len().is_multiple_of(6) || !values.iter().all(|v| v.is_finite()) {
        return Err("ONNX returned invalid detection values".into());
    }
    let mut data = Vec::new();
    for row in values.as_chunks::<6>().0 {
        let score = row[4];
        if !(0.0..=1.0).contains(&score) {
            return Err("ONNX returned an invalid confidence score".into());
        }
        if score < threshold {
            continue;
        }
        let cls = row[5] as usize;
        if row[5] < 0.0 || row[5].fract() != 0.0 || cls >= LABELS.len() {
            return Err("ONNX returned an invalid class index".into());
        }
        let x1 = ((row[0] - left as f32) / scale).clamp(0.0, w as f32);
        let y1 = ((row[1] - top as f32) / scale).clamp(0.0, h as f32);
        let x2 = ((row[2] - left as f32) / scale).clamp(0.0, w as f32);
        let y2 = ((row[3] - top as f32) / scale).clamp(0.0, h as f32);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        data.push(DetectionItem {
            object: "detection".into(),
            index: data.len(),
            label: LABELS[cls].into(),
            confidence: score as f64,
            box_: DetectionBox {
                x: ((x1 + x2) / 2.0 / w as f32) as f64,
                y: ((y1 + y2) / 2.0 / h as f32) as f64,
                width: ((x2 - x1) / w as f32) as f64,
                height: ((y2 - y1) / h as f32) as f64,
            },
            bbox_pixels: DetectionBboxPixels {
                x_min: x1.round() as u32,
                y_min: y1.round() as u32,
                x_max: x2.round() as u32,
                y_max: y2.round() as u32,
            },
            oriented_box: None,
            mask: None,
            pose: None,
            attributes: None,
        });
    }
    Ok(data)
}

fn decode_pose(
    values: &[f32],
    geometry: (u32, u32, f32, u32, u32),
    threshold: f32,
    iou_threshold: f32,
    include_keypoints: bool,
) -> Result<Vec<DetectionItem>, Box<dyn Error>> {
    if values.len() != POSE_CHANNELS * POSE_ANCHORS || !values.iter().all(|v| v.is_finite()) {
        return Err("ONNX returned invalid pose values".into());
    }
    let (w, h, scale, left, top) = geometry;
    let at = |channel: usize, anchor: usize| values[channel * POSE_ANCHORS + anchor];
    let mut candidates = Vec::new();
    for anchor in 0..POSE_ANCHORS {
        let score = at(4, anchor);
        if !(0.0..=1.0).contains(&score) {
            return Err("ONNX returned an invalid pose confidence".into());
        }
        if score < threshold {
            continue;
        }
        let (cx, cy, bw, bh) = (at(0, anchor), at(1, anchor), at(2, anchor), at(3, anchor));
        if bw <= 0.0 || bh <= 0.0 {
            continue;
        }
        candidates.push((
            anchor,
            score,
            [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
        ));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected: Vec<(usize, f32, [f32; 4])> = Vec::new();
    for candidate in candidates {
        let a = candidate.2;
        let area_a = (a[2] - a[0]) * (a[3] - a[1]);
        if selected.iter().any(|other| {
            let b = other.2;
            let intersection = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let union = area_a + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
            union > 0.0 && intersection / union > iou_threshold
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() == 300 {
            break;
        }
    }
    let mut data = Vec::new();
    for (anchor, score, b) in selected {
        let x1 = ((b[0] - left as f32) / scale).clamp(0.0, w as f32);
        let y1 = ((b[1] - top as f32) / scale).clamp(0.0, h as f32);
        let x2 = ((b[2] - left as f32) / scale).clamp(0.0, w as f32);
        let y2 = ((b[3] - top as f32) / scale).clamp(0.0, h as f32);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let pose = if include_keypoints {
            let mut points = Vec::with_capacity(KEYPOINTS.len());
            for (index, name) in KEYPOINTS.iter().enumerate() {
                let confidence = at(5 + index * 3 + 2, anchor);
                if !(0.0..=1.0).contains(&confidence) {
                    return Err("ONNX returned an invalid keypoint confidence".into());
                }
                let x = ((at(5 + index * 3, anchor) - left as f32) / scale).clamp(0.0, w as f32)
                    / w as f32;
                let y = ((at(5 + index * 3 + 1, anchor) - top as f32) / scale).clamp(0.0, h as f32)
                    / h as f32;
                points.push(DetectionPoseKeypoint {
                    name: (*name).into(),
                    x: x as f64,
                    y: y as f64,
                    confidence: confidence as f64,
                });
            }
            Some(points)
        } else {
            None
        };
        data.push(DetectionItem {
            object: "detection".into(),
            index: data.len(),
            label: "person".into(),
            confidence: score as f64,
            box_: DetectionBox {
                x: ((x1 + x2) / 2.0 / w as f32) as f64,
                y: ((y1 + y2) / 2.0 / h as f32) as f64,
                width: ((x2 - x1) / w as f32) as f64,
                height: ((y2 - y1) / h as f32) as f64,
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
    Ok(data)
}

fn decode_segmentation(
    values: &[f32],
    geometry: (u32, u32, f32, u32, u32),
    threshold: f32,
    iou_threshold: f32,
    include_masks: bool,
) -> Result<Vec<DetectionItem>, Box<dyn Error>> {
    use base64::Engine as _;
    let predictions = 116 * POSE_ANCHORS;
    if values.len() != predictions + 32 * 120 * 160 || !values.iter().all(|v| v.is_finite()) {
        return Err("ONNX returned invalid segmentation values".into());
    }
    let (w, h, scale, left, top) = geometry;
    let at = |c: usize, a: usize| values[c * POSE_ANCHORS + a];
    let mut candidates = Vec::new();
    for anchor in 0..POSE_ANCHORS {
        let (class, score) = (0..80)
            .map(|c| (c, at(4 + c, anchor)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("missing classes")?;
        if !(0.0..=1.0).contains(&score) {
            return Err("invalid segmentation confidence".into());
        }
        if score < threshold {
            continue;
        }
        let (cx, cy, bw, bh) = (at(0, anchor), at(1, anchor), at(2, anchor), at(3, anchor));
        if bw > 0.0 && bh > 0.0 {
            candidates.push((
                anchor,
                class,
                score,
                [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
            ));
        }
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut selected: Vec<(usize, usize, f32, [f32; 4])> = Vec::new();
    for c in candidates {
        let a = c.3;
        let area = (a[2] - a[0]) * (a[3] - a[1]);
        if selected.iter().any(|o| {
            let b = o.3;
            let i = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let u = area + (b[2] - b[0]) * (b[3] - b[1]) - i;
            c.1 == o.1 && u > 0.0 && i / u > iou_threshold
        }) {
            continue;
        }
        selected.push(c);
        if selected.len() == 300 {
            break;
        }
    }
    if include_masks && (w as u64) * (h as u64) * (selected.len() as u64) > 64_000_000 {
        return Err("segmentation masks exceed the 64-million-pixel response limit".into());
    }
    let mut data = Vec::new();
    for (anchor, class, score, b) in selected {
        let x1 = ((b[0] - left as f32) / scale).clamp(0.0, w as f32);
        let y1 = ((b[1] - top as f32) / scale).clamp(0.0, h as f32);
        let x2 = ((b[2] - left as f32) / scale).clamp(0.0, w as f32);
        let y2 = ((b[3] - top as f32) / scale).clamp(0.0, h as f32);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let mask = if include_masks {
            let mut proto = image::GrayImage::new(160, 120);
            for py in 0..120usize {
                for px in 0..160usize {
                    let fx = (px as f32 + 0.5) * 4.0;
                    let fy = (py as f32 + 0.5) * 4.0;
                    if fx >= b[0] && fx < b[2] && fy >= b[1] && fy < b[3] {
                        let i = py * 160 + px;
                        let sum: f32 = (0..32)
                            .map(|c| at(84 + c, anchor) * values[predictions + c * 120 * 160 + i])
                            .sum();
                        if sum > 0.0 {
                            proto.put_pixel(px as u32, py as u32, image::Luma([255]));
                        }
                    }
                }
            }
            let scaled_w = ((w as f32 * scale).round() as u32).clamp(1, 640);
            let scaled_h = ((h as f32 * scale).round() as u32).clamp(1, 480);
            let full =
                image::imageops::resize(&proto, 640, 480, image::imageops::FilterType::Triangle);
            let crop = image::imageops::crop_imm(&full, left, top, scaled_w, scaled_h).to_image();
            let mut mask =
                image::imageops::resize(&crop, w, h, image::imageops::FilterType::Triangle);
            for p in mask.pixels_mut() {
                p.0[0] = if p.0[0] >= 128 { 255 } else { 0 };
            }
            let mut encoded = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageLuma8(mask)
                .write_to(&mut encoded, image::ImageFormat::Png)?;
            Some(format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(encoded.into_inner())
            ))
        } else {
            None
        };
        data.push(DetectionItem {
            object: "detection".into(),
            index: data.len(),
            label: LABELS[class].into(),
            confidence: score as f64,
            box_: DetectionBox {
                x: ((x1 + x2) / 2.0 / w as f32) as f64,
                y: ((y1 + y2) / 2.0 / h as f32) as f64,
                width: ((x2 - x1) / w as f32) as f64,
                height: ((y2 - y1) / h as f32) as f64,
            },
            bbox_pixels: DetectionBboxPixels {
                x_min: x1.round() as u32,
                y_min: y1.round() as u32,
                x_max: x2.round() as u32,
                y_max: y2.round() as u32,
            },
            oriented_box: None,
            mask,
            pose: None,
            attributes: None,
        });
    }
    Ok(data)
}

const DOTA_LABELS: [&str; 15] = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
];
type OrientedCandidate = (usize, usize, f32, f32, [[f64; 2]; 4]);

fn oriented_corners(cx: f64, cy: f64, w: f64, h: f64, angle: f64) -> [[f64; 2]; 4] {
    let (sin, cos) = angle.sin_cos();
    [
        (-w / 2.0, -h / 2.0),
        (w / 2.0, -h / 2.0),
        (w / 2.0, h / 2.0),
        (-w / 2.0, h / 2.0),
    ]
    .map(|(x, y)| [cx + x * cos - y * sin, cy + x * sin + y * cos])
}

fn oriented_iou(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    fn cross(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    }
    fn area(p: &[[f64; 2]]) -> f64 {
        if p.len() < 3 {
            return 0.0;
        }
        (0..p.len())
            .map(|i| {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                a[0] * b[1] - a[1] * b[0]
            })
            .sum::<f64>()
            .abs()
            / 2.0
    }
    let mut polygon = a.to_vec();
    for i in 0..4 {
        if polygon.is_empty() {
            break;
        }
        let start = b[i];
        let end = b[(i + 1) % 4];
        let mut next = Vec::new();
        let mut prev = *polygon.last().unwrap();
        let mut prev_side = cross(start, end, prev);
        for &cur in &polygon {
            let side = cross(start, end, cur);
            if (side >= 0.0) != (prev_side >= 0.0) {
                let t = prev_side / (prev_side - side);
                next.push([
                    prev[0] + t * (cur[0] - prev[0]),
                    prev[1] + t * (cur[1] - prev[1]),
                ]);
            }
            if side >= 0.0 {
                next.push(cur);
            }
            prev = cur;
            prev_side = side;
        }
        polygon = next;
    }
    let intersection = area(&polygon);
    let union = area(a) + area(b) - intersection;
    if union > 0.0 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn decode_oriented(
    values: &[f32],
    geometry: (u32, u32, f32, u32, u32),
    threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<DetectionItem>, Box<dyn Error>> {
    const ANCHORS: usize = POSE_ANCHORS;
    if values.len() != 20 * ANCHORS || !values.iter().all(|v| v.is_finite()) {
        return Err("ONNX returned invalid oriented-box values".into());
    }
    let (w, h, scale, left, top) = geometry;
    let at = |c: usize, a: usize| values[c * ANCHORS + a];
    let mut candidates = Vec::new();
    for anchor in 0..ANCHORS {
        let (class, score) = (0..15)
            .map(|c| (c, at(4 + c, anchor)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("missing classes")?;
        if !(0.0..=1.0).contains(&score) {
            return Err("invalid oriented-box confidence".into());
        }
        if score < threshold {
            continue;
        }
        let (cx, cy, bw, bh, angle) = (
            at(0, anchor),
            at(1, anchor),
            at(2, anchor),
            at(3, anchor),
            at(19, anchor),
        );
        if bw <= 0.0 || bh <= 0.0 {
            continue;
        }
        let corners = oriented_corners(cx as f64, cy as f64, bw as f64, bh as f64, angle as f64);
        candidates.push((anchor, class, score, angle, corners));
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut selected: Vec<OrientedCandidate> = Vec::new();
    for candidate in candidates {
        if selected
            .iter()
            .any(|o| candidate.1 == o.1 && oriented_iou(&candidate.4, &o.4) > iou_threshold as f64)
        {
            continue;
        }
        selected.push(candidate);
        if selected.len() == 300 {
            break;
        }
    }
    let mut data = Vec::new();
    for (_, class, score, angle, corners) in selected {
        let pixels = corners.map(|p| {
            [
                (p[0] - left as f64) / scale as f64,
                (p[1] - top as f64) / scale as f64,
            ]
        });
        let x1 = pixels
            .iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min)
            .clamp(0.0, w as f64);
        let y1 = pixels
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
            .clamp(0.0, h as f64);
        let x2 = pixels
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            .clamp(0.0, w as f64);
        let y2 = pixels
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            .clamp(0.0, h as f64);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        data.push(DetectionItem {
            object: "detection".into(),
            index: data.len(),
            label: DOTA_LABELS[class].into(),
            confidence: score as f64,
            box_: DetectionBox {
                x: (x1 + x2) / 2.0 / w as f64,
                y: (y1 + y2) / 2.0 / h as f64,
                width: (x2 - x1) / w as f64,
                height: (y2 - y1) / h as f64,
            },
            bbox_pixels: DetectionBboxPixels {
                x_min: x1.round() as u32,
                y_min: y1.round() as u32,
                x_max: x2.round() as u32,
                y_max: y2.round() as u32,
            },
            oriented_box: Some(DetectionOrientedBox {
                corners_pixels: pixels,
                angle_radians: angle as f64,
            }),
            mask: None,
            pose: None,
            attributes: None,
        });
    }
    Ok(data)
}

fn decode_semantic(
    values: &[f32],
    geometry: (u32, u32, f32, u32, u32),
) -> Result<(String, Vec<crate::types::SemanticClass>), Box<dyn Error>> {
    use base64::Engine as _;
    if values.len() != (SIDE * SIDE) as usize {
        return Err("ONNX returned invalid semantic map dimensions".into());
    }
    let (w, h, scale, left, top) = geometry;
    let mut map = image::GrayImage::new(SIDE, SIDE);
    for (pixel, value) in map.pixels_mut().zip(values) {
        if !value.is_finite() || *value < 0.0 || *value >= 150.0 || value.fract() != 0.0 {
            return Err("ONNX returned an invalid semantic class ID".into());
        }
        pixel.0[0] = *value as u8;
    }
    let resized_w = ((w as f32 * scale).round() as u32).clamp(1, SIDE);
    let resized_h = ((h as f32 * scale).round() as u32).clamp(1, SIDE);
    let cropped = image::imageops::crop_imm(&map, left, top, resized_w, resized_h).to_image();
    let output = image::imageops::resize(&cropped, w, h, image::imageops::FilterType::Nearest);
    let mut seen = [false; 150];
    for pixel in output.pixels() {
        seen[pixel.0[0] as usize] = true;
    }
    let classes = seen
        .iter()
        .enumerate()
        .filter(|(_, present)| **present)
        .map(|(id, _)| crate::types::SemanticClass {
            id: id as u8,
            label: ade20k::LABELS[id].into(),
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

impl InferenceEngine for OnnxEngine {
    fn id(&self) -> &str {
        &self.id
    }
    fn execute(
        &self,
        task: &InferenceTaskRequest,
        _on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<InferenceOutput, Box<dyn Error>> {
        if let InferenceTaskRequest::PortraitMatting { image_bytes, .. } = task {
            if self.profile != Profile::Matting {
                return Err("ONNX model does not support portrait matting".into());
            }
            let (input, w, h, orientation_applied) = preprocess_matting(image_bytes)?;
            let (values, provider) = self.infer_preprocessed(&input)?;
            let alpha_matte = decode_matte(&values, w, h)?;
            return Ok(InferenceOutput {
                response: InferenceTaskResponse::PortraitMatting(
                    crate::types::PortraitMattingResponse {
                        model: self.id.clone(),
                        execution_provider: provider.into(),
                        width: w,
                        height: h,
                        exif_orientation_applied: orientation_applied,
                        inference_width: MATTING_SIDE,
                        inference_height: MATTING_SIDE,
                        max_source_edge: MATTING_MAX_EDGE,
                        max_source_pixels: MATTING_MAX_PIXELS,
                        alpha_matte,
                    },
                ),
                usage: Usage::new(0, 0),
            });
        }
        if self.profile == Profile::Matting {
            return Err("Portrait matting model requires a matting request".into());
        }
        if let InferenceTaskRequest::ImageSegmentation { image_bytes, .. } = task {
            if self.profile != Profile::Semantic {
                return Err("ONNX model does not support semantic segmentation".into());
            }
            let (input, w, h, scale, left, top) = preprocess(image_bytes, self.profile)?;
            let (values, provider) = self.infer_preprocessed(&input)?;
            let (class_map, classes) = decode_semantic(&values, (w, h, scale, left, top))?;
            return Ok(InferenceOutput {
                response: InferenceTaskResponse::Segmentation(
                    crate::types::ImageSegmentationResponse {
                        model: self.id.clone(),
                        execution_provider: provider.into(),
                        width: w,
                        height: h,
                        class_map,
                        classes,
                    },
                ),
                usage: Usage::new(0, 0),
            });
        }
        if self.profile == Profile::Semantic {
            return Err("Semantic segmentation model requires a segmentation request".into());
        }
        let InferenceTaskRequest::DetectObjects {
            image_bytes,
            confidence_threshold,
            iou_threshold,
            features,
            ..
        } = task
        else {
            return Err("ONNX engine supports object detection only".into());
        };
        if self.profile == Profile::Oriented {
            validate_oriented_task(task)?;
        } else if self.profile == Profile::Segmentation {
            validate_segmentation_task(task)?;
        } else {
            validate_task(task, self.profile == Profile::Pose)?;
        }
        let threshold = confidence_threshold.unwrap_or(0.25);
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("confidence_threshold must be between 0 and 1".into());
        }
        let (input, w, h, scale, left, top) = preprocess(image_bytes, self.profile)?;
        let (values, provider) = self.infer_preprocessed(&input)?;
        let data = match self.profile {
            Profile::Detection => decode_detections(&values, w, h, scale, left, top, threshold)?,
            Profile::Pose => decode_pose(
                &values,
                (w, h, scale, left, top),
                threshold,
                iou_threshold.unwrap_or(0.45),
                features.is_empty() || features.iter().any(|v| v == "keypoints"),
            )?,
            Profile::Segmentation => decode_segmentation(
                &values,
                (w, h, scale, left, top),
                threshold,
                iou_threshold.unwrap_or(0.45),
                features.is_empty() || features.iter().any(|v| v == "masks"),
            )?,
            Profile::Oriented => decode_oriented(
                &values,
                (w, h, scale, left, top),
                threshold,
                iou_threshold.unwrap_or(0.45),
            )?,
            Profile::Semantic => {
                return Err("Semantic segmentation model requires a segmentation request".into());
            }
            Profile::Matting => unreachable!(),
        };
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let count = data.len();
        Ok(InferenceOutput {
            response: InferenceTaskResponse::Detection(DetectObjectsResponse {
                id: format!("det_{created}"),
                object: "list".into(),
                created,
                model: self.id.clone(),
                execution_provider: Some(provider.into()),
                data,
                usage: DetectObjectsUsage {
                    image_width: w,
                    image_height: h,
                    total_objects: count,
                },
            }),
            usage: Usage::new(0, 0),
        })
    }
}

#[cfg(windows)]
pub fn worker(runtime: &Path, model: &Path, cpu: bool) -> anyhow::Result<()> {
    use ort::{session::Session, value::Tensor};
    verify_runtime(runtime, !cpu).map_err(|e| anyhow::anyhow!("{e}"))?;
    let profile = Profile::from_model(model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let environment = ort::init_from(runtime)?.build()?;
    let mut builder = Session::builder(&environment)?;
    if !cpu {
        let mut cuda = ort::ep::CUDA::default();
        if profile == Profile::Matting {
            cuda = cuda
                .with_memory_limit(MATTING_GPU_ARENA_BYTES)
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .with_conv_max_workspace(false);
        }
        builder = builder
            .with_execution_providers([cuda.build().error_on_failure()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let mut session = builder.commit_from_file(model)?;
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    loop {
        let width = if profile == Profile::Matting {
            MATTING_SIDE
        } else {
            SIDE
        };
        let mut bytes = vec![0; (width * profile.height() * 3 * 4) as usize];
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
            [1usize, 3, profile.height() as usize, width as usize],
            input
        ))?])?;
        if profile == Profile::Semantic {
            let (shape, values) = outputs[0].try_extract_tensor::<u8>()?;
            anyhow::ensure!(
                shape.as_ref() == [1, 640, 640],
                "unsupported YOLO semantic output shape: {shape:?}"
            );
            let mut encoded = Vec::with_capacity(values.len() * 4);
            for value in values {
                encoded.extend_from_slice(&(*value as f32).to_le_bytes());
            }
            stdout.write_all(&encoded)?;
            stdout.flush()?;
            continue;
        }
        let index = if profile == Profile::Matting {
            outputs.len() - 1
        } else {
            0
        };
        let (shape, values) = outputs[index].try_extract_tensor::<f32>()?;
        let expected: &[i64] = match profile {
            Profile::Detection => &[1, 300, 6],
            Profile::Pose => &[1, POSE_CHANNELS as i64, POSE_ANCHORS as i64],
            Profile::Segmentation => &[1, 116, 6300],
            Profile::Oriented => &[1, 20, 6300],
            Profile::Semantic => unreachable!(),
            Profile::Matting => &[1, 1, 1024, 1024],
        };
        anyhow::ensure!(
            shape.as_ref() == expected,
            "unsupported YOLO output shape: {shape:?}"
        );
        let mut encoded = Vec::with_capacity(values.len() * 4);
        for value in values {
            encoded.extend_from_slice(&value.to_le_bytes());
        }
        stdout.write_all(&encoded)?;
        if profile == Profile::Segmentation {
            let (shape, values) = outputs[1].try_extract_tensor::<f32>()?;
            anyhow::ensure!(
                shape.as_ref() == [1, 32, 120, 160],
                "unsupported YOLO mask shape: {shape:?}"
            );
            for value in values {
                stdout.write_all(&value.to_le_bytes())?;
            }
        }
        stdout.flush()?;
    }
    Ok(())
}
#[cfg(not(windows))]
pub fn worker(_runtime: &Path, _model: &Path, _cpu: bool) -> anyhow::Result<()> {
    anyhow::bail!("ONNX detection currently supports Windows x64 only")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portrait_matte_is_soft_and_matches_source_dimensions() {
        use base64::Engine as _;
        let mut values = vec![-4.0; (MATTING_SIDE * MATTING_SIDE) as usize];
        for y in 0..MATTING_SIDE as usize {
            for x in MATTING_SIDE as usize / 2..MATTING_SIDE as usize {
                values[y * MATTING_SIDE as usize + x] = 1.0;
            }
        }
        let uri = decode_matte(&values, 400, 300).unwrap();
        let png = base64::engine::general_purpose::STANDARD
            .decode(uri.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let image = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!(image.dimensions(), (400, 300));
        assert!((1..128).contains(&image.get_pixel(10, 150)[0]));
        assert!((128..255).contains(&image.get_pixel(390, 150)[0]));
        values[0] = f32::NAN;
        assert!(decode_matte(&values, 400, 300).is_err());
    }

    #[test]
    fn portrait_preprocessing_keeps_original_dimensions_and_rejects_invalid_images() {
        let image = image::RgbImage::from_pixel(120, 80, image::Rgb([128, 128, 128]));
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let (tensor, width, height, applied) = preprocess_matting(png.get_ref()).unwrap();
        assert_eq!((width, height), (120, 80));
        assert!(!applied);
        assert_eq!(tensor.len(), (MATTING_SIDE * MATTING_SIDE * 3 * 4) as usize);
        assert!(preprocess_matting(b"invalid image").is_err());
    }

    #[test]
    fn portrait_preprocessing_applies_exif_orientation() {
        let image = image::RgbImage::from_pixel(120, 80, image::Rgb([128, 128, 128]));
        let mut jpeg = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut jpeg, image::ImageFormat::Jpeg)
            .unwrap();
        let payload: &[u8] =
            b"Exif\0\0MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x06\0\0\0\0\0\0";
        let mut oriented = vec![0xff, 0xd8, 0xff, 0xe1];
        oriented.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        oriented.extend_from_slice(payload);
        oriented.extend_from_slice(&jpeg.get_ref()[2..]);
        let (_, width, height, applied) = preprocess_matting(&oriented).unwrap();
        assert_eq!((width, height), (80, 120));
        assert!(applied);
    }

    #[test]
    fn semantic_map_removes_padding_and_preserves_class_ids() {
        use base64::Engine as _;
        let mut values = vec![0.0; 640 * 640];
        for y in 160..480 {
            for x in 0..640 {
                values[y * 640 + x] = if x < 320 { 12.0 } else { 2.0 };
            }
        }
        let (uri, classes) = decode_semantic(&values, (4, 2, 160.0, 0, 160)).unwrap();
        assert_eq!(
            classes.iter().map(|c| c.id).collect::<Vec<_>>(),
            vec![2, 12]
        );
        assert_eq!(classes[1].label, "person");
        let png = base64::engine::general_purpose::STANDARD
            .decode(uri.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let map = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!(map.dimensions(), (4, 2));
        assert_eq!(map.get_pixel(0, 0).0[0], 12);
        assert_eq!(map.get_pixel(3, 1).0[0], 2);
        values[0] = 150.0;
        assert!(decode_semantic(&values, (4, 2, 160.0, 0, 160)).is_err());
    }

    #[test]
    fn oriented_boxes_use_rotated_overlap_and_map_letterboxing() {
        let a = oriented_corners(0.0, 0.0, 4.0, 2.0, 0.0);
        let b = oriented_corners(0.0, 0.0, 4.0, 2.0, std::f64::consts::FRAC_PI_2);
        assert!((oriented_iou(&a, &a) - 1.0).abs() < 1e-10);
        assert!((oriented_iou(&a, &b) - 1.0 / 3.0).abs() < 1e-10);
        assert_eq!(
            oriented_iou(&a, &oriented_corners(10.0, 0.0, 4.0, 2.0, 0.0)),
            0.0
        );
        let mut values = vec![0.0; 20 * POSE_ANCHORS];
        for (anchor, class, score) in [(0, 1, 0.9), (1, 1, 0.8), (2, 0, 0.7)] {
            for (channel, value) in [
                (0, 320.0),
                (1, 320.0),
                (2, 160.0),
                (3, 80.0),
                (4 + class, score),
            ] {
                values[channel * POSE_ANCHORS + anchor] = value;
            }
        }
        let data = decode_oriented(&values, (640, 320, 1.0, 0, 160), 0.25, 0.45).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].label, "ship");
        assert_eq!(data[1].label, "plane");
        assert_eq!(
            data[0].oriented_box.as_ref().unwrap().corners_pixels[0],
            [240.0, 120.0]
        );
        assert_eq!(data[0].bbox_pixels.x_min, 240);
        assert!(decode_oriented(&values[..100], (640, 320, 1.0, 0, 160), 0.25, 0.45).is_err());
    }

    #[test]
    fn segmentation_decodes_class_masks_and_suppresses_duplicates() {
        use base64::Engine as _;
        let mut values = vec![0.0; Profile::Segmentation.output_len()];
        for anchor in 0..2 {
            for (channel, value) in [
                (0, 320.0),
                (1, 240.0),
                (2, 160.0),
                (3, 160.0),
                (4, 0.9),
                (84, 1.0),
            ] {
                values[channel * POSE_ANCHORS + anchor] = value;
            }
        }
        values[116 * POSE_ANCHORS..116 * POSE_ANCHORS + 120 * 160].fill(1.0);
        let result = decode_segmentation(&values, (640, 480, 1.0, 0, 0), 0.25, 0.45, true).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "person");
        let data = result[0]
            .mask
            .as_ref()
            .unwrap()
            .strip_prefix("data:image/png;base64,")
            .unwrap();
        let png = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        let mask = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!(mask.dimensions(), (640, 480));
        assert_eq!(mask.get_pixel(320, 240).0[0], 255);
        assert_eq!(mask.get_pixel(0, 0).0[0], 0);
        assert!(
            decode_segmentation(&values[..100], (640, 480, 1.0, 0, 0), 0.25, 0.45, true).is_err()
        );
    }

    #[test]
    fn letterbox_preserves_aspect_ratio_and_rgb_channels() {
        let image = image::RgbImage::from_pixel(4, 2, image::Rgb([255, 0, 0]));
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let (input, width, height, scale, left, top) =
            preprocess(encoded.get_ref(), Profile::Detection).unwrap();
        assert_eq!((width, height, left, top), (4, 2, 0, 160));
        assert_eq!(scale, 160.0);
        let pixel = |channel: usize, x: usize, y: usize| {
            let offset = (channel * 640 * 640 + y * 640 + x) * 4;
            f32::from_le_bytes(input[offset..offset + 4].try_into().unwrap())
        };
        assert!((pixel(0, 0, 0) - 114.0 / 255.0).abs() < 0.00001);
        assert_eq!(pixel(0, 100, 200), 1.0);
        assert_eq!(pixel(1, 100, 200), 0.0);
        assert_eq!(pixel(2, 100, 200), 0.0);
        let (pose_input, _, _, pose_scale, pose_left, pose_top) =
            preprocess(encoded.get_ref(), Profile::Pose).unwrap();
        assert_eq!(pose_input.len(), (3 * 640 * 480 * 4) as usize);
        assert_eq!((pose_scale, pose_left, pose_top), (160.0, 0, 80));
    }

    #[test]
    fn rejects_detection_options_not_supported_by_export() {
        let mut task = InferenceTaskRequest::DetectObjects {
            model: None,
            image_bytes: vec![],
            prompt: None,
            categories: None,
            confidence_threshold: Some(0.25),
            iou_threshold: None,
            features: vec!["boxes".into()],
        };
        assert!(validate_task(&task, false).is_ok());
        if let InferenceTaskRequest::DetectObjects { features, .. } = &mut task {
            features.push("masks".into());
        }
        assert!(validate_task(&task, false).is_err());
    }

    #[test]
    fn pose_decoding_suppresses_overlapping_people_and_maps_keypoints() {
        let mut values = vec![0.0; POSE_CHANNELS * POSE_ANCHORS];
        for (anchor, cx, score) in [(0, 200.0, 0.9), (1, 202.0, 0.8), (2, 500.0, 0.7)] {
            for (channel, value) in [(0, cx), (1, 200.0), (2, 100.0), (3, 120.0), (4, score)] {
                values[channel * POSE_ANCHORS + anchor] = value;
            }
            for point in 0..17 {
                values[(5 + point * 3) * POSE_ANCHORS + anchor] = 128.0;
                values[(6 + point * 3) * POSE_ANCHORS + anchor] = 240.0;
                values[(7 + point * 3) * POSE_ANCHORS + anchor] = 0.75;
            }
        }
        let result = decode_pose(&values, (640, 480, 1.0, 0, 0), 0.25, 0.45, true).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "person");
        assert_eq!(result[0].bbox_pixels.x_min, 150);
        let points = result[0].pose.as_ref().unwrap();
        assert_eq!(points.len(), 17);
        assert_eq!(points[0].name, "nose");
        assert!((points[0].x - 0.2).abs() < 0.00001);
        assert_eq!(points[0].y, 0.5);
        assert_eq!(points[0].confidence, 0.75);
        assert_eq!(points[16].name, "right_ankle");
        assert!(decode_pose(&values[..100], (640, 480, 1.0, 0, 0), 0.25, 0.45, true).is_err());
        values[7 * POSE_ANCHORS] = 2.0;
        assert!(decode_pose(&values, (640, 480, 1.0, 0, 0), 0.25, 0.45, true).is_err());
    }

    #[test]
    fn pose_options_are_distinct_from_detection_options() {
        let task = InferenceTaskRequest::DetectObjects {
            model: None,
            image_bytes: vec![],
            prompt: None,
            categories: None,
            confidence_threshold: Some(0.25),
            iou_threshold: Some(0.45),
            features: vec!["boxes".into(), "keypoints".into()],
        };
        assert!(validate_task(&task, true).is_ok());
        assert!(validate_task(&task, false).is_err());
    }

    #[test]
    fn detections_remove_letterbox_padding_and_validate_output() {
        let values = [
            160.0, 240.0, 480.0, 400.0, 0.9, 0.0, 0.0, 0.0, 1.0, 1.0, 0.1, 0.0,
        ];
        let data = decode_detections(&values, 4, 2, 160.0, 0, 160, 0.25).unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].label, "person");
        assert_eq!(
            (
                data[0].box_.x,
                data[0].box_.y,
                data[0].box_.width,
                data[0].box_.height
            ),
            (0.5, 0.5, 0.5, 0.5)
        );
        assert_eq!(
            (
                data[0].bbox_pixels.x_min,
                data[0].bbox_pixels.y_min,
                data[0].bbox_pixels.x_max,
                data[0].bbox_pixels.y_max
            ),
            (1, 1, 3, 2)
        );
        assert!(decode_detections(&[0.0; 5], 4, 2, 160.0, 0, 160, 0.25).is_err());
        assert!(
            decode_detections(&[0.0, 0.0, 1.0, 1.0, 0.9, 80.0], 4, 2, 160.0, 0, 160, 0.25).is_err()
        );
        assert!(
            decode_detections(
                &[0.0, 0.0, 1.0, 1.0, f32::NAN, 0.0],
                4,
                2,
                160.0,
                0,
                160,
                0.25
            )
            .is_err()
        );
    }
}
