//! Live dashboard: model lifecycle, VRAM, throughput, requests and logs.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Clear, List, ListItem, Paragraph, Row, Sparkline, Table, Wrap};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::Level;

use super::theme::*;
use crate::telemetry::{LogLine, ModelEvent, ModelEventKind, RequestPhase, RequestRecord, UiEvent};

#[derive(Debug, Clone)]
pub struct ModelRow {
    pub id: String,
    pub name: String,
    pub modality: String,
    /// `None` for local models, provider name for cloud models.
    pub provider: Option<String>,
    /// Preferred engine for local models (e.g. `nemo-speech`).
    pub engine: Option<String>,
    pub locality: String,
    pub capabilities: Vec<String>,
    pub tasks: Vec<String>,
    pub total_params: Option<u64>,
    pub benchmarks: Option<crate::schema::BenchmarkMetrics>,
}

/// Model size classification per Artificial Analysis (https://artificialanalysis.ai/)
/// - Tiny: <= 4B parameters
/// - Small: 4B to 40B parameters
/// - Medium: 40B to 150B parameters
/// - Large: > 150B parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SizeCategory {
    Tiny,
    Small,
    Medium,
    Large,
    Unknown,
}

impl SizeCategory {
    pub fn from_model(id: &str, name: &str, explicit_params: Option<u64>) -> (Self, Option<u64>) {
        let p = infer_parameter_count(id, name, explicit_params);
        let cat = match p {
            Some(v) if v <= 4_000_000_000 => Self::Tiny,
            Some(v) if v <= 40_000_000_000 => Self::Small,
            Some(v) if v <= 150_000_000_000 => Self::Medium,
            Some(_) => Self::Large,
            None => Self::Unknown,
        };
        (cat, p)
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Tiny => "Tiny (≤4B)",
            Self::Small => "Small (4B–40B)",
            Self::Medium => "Medium (40B–150B)",
            Self::Large => "Large (>150B)",
            Self::Unknown => "Unknown",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Tiny => "TINY",
            Self::Small => "SMALL",
            Self::Medium => "MEDIUM",
            Self::Large => "LARGE",
            Self::Unknown => "-",
        }
    }
}

pub fn infer_parameter_count(id: &str, name: &str, explicit_params: Option<u64>) -> Option<u64> {
    if let Some(p) = explicit_params {
        if p > 0 {
            return Some(p);
        }
    }
    let lower = format!("{} {}", id.to_lowercase(), name.to_lowercase());
    if lower.contains("gemini-4") {
        return Some(200_000_000_000);
    }
    if lower.contains("gemini-3.8") || lower.contains("flash") {
        return Some(35_000_000_000);
    }
    if lower.contains("nemotron-3.5-super-vl") {
        return Some(120_000_000_000);
    }
    if lower.contains("minicpm-o-4") || lower.contains("minicpm-o-4_5") {
        return Some(4_500_000_000);
    }
    if lower.contains("space-bunny") || lower.contains("dots-3") || lower.contains("apodex") {
        return Some(8_000_000_000);
    }
    for token in lower.split(|c: char| !c.is_alphanumeric() && c != '.') {
        if let Some(num_str) = token.strip_suffix('b') {
            if let Ok(val) = num_str.parse::<f64>() {
                return Some((val * 1_000_000_000.0) as u64);
            }
        }
        if let Some(num_str) = token.strip_suffix('m') {
            if let Ok(val) = num_str.parse::<f64>() {
                return Some((val * 1_000_000.0) as u64);
            }
        }
    }
    if lower.contains("yolo")
        || lower.contains("depth")
        || lower.contains("sam2")
        || lower.contains("dino")
        || lower.contains("matting")
        || lower.contains("bert")
    {
        return Some(250_000_000);
    }
    None
}

/// Real-life task classification grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealLifeTask {
    ImageEditing,
    VideoAndMotion,
    ThreeDAndDepth,
    CodeAndCad,
    ComputerUseAndAgent,
    AudioAndSpeech,
    ImageGeneration,
    VisionAndMultimodal,
    DetectionAndSegmentation,
    EmbeddingAndSearch,
    GeneralText,
}

impl RealLifeTask {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ImageEditing => "Image Editing",
            Self::VideoAndMotion => "Video & Motion",
            Self::ThreeDAndDepth => "3D Modeling & Depth",
            Self::CodeAndCad => "Code & CAD",
            Self::ComputerUseAndAgent => "Computer Use & Tools",
            Self::AudioAndSpeech => "Audio & Voice",
            Self::ImageGeneration => "Image Generation",
            Self::VisionAndMultimodal => "Vision & Multimodal",
            Self::DetectionAndSegmentation => "Detection & Seg",
            Self::EmbeddingAndSearch => "Embeddings & Search",
            Self::GeneralText => "Text & Reasoning",
        }
    }

    pub fn short_badge(&self) -> &'static str {
        match self {
            Self::ImageEditing => "Image Edit",
            Self::VideoAndMotion => "Video",
            Self::ThreeDAndDepth => "3D & Depth",
            Self::CodeAndCad => "Code",
            Self::ComputerUseAndAgent => "Agent/Tools",
            Self::AudioAndSpeech => "Audio",
            Self::ImageGeneration => "Image Gen",
            Self::VisionAndMultimodal => "Vision",
            Self::DetectionAndSegmentation => "Detection",
            Self::EmbeddingAndSearch => "Embedding",
            Self::GeneralText => "Reasoning",
        }
    }

    pub fn sort_priority(&self) -> usize {
        match self {
            Self::ImageEditing => 1,
            Self::VideoAndMotion => 2,
            Self::ThreeDAndDepth => 3,
            Self::CodeAndCad => 4,
            Self::ComputerUseAndAgent => 5,
            Self::AudioAndSpeech => 6,
            Self::ImageGeneration => 7,
            Self::VisionAndMultimodal => 8,
            Self::DetectionAndSegmentation => 9,
            Self::EmbeddingAndSearch => 10,
            Self::GeneralText => 11,
        }
    }
}

pub fn classify_real_life_tasks(
    id: &str,
    tasks: &[String],
    capabilities: &[String],
) -> Vec<RealLifeTask> {
    let lower = format!("{} {} {}", id, tasks.join(" "), capabilities.join(" ")).to_lowercase();
    let mut list = Vec::new();

    if tasks
        .iter()
        .any(|t| t == "text-image-to-image" || t == "image-to-image")
        || lower.contains("image-edit")
    {
        list.push(RealLifeTask::ImageEditing);
    }
    if tasks.iter().any(|t| t == "text-video-to-text")
        || lower.contains("video tracking")
        || lower.contains("video")
    {
        list.push(RealLifeTask::VideoAndMotion);
    }
    if lower.contains("3d")
        || lower.contains("scad")
        || tasks.iter().any(|t| t == "image-to-depth")
        || capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case("depth estimation"))
    {
        list.push(RealLifeTask::ThreeDAndDepth);
    }
    if lower.contains("code") || lower.contains("coding") || lower.contains("coder") {
        list.push(RealLifeTask::CodeAndCad);
    }
    if lower.contains("agent")
        || lower.contains("terminal")
        || lower.contains("computer use")
        || lower.contains("needle")
        || (capabilities
            .iter()
            .any(|c| c.to_lowercase().contains("tool"))
            && lower.contains("reasoning"))
    {
        list.push(RealLifeTask::ComputerUseAndAgent);
    }
    if tasks.iter().any(|t| t.contains("audio"))
        || capabilities.iter().any(|c| {
            let cl = c.to_lowercase();
            cl.contains("audio")
                || cl.contains("voice")
                || cl.contains("speech")
                || cl.contains("diarization")
        })
    {
        list.push(RealLifeTask::AudioAndSpeech);
    }
    if tasks.iter().any(|t| t == "text-to-image")
        || capabilities
            .iter()
            .any(|c| c.contains("Photorealism") || c.contains("Turbo Generation"))
    {
        list.push(RealLifeTask::ImageGeneration);
    }
    if tasks.iter().any(|t| t == "text-image-to-text")
        || capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case("vision"))
    {
        list.push(RealLifeTask::VisionAndMultimodal);
    }
    if tasks
        .iter()
        .any(|t| t.contains("detection") || t.contains("segmentation") || t.contains("matting"))
        || capabilities.iter().any(|c| {
            let cl = c.to_lowercase();
            cl.contains("segmentation") || cl.contains("detection") || cl.contains("matting")
        })
    {
        list.push(RealLifeTask::DetectionAndSegmentation);
    }
    if tasks.iter().any(|t| t == "text-to-embedding")
        || capabilities.iter().any(|c| {
            let cl = c.to_lowercase();
            cl.contains("search") || cl.contains("retrieval") || cl.contains("embedding")
        })
    {
        list.push(RealLifeTask::EmbeddingAndSearch);
    }
    if list.is_empty() {
        list.push(RealLifeTask::GeneralText);
    }
    list
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSort {
    Default,
    IntelligenceIndex,
    FinanceIndex,
    Cost,
    Latency,
    TaskGroup,
    SizeTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskFilter {
    All,
    ImageEditing,
    VideoAndMotion,
    ThreeDAndDepth,
    CodeAndCad,
    ComputerUseAndAgent,
    AudioAndSpeech,
    ImageGeneration,
    VisionAndMultimodal,
    DetectionAndSegmentation,
    EmbeddingAndSearch,
}

impl TaskFilter {
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::All => "All Tasks",
            Self::ImageEditing => "Image Edit",
            Self::VideoAndMotion => "Video",
            Self::ThreeDAndDepth => "3D & Depth",
            Self::CodeAndCad => "Code",
            Self::ComputerUseAndAgent => "Agent/Tools",
            Self::AudioAndSpeech => "Audio",
            Self::ImageGeneration => "Image Gen",
            Self::VisionAndMultimodal => "Vision",
            Self::DetectionAndSegmentation => "Detection",
            Self::EmbeddingAndSearch => "Embedding",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeFilter {
    All,
    Large,
    Medium,
    Small,
    Tiny,
}

impl SizeFilter {
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::All => "All Sizes",
            Self::Large => "Large (>150B)",
            Self::Medium => "Medium (40B-150B)",
            Self::Small => "Small (4B-40B)",
            Self::Tiny => "Tiny (≤4B)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DashboardInfo {
    pub version: &'static str,
    pub addr: String,
    pub auth_enabled: bool,
    pub exposed_to_network: bool,
    pub config_path: PathBuf,
    pub root: PathBuf,
    pub server_path: PathBuf,
    pub model_registry_path: Option<PathBuf>,
    pub models: Vec<ModelRow>,
}

#[derive(Debug, Clone)]
enum Status {
    Idle,
    Loading(Instant),
    Loaded { load_ms: u64, detail: String },
    Failed(String),
}

#[derive(Debug, Clone)]
struct ModelState {
    row: ModelRow,
    status: Status,
    requests: u64,
    errors: u64,
    tps_sum: f64,
    tps_n: u64,
    last_used: Option<Instant>,
    original_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dashboard,
    Models,
    Engines,
    Logs,
}

const MAX_LOGS: usize = 2000;
const MAX_REQUESTS: usize = 100;
const HISTORY: usize = 80;

struct App {
    info: DashboardInfo,
    models: Vec<ModelState>,
    model_selected: usize,
    model_sort: ModelSort,
    benchmark_view: bool,
    task_filter: TaskFilter,
    size_filter: SizeFilter,
    model_action: Option<ModelAction>,
    model_action_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    model_action_status: Option<String>,
    model_registry_open: bool,
    model_registry_refresh: bool,
    model_registry_selected: usize,
    model_registry_models: Vec<crate::model_registry::LocalModel>,
    model_registry_rx:
        Option<std::sync::mpsc::Receiver<Result<Vec<crate::model_registry::LocalModel>, String>>>,
    model_registry_status: Option<String>,
    model_install_requested: bool,
    model_install_rx: Option<std::sync::mpsc::Receiver<ModelInstallEvent>>,
    model_install_cancel: Option<Arc<AtomicBool>>,
    model_download: Option<crate::inference::engines::download::DownloadProgress>,
    engines: Vec<EngineRow>,
    engine_selected: usize,
    refresh_engines: bool,
    update_requested: bool,
    update_cancel: Option<Arc<AtomicBool>>,
    engine_download: Option<crate::inference::engines::download::DownloadProgress>,
    update_status: Option<String>,
    update_engine: Option<String>,
    update_rx: Option<std::sync::mpsc::Receiver<crate::inference::engines::update::UpdateEvent>>,
    engine_rx: Option<std::sync::mpsc::Receiver<Vec<EngineRow>>>,
    index: HashMap<String, usize>,
    requests: VecDeque<RequestRecord>,
    logs: VecDeque<LogLine>,
    /// Fixed-point thousandths of a token/second preserve fractional rates.
    tps_hist: VecDeque<u64>,
    latency_hist: VecDeque<u64>,
    total_requests: u64,
    total_errors: u64,
    vram: Arc<Mutex<Option<(u64, u64)>>>,
    tab: Tab,
    errors_only: bool,
    help: bool,
    log_scroll: usize,
    follow: bool,
    started: Instant,
    tick: u64,
    quit: bool,
}

#[derive(Clone, Copy)]
enum ModelAction {
    Load,
    Unload,
}

enum ModelInstallEvent {
    Progress(crate::inference::engines::download::DownloadProgress),
    Finished(Result<String, String>),
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(cancel) = &self.update_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
        if let Some(cancel) = &self.model_install_cancel {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl App {
    fn begin_engine_operation(&mut self) -> Arc<AtomicBool> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.update_cancel = Some(cancel.clone());
        cancel
    }

    fn engine_cancelling(&self) -> bool {
        self.update_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Relaxed))
    }

    fn new(info: DashboardInfo, vram: Arc<Mutex<Option<(u64, u64)>>>) -> Self {
        let models: Vec<ModelState> = info
            .models
            .iter()
            .cloned()
            .enumerate()
            .map(|(original_index, row)| ModelState {
                row,
                status: Status::Idle,
                requests: 0,
                errors: 0,
                tps_sum: 0.0,
                tps_n: 0,
                last_used: None,
                original_index,
            })
            .collect();
        let index = models
            .iter()
            .enumerate()
            .map(|(i, m)| (m.row.id.clone(), i))
            .collect();
        Self {
            info,
            models,
            model_selected: 0,
            model_sort: ModelSort::Default,
            benchmark_view: false,
            task_filter: TaskFilter::All,
            size_filter: SizeFilter::All,
            model_action: None,
            model_action_rx: None,
            model_action_status: None,
            model_registry_open: false,
            model_registry_refresh: false,
            model_registry_selected: 0,
            model_registry_models: Vec::new(),
            model_registry_rx: None,
            model_registry_status: None,
            model_install_requested: false,
            model_install_rx: None,
            model_install_cancel: None,
            model_download: None,
            engines: Vec::new(),
            engine_selected: 0,
            refresh_engines: false,
            update_requested: false,
            update_cancel: None,
            engine_download: None,
            update_status: None,
            update_engine: None,
            update_rx: None,
            engine_rx: None,
            index,
            requests: VecDeque::new(),
            logs: VecDeque::new(),
            tps_hist: VecDeque::new(),
            latency_hist: VecDeque::new(),
            total_requests: 0,
            total_errors: 0,
            vram,
            tab: Tab::Dashboard,
            errors_only: false,
            help: false,
            log_scroll: 0,
            follow: true,
            started: Instant::now(),
            tick: 0,
            quit: false,
        }
    }

    fn model_mut(&mut self, id: &str) -> Option<&mut ModelState> {
        let i = *self.index.get(id)?;
        self.models.get_mut(i)
    }

    fn selected_model_action(&self) -> Option<ModelAction> {
        if self.model_action_rx.is_some() {
            return None;
        }
        let indices = self.filtered_model_indices();
        let idx = *indices.get(self.model_selected)?;
        match &self.models.get(idx)?.status {
            Status::Loaded { .. } => Some(ModelAction::Unload),
            Status::Loading(_) => None,
            Status::Idle | Status::Failed(_) => Some(ModelAction::Load),
        }
    }

    fn sync_configured_models(&mut self, state: Option<&Arc<crate::inference::AppState>>) {
        let Some(state) = state else {
            return;
        };
        let Ok(registry) = state.registry.lock() else {
            return;
        };
        for model in &registry.config.models {
            if self.index.contains_key(&model.id) {
                continue;
            }
            let row = ModelRow {
                id: model.id.clone(),
                name: model.name.clone(),
                modality: model.modality.as_str().to_string(),
                provider: (!model.is_local()).then(|| model.provider.as_str().to_string()),
                engine: model
                    .is_local()
                    .then(|| model.engine_plan()[0].as_str().to_string()),
                locality: if model.is_local() {
                    "LOCAL".into()
                } else {
                    "CLOUD".into()
                },
                capabilities: model.capabilities.clone().unwrap_or_default(),
                tasks: model.tasks.clone(),
                total_params: model.total_params,
                benchmarks: model.benchmarks.clone(),
            };
            let original_index = self.models.len();
            self.index.insert(row.id.clone(), self.models.len());
            self.models.push(ModelState {
                row,
                status: Status::Idle,
                requests: 0,
                errors: 0,
                tps_sum: 0.0,
                tps_n: 0,
                last_used: None,
                original_index,
            });
        }
    }

    fn cycle_model_sort(&mut self) {
        self.model_sort = match self.model_sort {
            ModelSort::Default => ModelSort::IntelligenceIndex,
            ModelSort::IntelligenceIndex => ModelSort::FinanceIndex,
            ModelSort::FinanceIndex => ModelSort::Cost,
            ModelSort::Cost => ModelSort::Latency,
            ModelSort::Latency => ModelSort::TaskGroup,
            ModelSort::TaskGroup => ModelSort::SizeTier,
            ModelSort::SizeTier => ModelSort::Default,
        };
        self.apply_model_sort();
    }

    fn cycle_task_filter(&mut self) {
        self.task_filter = match self.task_filter {
            TaskFilter::All => TaskFilter::ImageEditing,
            TaskFilter::ImageEditing => TaskFilter::VideoAndMotion,
            TaskFilter::VideoAndMotion => TaskFilter::ThreeDAndDepth,
            TaskFilter::ThreeDAndDepth => TaskFilter::CodeAndCad,
            TaskFilter::CodeAndCad => TaskFilter::ComputerUseAndAgent,
            TaskFilter::ComputerUseAndAgent => TaskFilter::AudioAndSpeech,
            TaskFilter::AudioAndSpeech => TaskFilter::ImageGeneration,
            TaskFilter::ImageGeneration => TaskFilter::VisionAndMultimodal,
            TaskFilter::VisionAndMultimodal => TaskFilter::DetectionAndSegmentation,
            TaskFilter::DetectionAndSegmentation => TaskFilter::EmbeddingAndSearch,
            TaskFilter::EmbeddingAndSearch => TaskFilter::All,
        };
        self.model_selected = 0;
    }

    fn cycle_size_filter(&mut self) {
        self.size_filter = match self.size_filter {
            SizeFilter::All => SizeFilter::Large,
            SizeFilter::Large => SizeFilter::Medium,
            SizeFilter::Medium => SizeFilter::Small,
            SizeFilter::Small => SizeFilter::Tiny,
            SizeFilter::Tiny => SizeFilter::All,
        };
        self.model_selected = 0;
    }

    fn filtered_model_indices(&self) -> Vec<usize> {
        self.models
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                let (size, _) =
                    SizeCategory::from_model(&m.row.id, &m.row.name, m.row.total_params);
                let size_match = match self.size_filter {
                    SizeFilter::All => true,
                    SizeFilter::Large => size == SizeCategory::Large,
                    SizeFilter::Medium => size == SizeCategory::Medium,
                    SizeFilter::Small => size == SizeCategory::Small,
                    SizeFilter::Tiny => size == SizeCategory::Tiny,
                };
                let tasks = classify_real_life_tasks(&m.row.id, &m.row.tasks, &m.row.capabilities);
                let task_match = match self.task_filter {
                    TaskFilter::All => true,
                    TaskFilter::ImageEditing => tasks.contains(&RealLifeTask::ImageEditing),
                    TaskFilter::VideoAndMotion => tasks.contains(&RealLifeTask::VideoAndMotion),
                    TaskFilter::ThreeDAndDepth => tasks.contains(&RealLifeTask::ThreeDAndDepth),
                    TaskFilter::CodeAndCad => tasks.contains(&RealLifeTask::CodeAndCad),
                    TaskFilter::ComputerUseAndAgent => {
                        tasks.contains(&RealLifeTask::ComputerUseAndAgent)
                    }
                    TaskFilter::AudioAndSpeech => tasks.contains(&RealLifeTask::AudioAndSpeech),
                    TaskFilter::ImageGeneration => tasks.contains(&RealLifeTask::ImageGeneration),
                    TaskFilter::VisionAndMultimodal => {
                        tasks.contains(&RealLifeTask::VisionAndMultimodal)
                    }
                    TaskFilter::DetectionAndSegmentation => {
                        tasks.contains(&RealLifeTask::DetectionAndSegmentation)
                    }
                    TaskFilter::EmbeddingAndSearch => {
                        tasks.contains(&RealLifeTask::EmbeddingAndSearch)
                    }
                };
                size_match && task_match
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn apply_model_sort(&mut self) {
        let selected_id = self
            .models
            .get(self.model_selected)
            .map(|m| m.row.id.clone());
        match self.model_sort {
            ModelSort::Default => {
                self.models.sort_by_key(|m| m.original_index);
            }
            ModelSort::IntelligenceIndex => {
                self.models.sort_by(|a, b| {
                    let score_a = a
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.score)
                        .unwrap_or(0.0);
                    let score_b = b
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.score)
                        .unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ModelSort::FinanceIndex => {
                self.models.sort_by(|a, b| {
                    let score_a = a
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.domains.as_ref().and_then(|d| d.finance_and_accounting))
                        .unwrap_or(0.0);
                    let score_b = b
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.domains.as_ref().and_then(|d| d.finance_and_accounting))
                        .unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ModelSort::Cost => {
                self.models.sort_by(|a, b| {
                    let cost_a = a
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.cost_per_task_usd)
                        .unwrap_or(999.0);
                    let cost_b = b
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.cost_per_task_usd)
                        .unwrap_or(999.0);
                    cost_a
                        .partial_cmp(&cost_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ModelSort::Latency => {
                self.models.sort_by(|a, b| {
                    let time_a = a
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.time_per_task_minutes)
                        .unwrap_or(999.0);
                    let time_b = b
                        .row
                        .benchmarks
                        .as_ref()
                        .and_then(|bm| bm.time_per_task_minutes)
                        .unwrap_or(999.0);
                    time_a
                        .partial_cmp(&time_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ModelSort::TaskGroup => {
                self.models.sort_by(|a, b| {
                    let tasks_a =
                        classify_real_life_tasks(&a.row.id, &a.row.tasks, &a.row.capabilities);
                    let tasks_b =
                        classify_real_life_tasks(&b.row.id, &b.row.tasks, &b.row.capabilities);
                    let prio_a = tasks_a
                        .first()
                        .copied()
                        .unwrap_or(RealLifeTask::GeneralText)
                        .sort_priority();
                    let prio_b = tasks_b
                        .first()
                        .copied()
                        .unwrap_or(RealLifeTask::GeneralText)
                        .sort_priority();
                    prio_a
                        .cmp(&prio_b)
                        .then_with(|| a.original_index.cmp(&b.original_index))
                });
            }
            ModelSort::SizeTier => {
                self.models.sort_by(|a, b| {
                    let (size_a, p_a) =
                        SizeCategory::from_model(&a.row.id, &a.row.name, a.row.total_params);
                    let (size_b, p_b) =
                        SizeCategory::from_model(&b.row.id, &b.row.name, b.row.total_params);
                    size_b
                        .cmp(&size_a)
                        .then_with(|| p_b.unwrap_or(0).cmp(&p_a.unwrap_or(0)))
                });
            }
        }
        self.index.clear();
        for (i, m) in self.models.iter().enumerate() {
            self.index.insert(m.row.id.clone(), i);
        }
        if let Some(id) = selected_id {
            if let Some(&new_pos) = self.index.get(&id) {
                self.model_selected = new_pos;
            }
        }
    }

    fn process_model_registry(&mut self) {
        if self.model_registry_refresh && self.model_registry_rx.is_none() {
            self.model_registry_refresh = false;
            self.model_registry_status = Some("Loading model registry…".into());
            let path = self.info.model_registry_path.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let result = match path {
                    Some(path) => crate::model_registry::load_from_path(&path),
                    None => crate::model_registry::fetch(),
                }
                .map_err(|error| format!("{error:#}"));
                let _ = tx.send(result);
            });
            self.model_registry_rx = Some(rx);
        }
        if let Some(rx) = &self.model_registry_rx {
            match rx.try_recv() {
                Ok(Ok(models)) => {
                    self.model_registry_models = models;
                    self.model_registry_selected = self
                        .model_registry_selected
                        .min(self.model_registry_models.len().saturating_sub(1));
                    self.model_registry_status = Some(if self.model_registry_models.is_empty() {
                        "No downloadable models listed".into()
                    } else {
                        "Select a model and press Enter to install".into()
                    });
                    self.model_registry_rx = None;
                }
                Ok(Err(error)) => {
                    self.model_registry_models.clear();
                    self.model_registry_status = Some(format!("Registry failed: {error}"));
                    self.model_registry_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.model_registry_status =
                        Some("Registry worker stopped unexpectedly".into());
                    self.model_registry_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if self.model_install_requested {
            self.model_install_requested = false;
            if self.model_registry_rx.is_some() {
                self.model_registry_status = Some("Wait for the registry to load".into());
            } else if let Some(model) = self
                .model_registry_models
                .get(self.model_registry_selected)
                .cloned()
            {
                if self.index.contains_key(&model.id) {
                    self.model_registry_status = Some(
                        "Model ID already configured; existing configuration preserved".into(),
                    );
                } else {
                    let root = self.info.root.clone();
                    let server = self.info.server_path.clone();
                    let catalog = self.info.config_path.clone();
                    let cancel = Arc::new(AtomicBool::new(false));
                    let worker_cancel = cancel.clone();
                    let (tx, rx) = std::sync::mpsc::channel();
                    self.model_registry_status = Some(format!("Installing {}…", model.id));
                    self.model_install_cancel = Some(cancel);
                    self.model_install_rx = Some(rx);
                    self.model_download = None;
                    std::thread::spawn(move || {
                        let progress_tx = tx.clone();
                        let result = crate::inference::engines::download::with_cancellation(
                            worker_cancel,
                            || {
                                crate::inference::engines::download::with_progress(
                                    move |progress| {
                                        let _ =
                                            progress_tx.send(ModelInstallEvent::Progress(progress));
                                    },
                                    || {
                                        crate::model_registry::install(
                                            &model, &root, &server, &catalog,
                                        )
                                    },
                                )
                            },
                        )
                        .map(|_| format!("Installed {}; waiting for catalog reload", model.id))
                        .map_err(|error| format!("{error:#}"));
                        let _ = tx.send(ModelInstallEvent::Finished(result));
                    });
                }
            } else {
                self.model_registry_status = Some("Select a model from the registry".into());
            }
        }
        if let Some(rx) = &self.model_install_rx {
            loop {
                match rx.try_recv() {
                    Ok(ModelInstallEvent::Progress(progress)) => {
                        self.model_download = Some(progress)
                    }
                    Ok(ModelInstallEvent::Finished(result)) => {
                        let cancelled = self
                            .model_install_cancel
                            .as_ref()
                            .is_some_and(|cancel| cancel.load(Ordering::Relaxed));
                        self.model_registry_status = Some(match result {
                            Ok(message) => message,
                            Err(_) if cancelled => {
                                "Installation cancelled; partial download saved for retry".into()
                            }
                            Err(error) => format!("Installation failed: {error}"),
                        });
                        self.model_install_rx = None;
                        self.model_install_cancel = None;
                        self.model_download = None;
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.model_registry_status =
                            Some("Model installer stopped unexpectedly".into());
                        self.model_install_rx = None;
                        self.model_install_cancel = None;
                        self.model_download = None;
                        break;
                    }
                }
            }
        }
    }

    fn process_model_action(&mut self, state: Option<&Arc<crate::inference::AppState>>) {
        if let Some(action) = self.model_action.take() {
            if self.model_action_rx.is_none() {
                match (self.models.get(self.model_selected), state) {
                    (Some(model), Some(state)) if model.row.provider.is_none() => {
                        let id = model.row.id.clone();
                        self.model_action_status = Some(format!(
                            "{}: {id}…",
                            match action {
                                ModelAction::Load => "Loading",
                                ModelAction::Unload => "Unloading",
                            }
                        ));
                        let state = state.clone();
                        let (tx, rx) = std::sync::mpsc::channel();
                        std::thread::spawn(move || {
                            let result = match action {
                                ModelAction::Load => {
                                    state.load_model(&id).map(|()| format!("Loaded: {id}"))
                                }
                                ModelAction::Unload => state.unload_model(&id).map(|unloaded| {
                                    format!(
                                        "{}: {id}",
                                        if unloaded {
                                            "Unloaded"
                                        } else {
                                            "Already unloaded"
                                        }
                                    )
                                }),
                            };
                            let _ = tx.send(result);
                        });
                        self.model_action_rx = Some(rx);
                    }
                    (Some(model), _) if model.row.provider.is_some() => {
                        self.model_action_status =
                            Some("Cloud models cannot be loaded or unloaded here".into());
                    }
                    _ => {
                        self.model_action_status =
                            Some("Select a local model; server state is required".into())
                    }
                }
            }
        }
        if let Some(rx) = &self.model_action_rx {
            match rx.try_recv() {
                Ok(result) => {
                    self.model_action_status =
                        Some(result.unwrap_or_else(|error| format!("Failed: {error}")));
                    self.model_action_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.model_action_status = Some("Model worker stopped unexpectedly".into());
                    self.model_action_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }

    fn on_event(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Log(l) => {
                if self.logs.len() >= MAX_LOGS {
                    self.logs.pop_front();
                }
                self.logs.push_back(l);
            }
            UiEvent::Model(m) => self.on_model(m),
            UiEvent::Request(r) => self.on_request(r),
        }
    }

    fn on_model(&mut self, e: ModelEvent) {
        if let Some(m) = self.model_mut(&e.model) {
            m.status = match e.kind {
                ModelEventKind::Loading => Status::Loading(Instant::now()),
                ModelEventKind::Loaded => Status::Loaded {
                    load_ms: e.duration_ms.unwrap_or(0),
                    detail: e.detail,
                },
                ModelEventKind::Evicted => Status::Idle,
                ModelEventKind::Failed => Status::Failed(e.detail),
            };
        }
    }

    fn on_request(&mut self, r: RequestRecord) {
        let existing = (r.id != 0)
            .then(|| self.requests.iter().position(|entry| entry.id == r.id))
            .flatten();
        if existing.is_some_and(|i| self.requests[i].phase.is_finished()) {
            return;
        }
        let finished = r.phase.is_finished();
        if !finished {
            if let Some(i) = existing {
                let timestamp = self.requests[i].ts.clone();
                self.requests[i] = r;
                self.requests[i].ts = timestamp;
            } else {
                self.requests.push_back(r);
            }
            return;
        }
        self.total_requests += 1;
        let failed = r.status >= 400 || r.error.is_some();
        if failed {
            self.total_errors += 1;
        }
        let tps = r.tokens_per_second();
        if let Some(m) = self.model_mut(&r.model) {
            m.requests += 1;
            m.last_used = Some(Instant::now());
            if failed {
                m.errors += 1;
            }
            if let Some(t) = tps {
                m.tps_sum += t;
                m.tps_n += 1;
            }
        }
        if let Some(t) = tps {
            push_capped(&mut self.tps_hist, (t * 1000.0).round() as u64, HISTORY);
        }
        if !failed {
            push_capped(&mut self.latency_hist, r.latency_ms, HISTORY);
        }
        if let Some(i) = existing {
            let timestamp = self.requests[i].ts.clone();
            self.requests[i] = r;
            self.requests[i].ts = timestamp;
        } else {
            self.requests.push_back(r);
        }
        while self.requests.len() > MAX_REQUESTS {
            let Some(i) = self.requests.iter().position(|r| r.phase.is_finished()) else {
                break;
            };
            self.requests.remove(i);
        }
    }

    fn on_key(&mut self, code: KeyCode, mods: KeyModifiers) {
        if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        if self.tab == Tab::Models && self.model_registry_open {
            match code {
                KeyCode::Char('i') | KeyCode::Esc => {
                    self.model_registry_open = false;
                    return;
                }
                KeyCode::Char('r') => {
                    self.model_registry_refresh = true;
                    return;
                }
                KeyCode::Up => {
                    self.model_registry_selected = self.model_registry_selected.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    self.model_registry_selected = (self.model_registry_selected + 1)
                        .min(self.model_registry_models.len().saturating_sub(1));
                    return;
                }
                KeyCode::Enter if self.model_install_rx.is_none() => {
                    self.model_install_requested = true;
                    return;
                }
                KeyCode::Char('c') => {
                    if let Some(cancel) = &self.model_install_cancel {
                        cancel.store(true, Ordering::Relaxed);
                        self.model_registry_status = Some("Cancelling model installation".into());
                    }
                    return;
                }
                _ => {}
            }
        }
        match code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('l') => {
                self.tab = if self.tab == Tab::Logs {
                    Tab::Dashboard
                } else {
                    Tab::Logs
                }
            }
            KeyCode::Char('m') => {
                self.tab = if self.tab == Tab::Models {
                    Tab::Dashboard
                } else {
                    Tab::Models
                }
            }
            KeyCode::Char('e') => {
                self.tab = if self.tab == Tab::Engines {
                    Tab::Dashboard
                } else {
                    Tab::Engines
                }
            }
            KeyCode::Char('r') if self.tab == Tab::Engines => self.refresh_engines = true,
            KeyCode::Char('i') if self.tab == Tab::Models => {
                self.model_registry_open = true;
                self.model_registry_refresh = true;
            }
            KeyCode::Char('s') if self.tab == Tab::Models => {
                self.cycle_model_sort();
            }
            KeyCode::Char('b') if self.tab == Tab::Models => {
                self.benchmark_view = !self.benchmark_view;
            }
            KeyCode::Char('g') if self.tab == Tab::Models => {
                self.cycle_task_filter();
            }
            KeyCode::Char('z') if self.tab == Tab::Models => {
                self.cycle_size_filter();
            }
            KeyCode::Enter if self.tab == Tab::Models && self.model_action_rx.is_none() => {
                self.model_action = self.selected_model_action();
            }
            KeyCode::Up if self.tab == Tab::Models => {
                self.model_selected = self.model_selected.saturating_sub(1);
            }
            KeyCode::Down if self.tab == Tab::Models => {
                let total = self.filtered_model_indices().len();
                self.model_selected = (self.model_selected + 1).min(total.saturating_sub(1));
            }
            KeyCode::Char('u') if self.tab == Tab::Engines => self.update_requested = true,
            KeyCode::Char('c') if self.tab == Tab::Engines => {
                if let Some(cancel) = &self.update_cancel {
                    cancel.store(true, Ordering::Relaxed);
                    self.update_status =
                        Some("Cancelling engine operation; waiting for current step".into());
                    self.engine_download = None;
                }
            }
            KeyCode::Up if self.tab == Tab::Engines => {
                self.engine_selected = self.engine_selected.saturating_sub(1)
            }
            KeyCode::Down if self.tab == Tab::Engines => {
                self.engine_selected =
                    (self.engine_selected + 1).min(self.engines.len().saturating_sub(1))
            }
            KeyCode::Char('w') if self.tab == Tab::Logs => {
                self.errors_only = !self.errors_only;
                self.follow = true;
            }
            KeyCode::Char('c') if self.tab == Tab::Logs => {
                self.logs.clear();
                self.log_scroll = 0;
            }
            KeyCode::Up if self.tab == Tab::Logs => {
                self.follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(1);
            }
            KeyCode::Down if self.tab == Tab::Logs => self.log_scroll += 1,
            KeyCode::PageUp if self.tab == Tab::Logs => {
                self.follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(15);
            }
            KeyCode::PageDown if self.tab == Tab::Logs => self.log_scroll += 15,
            KeyCode::Char('f') if self.tab == Tab::Logs => self.follow = true,
            _ => {}
        }
    }
}

fn push_capped(q: &mut VecDeque<u64>, v: u64, cap: usize) {
    if q.len() >= cap {
        q.pop_front();
    }
    q.push_back(v);
}

/// Run the dashboard until the user quits. Blocks the calling thread.
pub fn run_dashboard(
    terminal: &mut DefaultTerminal,
    info: DashboardInfo,
    rx: UnboundedReceiver<UiEvent>,
) -> Result<()> {
    run_dashboard_inner(terminal, info, rx, None)
}

pub fn run_dashboard_with_state(
    terminal: &mut DefaultTerminal,
    info: DashboardInfo,
    rx: UnboundedReceiver<UiEvent>,
    state: Arc<crate::inference::AppState>,
) -> Result<()> {
    run_dashboard_inner(terminal, info, rx, Some(state))
}

fn run_dashboard_inner(
    terminal: &mut DefaultTerminal,
    info: DashboardInfo,
    mut rx: UnboundedReceiver<UiEvent>,
    state: Option<Arc<crate::inference::AppState>>,
) -> Result<()> {
    let vram = Arc::new(Mutex::new(None));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    spawn_vram_poller(vram.clone(), stop.clone());

    let mut app = App::new(info, vram);
    app.engine_rx = Some(start_engine_inventory());
    while !app.quit {
        app.process_model_action(state.as_ref());
        app.process_model_registry();
        if app.tick.is_multiple_of(15) {
            app.sync_configured_models(state.as_ref());
        }
        if app.refresh_engines && app.engine_rx.is_none() {
            app.engine_rx = Some(start_engine_inventory());
            app.refresh_engines = false;
        }
        if app.update_requested && app.update_rx.is_some() {
            app.update_requested = false;
        }
        if app.update_requested && app.update_rx.is_none() {
            app.update_requested = false;
            let selected = app.engines.get(app.engine_selected).cloned();
            app.update_engine = selected.as_ref().map(|engine| engine.name.to_string());
            app.engine_download = None;
            match (
                selected.as_ref().map(engine_action),
                selected,
                state.clone(),
            ) {
                (Some(EngineAction::Unavailable(reason)), _, _) => {
                    app.update_status = Some(reason);
                }
                (Some(EngineAction::Install), Some(engine), Some(state)) => {
                    if let Some(kind) = install_engine_kind(engine.name) {
                        let (tx, rx) = std::sync::mpsc::channel();
                        let cancel = app.begin_engine_operation();
                        std::thread::spawn(move || {
                            let result = crate::inference::engines::download::with_cancellation(
                                cancel.clone(),
                                || {
                                    crate::inference::engines::update::install_engine(
                                        state,
                                        cancel,
                                        kind,
                                        tx.clone(),
                                    )
                                },
                            );
                            let _ = tx.send(
                                crate::inference::engines::update::UpdateEvent::Finished(result),
                            );
                        });
                        app.update_rx = Some(rx);
                        app.update_status = Some("Preparing installation".into());
                    } else {
                        app.update_status =
                            Some("Installation is unavailable for this engine".into());
                    }
                }

                (Some(EngineAction::Update), Some(engine), Some(state)) => {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let cancel = app.begin_engine_operation();
                    std::thread::spawn(move || {
                        let result = crate::inference::engines::download::with_cancellation(
                            cancel.clone(),
                            || {
                                if matches!(engine.name, "llama-server" | "llama-server-prism") {
                                    let expected_registry_release =
                                        if engine.name == "llama-server-prism" {
                                            engine.registry_revisions.as_ref().and_then(
                                                |versions| versions.get(&engine.latest).copied(),
                                            )
                                        } else {
                                            engine.registry_versions.as_ref().and_then(|versions| {
                                                engine_build_identity(&engine.latest).and_then(
                                                    |(build, _)| versions.get(&build).copied(),
                                                )
                                            })
                                        };
                                    crate::inference::engines::update::update_llama(
                                        state,
                                        if engine.name == "llama-server-prism" {
                                            crate::config::EngineKind::LlamaServerPrism
                                        } else {
                                            crate::config::EngineKind::LlamaServer
                                        },
                                        cancel,
                                        engine.latest,
                                        engine.build_tag,
                                        expected_registry_release,
                                        tx.clone(),
                                    )
                                } else if engine.name == "needle" {
                                    crate::inference::engines::update::update_needle(
                                        state,
                                        cancel,
                                        engine.latest,
                                        tx.clone(),
                                    )
                                } else if engine.name == "sd-server" {
                                    let expected_release =
                                        engine.registry_revisions.as_ref().and_then(|versions| {
                                            crate::inference::engines::sd_server::release_commit(
                                                &engine.latest,
                                            )
                                            .and_then(|commit| versions.get(commit).copied())
                                        });
                                    crate::inference::engines::update::update_sd(
                                        state,
                                        cancel,
                                        engine.latest,
                                        expected_release,
                                        tx.clone(),
                                    )
                                } else if engine.name == "nemo-speech" {
                                    let expected_release =
                                        engine.registry_revisions.as_ref().and_then(|versions| {
                                            versions
                                                .get(engine.latest.trim_start_matches('v'))
                                                .copied()
                                        });
                                    crate::inference::engines::update::update_nemo(
                                        state,
                                        cancel,
                                        engine.latest,
                                        expected_release,
                                        tx.clone(),
                                    )
                                } else if engine.name == "whisper-cpp" {
                                    crate::inference::engines::update::update_whisper(
                                        state,
                                        cancel,
                                        engine.latest,
                                        tx.clone(),
                                    )
                                } else if engine.name == "crispasr" {
                                    let expected_release =
                                        engine.registry_revisions.as_ref().and_then(|versions| {
                                            versions
                                                .get(engine.latest.trim_start_matches('v'))
                                                .copied()
                                        });
                                    crate::inference::engines::update::update_crispasr(
                                        state,
                                        cancel,
                                        engine.latest,
                                        expected_release,
                                        tx.clone(),
                                    )
                                } else {
                                    let expected_release =
                                        engine.registry_revisions.as_ref().and_then(|versions| {
                                            versions
                                                .get(engine.latest.trim_start_matches('v'))
                                                .copied()
                                        });
                                    crate::inference::engines::update::update_audio(
                                        state,
                                        cancel,
                                        engine.latest,
                                        expected_release,
                                        tx.clone(),
                                    )
                                }
                            },
                        );
                        let _ = tx.send(crate::inference::engines::update::UpdateEvent::Finished(
                            result,
                        ));
                    });
                    app.update_rx = Some(rx);
                    app.update_status = Some("Preparing update".into());
                }
                (_, None, _) => app.update_status = Some("Select an engine".into()),
                _ => {
                    app.update_status = Some("Managed server state is required".into());
                }
            }
        }
        if app.update_rx.is_some() {
            loop {
                match app.update_rx.as_ref().unwrap().try_recv() {
                    Ok(crate::inference::engines::update::UpdateEvent::Progress(status)) => {
                        if !app.engine_cancelling() {
                            app.update_status = Some(status);
                            app.engine_download = None;
                        }
                    }
                    Ok(crate::inference::engines::update::UpdateEvent::Download(progress)) => {
                        if !app.engine_cancelling() {
                            app.engine_download = Some(progress);
                        }
                    }
                    Ok(crate::inference::engines::update::UpdateEvent::Finished(result)) => {
                        app.update_status = Some(match result {
                            Ok(message) => {
                                app.refresh_engines = true;
                                message
                            }
                            Err(error)
                                if matches!(
                                    error.as_str(),
                                    "Update cancelled"
                                        | "Installation cancelled"
                                        | "Engine operation cancelled"
                                ) =>
                            {
                                error
                            }
                            Err(error) => {
                                tracing::error!(engine = ?app.update_engine, %error, "Engine installation/update failed");
                                format!("Failed: {error}")
                            }
                        });
                        app.update_rx = None;
                        app.update_cancel = None;
                        app.engine_download = None;
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        app.update_status = Some("Update worker stopped unexpectedly".into());
                        app.update_rx = None;
                        app.update_cancel = None;
                        app.engine_download = None;
                        break;
                    }
                }
            }
        }
        while let Some(result) = app.engine_rx.as_ref().map(|rx| rx.try_recv()) {
            match result {
                Ok(rows) => {
                    app.engines = rows;
                    app.engine_selected =
                        app.engine_selected.min(app.engines.len().saturating_sub(1));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    app.engine_rx = None;
                    break;
                }
            }
        }
        while let Ok(ev) = rx.try_recv() {
            app.on_event(ev);
        }
        app.tick += 1;
        terminal.draw(|f| draw(f, &mut app))?;

        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    app.on_key(k.code, k.modifiers);
                }
            }
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

fn spawn_vram_poller(
    slot: Arc<Mutex<Option<(u64, u64)>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        use crate::inference::registry::ModelRegistry as R;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            let reading = R::detect_used_gpu_vram().zip(R::detect_total_gpu_vram());
            if let Ok(mut g) = slot.lock() {
                *g = reading;
            }
            for _ in 0..20 {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

fn draw(f: &mut Frame, app: &mut App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(1),
    ])
    .areas(f.area());

    draw_header(f, header, app);
    match app.tab {
        Tab::Dashboard => draw_dashboard(f, body, app),
        Tab::Models => draw_all_models(f, body, app),
        Tab::Engines => draw_engines(f, body, app),
        Tab::Logs => draw_logs(f, body, app),
    }
    draw_footer(f, footer, app);
    if app.help {
        draw_help(f);
    }
    terminal_compatibility(f);
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = vec![
        Span::styled(
            " ᚱ Rune Infer ",
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("v{}  ", app.info.version), Style::new().fg(DIM)),
        Span::styled("● ", Style::new().fg(OK)),
        Span::raw(format!("http://{}  ", app.info.addr)),
    ];
    if app.info.auth_enabled {
        spans.push(Span::styled("🔒 API key required  ", Style::new().fg(OK)));
    } else if app.info.exposed_to_network {
        spans.push(Span::styled(
            "⚠ OPEN TO NETWORK WITHOUT API KEY  ",
            Style::new().fg(ERR).add_modifier(Modifier::BOLD),
        ));
    } else {
        spans.push(Span::styled(
            "no API key (localhost only)  ",
            Style::new().fg(DIM),
        ));
    }
    spans.push(Span::styled(
        format!("up {}", human_duration(app.started.elapsed().as_secs())),
        Style::new().fg(DIM),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)).block(panel("")), area);
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = Vec::new();
    spans.extend(key_hint(
        "l",
        if app.tab == Tab::Logs {
            "dashboard"
        } else {
            "logs"
        },
    ));
    spans.extend(key_hint(
        "m",
        if app.tab == Tab::Models {
            "dashboard"
        } else {
            "models"
        },
    ));
    spans.extend(key_hint(
        "e",
        if app.tab == Tab::Engines {
            "dashboard"
        } else {
            "engines"
        },
    ));
    if app.tab == Tab::Engines {
        spans.extend(key_hint("↑↓", "select"));
        spans.extend(key_hint("r", "check versions"));
        if app.update_rx.is_some() {
            if !app.engine_cancelling() {
                spans.extend(key_hint("c", "cancel engine operation"));
            }
            spans.push(Span::styled(
                if app.engine_cancelling() {
                    " cancelling engine operation "
                } else {
                    " engine operation in progress "
                },
                Style::new().fg(DIM),
            ));
        } else if let Some(engine) = app.engines.get(app.engine_selected) {
            match engine_action(engine) {
                EngineAction::Install => spans.extend(key_hint("u", "install")),
                EngineAction::Update => spans.extend(key_hint("u", "update")),
                EngineAction::Unavailable(_) => {}
            }
        }
    }
    if app.tab == Tab::Models {
        spans.extend(key_hint("↑↓", "select"));
        if app.model_registry_open {
            spans.extend(key_hint("i/Esc", "configured"));
            spans.extend(key_hint("r", "refresh registry"));
            if app.model_install_rx.is_some() {
                spans.extend(key_hint("c", "cancel"));
            } else {
                spans.extend(key_hint("Enter", "install"));
            }
        } else {
            spans.extend(key_hint("i", "install models"));
            match app.selected_model_action() {
                Some(ModelAction::Load) => spans.extend(key_hint("Enter", "load")),
                Some(ModelAction::Unload) => spans.extend(key_hint("Enter", "unload")),
                None => spans.push(Span::styled(" model busy ", Style::new().fg(DIM))),
            }
        }
    }
    if app.tab == Tab::Logs {
        spans.extend(key_hint("↑↓ PgUp/Dn", "scroll"));
        spans.extend(key_hint("f", "follow"));
        spans.extend(key_hint(
            "w",
            if app.errors_only {
                "all levels"
            } else {
                "warnings + errors"
            },
        ));
        spans.extend(key_hint("c", "clear"));
    }
    spans.extend(key_hint("?", "help"));
    spans.extend(key_hint("q", "quit"));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[derive(Clone, Debug)]
struct EngineRow {
    name: &'static str,
    path: PathBuf,
    managed: bool,
    installed: bool,
    version: String,
    latest: String,
    build_tag: Option<String>,
    registry_versions: Option<std::collections::BTreeMap<u64, u64>>,
    registry_revisions: Option<std::collections::BTreeMap<String, u64>>,
    published_at: Option<String>,
    gpu_upgrade: bool,
    check_detail: Option<String>,
}

fn discover_engines() -> Vec<EngineRow> {
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let definitions = [
        ("llama-server", "llama-upstream", "llama-server", true),
        ("llama-server-prism", "llama-prism", "llama-server", true),
        ("audio-cpp", "audiocpp", "audiocpp_cli", true),
        ("sd-server", "sd-server", "sd-server", true),
        ("crispasr", "crispasr", "crispasr", true),
        ("nemo-speech", "nemo-speech", "nemo-speech", true),
        ("needle", "needle", "needle", false),
        ("whisper-cpp", "whisper-cpp", "whisper-cli", true),
        ("litert", "litert", "libLiteRt", false),
        ("onnx", "onnxruntime", "onnxruntime", false),
    ];
    let mut rows: Vec<EngineRow> = definitions
        .into_iter()
        .map(|(name, directory, executable, version_supported)| {
            let filename = if name == "onnx" {
                "onnxruntime.dll".into()
            } else if name == "litert" {
                "libLiteRt.dll".into()
            } else {
                format!("{executable}{suffix}")
            };
            let custom_vars: &[&str] = match name {
                "llama-server" | "llama-server-prism" => &["LLAMA_SERVER_PATH"],
                "audio-cpp" => &["AUDIOCPP_PATH", "AUDIO8_PATH"],
                "sd-server" => &["SD_SERVER_PATH"],
                "crispasr" => &["CRISPASR_PATH"],
                "nemo-speech" => &["NEMO_SPEECH_PATH"],
                "whisper-cpp" => &["WHISPER_CPP_PATH"],
                "litert" => &["LITERT_LIBRARY_PATH"],
                "onnx" => &["ONNX_RUNTIME_LIBRARY_PATH"],
                _ => &[],
            };
            let custom = custom_vars
                .iter()
                .find_map(std::env::var_os)
                .map(PathBuf::from)
                .filter(|p| matches!(name, "litert" | "onnx") || p.is_file());
            let base = PathBuf::from("bin").join(directory);
            let direct = base.join(&filename);
            let managed = direct.is_file().then_some(direct).or_else(|| {
                crate::inference::engines::download::find_executable_recursive(&base, &filename)
            });
            let root = PathBuf::from("bin").join(&filename);
            let root = root.is_file().then_some(root);
            let on_path = || {
                std::env::var_os("PATH").and_then(|paths| {
                    std::env::split_paths(&paths)
                        .map(|dir| dir.join(&filename))
                        .find(|p| p.is_file())
                })
            };
            let candidate = if let Some(path) = custom {
                Some((path, false))
            } else if name == "llama-server" || name == "llama-server-prism" {
                root.map(|p| (p, false))
                    .or_else(|| managed.map(|p| (p, true)))
                    .or_else(|| on_path().map(|p| (p, false)))
            } else {
                managed.map(|p| (p, true)).or_else(|| {
                    if matches!(name, "needle" | "litert" | "onnx") {
                        None
                    } else {
                        root.map(|p| (p, false))
                            .or_else(|| on_path().map(|p| (p, false)))
                    }
                })
            };
            let (path, managed, installed) = match candidate {
                Some((path, managed)) => {
                    let installed = path.is_file();
                    (path, managed, installed)
                }
                None => (base.join(&filename), matches!(name, "litert" | "onnx"), false),
            };
            let version = if name == "onnx" && installed {
                crate::inference::engines::onnx::runtime::runtime_version(&path)
            } else if name == "litert" && installed {
                crate::inference::engines::litert::runtime_version(&path)
            } else if name == "needle" && installed && managed {
                crate::inference::engines::needle::installed_revision(&path)
            } else if version_supported && managed {
                probe_engine_version(&path).and_then(|version| {
                    if name == "sd-server" {
                        crate::inference::engines::sd_server::reported_version(&version)
                    } else {
                        Some(version)
                    }
                })
            } else {
                None
            }
            .unwrap_or_else(|| {
                if installed {
                    "Unknown".into()
                } else {
                    "Not installed".into()
                }
            });
            EngineRow {
                name,
                path,
                managed,
                installed,
                version,
                latest: if name == "onnx" {
                    "1.31.1 · pinned".into()
                } else if name == "litert" {
                    "2.3.0 · pinned".into()
                } else if managed || !installed {
                    "Checking…".into()
                } else {
                    "External".into()
                },
                build_tag: None,
                registry_versions: None,
registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: match name {
                    "litert" => Some("Pinned Windows x64 CPU + WebGPU/DXC runtime · GPU requires a parity-verified model export".into()),
                    "onnx" => Some("Pinned Windows x64 ONNX Runtime + CUDA 12/cuDNN · approximately 1.7 GB download".into()),
                    _ => None,
                },
            }
        })
        .collect();
    rows.push(EngineRow {
        name: "encoder",
        path: "Built into Rune Infer".into(),
        managed: false,
        installed: true,
        version: "Built in".into(),
        latest: "Built in".into(),
        build_tag: None,
        registry_versions: None,
        registry_revisions: None,
        published_at: None,
        gpu_upgrade: false,
        check_detail: Some("Built-in encoder uses synthetic outputs".into()),
    });
    rows
}

fn start_engine_inventory() -> std::sync::mpsc::Receiver<Vec<EngineRow>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut rows = discover_engines();
        if tx.send(rows.clone()).is_err() {
            return;
        }
        for index in 0..rows.len() {
            if matches!(rows[index].name, "litert" | "onnx") {
                continue;
            }
            if rows[index].installed && !rows[index].managed {
                continue;
            }
            match check_latest_release(rows[index].name) {
                Ok(release) => {
                    rows[index].latest = release.tag;
                    rows[index].build_tag = release.build_tag;
                    rows[index].registry_versions = release.registry_versions;
                    rows[index].registry_revisions = release.registry_revisions;
                    rows[index].published_at = release.published_at;
                    let mut whisper_verified = false;
                    if rows[index].name == "whisper-cpp"
                        && rows[index].installed
                        && rows[index].managed
                    {
                        if let Some(package) = &release.whisper_gpu {
                            let base = PathBuf::from("bin/whisper-cpp");
                            let in_managed_folder = rows[index]
                                .path
                                .canonicalize()
                                .ok()
                                .zip(base.canonicalize().ok())
                                .is_some_and(|(path, root)| path.starts_with(root));
                            whisper_verified = in_managed_folder
                                && crate::inference::engines::whisper::installation_matches(
                                    &base, package,
                                );
                            rows[index].gpu_upgrade = in_managed_folder
                                && !whisper_verified
                                && release_status(&rows[index]) == "Up to date";
                        }
                    }
                    rows[index].check_detail = Some(if let Some(package) = release.whisper_gpu {
                        format!(
                            "{} · GPU package: {} · {}",
                            release.detail,
                            package.archive,
                            if whisper_verified {
                                "installed package verified"
                            } else if !rows[index].installed {
                                "not installed"
                            } else if rows[index].gpu_upgrade {
                                "installed package unverified; verified GPU refresh available"
                            } else {
                                "installed package unverified; version rules apply"
                            }
                        )
                    } else {
                        release.detail
                    });
                }
                Err(error) => {
                    rows[index].latest = if error.contains("GitHub rate limited until") {
                        "Rate limited"
                    } else {
                        "Check failed"
                    }
                    .into();
                    rows[index].build_tag = None;
                    rows[index].registry_versions = None;
                    rows[index].registry_revisions = None;
                    rows[index].check_detail = Some(error);
                }
            }
            if tx.send(rows.clone()).is_err() {
                return;
            }
        }
    });
    rx
}

#[derive(Debug, PartialEq, Eq)]
enum EngineAction {
    Install,
    Update,
    Unavailable(String),
}

fn engine_action(engine: &EngineRow) -> EngineAction {
    engine_action_on(engine, std::env::consts::OS, std::env::consts::ARCH)
}

// Share eligibility between the action hint and dispatch. The installer still
// selects and verifies a compatible package before replacing engine files.
fn engine_action_on(engine: &EngineRow, os: &str, arch: &str) -> EngineAction {
    let unavailable = |reason: &str| EngineAction::Unavailable(reason.into());
    if engine.name == "encoder" {
        return unavailable("Built in; no installation or update required");
    }
    if engine.name == "onnx" {
        if os != "windows" || arch != "x86_64" {
            return unavailable("ONNX Runtime installation currently supports Windows x64 only");
        }
        if !engine.managed {
            return unavailable(
                "External ONNX Runtime: remove ONNX_RUNTIME_LIBRARY_PATH to use the managed installation",
            );
        }
        if !engine.installed || engine.version == "ONNX Runtime 1.31.1 CPU" {
            return EngineAction::Install;
        }
        return if engine.version == "ONNX Runtime 1.31.1 CPU + CUDA" {
            unavailable("Pinned ONNX CPU and CUDA components are installed")
        } else {
            unavailable("Existing ONNX Runtime is unverified; preserve it before reinstalling")
        };
    }
    if engine.name == "litert" {
        if os != "windows" || arch != "x86_64" {
            return unavailable("LiteRT installation currently supports Windows x64 only");
        }
        if !engine.managed {
            return unavailable(
                "External LiteRT runtime: fix or remove LITERT_LIBRARY_PATH to use the managed installation",
            );
        }
        if !engine.installed {
            return EngineAction::Install;
        }
        return if engine.version == "LiteRT 2.3.0 CPU" {
            EngineAction::Install
        } else if engine.version == "LiteRT 2.3.0 GPU + CPU" {
            unavailable("Pinned CPU and GPU runtime components are installed")
        } else {
            unavailable("Existing LiteRT DLL is unverified; preserve it before reinstalling")
        };
    }
    if engine.installed && !engine.managed {
        return unavailable("Externally managed; update using its original installation method");
    }
    if engine.installed {
        let supported = match engine.name {
            "audio-cpp" => os == "windows" && arch == "x86_64",
            "llama-server" | "llama-server-prism" | "sd-server" | "nemo-speech" | "whisper-cpp"
            | "crispasr" | "needle" => os == "windows" && arch == "x86_64",
            _ => return unavailable("Automatic update is not implemented for this engine"),
        };
        if !supported {
            return unavailable("Automatic update is unavailable on this platform");
        }
    } else if install_engine_kind(engine.name).is_none() {
        return unavailable("Installation is unavailable for this engine");
    } else if !matches!(os, "windows" | "linux" | "android" | "macos") {
        return unavailable("Installation is unavailable on this platform");
    }
    match engine.latest.as_str() {
        "Checking…" => {
            return unavailable("Checking release metadata; wait for the check to finish");
        }
        "Rate limited" => {
            return unavailable(engine.check_detail.as_deref().unwrap_or(
                "GitHub rate limited; wait for the reset before checking versions again",
            ));
        }
        "Check failed" => {
            return unavailable(
                engine
                    .check_detail
                    .as_deref()
                    .unwrap_or("Release check failed; press r to check versions again"),
            );
        }
        "" | "Unknown" | "External" | "Not checked" => {
            return unavailable("Check versions before installing or updating");
        }
        _ => {}
    }
    if !engine.installed {
        return EngineAction::Install;
    }
    if engine.name == "needle" && release_status(engine) == "Revision unknown" {
        return unavailable("Cannot verify Needle installation receipt; update refused");
    }
    if engine.name == "sd-server" && release_status(engine) == "Unknown" {
        return unavailable(
            "Cannot verify SD-server commit identities; check versions before updating",
        );
    }
    if matches!(engine.name, "llama-server" | "llama-server-prism")
        && (engine_build_identity(&engine.version).is_none()
            || engine_build_identity(engine.build_tag.as_deref().unwrap_or(&engine.latest))
                .is_none())
    {
        return unavailable(
            "Cannot verify llama-server build identities; check versions before updating",
        );
    }
    if matches!(engine.name, "llama-server" | "llama-server-prism")
        && release_status(engine) == "Unknown"
    {
        return unavailable("Installed llama-server build is not in the registry; update refused");
    }
    if matches!(engine.name, "nemo-speech" | "audio-cpp" | "crispasr")
        && release_status(engine) == "Unknown"
    {
        return unavailable("Installed engine version is not in the registry; update refused");
    }
    match release_status(engine) {
        "Up to date" => unavailable("Already up to date; no download needed"),
        "Ahead" => unavailable("Installed version is newer; no downgrade will be performed"),
        _ => EngineAction::Update,
    }
}

fn install_engine_kind(name: &str) -> Option<crate::config::EngineKind> {
    use crate::config::EngineKind;
    Some(match name {
        "llama-server" => EngineKind::LlamaServer,
        "llama-server-prism" => EngineKind::LlamaServerPrism,
        "audio-cpp" => EngineKind::AudioCpp,
        "sd-server" => EngineKind::SdServer,
        "crispasr" => EngineKind::Crispasr,
        "nemo-speech" => EngineKind::NemoSpeech,
        "needle" => EngineKind::Needle,
        "whisper-cpp" => EngineKind::WhisperCpp,
        "litert" => EngineKind::Litert,
        "onnx" => EngineKind::Onnx,
        _ => return None,
    })
}

fn release_url(name: &str) -> Option<&'static str> {
    match name {
        "needle" => Some("https://huggingface.co/api/models/Cactus-Compute/needle3"),
        "whisper-cpp" => Some("https://api.github.com/repos/ggml-org/whisper.cpp/releases"),
        _ => None,
    }
}

struct AvailableRelease {
    tag: String,
    build_tag: Option<String>,
    registry_versions: Option<std::collections::BTreeMap<u64, u64>>,
    registry_revisions: Option<std::collections::BTreeMap<String, u64>>,
    published_at: Option<String>,
    detail: String,
    whisper_gpu: Option<crate::inference::engines::whisper::PackageIdentity>,
}

fn check_latest_release(name: &str) -> Result<AvailableRelease, String> {
    use crate::inference::engines::download;
    crate::inference::engines::release_cache::begin_check();
    let url = if matches!(
        name,
        "llama-server"
            | "llama-server-prism"
            | "nemo-speech"
            | "audio-cpp"
            | "crispasr"
            | "sd-server"
    ) {
        ""
    } else {
        release_url(name).ok_or("No release source is available for this engine")?
    };
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|_| "Could not create release client")?;
    if name == "needle" {
        let value: serde_json::Value = client
            .get(url)
            .send()
            .map_err(|_| "Cannot reach release source; press r to retry")?
            .error_for_status()
            .map_err(|e| format!("Needle metadata unavailable: {e}"))?
            .json()
            .map_err(|_| "Invalid Needle metadata")?;
        let revision = value["sha"]
            .as_str()
            .filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or("No immutable Needle revision found")?;
        return Ok(AvailableRelease {
            tag: format!("revision {revision}"),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: value["lastModified"].as_str().map(str::to_owned),
            detail:
                "Approved Needle distribution · revision and SHA-256 verified during installation"
                    .into(),
            whisper_gpu: None,
        });
    }
    let value = if name == "llama-server" {
        crate::inference::engines::registry::llama_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "llama-server-prism" {
        crate::inference::engines::registry::prism_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "nemo-speech" {
        crate::inference::engines::registry::nemo_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "audio-cpp" {
        crate::inference::engines::registry::audio_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "crispasr" {
        crate::inference::engines::registry::crispasr_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "sd-server" {
        crate::inference::engines::registry::sd_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "whisper-cpp" {
        crate::inference::engines::whisper::resolve_release(&client)
            .map_err(|e| format!("Release check failed: {e}; press r to retry"))?
    } else {
        download::resolve_release(&client, url)
            .map_err(|e| format!("Release check failed: {e}; press r to retry"))?
    };
    let tag = value["tag_name"]
        .as_str()
        .ok_or("Release tag missing")?
        .to_owned();
    let channel = if value["rune_registry"].as_bool() == Some(true) {
        "Reviewed registry build"
    } else if download::is_stable_release(&value) {
        "Stable release"
    } else {
        "Latest published build · no stable release available"
    };
    let mut detail = if let Some(build) = value["rune_build_tag"].as_str() {
        format!("{channel} {tag} · upstream binary build {build}")
    } else {
        format!("{channel} {tag}")
    };
    if let Some(note) = crate::inference::engines::release_cache::check_note() {
        detail.push_str(" · ");
        detail.push_str(&note);
    }
    let published_at = value["rune_build_published_at"]
        .as_str()
        .or_else(|| value["published_at"].as_str())
        .map(str::to_owned);
    Ok(AvailableRelease {
        tag,
        build_tag: value["rune_build_tag"].as_str().map(str::to_owned),
        registry_versions: if name == "llama-server" {
            Some(
                serde_json::from_value(value["rune_registry_versions"].clone())
                    .map_err(|_| "Invalid registry version order")?,
            )
        } else {
            None
        },
        registry_revisions: if matches!(
            name,
            "nemo-speech" | "audio-cpp" | "crispasr" | "sd-server" | "llama-server-prism"
        ) {
            Some(
                serde_json::from_value(value["rune_registry_revisions"].clone())
                    .map_err(|_| "Invalid registry version order")?,
            )
        } else {
            None
        },
        published_at,
        detail,
        whisper_gpu: if name == "whisper-cpp" {
            crate::inference::engines::whisper::platform_asset(&value)
                .and_then(|asset| {
                    crate::inference::engines::whisper::PackageIdentity::from_asset(asset).ok()
                })
                .filter(|package| package.is_gpu())
        } else {
            None
        },
    })
}

#[cfg(test)]
fn latest_release_tag(value: &serde_json::Value) -> Option<String> {
    crate::inference::engines::download::preferred_release(value).ok()?["tag_name"]
        .as_str()
        .map(str::to_owned)
}

fn probe_engine_version(path: &std::path::Path) -> Option<String> {
    use std::process::{Command, Stdio};
    let mut command = Command::new(path);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                use std::io::Read;
                let mut stdout = String::new();
                let mut stderr = String::new();
                child
                    .stdout
                    .take()?
                    .take(4096)
                    .read_to_string(&mut stdout)
                    .ok()?;
                child
                    .stderr
                    .take()?
                    .take(4096)
                    .read_to_string(&mut stderr)
                    .ok()?;
                return parse_engine_version(&stdout, &stderr);
            }
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if start.elapsed() >= Duration::from_secs(3) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(40)),
        }
    }
}

fn parse_engine_version(stdout: &str, stderr: &str) -> Option<String> {
    let lines: Vec<_> = stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines
        .iter()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim().eq_ignore_ascii_case("version") && !value.trim().is_empty() {
                Some(value.trim().to_owned())
            } else {
                None
            }
        })
        .or_else(|| lines.first().map(|line| (*line).to_owned()))
}

fn release_status(engine: &EngineRow) -> &'static str {
    if engine.name == "encoder" {
        return "Built in · synthetic";
    }
    if !engine.installed {
        return "Not installed";
    }
    if engine.name == "onnx" {
        return if engine.version == "ONNX Runtime 1.31.1 CPU + CUDA" {
            "Ready · CUDA + CPU"
        } else if engine.version == "ONNX Runtime 1.31.1 CPU" {
            "Ready · CPU"
        } else {
            "Unverified runtime"
        };
    }
    if engine.name == "litert" {
        return if engine.version == "LiteRT 2.3.0 GPU + CPU" {
            "Ready · GPU + CPU"
        } else if engine.version == "LiteRT 2.3.0 CPU" {
            "Ready · CPU"
        } else {
            "Unverified runtime"
        };
    }
    if engine.name == "needle" {
        if engine.version.starts_with("revision ") && engine.latest.starts_with("revision ") {
            return if engine.version == engine.latest {
                "Up to date"
            } else {
                "Different revision"
            };
        }
        return "Revision unknown";
    }
    if engine.latest == "Check failed" {
        return "Check failed";
    }
    if engine.latest == "Rate limited" {
        return "Rate limited";
    }
    if engine.name == "sd-server" {
        return match engine.registry_revisions.as_ref().and_then(|versions| {
            crate::inference::engines::registry::compare_sd_registry(
                versions,
                &engine.version,
                &engine.latest,
            )
        }) {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if matches!(engine.name, "nemo-speech" | "audio-cpp" | "crispasr") {
        if engine.name == "audio-cpp" {
            let Some((version, custom)) =
                crate::inference::engines::registry::audio_identity(&engine.version)
            else {
                return "Unknown";
            };
            let order = engine.registry_revisions.as_ref().and_then(|versions| {
                crate::inference::engines::registry::compare_nemo_revision(
                    versions,
                    version,
                    &engine.latest,
                )
            });
            return match order {
                Some(std::cmp::Ordering::Less) => "New release",
                Some(std::cmp::Ordering::Equal) if custom => "Different revision",
                Some(std::cmp::Ordering::Equal) => "Up to date",
                Some(std::cmp::Ordering::Greater) => "Ahead",
                None => "Unknown",
            };
        }
        let installed = engine
            .version
            .trim()
            .strip_prefix("nemo-speech ")
            .unwrap_or(engine.version.trim())
            .split_whitespace()
            .next()
            .unwrap_or("");
        let order = engine.registry_revisions.as_ref().and_then(|versions| {
            crate::inference::engines::registry::compare_nemo_revision(
                versions,
                installed,
                &engine.latest,
            )
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if engine.name == "llama-server" {
        let order = engine.registry_versions.as_ref().and_then(|versions| {
            let installed = engine_build_identity(&engine.version)?.0;
            let selected = engine_build_identity(&engine.latest)?.0;
            crate::inference::engines::registry::compare_llama_build(versions, installed, selected)
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if engine.name == "llama-server-prism" {
        let order = engine.registry_revisions.as_ref().and_then(|versions| {
            let (build, commit) = engine_build_identity(&engine.version)?;
            crate::inference::engines::registry::compare_prism_revision(
                versions,
                build,
                commit?,
                &engine.latest,
            )
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    let lower = engine.version.to_ascii_lowercase();
    if lower.contains("nightly") || lower.contains("-dev") {
        return "Development build";
    }
    let compare = if matches!(engine.name, "llama-server" | "llama-server-prism")
        && engine.latest.starts_with('b')
    {
        let installed = engine
            .version
            .split("build ")
            .nth(1)
            .and_then(|tail| tail.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|v| v.parse::<u64>().ok());
        let published = engine
            .latest
            .strip_prefix('b')
            .and_then(|v| v.parse::<u64>().ok());
        installed.zip(published).map(|(a, b)| a.cmp(&b))
    } else {
        let parse = |text: &str| {
            let text = text.trim_start_matches('v');
            let mut parts = text.split('.');
            let major = parts.next()?.parse::<u64>().ok()?;
            let minor = parts.next()?.parse::<u64>().ok()?;
            let patch = parts.next()?.parse::<u64>().ok()?;
            if parts.next().is_some() {
                return None;
            }
            Some((major, minor, patch))
        };
        let installed = engine.version.split_whitespace().find_map(parse);
        installed.zip(parse(&engine.latest)).map(|(a, b)| a.cmp(&b))
    };
    match compare {
        Some(std::cmp::Ordering::Equal) if engine.name == "whisper-cpp" && engine.gpu_upgrade => {
            "GPU build available"
        }
        Some(std::cmp::Ordering::Equal) => "Up to date",
        Some(std::cmp::Ordering::Less) => "New release",
        Some(std::cmp::Ordering::Greater) => "Ahead",
        None => "Unknown",
    }
}

fn engine_build_identity(text: &str) -> Option<(u64, Option<&str>)> {
    let (build, commit) = if let Some((_, tail)) = text.split_once("build ") {
        let build = tail.split(|c: char| !c.is_ascii_digit()).next()?;
        let commit = tail.split_once("commit ").map(|(_, commit)| commit);
        (build, commit)
    } else {
        let tail = text
            .strip_prefix("prism-")
            .unwrap_or(text)
            .strip_prefix('b')?;
        tail.split_once('-')
            .map_or((tail, None), |(build, commit)| (build, Some(commit)))
    };
    let commit = commit.and_then(|value| {
        let value = value.split(|c: char| !c.is_ascii_hexdigit()).next()?;
        (value.len() >= 7).then_some(value)
    });
    Some((build.parse().ok()?, commit))
}

fn release_age(published_at: &str, now: chrono::DateTime<chrono::Utc>) -> Option<String> {
    let published = chrono::DateTime::parse_from_rfc3339(published_at).ok()?;
    let hours = now.signed_duration_since(published).num_hours().max(0);
    Some(if hours < 1 {
        "less than 1 hour old".into()
    } else if hours < 24 {
        format!("{hours} {} old", if hours == 1 { "hour" } else { "hours" })
    } else if hours < 24 * 30 {
        let days = hours / 24;
        format!("{days} {} old", if days == 1 { "day" } else { "days" })
    } else {
        let months = hours / (24 * 30);
        format!(
            "{months} {} old",
            if months == 1 { "month" } else { "months" }
        )
    })
}

fn engine_status_label(engine: &EngineRow) -> String {
    let status = release_status(engine);
    let cached = if engine
        .check_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("Cached release metadata"))
    {
        " · cached"
    } else {
        ""
    };
    match engine
        .published_at
        .as_deref()
        .and_then(|date| release_age(date, chrono::Utc::now()))
    {
        Some(age) => format!("{status}{cached} · {age}"),
        None => format!("{status}{cached}"),
    }
}

fn download_progress_line(
    progress: &crate::inference::engines::download::DownloadProgress,
    tick: u64,
) -> Line<'static> {
    use crate::inference::engines::download::format_bytes;
    if progress.extracting {
        return Line::from(Span::styled(
            format!("{} Extracting {}", spinner(tick), progress.filename),
            Style::new().fg(ACCENT),
        ));
    }
    let speed = format!(
        "{}/s",
        format_bytes(progress.bytes_per_second.max(0.0) as u64)
    );
    if let Some(total) = progress.total.filter(|total| *total > 0) {
        let ratio = (progress.downloaded as f64 / total as f64).clamp(0.0, 1.0);
        let filled = (ratio * 16.0).floor() as usize;
        let dots = "•".repeat(filled) + &"·".repeat(16 - filled);
        Line::from(vec![
            Span::styled(dots, Style::new().fg(ACCENT)),
            Span::styled(
                format!(
                    " {:3.0}% · {} / {} · {speed}",
                    ratio * 100.0,
                    format_bytes(progress.downloaded),
                    format_bytes(total)
                ),
                Style::new().fg(DIM),
            ),
        ])
    } else {
        Line::from(Span::styled(
            format!(
                "{} {} downloaded · {speed} · size unknown",
                spinner(tick),
                format_bytes(progress.downloaded)
            ),
            Style::new().fg(ACCENT),
        ))
    }
}

fn draw_engines(f: &mut Frame, area: Rect, app: &App) {
    let progress = app.engine_download.as_ref().filter(|_| {
        app.engines
            .get(app.engine_selected)
            .is_some_and(|engine| app.update_engine.as_deref() == Some(engine.name))
    });
    let [table_area, details] = Layout::vertical([
        Constraint::Min(4),
        Constraint::Length(if progress.is_some() { 8 } else { 7 }),
    ])
    .areas(area);
    let visible = table_area.height.saturating_sub(3) as usize;
    let start = app
        .engine_selected
        .saturating_add(1)
        .saturating_sub(visible);
    let rows = app
        .engines
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, engine)| {
            let marker = if index == app.engine_selected {
                ">"
            } else {
                " "
            };
            Row::new(vec![
                marker.to_string(),
                engine.name.to_string(),
                engine.version.clone(),
                engine.latest.clone(),
                engine_status_label(engine),
            ])
        });
    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Percentage(20),
            Constraint::Percentage(26),
            Constraint::Percentage(24),
            Constraint::Percentage(30),
        ],
    )
    .header(
        Row::new([
            "",
            "Engine",
            "Installed version",
            "Available version",
            "Status",
        ])
        .style(Style::new().fg(DIM)),
    )
    .block(panel(if app.engine_rx.is_some() {
        " Engines · checking versions "
    } else {
        " Engines "
    }));
    f.render_widget(table, table_area);
    let detail = app
        .engines
        .get(app.engine_selected)
        .map(|engine| {
            let models = app
                .models
                .iter()
                .filter(|model| model.row.engine.as_deref() == Some(engine.name))
                .map(|model| model.row.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let status = app
                .update_status
                .as_deref()
                .filter(|_| app.update_engine.as_deref() == Some(engine.name))
                .or(engine.check_detail.as_deref())
                .unwrap_or("Release compatibility is checked before updating.");
            let status_line = if progress.is_some() {
                String::new()
            } else {
                let action = match engine_action(engine) {
                    EngineAction::Install => {
                        "Install: press u (compatible package required)".into()
                    }
                    EngineAction::Update => "Update: press u".into(),
                    EngineAction::Unavailable(reason) => reason,
                };
                format!("{action}\n{status}\n")
            };
            format!(
                "{}Path: {} · {}\nPreferred by: {}",
                status_line,
                engine.path.display(),
                if !engine.installed {
                    "not installed"
                } else if engine.name == "encoder" {
                    "built in"
                } else if engine.managed {
                    "managed"
                } else {
                    "external"
                },
                if models.is_empty() { "none" } else { &models }
            )
        })
        .unwrap_or_else(|| {
            if app.engine_rx.is_some() {
                "Checking installed engines…".into()
            } else {
                "No engine binaries found.".into()
            }
        });
    let block = panel(" Details ");
    let inner = block.inner(details);
    f.render_widget(block, details);
    let [text_area, progress_area] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(u16::from(progress.is_some())),
    ])
    .areas(inner);
    f.render_widget(Paragraph::new(detail).wrap(Wrap { trim: false }), text_area);
    if let Some(progress) = progress {
        f.render_widget(
            Paragraph::new(download_progress_line(progress, app.tick)),
            progress_area,
        );
    }
}

fn draw_engine_overview(f: &mut Frame, area: Rect, app: &App) {
    let mut active = Vec::<&str>::new();
    for model in &app.models {
        if let Status::Loaded { detail, .. } = &model.status {
            let name = detail.split(" · ").next().unwrap_or("").trim();
            if !name.is_empty() && !active.contains(&name) {
                active.push(name);
            }
        }
    }
    let title = format!(" Engines · {} active · e to list all ", active.len());
    if active.is_empty() {
        f.render_widget(
            Paragraph::new("No engines active.").block(panel(title)),
            area,
        );
        return;
    }
    let rows = active.into_iter().map(|name| {
        let version = app
            .engines
            .iter()
            .find(|engine| engine.name == name)
            .map(|engine| engine.version.as_str())
            .unwrap_or("Unknown");
        let latest = app
            .engines
            .iter()
            .find(|engine| engine.name == name)
            .map(|engine| engine.latest.as_str())
            .unwrap_or("Unknown");
        Row::new([name, version, latest])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Percentage(30),
            Constraint::Percentage(40),
            Constraint::Percentage(30),
        ],
    )
    .header(Row::new(["Engine", "Installed", "Latest"]).style(Style::new().fg(DIM)))
    .block(panel(title));
    f.render_widget(table, area);
}

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let [top, mid, bottom] = Layout::vertical([
        Constraint::Percentage(42),
        Constraint::Percentage(30),
        Constraint::Percentage(28),
    ])
    .areas(area);
    let [models, engines] =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(top);
    let [requests, stats] =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(mid);

    draw_models(f, models, app);
    draw_engine_overview(f, engines, app);
    draw_requests(f, requests, app);
    draw_stats(f, stats, app);
    draw_recent_logs(f, bottom, app);
}

fn render_model_row<'a>(app: &App, idx: usize) -> Row<'a> {
    let m = &app.models[idx];
    let (icon, color, extra) = match &m.status {
        Status::Idle => ("·".to_string(), DIM, String::new()),
        Status::Loading(since) => (
            spinner(app.tick).to_string(),
            ACCENT,
            format!(" loading {:.1}s", since.elapsed().as_secs_f32()),
        ),
        Status::Loaded { load_ms, detail } => (
            "●".to_string(),
            OK,
            format!(
                " · loaded in {:.1}s{}",
                *load_ms as f64 / 1000.0,
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(" · {detail}")
                }
            ),
        ),
        Status::Failed(_) => ("×".to_string(), ERR, " failed".to_string()),
    };
    let avg = if m.tps_n > 0 {
        format!("{:.1}", m.tps_sum / m.tps_n as f64)
    } else {
        "–".into()
    };
    Row::new([
        Cell::from(icon).style(Style::new().fg(color)),
        Cell::from(format!("{}{}", m.row.id, extra)),
        Cell::from(m.row.modality.clone()).style(Style::new().fg(DIM)),
        Cell::from(
            m.row
                .provider
                .clone()
                .or_else(|| m.row.engine.clone())
                .unwrap_or_else(|| "local".into()),
        )
        .style(Style::new().fg(DIM)),
        Cell::from(m.requests.to_string()),
        Cell::from(m.errors.to_string()).style(if m.errors > 0 {
            Style::new().fg(ERR)
        } else {
            Style::new()
        }),
        Cell::from(avg),
    ])
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let loaded = app
        .models
        .iter()
        .filter(|m| matches!(m.status, Status::Loaded { .. }))
        .count();
    let block = panel(format!(
        " Models · {} configured · {} loaded ",
        app.models.len(),
        loaded
    ));

    if app.models.is_empty() {
        let msg = Paragraph::new(vec![
            Line::from(""),
            Line::styled("  No models configured yet.", Style::new().fg(WARN)),
            Line::from(""),
            Line::from(format!("  Edit {}", app.info.config_path.display())),
            Line::from("  or run `rune-infer setup` to add models and API keys."),
        ])
        .block(block);
        f.render_widget(msg, area);
        return;
    }

    let header = Row::new(["", "Model", "Type", "Source", "Reqs", "Err", "tok/s"])
        .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));

    // Prioritize display order: Loading -> Loaded -> Failed -> Active (requests > 0) -> Idle
    let mut display_indices: Vec<usize> = (0..app.models.len()).collect();
    display_indices.sort_by(|&a, &b| {
        let ma = &app.models[a];
        let mb = &app.models[b];

        let priority_a = match &ma.status {
            Status::Loading(_) => 0,
            Status::Loaded { .. } => 1,
            Status::Failed(_) => 2,
            Status::Idle if ma.requests > 0 => 3,
            Status::Idle => 4,
        };
        let priority_b = match &mb.status {
            Status::Loading(_) => 0,
            Status::Loaded { .. } => 1,
            Status::Failed(_) => 2,
            Status::Idle if mb.requests > 0 => 3,
            Status::Idle => 4,
        };

        priority_a
            .cmp(&priority_b)
            .then_with(|| mb.last_used.cmp(&ma.last_used))
            .then_with(|| mb.requests.cmp(&ma.requests))
            .then_with(|| a.cmp(&b))
    });

    let mut active_indices = Vec::new();
    let mut idle_indices = Vec::new();
    for &idx in &display_indices {
        let m = &app.models[idx];
        if matches!(
            m.status,
            Status::Loading(_) | Status::Loaded { .. } | Status::Failed(_)
        ) || m.requests > 0
        {
            active_indices.push(idx);
        } else {
            idle_indices.push(idx);
        }
    }

    let mut rows: Vec<Row> = Vec::new();
    for idx in active_indices {
        rows.push(render_model_row(app, idx));
    }

    if idle_indices.len() > 1 {
        rows.push(Row::new([
            Cell::from("·").style(Style::new().fg(DIM)),
            Cell::from(format!("+ {} MORE · NO TRAFFIC", idle_indices.len()))
                .style(Style::new().fg(DIM)),
            Cell::from("–").style(Style::new().fg(DIM)),
            Cell::from("m to list all").style(Style::new().fg(DIM)),
            Cell::from("0").style(Style::new().fg(DIM)),
            Cell::from("0").style(Style::new().fg(DIM)),
            Cell::from("–").style(Style::new().fg(DIM)),
        ]));
    } else {
        for idx in idle_indices {
            rows.push(render_model_row(app, idx));
        }
    }
    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Min(18),
            Constraint::Length(14),
            Constraint::Length(18),
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Length(7),
        ],
    )
    .header(header)
    .block(block);
    f.render_widget(table, area);

    // Surface the most recent load failure inline, where the user is looking.
    if let Some(m) = app
        .models
        .iter()
        .find(|m| matches!(m.status, Status::Failed(_)))
    {
        if let Status::Failed(msg) = &m.status {
            let r = Rect {
                x: area.x + 2,
                y: area.bottom().saturating_sub(2),
                width: area.width.saturating_sub(4),
                height: 1,
            };
            f.render_widget(
                Paragraph::new(Span::styled(
                    format!("✗ {}: {}", m.row.id, msg),
                    Style::new().fg(ERR),
                )),
                r,
            );
        }
    }
}

fn draw_stats(f: &mut Frame, area: Rect, app: &App) {
    let block = panel(" Performance ");
    let inner = block.inner(area);
    f.render_widget(block, area);

    let [vram, nums, tps, lat] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Min(3),
        Constraint::Min(3),
    ])
    .areas(inner);

    // VRAM dot gauge
    let reading = app.vram.lock().ok().and_then(|g| *g);
    match reading {
        Some((used, total)) if total > 0 => {
            let ratio = (used as f64 / total as f64).clamp(0.0, 1.0);
            let color = if ratio > 0.95 { ERR } else { ACCENT };
            let dot_count = 16usize;
            let filled = ((ratio * dot_count as f64).round() as usize).min(dot_count);
            let dots = "•".repeat(filled) + &"·".repeat(dot_count - filled);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("VRAM  ", Style::new().fg(DIM)),
                    Span::styled(dots, Style::new().fg(color)),
                    Span::styled(
                        format!("  {} / {}", human_bytes(used), human_bytes(total)),
                        Style::new().fg(DIM),
                    ),
                ])),
                Rect { height: 1, ..vram },
            );
        }
        _ => f.render_widget(
            Paragraph::new(Span::styled("VRAM: no GPU detected", Style::new().fg(DIM))),
            Rect { height: 1, ..vram },
        ),
    }

    let avg_latency = if app.latency_hist.is_empty() {
        0.0
    } else {
        (app.latency_hist.iter().sum::<u64>() as f64 / app.latency_hist.len() as f64) / 1000.0
    };
    let err_style = if app.total_errors > 0 {
        Style::new().fg(ERR)
    } else {
        Style::new().fg(DIM)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::raw(format!("Requests {}  ", app.total_requests)),
            Span::styled(format!("Errors {}  ", app.total_errors), err_style),
            Span::styled(format!("Avg {:.2}s", avg_latency), Style::new().fg(DIM)),
        ])),
        nums,
    );

    let latest_tps = app
        .tps_hist
        .back()
        .map(|value| format!("{:.1}", *value as f64 / 1000.0))
        .unwrap_or_else(|| "–".into());
    let tps_data: Vec<u64> = app.tps_hist.iter().copied().collect();
    f.render_widget(
        Sparkline::default()
            .bar_set(sparkline_bar_set())
            .block(ratatui::widgets::Block::default().title(Span::styled(
                format!("Generation speed · last {latest_tps} tok/s"),
                Style::new().fg(DIM),
            )))
            .style(Style::new().fg(ACCENT))
            .data(&tps_data),
        tps,
    );
    let lat_data: Vec<u64> = app.latency_hist.iter().copied().collect();
    f.render_widget(
        Sparkline::default()
            .bar_set(sparkline_bar_set())
            .block(
                ratatui::widgets::Block::default()
                    .title(Span::styled("Latency (ms)", Style::new().fg(DIM))),
            )
            .style(Style::new().fg(DIM))
            .data(&lat_data),
        lat,
    );
}

fn draw_all_models(f: &mut Frame, area: Rect, app: &App) {
    if app.model_registry_open {
        draw_model_registry(f, area, app);
        return;
    }
    let [table_area, details] =
        Layout::vertical([Constraint::Min(4), Constraint::Length(7)]).areas(area);
    let sort_label = match app.model_sort {
        ModelSort::Default => "Default",
        ModelSort::IntelligenceIndex => "Intel Index ↓",
        ModelSort::FinanceIndex => "Finance Index ↓",
        ModelSort::Cost => "Cost / Task ↑",
        ModelSort::Latency => "Time / Task ↑",
        ModelSort::TaskGroup => "Task Group",
        ModelSort::SizeTier => "AA Size Tier ↓",
    };
    let view_label = if app.benchmark_view {
        "Benchmarks"
    } else {
        "Capabilities"
    };

    let indices = app.filtered_model_indices();
    let total_shown = indices.len();
    let selected_pos = if total_shown == 0 {
        0
    } else {
        app.model_selected.min(total_shown.saturating_sub(1))
    };

    let filter_info = match (app.task_filter, app.size_filter) {
        (TaskFilter::All, SizeFilter::All) => format!("{} configured", app.models.len()),
        (tf, SizeFilter::All) => format!(
            "{}/{} shown · [g] Task: {}",
            total_shown,
            app.models.len(),
            tf.short_label()
        ),
        (TaskFilter::All, sf) => format!(
            "{}/{} shown · [z] Size: {}",
            total_shown,
            app.models.len(),
            sf.short_label()
        ),
        (tf, sf) => format!(
            "{}/{} shown · [g] {} · [z] {}",
            total_shown,
            app.models.len(),
            tf.short_label(),
            sf.short_label()
        ),
    };
    let block = panel(format!(
        " Models Catalog · {} · [s] Sort: {} · [b] View: {} ",
        filter_info, sort_label, view_label
    ));

    let header = if app.benchmark_view {
        Row::new([
            "",
            "Model ID / Name",
            "Locality",
            "AA Size",
            "Primary Task",
            "Benchmark Index",
            "Mode (Effort)",
            "Cost / Task",
            "Time / Task",
            "Status",
        ])
        .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD))
    } else {
        Row::new([
            "",
            "Model ID / Name",
            "Locality",
            "AA Size",
            "Primary Task",
            "Capabilities",
            "Provider / Engine",
            "Reqs",
            "Status",
        ])
        .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD))
    };

    let visible = table_area.height.saturating_sub(3) as usize;
    let start = selected_pos.saturating_add(1).saturating_sub(visible);
    let rows = indices
        .iter()
        .copied()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(list_idx, model_idx)| {
            let m = &app.models[model_idx];
            let (icon, color, status_text) = match &m.status {
                Status::Idle => ("·", DIM, "IDLE"),
                Status::Loading(_) => (spinner(app.tick), ACCENT, "LOADING"),
                Status::Loaded { .. } => ("●", OK, "LOADED"),
                Status::Failed(_) => ("×", ERR, "FAILED"),
            };

            let locality_style = if m.row.locality == "LOCAL" {
                Style::new().fg(OK)
            } else {
                Style::new().fg(ACCENT)
            };

            let indicator = if list_idx == selected_pos { ">" } else { icon };

            let (size_cat, _) =
                SizeCategory::from_model(&m.row.id, &m.row.name, m.row.total_params);
            let size_style = match size_cat {
                SizeCategory::Large => Style::new().fg(ERR),
                SizeCategory::Medium => Style::new().fg(WARN),
                SizeCategory::Small => Style::new().fg(ACCENT),
                SizeCategory::Tiny => Style::new().fg(OK),
                SizeCategory::Unknown => Style::new().fg(DIM),
            };
            let tasks = classify_real_life_tasks(&m.row.id, &m.row.tasks, &m.row.capabilities);
            let primary_task = tasks.first().copied().unwrap_or(RealLifeTask::GeneralText);
            let task_style = Style::new().fg(ACCENT);

            if app.benchmark_view {
                let bench_str = if let Some(b) = &m.row.benchmarks {
                    if let Some(score) = b.score {
                        let unit = b.unit.as_deref().unwrap_or("");
                        let name = b.index_name.as_deref().unwrap_or("Index");
                        if unit.is_empty() {
                            format!("{score:.1} ({name})")
                        } else {
                            format!("{score:.1} {unit}")
                        }
                    } else {
                        "-".into()
                    }
                } else {
                    "-".into()
                };

                let mode_str = if let Some(b) = &m.row.benchmarks {
                    let m_str = b.mode.as_deref().unwrap_or("-");
                    let eff = b.reasoning_effort.as_deref().unwrap_or("-");
                    if eff != "-" && eff != "none" {
                        format!("{m_str} ({eff})")
                    } else {
                        m_str.into()
                    }
                } else {
                    "-".into()
                };

                let cost_str = if let Some(b) = &m.row.benchmarks {
                    if let Some(c) = b.cost_per_task_usd {
                        if c == 0.0 {
                            "$0.00".into()
                        } else {
                            format!("${:.2}", c)
                        }
                    } else {
                        "-".into()
                    }
                } else {
                    "-".into()
                };

                let time_str = if let Some(b) = &m.row.benchmarks {
                    if let Some(t) = b.time_per_task_minutes {
                        format!("{:.2}m", t)
                    } else {
                        "-".into()
                    }
                } else {
                    "-".into()
                };

                Row::new([
                    Cell::from(indicator).style(Style::new().fg(color)),
                    Cell::from(format!("{} ({})", m.row.id, m.row.name)),
                    Cell::from(m.row.locality.clone()).style(locality_style),
                    Cell::from(size_cat.badge()).style(size_style),
                    Cell::from(primary_task.short_badge()).style(task_style),
                    Cell::from(bench_str).style(Style::new().fg(ACCENT)),
                    Cell::from(mode_str).style(Style::new().fg(DIM)),
                    Cell::from(cost_str),
                    Cell::from(time_str),
                    Cell::from(status_text).style(Style::new().fg(color)),
                ])
            } else {
                let has_cap = |prefix: &str| {
                    m.row
                        .capabilities
                        .iter()
                        .any(|c| c.to_lowercase().contains(prefix))
                };
                let chat_dot = if has_cap("chat") || has_cap("text") {
                    "●"
                } else {
                    "○"
                };
                let vision_dot = if has_cap("vision") { "●" } else { "○" };
                let tools_dot = if has_cap("tool") { "●" } else { "○" };
                let audio_dot = if has_cap("audio") || has_cap("speech") {
                    "●"
                } else {
                    "○"
                };
                let image_dot = if has_cap("image") { "●" } else { "○" };
                let cap_str = format!(
                    "C:{chat_dot} V:{vision_dot} T:{tools_dot} A:{audio_dot} I:{image_dot}"
                );

                Row::new([
                    Cell::from(indicator).style(Style::new().fg(color)),
                    Cell::from(format!("{} ({})", m.row.id, m.row.name)),
                    Cell::from(m.row.locality.clone()).style(locality_style),
                    Cell::from(size_cat.badge()).style(size_style),
                    Cell::from(primary_task.short_badge()).style(task_style),
                    Cell::from(cap_str).style(Style::new().fg(DIM)),
                    Cell::from(
                        m.row
                            .provider
                            .clone()
                            .or_else(|| m.row.engine.clone())
                            .unwrap_or_else(|| "local".into()),
                    )
                    .style(Style::new().fg(DIM)),
                    Cell::from(m.requests.to_string()),
                    Cell::from(status_text).style(Style::new().fg(color)),
                ])
            }
        });

    let table = Table::new(
        rows,
        if app.benchmark_view {
            vec![
                Constraint::Length(2),
                Constraint::Min(22),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(13),
                Constraint::Length(16),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
            ]
        } else {
            vec![
                Constraint::Length(2),
                Constraint::Min(22),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(13),
                Constraint::Length(20),
                Constraint::Length(15),
                Constraint::Length(5),
                Constraint::Length(8),
            ]
        },
    )
    .header(header)
    .block(block);
    f.render_widget(table, table_area);
    let selected = indices
        .get(selected_pos)
        .and_then(|&idx| app.models.get(idx));
    let detail = selected
        .map(|model| {
            let status = match &model.status {
                Status::Failed(error) => format!("Failed: {error}"),
                Status::Loaded { detail, .. } => format!("Loaded: {detail}"),
                _ => String::new(),
            };
            let (size_cat, inferred_params) =
                SizeCategory::from_model(&model.row.id, &model.row.name, model.row.total_params);
            let size_detail = if let Some(p) = inferred_params {
                let p_str = if p >= 1_000_000_000 {
                    format!("{:.1}B", p as f64 / 1_000_000_000.0)
                } else {
                    format!("{:.0}M", p as f64 / 1_000_000.0)
                };
                format!("AA Size Tier: {} ({} params)", size_cat.display_name(), p_str)
            } else {
                format!("AA Size Tier: {}", size_cat.display_name())
            };
            let all_tasks = classify_real_life_tasks(&model.row.id, &model.row.tasks, &model.row.capabilities);
            let tasks_str = all_tasks.iter().map(|t| t.display_name()).collect::<Vec<_>>().join(", ");
            let task_detail = format!("Real-Life Tasks: {}", tasks_str);
            let meta_line = format!("{} · {}", size_detail, task_detail);

            let bench_str = if let Some(b) = &model.row.benchmarks {
                let primary = if let Some(score) = b.score {
                    let name = b.index_name.as_deref().unwrap_or("Index");
                    let unit = b.unit.as_deref().unwrap_or("");
                    if unit.is_empty() {
                        format!("{name}: {score:.1}")
                    } else {
                        format!("{name}: {score:.1} {unit}")
                    }
                } else {
                    "No benchmark score".into()
                };
                let domains = b.domains.as_ref().map(|d| {
                    format!(
                        " | Domains: Fin: {:.0}, Ops: {:.0}, Legal: {:.0}, Eng: {:.0}, Med: {:.0}",
                        d.finance_and_accounting.unwrap_or(0.0),
                        d.strategy_and_ops.unwrap_or(0.0),
                        d.legal.unwrap_or(0.0),
                        d.engineering.unwrap_or(0.0),
                        d.healthcare_and_medical.unwrap_or(0.0)
                    )
                }).unwrap_or_default();
                format!(
                    "{} · Mode: {} ({}) · Cost: ${:.2} · Latency: {:.2}m · Tokens: {}{}",
                    primary,
                    b.mode.as_deref().unwrap_or("default"),
                    b.reasoning_effort.as_deref().unwrap_or("none"),
                    b.cost_per_task_usd.unwrap_or(0.0),
                    b.time_per_task_minutes.unwrap_or(0.0),
                    b.output_tokens_per_task.unwrap_or(0),
                    domains
                )
            } else {
                "No benchmark metrics recorded".into()
            };
            let action_status = app.model_action_status.as_deref().unwrap_or(
                "[s: Sort | b: View | g: Filter Task | z: Filter AA Size | Enter: Load/Unload | i: Registry]"
            );
            if status.is_empty() {
                format!("{}\n{}\n{}\n{}", model.row.id, meta_line, bench_str, action_status)
            } else {
                format!("{} ({})\n{}\n{}\n{}", model.row.id, status, meta_line, bench_str, action_status)
            }
        })
        .unwrap_or_else(|| "No models matching current filters".into());
    f.render_widget(
        Paragraph::new(detail)
            .wrap(Wrap { trim: false })
            .block(panel(" Model controls ")),
        details,
    );
}

fn draw_model_registry(f: &mut Frame, area: Rect, app: &App) {
    let [table_area, details] =
        Layout::vertical([Constraint::Min(4), Constraint::Length(5)]).areas(area);
    let visible = table_area.height.saturating_sub(3) as usize;
    let start = app
        .model_registry_selected
        .saturating_add(1)
        .saturating_sub(visible);
    let rows = app
        .model_registry_models
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, model)| {
            let size: u64 = model.versions[&model.latest]
                .files
                .iter()
                .fold(0_u64, |sum, file| sum.saturating_add(file.size_bytes));
            Row::new([
                if index == app.model_registry_selected {
                    ">"
                } else {
                    " "
                }
                .to_string(),
                model.id.clone(),
                model.name.clone(),
                crate::inference::engines::download::format_bytes(size),
                if app.index.contains_key(&model.id) {
                    "Configured"
                } else {
                    "Available"
                }
                .to_string(),
            ])
        });
    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Min(20),
            Constraint::Min(20),
            Constraint::Length(12),
            Constraint::Length(11),
        ],
    )
    .header(
        Row::new(["", "Model ID", "Name", "Download", "Status"])
            .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD)),
    )
    .block(panel(if app.info.model_registry_path.is_some() {
        " Available models · local registry "
    } else {
        " Available models · remote registry "
    }));
    f.render_widget(table, table_area);
    let status = app
        .model_registry_status
        .as_deref()
        .unwrap_or("Open the registry to browse models");
    let detail = if let Some(progress) = &app.model_download {
        format!("{status}\n{}", progress.filename)
    } else if let Some(model) = app.model_registry_models.get(app.model_registry_selected) {
        format!(
            "{status}\nEngine: {} · License: {}\n{}",
            model.engine, model.license, model.description
        )
    } else {
        status.to_string()
    };
    let block = panel(" Model installation ");
    let inner = block.inner(details);
    f.render_widget(block, details);
    let [text_area, progress_area] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(u16::from(app.model_download.is_some())),
    ])
    .areas(inner);
    f.render_widget(Paragraph::new(detail).wrap(Wrap { trim: false }), text_area);
    if let Some(progress) = &app.model_download {
        f.render_widget(
            Paragraph::new(download_progress_line(progress, app.tick)),
            progress_area,
        );
    }
}

fn draw_requests(f: &mut Frame, area: Rect, app: &App) {
    let block = panel(" Recent requests ");
    if app.requests.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Waiting for requests… point an OpenAI client at this server.",
                Style::new().fg(DIM),
            ))
            .block(block),
            area,
        );
        return;
    }
    let detailed = area.width >= 100;
    let columns = if detailed {
        vec!["Time", "Model", "Status", "Latency", "In", "Out", "tok/s"]
    } else {
        vec!["Time", "Model", "Status", "Latency"]
    };
    let header = Row::new(columns).style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));
    let visible = area.height.saturating_sub(3) as usize;
    let rows: Vec<Row> = app
        .requests
        .iter()
        .rev()
        .take(visible)
        .map(|r| {
            let finished = r.phase.is_finished();
            let ok = r.status < 400 && r.error.is_none();
            let status = match r.phase {
                RequestPhase::Incoming => "INCOMING".to_string(),
                RequestPhase::Processing => "PROCESSING".to_string(),
                _ if ok => format!("{} SUCCEEDED", r.status),
                _ => format!("{} FAILED", r.status),
            };
            let mut cells = vec![
                Cell::from(r.ts.clone()).style(Style::new().fg(DIM)),
                Cell::from(r.model.clone()),
                Cell::from(status).style(Style::new().fg(if !finished {
                    ACCENT
                } else if ok {
                    OK
                } else {
                    ERR
                })),
                Cell::from(if finished {
                    format!("{:.2}s", r.latency_ms as f64 / 1000.0)
                } else {
                    "–".into()
                }),
            ];
            if detailed {
                cells.extend([
                    Cell::from(if finished && r.prompt_tokens > 0 {
                        r.prompt_tokens.to_string()
                    } else {
                        "–".into()
                    }),
                    Cell::from(if finished && r.completion_tokens > 0 {
                        r.completion_tokens.to_string()
                    } else {
                        "–".into()
                    }),
                    Cell::from(
                        r.tokens_per_second()
                            .filter(|_| finished)
                            .map(|t| format!("{t:.1}"))
                            .unwrap_or("–".into()),
                    ),
                ]);
            }
            Row::new(cells)
        })
        .collect();
    let mut widths = vec![
        Constraint::Length(9),
        Constraint::Min(12),
        Constraint::Length(14),
        Constraint::Length(8),
    ];
    if detailed {
        widths.extend([
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
        ]);
    }
    f.render_widget(Table::new(rows, widths).header(header).block(block), area);
}

fn level_style(l: Level, msg: &str) -> Style {
    if msg.to_lowercase().contains("idle") {
        return Style::new().fg(DIM);
    }
    match l {
        Level::ERROR => Style::new().fg(ERR).add_modifier(Modifier::BOLD),
        Level::WARN => Style::new().fg(DIM),
        Level::INFO => Style::new(),
        _ => Style::new().fg(DIM),
    }
}

fn log_item(l: &LogLine) -> ListItem<'static> {
    ListItem::new(Line::from(vec![
        Span::styled(format!("{} ", l.ts), Style::new().fg(DIM)),
        Span::styled(format!("{:<5} ", l.level), level_style(l.level, &l.message)),
        Span::styled(l.message.clone(), level_style(l.level, &l.message)),
    ]))
}

fn draw_recent_logs(f: &mut Frame, area: Rect, app: &App) {
    let h = area.height.saturating_sub(2) as usize;
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .filter(|l| {
            l.level <= Level::INFO
                && l.target != "llama_server"
                && !l.target.starts_with("tower_http")
        })
        .rev()
        .take(h)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(log_item)
        .collect();
    f.render_widget(List::new(items).block(panel(" Activity ")), area);
}

fn draw_logs(f: &mut Frame, area: Rect, app: &mut App) {
    let filtered: Vec<&LogLine> = app
        .logs
        .iter()
        .filter(|l| {
            if app.errors_only {
                l.level <= Level::WARN
            } else {
                true
            }
        })
        .collect();
    let h = area.height.saturating_sub(2) as usize;
    let max_start = filtered.len().saturating_sub(h);
    if app.follow {
        app.log_scroll = max_start;
    } else {
        app.log_scroll = app.log_scroll.min(max_start);
        if app.log_scroll == max_start {
            app.follow = true;
        }
    }
    let items: Vec<ListItem> = filtered
        .iter()
        .skip(app.log_scroll)
        .take(h)
        .map(|l| log_item(l))
        .collect();
    let title = format!(
        " Logs · {} lines{}{} ",
        filtered.len(),
        if app.errors_only {
            " · errors+warnings only"
        } else {
            ""
        },
        if app.follow {
            " · following"
        } else {
            " · paused"
        }
    );
    f.render_widget(List::new(items).block(panel(title)), area);
}

fn draw_help(f: &mut Frame) {
    let area = centered(f.area(), 60, 22);
    f.render_widget(Clear, area);
    let text = vec![
        Line::from(""),
        Line::from("  l               toggle logs"),
        Line::from("  m               toggle models"),
        Line::from("  ↑ ↓             select a model in Models"),
        Line::from("  Enter           toggle selected model load/unload"),
        Line::from("  i               browse/install registry models"),
        Line::from("  Enter / c       install / cancel in registry"),
        Line::from("  e               toggle engines"),
        Line::from("  ↑ ↓             select an engine"),
        Line::from("  r               check engine releases"),
        Line::from("  u               install or update selected engine"),
        Line::from("  c               cancel engine operation in Engines"),
        Line::from("  ↑ ↓  PgUp PgDn  scroll the log view"),
        Line::from("  f               follow the newest log line"),
        Line::from("  w               toggle warnings and errors"),
        Line::from("  c               clear logs"),
        Line::from("  q / Ctrl-C      stop the server and quit"),
        Line::from(""),
        Line::styled(
            "  Models load on first request and are evicted",
            Style::new().fg(DIM),
        ),
        Line::styled(
            "  least-recently-used when the limit is reached.",
            Style::new().fg(DIM),
        ),
        Line::from(""),
        Line::styled("  press any key to close", Style::new().fg(DIM)),
    ];
    f.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .alignment(Alignment::Left)
            .block(panel(" Help ")),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn export_review_frames() {
        let Ok(directory) = std::env::var("RUNE_TUI_SNAPSHOT_DIR") else {
            return;
        };
        std::fs::create_dir_all(&directory).unwrap();
        for scenario in ["idle", "busy", "loading", "models", "logs", "80-columns"] {
            let mut a = app();
            a.info.exposed_to_network = false;
            a.started = Instant::now();
            if matches!(scenario, "busy" | "models" | "logs" | "80-columns") {
                a.models[0].status = Status::Loaded {
                    load_ms: 1500,
                    detail: "fixture".into(),
                };
                for i in 0..20 {
                    a.on_event(UiEvent::Request(RequestRecord {
                        id: 0,
                        phase: RequestPhase::Succeeded,
                        ts: ts(),
                        model: "a".into(),
                        status: 200,
                        latency_ms: 500 + i * 20,
                        prompt_tokens: 64,
                        completion_tokens: 32,
                        generation_ms: 500,
                        error: None,
                    }));
                }
            }
            if scenario == "loading" {
                a.models[0].status = Status::Loading(Instant::now());
            }
            a.logs.push_back(LogLine {
                ts: ts(),
                level: Level::INFO,
                target: "fixture".into(),
                message: "Synthetic review state; no model benchmark".into(),
            });
            if scenario == "models" {
                a.tab = Tab::Models;
            }
            if scenario == "logs" {
                a.tab = Tab::Logs;
            }
            let width = if scenario == "80-columns" { 80 } else { 120 };
            let mut terminal = Terminal::new(TestBackend::new(width, 32)).unwrap();
            terminal.draw(|frame| draw(frame, &mut a)).unwrap();
            let buffer = terminal.backend().buffer();
            if is_no_color() {
                assert!(
                    buffer
                        .content
                        .iter()
                        .all(|cell| cell.fg == ratatui::style::Color::Reset
                            && cell.bg == ratatui::style::Color::Reset)
                );
            }
            if std::env::var_os("RUNE_ASCII").is_some() {
                assert!(buffer.content.iter().all(|cell| cell.symbol().is_ascii()));
            }
            let cells: Vec<_> = buffer.content.iter().map(|cell| serde_json::json!({"text":cell.symbol(), "fg":format!("{:?}",cell.fg), "bg":format!("{:?}",cell.bg)})).collect();
            std::fs::write(
                std::path::Path::new(&directory).join(format!("{scenario}.json")),
                serde_json::to_vec(&serde_json::json!({"width":width,"height":32,"cells":cells}))
                    .unwrap(),
            )
            .unwrap();
        }
    }

    fn app() -> App {
        let info = DashboardInfo {
            version: "0.0.0",
            addr: "127.0.0.1:8080".into(),
            auth_enabled: false,
            exposed_to_network: true,
            config_path: "x".into(),
            root: "fixture-root".into(),
            server_path: "fixture-server.json".into(),
            model_registry_path: None,
            models: vec![
                ModelRow {
                    id: "a".into(),
                    name: "A".into(),
                    modality: "text".into(),
                    provider: None,
                    engine: Some("llama-server".into()),
                    locality: "LOCAL".into(),
                    capabilities: vec!["Chat".into()],
                    tasks: vec!["text-to-text".into()],
                    total_params: Some(7_000_000_000),
                    benchmarks: None,
                },
                ModelRow {
                    id: "g".into(),
                    name: "G".into(),
                    modality: "vision-text".into(),
                    provider: Some("google".into()),
                    engine: None,
                    locality: "CLOUD".into(),
                    capabilities: vec!["Chat".into(), "Vision".into()],
                    tasks: vec!["text-image-to-text".into(), "text-video-to-text".into()],
                    total_params: Some(200_000_000_000),
                    benchmarks: None,
                },
            ],
        };
        App::new(info, Arc::new(Mutex::new(Some((8 << 30, 16 << 30)))))
    }

    fn ts() -> String {
        "00:00:00".into()
    }

    #[test]
    fn model_registry_view_loads_the_explicit_file_and_navigates() {
        let mut app = app();
        app.info.model_registry_path = Some(
            std::env::current_dir()
                .unwrap()
                .join("registry/models.json"),
        );
        app.on_key(KeyCode::Char('m'), KeyModifiers::NONE);
        app.on_key(KeyCode::Char('i'), KeyModifiers::NONE);
        assert!(app.model_registry_open && app.model_registry_refresh);
        app.process_model_registry();
        let received = app
            .model_registry_rx
            .take()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(received).unwrap();
        app.model_registry_rx = Some(rx);
        app.process_model_registry();
        assert!(
            app.model_registry_models
                .iter()
                .any(|model| model.id == "lfm3-1.7b-q4-k-m")
        );
        app.model_registry_selected = app
            .model_registry_models
            .iter()
            .position(|model| model.id == "lfm3-1.7b-q4-k-m")
            .unwrap();
        let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let rendered: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(rendered.contains("lfm3-1.7b-q4-k-m"));
        assert!(rendered.contains("LFM Open License"));
        app.model_registry_selected = 0;
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.model_registry_selected, 1);
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.model_install_requested);
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.model_registry_open);
    }

    #[test]
    fn model_catalog_benchmark_view_and_sorting() {
        let mut app = app();
        app.on_key(KeyCode::Char('m'), KeyModifiers::NONE);
        assert_eq!(app.tab, Tab::Models);
        assert!(!app.benchmark_view);
        assert_eq!(app.model_sort, ModelSort::Default);

        // Toggle benchmark view
        app.on_key(KeyCode::Char('b'), KeyModifiers::NONE);
        assert!(app.benchmark_view);

        // Cycle sorting
        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::IntelligenceIndex);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::FinanceIndex);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::Cost);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::Latency);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::TaskGroup);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::SizeTier);

        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.model_sort, ModelSort::Default);

        // Test Task Filter cycling [g]
        assert_eq!(app.task_filter, TaskFilter::All);
        app.on_key(KeyCode::Char('g'), KeyModifiers::NONE);
        assert_eq!(app.task_filter, TaskFilter::ImageEditing);

        // Test Size Filter cycling [z]
        assert_eq!(app.size_filter, SizeFilter::All);
        app.on_key(KeyCode::Char('z'), KeyModifiers::NONE);
        assert_eq!(app.size_filter, SizeFilter::Large);

        // Verify Artificial Analysis size classification
        let (size_a, p_a) =
            SizeCategory::from_model("qwen3.8-27b", "Qwen3.8 27B", Some(27_000_000_000));
        assert_eq!(size_a, SizeCategory::Small);
        assert_eq!(p_a, Some(27_000_000_000));

        let (size_gemini, _) = SizeCategory::from_model("gemini-4-carbon", "Gemini 4 Carbon", None);
        assert_eq!(size_gemini, SizeCategory::Large);

        let (size_08b, _) =
            SizeCategory::from_model("qwen3.5-0.8b", "Qwen3.5 0.8B", Some(873_000_000));
        assert_eq!(size_08b, SizeCategory::Tiny);

        let (size_nemo, _) =
            SizeCategory::from_model("nemotron-3.5-super-vl", "Nemotron 3.5 Super-VL", None);
        assert_eq!(size_nemo, SizeCategory::Medium);

        // Verify Real-Life Task classification
        let tasks_edit =
            classify_real_life_tasks("qwen-image-edit-2511", &["image-to-image".into()], &[]);
        assert!(tasks_edit.contains(&RealLifeTask::ImageEditing));

        let tasks_ox = classify_real_life_tasks(
            "oxcoder-9b",
            &["text-to-text".into()],
            &["Agentic Coding".into()],
        );
        assert!(tasks_ox.contains(&RealLifeTask::CodeAndCad));
        assert!(tasks_ox.contains(&RealLifeTask::ComputerUseAndAgent));

        let tasks_scad = classify_real_life_tasks(
            "scadcoder-qwen3.5-9b-v11-q5-k-m",
            &["text-to-text".into()],
            &["Chat".into()],
        );
        assert!(tasks_scad.contains(&RealLifeTask::ThreeDAndDepth));
        assert!(tasks_scad.contains(&RealLifeTask::CodeAndCad));
    }

    #[test]
    fn dashboard_model_install_uses_selected_registry_entry() {
        use sha2::{Digest, Sha256};
        let mut app = app();
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "dashboard-model-{}",
                crate::setup::generate_api_key()
            ));
        app.info.root = root.clone();
        app.info.server_path = root.join("config/server.json");
        app.info.config_path = root.join("config/catalog.json");
        let mut model = crate::model_registry::load_from_path(
            &std::env::current_dir()
                .unwrap()
                .join("registry/models.json"),
        )
        .unwrap()
        .into_iter()
        .find(|model| model.id == "qwen3.5-0.8b")
        .unwrap();
        let file = &mut model.versions.get_mut(&model.latest).unwrap().files[0];
        let bytes = b"GGUF fixture dashboard";
        file.sha256 = hex::encode(Sha256::digest(bytes));
        file.size_bytes = bytes.len() as u64;
        let directory = root
            .join("models")
            .join(model.install_dir.as_deref().unwrap());
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(&file.filename), bytes).unwrap();
        app.model_registry_models = vec![model.clone()];
        app.model_registry_open = true;
        app.tab = Tab::Models;
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        app.process_model_registry();
        let received = app
            .model_install_rx
            .take()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(received).unwrap();
        app.model_install_rx = Some(rx);
        app.process_model_registry();
        assert!(
            app.model_registry_status
                .as_deref()
                .unwrap()
                .contains("Installed")
        );
        let loaded =
            crate::config::ModelRegistry::from_files(&app.info.server_path, &app.info.config_path)
                .unwrap();
        assert!(loaded.models.iter().any(|entry| entry.id == model.id));
        let state = Arc::new(crate::inference::AppState::new(&loaded));
        app.sync_configured_models(Some(&state));
        assert!(app.index.contains_key(&model.id));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_controls_select_scroll_and_report_results_without_blocking() {
        let mut app = app();
        app.on_key(KeyCode::Char('m'), KeyModifiers::NONE);
        app.on_key(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(app.model_selected, 0);
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(matches!(app.model_action, Some(ModelAction::Load)));
        app.process_model_action(None);
        assert!(
            app.model_action_status
                .as_ref()
                .unwrap()
                .contains("server state")
        );
        app.models[0].status = Status::Loaded {
            load_ms: 0,
            detail: String::new(),
        };
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(matches!(app.model_action.take(), Some(ModelAction::Unload)));
        app.models[0].status = Status::Loading(Instant::now());
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.model_action.is_none());
        app.models[0].status = Status::Idle;
        for _ in 0..3 {
            app.on_key(KeyCode::Down, KeyModifiers::NONE);
        }
        assert_eq!(app.model_selected, 1);
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        let state = Arc::new(crate::inference::AppState::new(
            &crate::config::ModelRegistry::default(),
        ));
        app.process_model_action(Some(&state));
        assert!(
            app.model_action_status
                .as_ref()
                .unwrap()
                .contains("Cloud models")
        );
        let (tx, rx) = std::sync::mpsc::channel();
        app.model_action_rx = Some(rx);
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.model_action.is_none());
        tx.send(Err("in use".into())).unwrap();
        app.process_model_action(Some(&state));
        assert_eq!(app.model_action_status.as_deref(), Some("Failed: in use"));
        assert!(app.model_action_rx.is_none());
        for index in 2..30 {
            let mut model = app.models[0].clone();
            model.row.id = format!("model-{index}");
            app.models.push(model);
        }
        app.model_selected = 29;
        let mut terminal = Terminal::new(TestBackend::new(140, 25)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        for text in [">", "model-29", "Model controls", "load", "Failed: in use"] {
            assert!(screen.contains(text), "{text}");
        }
        app.models[29].status = Status::Loaded {
            load_ms: 0,
            detail: String::new(),
        };
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(screen.contains("unload"));
        draw_all(&mut app);
    }

    fn draw_all(app: &mut App) {
        for (w, h) in [(20, 6), (60, 20), (120, 40)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            for tab in [Tab::Dashboard, Tab::Models, Tab::Logs, Tab::Engines] {
                app.tab = tab;
                t.draw(|f| draw(f, app)).unwrap();
            }
        }
        app.tab = Tab::Dashboard;
    }

    #[test]
    fn dashboard_places_engines_above_performance_and_requests_on_the_left() {
        let mut app = app();
        app.engines.push(EngineRow {
            name: "audio-cpp",
            path: "bin/audio".into(),
            managed: true,
            installed: true,
            version: "0.9.2".into(),
            latest: "Not checked".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        });
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let locate = |word: &str| -> (u16, u16) {
            for y in 0..buffer.area.height {
                let line: String = (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect();
                if let Some(x) = line.find(word) {
                    return (x as u16, y);
                }
            }
            panic!("missing {word}");
        };
        let engines = locate("Engines");
        let performance = locate("Performance");
        let requests = locate("Recent requests");
        assert!(engines.0 > requests.0 && performance.0 > requests.0);
        assert!(engines.1 < performance.1 && requests.1 == performance.1);
    }

    #[test]
    fn dashboard_shows_only_actual_loaded_engines_and_navigation_toggles() {
        let mut app = app();
        app.engines = vec![
            EngineRow {
                name: "audio-cpp",
                path: "audio".into(),
                managed: true,
                installed: true,
                version: "0.9.2".into(),
                latest: "Not checked".into(),
                build_tag: None,
                registry_versions: None,
                registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: None,
            },
            EngineRow {
                name: "llama-server",
                path: "llama".into(),
                managed: true,
                installed: true,
                version: "1.0".into(),
                latest: "Not checked".into(),
                build_tag: None,
                registry_versions: None,
                registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: None,
            },
        ];
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let render = |terminal: &mut Terminal<TestBackend>, app: &App| {
            terminal
                .draw(|f| draw_engine_overview(f, f.area(), app))
                .unwrap();
            terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
        };
        let idle = render(&mut terminal, &app);
        assert!(idle.contains("No engines active"));
        assert!(!idle.contains("audio-cpp"));
        app.models[0].status = Status::Loaded {
            load_ms: 10,
            detail: "audio-cpp · +1 GiB VRAM".into(),
        };
        let active = render(&mut terminal, &app);
        assert!(active.contains("audio-cpp"));
        assert!(!active.contains("llama-server"));
        app.models[0].status = Status::Idle;
        assert!(render(&mut terminal, &app).contains("No engines active"));
        for (key, tab) in [('l', Tab::Logs), ('m', Tab::Models), ('e', Tab::Engines)] {
            app.on_key(KeyCode::Char(key), KeyModifiers::NONE);
            assert!(app.tab == tab);
            app.on_key(KeyCode::Char(key), KeyModifiers::NONE);
            assert!(app.tab == Tab::Dashboard);
        }
        app.on_key(KeyCode::Tab, KeyModifiers::NONE);
        app.on_key(KeyCode::Char('g'), KeyModifiers::NONE);
        assert!(app.tab == Tab::Dashboard);
    }

    #[test]
    fn compares_only_recognised_release_versions() {
        let mut row = EngineRow {
            name: "audio-cpp",
            path: "bin".into(),
            managed: true,
            installed: true,
            version: "audio.cpp 0.9.2".into(),
            latest: "v0.9.3".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: Some([("0.9.2".into(), 1), ("0.9.3".into(), 2)].into()),
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        };
        assert_eq!(release_status(&row), "New release");
        row.latest = "v0.9.2".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "nightly".into();
        assert_eq!(release_status(&row), "Unknown");
        row.latest = "v0.9.3".into();
        row.version = "audio.cpp 0.9.3 (rune-gguf1)".into();
        assert_eq!(release_status(&row), "Different revision");
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Update
        );
        row.version = "audio.cpp 0.9.3 (unknown)".into();
        assert_eq!(release_status(&row), "Unknown");
        row.name = "whisper-cpp";
        row.version = "whisper.cpp version: 1.9.2".into();
        row.latest = "v1.9.2".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "v1.9.3".into();
        assert_eq!(release_status(&row), "New release");
        row.latest = "v1.9.1".into();
        assert_eq!(release_status(&row), "Ahead");
        row.name = "nemo-speech";
        row.registry_revisions = Some(
            [
                ("9.0.0".into(), 1),
                ("1.0.0".into(), 2),
                ("3.0.0".into(), 3),
            ]
            .into(),
        );
        row.latest = "v1.0.0".into();
        row.version = "nemo-speech 9.0.0".into();
        assert_eq!(release_status(&row), "New release");
        row.version = "nemo-speech 1.0.0".into();
        assert_eq!(release_status(&row), "Up to date");
        row.version = "nemo-speech 3.0.0".into();
        assert_eq!(release_status(&row), "Ahead");
        row.version = "nemo-speech 4.0.0".into();
        assert_eq!(release_status(&row), "Unknown");
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.name = "llama-server";
        row.version = "0.6.0-dev (build 11512, commit abc)".into();
        row.latest = "b11512".into();
        row.registry_versions = Some([(11512, 1)].into());
        assert_eq!(release_status(&row), "Up to date");
        row.registry_versions = Some([(900, 1), (3, 2), (100, 3)].into());
        row.latest = "b3".into();
        row.version = "version: test (build 900)".into();
        assert_eq!(release_status(&row), "New release");
        row.version = "version: test (build 3)".into();
        assert_eq!(release_status(&row), "Up to date");
        row.version = "version: test (build 100)".into();
        assert_eq!(release_status(&row), "Ahead");
        row.version = "version: test (build 42)".into();
        assert_eq!(release_status(&row), "Unknown");
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        for (name, version, latest, build, expected) in [
            (
                "llama-server",
                "0.7.0-dev (build 11669, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "Unknown",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11668, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "Unknown",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11670, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "Unknown",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11669, commit 620d2225c)",
                "v0.7.0",
                None,
                "Unknown",
            ),
            (
                "llama-server-prism",
                "0.2.0-dev (build 10826, commit c9c34bc9c)",
                "prism-b10826-c9c34bc",
                None,
                "Up to date",
            ),
            (
                "llama-server-prism",
                "0.2.0-dev (build 10826, commit aaaaaaa)",
                "prism-b10826-c9c34bc",
                None,
                "Unknown",
            ),
            (
                "llama-server-prism",
                "0.2.0-dev (build 10826)",
                "prism-b10826-c9c34bc",
                None,
                "Unknown",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11669)",
                "Check failed",
                Some("b11669"),
                "Check failed",
            ),
        ] {
            row.name = name;
            row.version = version.into();
            row.latest = latest.into();
            row.build_tag = build.map(str::to_owned);
            row.registry_revisions = Some([("prism-b10826-c9c34bc".to_string(), 1)].into());
            assert_eq!(
                release_status(&row),
                expected,
                "{name}: {version} / {latest}"
            );
        }
        row.check_detail = Some("Cached release metadata · checked 1h ago".into());
        assert!(engine_status_label(&row).contains("cached"));
        row.latest = "Rate limited".into();
        assert_eq!(release_status(&row), "Rate limited");
    }

    #[test]
    fn release_age_covers_hours_days_months_and_missing_dates() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-10-28T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(
            release_age("2026-10-28T10:00:00Z", now).as_deref(),
            Some("2 hours old")
        );
        assert_eq!(
            release_age("2026-10-26T12:00:00Z", now).as_deref(),
            Some("2 days old")
        );
        assert_eq!(
            release_age("2026-09-28T12:00:00Z", now).as_deref(),
            Some("1 month old")
        );
        assert_eq!(
            release_age("2026-10-28T13:00:00Z", now).as_deref(),
            Some("less than 1 hour old")
        );
        assert_eq!(release_age("invalid", now), None);
    }

    #[test]
    fn engine_download_progress_renders_a_dotted_bar_and_unknown_size_honestly() {
        let mut app = app();
        app.engines.push(EngineRow {
            name: "whisper-cpp",
            path: "bin/whisper-cpp".into(),
            managed: false,
            installed: false,
            version: "Not installed".into(),
            latest: "v1.9.2".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        });
        app.update_engine = Some("whisper-cpp".into());
        app.update_status = Some("Downloading engine.zip · 512 B / 1.00 KiB".into());
        app.engine_download = Some(crate::inference::engines::download::DownloadProgress {
            filename: "engine.zip".into(),
            extracting: false,
            downloaded: 512,
            total: Some(1024),
            bytes_per_second: 256.0,
        });
        let mut terminal = Terminal::new(TestBackend::new(120, 16)).unwrap();
        terminal
            .draw(|frame| draw_engines(frame, frame.area(), &app))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("••••••••········"));
        assert!(text.contains("50%"));
        assert!(!text.contains("Downloading engine.zip"));
        assert!(text.contains("Path: bin/whisper-cpp"));
        let progress = app.engine_download.as_mut().unwrap();
        progress.total = None;
        let line = download_progress_line(progress, 0);
        let text: String = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("size unknown"));
        assert!(!text.contains('%'));
        app.on_key(KeyCode::Char('e'), KeyModifiers::NONE);
        app.on_key(KeyCode::Char('i'), KeyModifiers::NONE);
        assert!(!app.update_requested);
        app.on_key(KeyCode::Char('u'), KeyModifiers::NONE);
        assert!(app.update_requested);
    }

    #[test]
    fn model_download_uses_engine_progress_bar_and_real_filename() {
        let mut app = app();
        app.tab = Tab::Models;
        app.model_registry_open = true;
        app.model_registry_status = Some("Installing lfm3…".into());
        app.model_download = Some(crate::inference::engines::download::DownloadProgress {
            filename: "LFM3-1.7B-Q4_K_M.gguf".into(),
            extracting: false,
            downloaded: 512,
            total: Some(1024),
            bytes_per_second: 256.0,
        });
        let mut terminal = Terminal::new(TestBackend::new(120, 16)).unwrap();
        terminal
            .draw(|frame| draw_model_registry(frame, frame.area(), &app))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("LFM3-1.7B-Q4_K_M.gguf"));
        assert!(text.contains("••••••••········"));
        assert!(text.contains("50%"));
        assert!(text.contains("256 B/s"));
        let progress = app.model_download.as_mut().unwrap();
        progress.total = None;
        let line = download_progress_line(progress, 0);
        let text: String = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("size unknown"));
    }

    #[test]
    fn engine_status_distinguishes_not_installed_development_and_built_in() {
        let mut row = EngineRow {
            name: "whisper-cpp",
            path: "bin/whisper-cpp".into(),
            managed: false,
            installed: false,
            version: "Not installed".into(),
            latest: "v1.9.2".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        };
        assert_eq!(release_status(&row), "Not installed");
        row.installed = true;
        row.version = "nemo-speech 0.2.0+nightly.123".into();
        assert_eq!(release_status(&row), "Development build");
        row.name = "encoder";
        assert_eq!(release_status(&row), "Built in · synthetic");
        assert!(install_engine_kind("needle").is_some());
        row.name = "needle";
        row.version = format!("revision {}", "a".repeat(40));
        row.latest = row.version.clone();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = format!("revision {}", "b".repeat(40));
        assert_eq!(release_status(&row), "Different revision");
        assert!(install_engine_kind("encoder").is_none());
        row.name = "sd-server";
        row.registry_revisions = Some([("be48649".into(), 1), ("37bc88b".into(), 2)].into());
        row.version = "commit be48649".into();
        row.latest = "master-1017-be48649".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "master-1028-37bc88b".into();
        assert_eq!(release_status(&row), "New release");
        row.version = "Unknown".into();
        assert_eq!(release_status(&row), "Unknown");
        let mut app = app();
        app.on_key(KeyCode::Char('e'), KeyModifiers::NONE);
        app.on_key(KeyCode::Char('u'), KeyModifiers::NONE);
        assert!(app.update_requested);
    }

    #[test]
    fn release_tag_check_skips_unpublished_releases() {
        let data = serde_json::json!([
            {"tag_name":"v3","draft":true}, {"tag_name":"v2-rc","prerelease":true},
            {"tag_name":"v1.0.0","draft":false,"prerelease":false}
        ]);
        assert_eq!(latest_release_tag(&data).as_deref(), Some("v1.0.0"));
        assert_eq!(latest_release_tag(&serde_json::json!([])), None);
    }

    #[test]
    fn engine_versions_are_read_from_stdout_and_stderr() {
        assert_eq!(
            parse_engine_version("", "version: 0.6.0-dev (build 11512)\n"),
            Some("0.6.0-dev (build 11512)".into())
        );
        assert_eq!(
            parse_engine_version("=== build info ===\n version : 0.8.43\n", ""),
            Some("0.8.43".into())
        );
        assert_eq!(
            parse_engine_version("audio.cpp 0.9.2\n", ""),
            Some("audio.cpp 0.9.2".into())
        );
        assert_eq!(parse_engine_version("", ""), None);
    }

    #[test]
    fn engine_cancel_key_signals_only_the_active_operation_and_keeps_infer_running() {
        let mut app = app();
        app.tab = Tab::Engines;
        let token = app.begin_engine_operation();
        let (_tx, rx) = std::sync::mpsc::channel();
        app.update_rx = Some(rx);
        let mut terminal = Terminal::new(TestBackend::new(180, 3)).unwrap();
        terminal.draw(|f| draw_footer(f, f.area(), &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("cancel engine operation"));
        app.on_key(KeyCode::Char('c'), KeyModifiers::NONE);
        assert!(token.load(Ordering::Relaxed));
        assert!(!app.quit);
        assert!(app.update_status.as_deref().unwrap().contains("Cancelling"));
        terminal.draw(|f| draw_footer(f, f.area(), &app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("cancelling engine operation"));
        app.update_rx = None;
        app.update_cancel = None;
        let next = app.begin_engine_operation();
        assert!(!next.load(Ordering::Relaxed));
        assert!(token.load(Ordering::Relaxed));
        drop(app);
        assert!(
            next.load(Ordering::Relaxed),
            "quitting also cancels the active engine worker"
        );
    }

    #[test]
    fn onnx_runtime_install_requires_verified_managed_windows_bundle() {
        let mut row = EngineRow {
            name: "onnx",
            path: "bin/onnxruntime/onnxruntime.dll".into(),
            managed: true,
            installed: false,
            version: "Not installed".into(),
            latest: "1.31.1 · pinned".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        };
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Install
        );
        assert!(matches!(
            engine_action_on(&row, "linux", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.installed = true;
        row.version = "ONNX Runtime 1.31.1 CPU".into();
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Install
        );
        assert_eq!(release_status(&row), "Ready · CPU");
        row.version = "ONNX Runtime 1.31.1 CPU + CUDA".into();
        assert_eq!(release_status(&row), "Ready · CUDA + CPU");
        row.version = "Unknown".into();
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.managed = false;
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        assert_eq!(
            install_engine_kind("onnx"),
            Some(crate::config::EngineKind::Onnx)
        );
    }

    #[test]
    fn missing_litert_runtime_offers_install_on_windows_x64_and_preserves_external_files() {
        let mut row = EngineRow {
            name: "litert",
            path: "bin/litert/libLiteRt.dll".into(),
            managed: true,
            installed: false,
            version: "Not installed".into(),
            latest: "2.3.0 · pinned".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        };
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Install
        );
        assert_eq!(
            install_engine_kind("litert"),
            Some(crate::config::EngineKind::Litert)
        );
        for (os, arch) in [
            ("linux", "x86_64"),
            ("windows", "aarch64"),
            ("macos", "aarch64"),
        ] {
            assert!(matches!(
                engine_action_on(&row, os, arch),
                EngineAction::Unavailable(_)
            ));
        }
        row.managed = false;
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.managed = true;
        let mut app = app();
        app.tab = Tab::Engines;
        app.engines = vec![row.clone()];
        let mut terminal = Terminal::new(TestBackend::new(160, 3)).unwrap();
        terminal
            .draw(|frame| draw_footer(frame, frame.area(), &app))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert_eq!(
            text.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .contains("u install"),
            cfg!(all(windows, target_arch = "x86_64"))
        );
        row.installed = true;
        row.version = "Unknown".into();
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.version = "LiteRT 2.3.0 CPU".into();
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Install
        ));
        assert_eq!(release_status(&row), "Ready · CPU");
        row.version = "LiteRT 2.3.0 GPU + CPU".into();
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        assert_eq!(release_status(&row), "Ready · GPU + CPU");
    }

    #[test]
    fn engine_actions_match_implemented_updaters_and_release_readiness() {
        let mut row = EngineRow {
            name: "audio-cpp",
            path: "bin/test".into(),
            managed: true,
            installed: true,
            version: "0.9.2".into(),
            latest: "v0.9.3".into(),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: false,
            check_detail: None,
        };
        for (name, os, arch, allowed) in [
            ("audio-cpp", "windows", "x86_64", true),
            ("audio-cpp", "linux", "x86_64", false),
            ("whisper-cpp", "windows", "x86_64", true),
            ("whisper-cpp", "windows", "aarch64", false),
            ("whisper-cpp", "linux", "x86_64", false),
            ("crispasr", "windows", "x86_64", true),
            ("crispasr", "windows", "aarch64", false),
            ("crispasr", "macos", "aarch64", false),
            ("llama-server", "windows", "x86_64", true),
            ("llama-server", "windows", "aarch64", false),
            ("llama-server", "linux", "x86_64", false),
            ("llama-server-prism", "windows", "x86_64", true),
            ("llama-server-prism", "windows", "aarch64", false),
            ("llama-server-prism", "linux", "x86_64", false),
            ("sd-server", "windows", "x86_64", true),
            ("sd-server", "windows", "aarch64", false),
            ("sd-server", "linux", "x86_64", false),
            ("nemo-speech", "windows", "x86_64", true),
            ("nemo-speech", "windows", "aarch64", false),
            ("nemo-speech", "linux", "x86_64", false),
            ("needle", "windows", "x86_64", true),
            ("needle", "windows", "aarch64", false),
            ("needle", "linux", "x86_64", false),
            ("encoder", "windows", "x86_64", false),
            ("litert", "windows", "x86_64", false),
        ] {
            row.name = name;
            row.version = if matches!(name, "llama-server" | "llama-server-prism") {
                "0.9.2-dev (build 12000, commit abcdef123)".into()
            } else {
                "0.9.2".into()
            };
            row.build_tag = Some("b12100".into());
            row.latest = "v0.9.3".into();
            if name == "llama-server" {
                row.latest = "b12100".into();
                row.registry_versions = Some([(12000, 1), (12100, 2)].into());
            }
            if name == "llama-server-prism" {
                row.latest = "prism-b12100-bbbbbbb".into();
                row.build_tag = None;
                row.registry_revisions = Some(
                    [
                        ("prism-b12000-abcdef1".into(), 1),
                        ("prism-b12100-bbbbbbb".into(), 2),
                    ]
                    .into(),
                );
            }
            if name == "nemo-speech" {
                row.registry_revisions = Some([("0.9.2".into(), 1), ("0.9.3".into(), 2)].into());
            }
            if name == "audio-cpp" {
                row.version = "audio.cpp 0.9.2".into();
                row.registry_revisions = Some([("0.9.2".into(), 1), ("0.9.3".into(), 2)].into());
            }
            if name == "crispasr" {
                row.registry_revisions = Some([("0.9.2".into(), 1), ("0.9.3".into(), 2)].into());
            }
            if name == "sd-server" {
                row.version = "commit be48649".into();
                row.latest = "master-1028-37bc88b".into();
                row.registry_revisions =
                    Some([("be48649".into(), 1), ("37bc88b".into(), 2)].into());
            }
            if name == "needle" {
                row.version = format!("revision {}", "a".repeat(40));
                row.latest = format!("revision {}", "b".repeat(40));
            }
            assert_eq!(
                engine_action_on(&row, os, arch) == EngineAction::Update,
                allowed,
                "{name}/{os}/{arch}"
            );
        }
        row.name = "whisper-cpp";
        for installed in [true, false] {
            row.installed = installed;
            for latest in [
                "Checking…",
                "Check failed",
                "Rate limited",
                "Not checked",
                "Unknown",
                "",
            ] {
                row.latest = latest.into();
                assert!(
                    matches!(
                        engine_action_on(&row, "windows", "x86_64"),
                        EngineAction::Unavailable(_)
                    ),
                    "{latest}, installed={installed}"
                );
            }
        }
        row.latest = "v0.9.3".into();
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Install
        );
        row.installed = true;
        row.managed = false;
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.managed = true;
        for version in ["0.9.3", "0.9.4"] {
            row.version = version.into();
            assert!(matches!(
                engine_action_on(&row, "windows", "x86_64"),
                EngineAction::Unavailable(_)
            ));
        }
        row.version = "0.9.2".into();
        row.latest = "Rate limited".into();
        row.check_detail = Some("GitHub rate limited until reset".into());
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable("GitHub rate limited until reset".into())
        );
    }

    #[test]
    fn whisper_same_version_gpu_refresh_is_actionable_without_allowing_downgrades_or_external_updates()
     {
        let mut row = EngineRow {
            name: "whisper-cpp",
            path: "bin/whisper-cpp/Release/whisper-cli.exe".into(),
            managed: true,
            installed: true,
            version: "whisper.cpp version: 1.9.5".into(),
            latest: "v1.9.5".into(),
            build_tag: Some("b5454".into()),
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            gpu_upgrade: true,
            check_detail: None,
        };
        assert_eq!(release_status(&row), "GPU build available");
        assert_eq!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Update
        );
        row.gpu_upgrade = false;
        assert_eq!(release_status(&row), "Up to date");
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.gpu_upgrade = true;
        row.version = "whisper.cpp version: 1.9.6".into();
        assert_eq!(release_status(&row), "Ahead");
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
        row.version = "whisper.cpp version: 1.9.5".into();
        row.managed = false;
        assert!(matches!(
            engine_action_on(&row, "windows", "x86_64"),
            EngineAction::Unavailable(_)
        ));
    }

    #[test]
    fn engine_inventory_renders_selection_and_unknown_update_status() {
        let mut app = app();
        app.engines = vec![
            EngineRow {
                name: "audio-cpp",
                path: "bin/audiocpp/audiocpp_cli".into(),
                managed: true,
                installed: true,
                version: "audio.cpp 0.9.2".into(),
                latest: "Not checked".into(),
                build_tag: None,
                registry_versions: None,
                registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: None,
            },
            EngineRow {
                name: "needle",
                path: "bin/needle/needle".into(),
                managed: true,
                installed: true,
                version: "Unknown".into(),
                latest: "Not checked".into(),
                build_tag: None,
                registry_versions: None,
                registry_revisions: None,
                published_at: None,
                gpu_upgrade: false,
                check_detail: None,
            },
        ];
        app.on_key(KeyCode::Char('e'), KeyModifiers::NONE);
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert!(app.tab == Tab::Engines);
        assert_eq!(app.engine_selected, 1);
        let mut terminal = Terminal::new(TestBackend::new(100, 16)).unwrap();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("needle"));
        assert!(text.contains("Not checked"));
        assert!(text.contains("bin/needle/needle"));
        let expected_action = if cfg!(all(windows, target_arch = "x86_64")) {
            "Check versions before installing or updating"
        } else {
            "Automatic update is unavailable on this platform"
        };
        assert!(text.contains(expected_action));
        assert!(!text.contains("u update"));
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.engine_selected, 1);
    }

    #[test]
    fn fractional_speed_and_unknown_usage_are_displayed_honestly() {
        let mut a = app();
        let mut terminal = Terminal::new(TestBackend::new(120, 20)).unwrap();
        terminal.draw(|f| draw_stats(f, f.area(), &a)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("last – tok/s"));
        a.on_request(RequestRecord {
            id: 1,
            phase: RequestPhase::Succeeded,
            ts: ts(),
            model: "a".into(),
            status: 200,
            latency_ms: 5000,
            prompt_tokens: 10,
            completion_tokens: 1,
            generation_ms: 5000,
            error: None,
        });
        assert_eq!(a.tps_hist.back(), Some(&200));
        terminal.draw(|f| draw_stats(f, f.area(), &a)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains("last 0.2 tok/s"));
        a.on_request(RequestRecord {
            id: 2,
            phase: RequestPhase::Succeeded,
            ts: ts(),
            model: "a".into(),
            status: 200,
            latency_ms: 5000,
            prompt_tokens: 0,
            completion_tokens: 0,
            generation_ms: 5000,
            error: None,
        });
        assert_eq!(
            a.tps_hist.len(),
            1,
            "unknown counts do not add a fake zero rate"
        );
        terminal.draw(|f| draw_requests(f, f.area(), &a)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(text.contains('–'));
    }

    #[test]
    fn lifecycle_updates_same_rows_without_double_counting_or_reordering() {
        let mut a = app();
        let record = |id, phase, model: &str| RequestRecord {
            id,
            phase,
            ts: ts(),
            model: model.into(),
            status: if phase.is_finished() { 200 } else { 0 },
            latency_ms: 100,
            prompt_tokens: 10,
            completion_tokens: 20,
            generation_ms: 100,
            error: None,
        };
        a.on_request(record(1, RequestPhase::Incoming, "a"));
        assert_eq!(a.requests.len(), 1);
        assert_eq!(a.total_requests, 0);
        draw_all(&mut a);
        a.on_request(record(2, RequestPhase::Incoming, "a"));
        a.on_request(record(1, RequestPhase::Processing, "a"));
        assert_eq!(a.requests.len(), 2);
        assert_eq!(a.requests[0].phase, RequestPhase::Processing);
        draw_all(&mut a);
        let done = record(1, RequestPhase::Succeeded, "a");
        a.on_request(done.clone());
        a.on_request(done);
        assert_eq!(a.total_requests, 1);
        assert_eq!(a.models[0].requests, 1);
        let mut failed = record(2, RequestPhase::Failed, "a");
        failed.status = 400;
        failed.error = Some("unsupported image".into());
        a.on_request(failed);
        assert_eq!((a.total_requests, a.total_errors), (2, 1));
        assert_eq!(a.requests.len(), 2);
        assert_eq!(a.requests[0].id, 1);
        assert_eq!(a.requests[1].id, 2);
        draw_all(&mut a);
    }

    #[test]
    fn state_tracks_model_lifecycle_and_stats() {
        let mut a = app();
        draw_all(&mut a); // empty state
        a.on_event(UiEvent::Model(ModelEvent {
            ts: ts(),
            model: "a".into(),
            kind: ModelEventKind::Loading,
            duration_ms: None,
            detail: String::new(),
        }));
        assert!(matches!(a.models[0].status, Status::Loading(_)));
        draw_all(&mut a);
        a.on_event(UiEvent::Model(ModelEvent {
            ts: ts(),
            model: "a".into(),
            kind: ModelEventKind::Loaded,
            duration_ms: Some(1500),
            detail: "+2 GiB VRAM".into(),
        }));
        assert!(matches!(a.models[0].status, Status::Loaded { .. }));

        for i in 0..3u64 {
            a.on_event(UiEvent::Request(RequestRecord {
                id: 0,
                phase: RequestPhase::Succeeded,
                ts: ts(),
                model: "a".into(),
                status: 200,
                latency_ms: 1000 + i,
                prompt_tokens: 10,
                completion_tokens: 40,
                generation_ms: 2000,
                error: None,
            }));
        }
        a.on_event(UiEvent::Request(RequestRecord {
            id: 0,
            phase: RequestPhase::Failed,
            ts: ts(),
            model: "g".into(),
            status: 429,
            latency_ms: 50,
            prompt_tokens: 0,
            completion_tokens: 0,
            generation_ms: 0,
            error: Some("rate limited".into()),
        }));
        assert_eq!((a.total_requests, a.total_errors), (4, 1));
        assert_eq!(a.models[0].requests, 3);
        assert_eq!(a.models[1].errors, 1);
        assert_eq!(a.tps_hist.back(), Some(&20_000));
        assert_eq!(
            a.latency_hist.len(),
            3,
            "failed requests excluded from latency"
        );

        a.on_event(UiEvent::Model(ModelEvent {
            ts: ts(),
            model: "a".into(),
            kind: ModelEventKind::Evicted,
            duration_ms: None,
            detail: String::new(),
        }));
        assert!(matches!(a.models[0].status, Status::Idle));
        a.on_event(UiEvent::Model(ModelEvent {
            ts: ts(),
            model: "g".into(),
            kind: ModelEventKind::Failed,
            duration_ms: None,
            detail: "boom".into(),
        }));
        draw_all(&mut a);
    }

    #[test]
    fn logs_view_handles_scroll_filter_and_overflow() {
        let mut a = app();
        for i in 0..(MAX_LOGS + 50) {
            a.on_event(UiEvent::Log(LogLine {
                ts: ts(),
                level: if i % 10 == 0 {
                    Level::ERROR
                } else {
                    Level::INFO
                },
                target: "t".into(),
                message: format!("line {i}"),
            }));
        }
        assert_eq!(a.logs.len(), MAX_LOGS);
        a.tab = Tab::Logs;
        a.on_key(KeyCode::Char('w'), KeyModifiers::NONE);
        for code in [
            KeyCode::Up,
            KeyCode::PageUp,
            KeyCode::PageDown,
            KeyCode::Down,
            KeyCode::Char('f'),
            KeyCode::Char('?'),
        ] {
            a.on_key(code, KeyModifiers::NONE);
            draw_all(&mut a);
        }
        a.on_key(KeyCode::Char('c'), KeyModifiers::NONE);
        draw_all(&mut a);
        a.on_key(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(a.quit);
    }
}
