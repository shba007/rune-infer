//! First-run setup wizard.
//!
//! Configure at least one model before saving. Engine downloads may write files
//! before the final configuration review.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc::UnboundedReceiver;
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use super::theme::{DIM, ERR, OK, WARN, centered, human_bytes, spinner, terminal_compatibility};
use crate::config::EngineKind;
use crate::home::Workspace;
use crate::setup::{
    self, CloudChoice, CloudModel, Exposure, LocalCandidate, LocalPreparationEvent, PROVIDERS,
    PreparedLocal, Prerequisites, SetupPlan,
};
use crate::telemetry::UiEvent;
use ratatui::style::Color;

const ACCENT: Color = Color::White;
fn panel<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::default()
        .title(title.into())
        .title_style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WizardOutcome {
    Completed,
    Cancelled,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Sources,
    Network,
    Cloud,
    Local,
    Engine,
    Review,
}

const STEPS: [(&str, Step); 6] = [
    ("Network", Step::Network),
    ("Sources", Step::Sources),
    ("Local models", Step::Local),
    ("Engine", Step::Engine),
    ("Cloud", Step::Cloud),
    ("Review", Step::Review),
];

enum Flow {
    Stay,
    Cancel,
    Finish,
}

enum Validation {
    Idle,
    Running(Receiver<Result<Vec<CloudModel>, String>>),
    Failed(String),
}

#[derive(PartialEq, Eq)]
enum CloudPhase {
    Pick,
    Details,
    Models,
}

#[derive(PartialEq, Eq)]
enum LocalPhase {
    Path,
    Pick,
}

enum EnginePhase {
    Pick,
    Installing(Receiver<LocalPreparationEvent>),
    Done(Result<PreparedLocal, String>),
}

struct Wizard<'a> {
    ws: &'a Workspace,
    step: Step,
    tick: u64,

    prereq: Option<Prerequisites>,
    prereq_rx: Receiver<Prerequisites>,

    sources: [bool; 2],
    source_cursor: usize,
    source_error: Option<String>,

    // network
    exposure: Exposure,
    port: Input,
    want_key: bool,
    net_focus: usize,
    server_key: Option<String>,
    net_error: Option<String>,

    // cloud
    cloud_phase: CloudPhase,
    cloud_checked: [bool; PROVIDERS.len()],
    cloud_cursor: usize,
    cloud_queue: Vec<usize>,
    cloud_pos: usize,
    key_input: Input,
    cloud_models: Vec<CloudModel>,
    cloud_model_picks: Vec<bool>,
    cloud_model_cursor: usize,
    cloud_filter: Input,
    cloud_model_error: Option<String>,
    validation: Validation,
    cloud_done: Vec<CloudChoice>,

    // local
    local_phase: LocalPhase,
    path_input: Input,
    candidates: Vec<LocalCandidate>,
    skipped: Vec<setup::Skipped>,
    picks: Vec<bool>,
    local_cursor: usize,
    chosen: Vec<LocalCandidate>,
    local_error: Option<String>,

    // engine installation and real inference checks
    engine: EnginePhase,
    engine_choices: Vec<Vec<EngineKind>>,
    engine_available: Vec<EngineKind>,
    engine_selected: Vec<bool>,
    engine_cursor: usize,
    engine_option: usize,
    engine_status: String,
    engine_download: Option<crate::inference::engines::download::DownloadProgress>,

    // review
    review_error: Option<String>,
}

pub fn run_wizard(
    terminal: &mut DefaultTerminal,
    ws: &Workspace,
    rx: &mut UnboundedReceiver<UiEvent>,
) -> Result<WizardOutcome> {
    let (ptx, prx) = channel();
    let home = ws.root.clone();
    std::thread::spawn(move || {
        let _ = ptx.send(setup::detect_prerequisites(&home));
    });

    let mut w = Wizard::new(ws, prx);

    loop {
        while rx.try_recv().is_ok() {}
        w.poll_workers();
        w.tick += 1;
        terminal.draw(|f| w.draw(f))?;

        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                if k.modifiers.contains(KeyModifiers::CONTROL) && k.code == KeyCode::Char('c') {
                    return Ok(WizardOutcome::Cancelled);
                }
                match w.on_key(k) {
                    Flow::Stay => {}
                    Flow::Cancel => return Ok(WizardOutcome::Cancelled),
                    Flow::Finish => return Ok(WizardOutcome::Completed),
                }
            }
        }
    }
}

fn strip_quotes(s: &str) -> &str {
    s.trim().trim_matches(|c| c == '"' || c == '\'')
}

fn expand_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/").or_else(|| s.strip_prefix("~\\")) {
        let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        if let Some(h) = std::env::var_os(var) {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(s)
}

impl<'a> Wizard<'a> {
    fn new(ws: &'a Workspace, prereq_rx: Receiver<Prerequisites>) -> Self {
        Wizard {
            ws,
            step: Step::Network,
            sources: [false; 2],
            source_cursor: 0,
            source_error: None,
            tick: 0,
            prereq: None,
            prereq_rx,
            exposure: Exposure::Local,
            port: Input::new("8080".into()),
            want_key: true,
            net_focus: 0,
            server_key: None,
            net_error: None,
            cloud_phase: CloudPhase::Pick,
            cloud_checked: [false; PROVIDERS.len()],
            cloud_cursor: 0,
            cloud_queue: Vec::new(),
            cloud_pos: 0,
            key_input: Input::default(),
            cloud_models: Vec::new(),
            cloud_model_picks: Vec::new(),
            cloud_model_cursor: 0,
            cloud_filter: Input::default(),
            cloud_model_error: None,
            validation: Validation::Idle,
            cloud_done: Vec::new(),
            local_phase: LocalPhase::Path,
            path_input: Input::default(),
            candidates: Vec::new(),
            skipped: Vec::new(),
            picks: Vec::new(),
            local_cursor: 0,
            chosen: Vec::new(),
            local_error: None,
            engine: EnginePhase::Pick,
            engine_choices: Vec::new(),
            engine_available: Vec::new(),
            engine_selected: Vec::new(),
            engine_cursor: 0,
            engine_option: 0,
            engine_status: String::new(),
            engine_download: None,
            review_error: None,
        }
    }

    fn visible_steps(&self) -> Vec<(&'static str, Step)> {
        STEPS
            .iter()
            .copied()
            .filter(|(_, step)| match step {
                Step::Local => self.sources[0],
                Step::Engine => self.sources[0] && self.engine_needed(),
                Step::Cloud => self.sources[1],
                _ => true,
            })
            .collect()
    }

    fn gpu_available(&self) -> bool {
        self.prereq
            .as_ref()
            .is_some_and(|p| p.gpu.is_some() || (p.os == "macos" && p.arch == "aarch64"))
    }

    fn poll_workers(&mut self) {
        if self.prereq.is_none() {
            if let Ok(p) = self.prereq_rx.try_recv() {
                self.prereq = Some(p);
            }
        }
        let model_result = if let Validation::Running(rx) = &self.validation {
            match rx.try_recv() {
                Ok(result) => Some(result),
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    Some(Err("Model discovery stopped unexpectedly. Retry.".into()))
                }
            }
        } else {
            None
        };
        if let Some(result) = model_result {
            match result {
                Ok(models) => {
                    let provider = PROVIDERS[self.cloud_queue[self.cloud_pos]].provider;
                    self.cloud_model_picks = models
                        .iter()
                        .map(|model| {
                            model.selection_error().is_none()
                                && self.cloud_done.iter().any(|c| {
                                    c.preset.provider == provider && c.model.id == model.id
                                })
                        })
                        .collect();
                    self.cloud_models = models;
                    self.cloud_model_cursor = 0;
                    self.cloud_filter = Input::default();
                    self.cloud_model_error = None;
                    self.cloud_phase = CloudPhase::Models;
                    self.validation = Validation::Idle;
                }
                Err(error) => self.validation = Validation::Failed(error),
            }
        }
        let mut finished = None;
        if let EnginePhase::Installing(rx) = &self.engine {
            loop {
                match rx.try_recv() {
                    Ok(LocalPreparationEvent::Progress(status)) => {
                        self.engine_status = status;
                        self.engine_download = None;
                    }
                    Ok(LocalPreparationEvent::Download(progress)) => {
                        self.engine_download = Some(progress)
                    }
                    Ok(LocalPreparationEvent::Finished(result)) => {
                        finished = Some(result);
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        finished = Some(Err(
                            "Engine check worker stopped unexpectedly. Retry or go back.".into(),
                        ));
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished {
            self.engine = EnginePhase::Done(result);
        }
    }

    // ------------------------------------------------------------------ keys

    fn go(&mut self, step: Step) {
        self.step = step;
    }

    fn engine_needed(&self) -> bool {
        !self.chosen.is_empty()
    }

    fn next_after_local(&mut self) {
        if self.engine_needed() {
            self.engine_choices = self.chosen.iter().map(|m| vec![m.engine()]).collect();
            self.engine_available.clear();
            for model in &self.chosen {
                for &engine in &model.engines {
                    if engine != EngineKind::Encoder && !self.engine_available.contains(&engine) {
                        self.engine_available.push(engine);
                    }
                }
            }
            self.engine_selected = self
                .engine_available
                .iter()
                .map(|engine| {
                    self.engine_choices
                        .iter()
                        .any(|choices| choices.first() == Some(engine))
                })
                .collect();
            self.refresh_engine_choices();
            self.engine_status.clear();
            self.engine_cursor = 0;
            self.engine_option = 0;
            self.engine = EnginePhase::Pick;
            self.go(Step::Engine);
        } else {
            self.next_after_engine();
        }
    }

    fn next_after_engine(&mut self) {
        self.go(if self.sources[1] {
            Step::Cloud
        } else {
            Step::Review
        });
    }

    fn on_key(&mut self, k: KeyEvent) -> Flow {
        match self.step {
            Step::Sources => match k.code {
                KeyCode::Esc => self.go(Step::Network),
                KeyCode::Up | KeyCode::Down | KeyCode::Tab => {
                    self.source_cursor = 1 - self.source_cursor
                }
                KeyCode::Char(' ') => {
                    self.sources[self.source_cursor] = !self.sources[self.source_cursor];
                    self.source_error = None;
                }
                KeyCode::Enter => {
                    if !self.sources.iter().any(|selected| *selected) {
                        self.source_error = Some("Select Local, Cloud, or both.".into());
                    } else {
                        if !self.sources[0] {
                            self.chosen.clear();
                        }
                        if !self.sources[1] {
                            self.cloud_done.clear();
                        }
                        self.cloud_phase = CloudPhase::Pick;
                        self.local_phase = LocalPhase::Path;
                        self.source_error = None;
                        self.review_error = None;
                        self.go(if self.sources[0] {
                            Step::Local
                        } else {
                            Step::Cloud
                        });
                    }
                }
                _ => {}
            },
            Step::Network => return self.key_network(k),
            Step::Cloud => self.key_cloud(k),
            Step::Local => self.key_local(k),
            Step::Engine => self.key_engine(k),
            Step::Review => return self.key_review(k),
        }
        Flow::Stay
    }

    fn key_network(&mut self, k: KeyEvent) -> Flow {
        match k.code {
            KeyCode::Esc => return Flow::Cancel,
            KeyCode::Up => self.net_focus = self.net_focus.saturating_sub(1),
            KeyCode::Down | KeyCode::Tab => self.net_focus = (self.net_focus + 1).min(2),
            KeyCode::Enter => {
                let Ok(port) = self.port.value().trim().parse::<u16>() else {
                    self.net_error = Some("Port must be a number between 1 and 65535.".into());
                    self.net_focus = 1;
                    return Flow::Stay;
                };
                if port == 0 {
                    self.net_error = Some("Port must be between 1 and 65535.".into());
                    return Flow::Stay;
                }
                if self.exposure == Exposure::Network && !self.want_key {
                    self.net_error = Some(
                        "Network access without an API key would expose your models to everyone on the network. Enable an API key or choose 'this computer only'."
                            .into(),
                    );
                    self.net_focus = 2;
                    return Flow::Stay;
                }
                self.net_error = None;
                if self.want_key && self.server_key.is_none() {
                    self.server_key = Some(setup::generate_api_key());
                }
                if !self.want_key {
                    self.server_key = None;
                }
                self.go(Step::Sources);
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') if self.net_focus != 1 => {
                if self.net_focus == 0 {
                    self.exposure = match self.exposure {
                        Exposure::Local => Exposure::Network,
                        Exposure::Network => Exposure::Local,
                    };
                    if self.exposure == Exposure::Network {
                        self.want_key = true;
                    }
                } else {
                    self.want_key = !self.want_key;
                }
                self.net_error = None;
            }
            _ if self.net_focus == 1 => {
                self.port.handle_event(&Event::Key(k));
            }
            _ => {}
        }
        Flow::Stay
    }

    fn key_cloud(&mut self, k: KeyEvent) {
        match self.cloud_phase {
            CloudPhase::Pick => match k.code {
                KeyCode::Esc => self.go(Step::Sources),
                KeyCode::Up => self.cloud_cursor = self.cloud_cursor.saturating_sub(1),
                KeyCode::Down => {
                    self.cloud_cursor = (self.cloud_cursor + 1).min(PROVIDERS.len() - 1)
                }
                KeyCode::Char(' ') => {
                    self.cloud_checked[self.cloud_cursor] = !self.cloud_checked[self.cloud_cursor]
                }
                KeyCode::Enter => {
                    self.cloud_queue = (0..PROVIDERS.len())
                        .filter(|i| self.cloud_checked[*i])
                        .collect();
                    self.cloud_done.retain(|choice| {
                        self.cloud_queue
                            .iter()
                            .any(|i| PROVIDERS[*i].provider == choice.preset.provider)
                    });
                    self.cloud_pos = 0;
                    if self.cloud_queue.is_empty() {
                        self.review_error = Some("Select at least one cloud provider.".into());
                    } else {
                        self.review_error = None;
                        self.begin_provider();
                        self.cloud_phase = CloudPhase::Details;
                    }
                }
                _ => {}
            },
            CloudPhase::Details => self.key_cloud_details(k),
            CloudPhase::Models => self.key_cloud_models(k),
        }
    }

    fn begin_provider(&mut self) {
        let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
        self.key_input = Input::new(
            self.cloud_done
                .iter()
                .find(|c| c.preset.provider == preset.provider)
                .map(|c| c.api_key.clone())
                .unwrap_or_default(),
        );
        self.cloud_models.clear();
        self.cloud_model_picks.clear();
        self.cloud_model_error = None;
        self.cloud_phase = CloudPhase::Details;
        self.validation = Validation::Idle;
    }

    fn fetch_cloud_models(&mut self) {
        let key = self.key_input.value().trim().to_string();
        if key.is_empty() {
            self.validation = Validation::Failed("Enter an API key before fetching models.".into());
            return;
        }
        let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(setup::list_provider_models(&preset, &key));
        });
        self.validation = Validation::Running(rx);
    }

    fn advance_provider(&mut self) {
        self.cloud_pos += 1;
        if self.cloud_pos >= self.cloud_queue.len() {
            self.cloud_phase = CloudPhase::Pick;
            self.go(Step::Review);
        } else {
            self.begin_provider();
        }
    }

    fn key_cloud_details(&mut self, k: KeyEvent) {
        match &self.validation {
            Validation::Running(_) => {}
            Validation::Failed(_) => match k.code {
                KeyCode::Enter => self.fetch_cloud_models(),
                KeyCode::Char('r') => self.validation = Validation::Idle,
                KeyCode::Esc => {
                    self.cloud_phase = CloudPhase::Pick;
                    self.validation = Validation::Idle;
                }
                _ => {}
            },
            Validation::Idle => match k.code {
                KeyCode::Esc => self.cloud_phase = CloudPhase::Pick,
                KeyCode::Enter => self.fetch_cloud_models(),
                _ => {
                    self.key_input.handle_event(&Event::Key(k));
                }
            },
        }
    }

    fn filtered_cloud_models(&self) -> Vec<usize> {
        let query = self.cloud_filter.value().to_lowercase();
        self.cloud_models
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.id.to_lowercase().contains(&query) || m.name.to_lowercase().contains(&query)
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn key_cloud_models(&mut self, k: KeyEvent) {
        let visible = self.filtered_cloud_models();
        match k.code {
            KeyCode::Up => self.cloud_model_cursor = self.cloud_model_cursor.saturating_sub(1),
            KeyCode::Down => {
                self.cloud_model_cursor =
                    (self.cloud_model_cursor + 1).min(visible.len().saturating_sub(1))
            }
            KeyCode::Esc => {
                self.cloud_phase = CloudPhase::Details;
                self.validation = Validation::Idle;
            }
            KeyCode::Char(' ') => {
                if let Some(i) = visible.get(self.cloud_model_cursor).copied() {
                    if let Some(error) = self.cloud_models[i].selection_error() {
                        self.cloud_model_error = Some(error.into());
                    } else {
                        self.cloud_model_picks[i] = !self.cloud_model_picks[i];
                        self.cloud_model_error = None;
                    }
                }
            }
            KeyCode::Enter => {
                let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
                let mut choices = Vec::new();
                for (model, picked) in self.cloud_models.iter().zip(&self.cloud_model_picks) {
                    if !picked {
                        continue;
                    }
                    if let Some(error) = model.selection_error() {
                        self.cloud_model_error = Some(error.into());
                        return;
                    }
                    let Some(metadata) = model.metadata.clone() else {
                        self.cloud_model_error = Some("Add registry metadata first.".into());
                        return;
                    };
                    choices.push(CloudChoice {
                        preset,
                        api_key: self.key_input.value().trim().to_string(),
                        model: metadata,
                    });
                }
                if choices.is_empty() {
                    self.cloud_model_error =
                        Some("Select at least one registered chat model for this provider.".into());
                    return;
                }
                self.cloud_done
                    .retain(|c| c.preset.provider != preset.provider);
                self.cloud_done.extend(choices);
                self.advance_provider();
            }
            _ => {
                self.cloud_filter.handle_event(&Event::Key(k));
                self.cloud_model_cursor = 0;
                self.cloud_model_error = None;
            }
        }
    }

    fn key_local(&mut self, k: KeyEvent) {
        match self.local_phase {
            LocalPhase::Path => match k.code {
                KeyCode::Esc => self.go(Step::Sources),
                KeyCode::Enter => {
                    let raw = strip_quotes(self.path_input.value()).to_string();
                    if raw.is_empty() {
                        if self.chosen.is_empty() {
                            self.local_error =
                                Some("Select at least one local model before continuing.".into());
                            return;
                        }
                        self.local_error = None;
                        self.next_after_local();
                        return;
                    }
                    match setup::discover_models(&expand_tilde(&raw)) {
                        Ok(d) if d.models.is_empty() => {
                            let why = d
                                .skipped
                                .first()
                                .map(|s| format!(" First skipped file: {}", s.reason))
                                .unwrap_or_default();
                            self.local_error = Some(format!(
                                "No models rune-infer can set up were found there.{why}"
                            ))
                        }
                        Ok(d) => {
                            // Models that still need companion files start unticked.
                            self.picks = d
                                .models
                                .iter()
                                .map(|c| {
                                    c.is_ready()
                                        && c.engines.iter().any(|e| *e != EngineKind::Encoder)
                                })
                                .collect();
                            self.candidates = d.models;
                            self.skipped = d.skipped;
                            self.local_cursor = 0;
                            self.local_error = None;
                            self.local_phase = LocalPhase::Pick;
                        }
                        Err(e) => self.local_error = Some(e.to_string()),
                    }
                }
                _ => {
                    self.path_input.handle_event(&Event::Key(k));
                }
            },
            LocalPhase::Pick => {
                match k.code {
                    KeyCode::Esc => self.local_phase = LocalPhase::Path,
                    KeyCode::Up => self.local_cursor = self.local_cursor.saturating_sub(1),
                    KeyCode::Down => {
                        self.local_cursor = (self.local_cursor + 1).min(self.candidates.len() - 1)
                    }
                    KeyCode::Char(' ') => {
                        self.picks[self.local_cursor] = !self.picks[self.local_cursor]
                    }
                    KeyCode::Char('a') => {
                        let all = self.picks.iter().all(|p| *p);
                        self.picks.iter_mut().for_each(|p| *p = !all);
                    }
                    KeyCode::Enter => {
                        for (c, picked) in self.candidates.iter().zip(&self.picks) {
                            if *picked && !self.chosen.iter().any(|x| x.path == c.path) {
                                let mut c = c.clone();
                                c.engines.retain(|engine| *engine != EngineKind::Encoder);
                                if c.engines.is_empty() {
                                    self.local_error = Some("This model has no real inference engine available in setup.".into());
                                    return;
                                }
                                self.chosen.push(c);
                            }
                        }
                        self.path_input = Input::default();
                        self.local_phase = LocalPhase::Path;
                    }
                    _ => {}
                }
            }
        }
    }

    fn refresh_engine_choices(&mut self) {
        for (model, choices) in self.chosen.iter().zip(&mut self.engine_choices) {
            choices.retain(|engine| {
                model.engines.contains(engine)
                    && self
                        .engine_available
                        .iter()
                        .zip(&self.engine_selected)
                        .any(|(available, selected)| *selected && available == engine)
            });
            for &engine in &model.engines {
                if self
                    .engine_available
                    .iter()
                    .zip(&self.engine_selected)
                    .any(|(available, selected)| *selected && *available == engine)
                    && !choices.contains(&engine)
                {
                    choices.push(engine);
                }
            }
        }
    }

    fn engine_models(&self) -> Vec<usize> {
        self.chosen
            .iter()
            .enumerate()
            .filter_map(|(index, model)| {
                self.engine_available
                    .get(self.engine_cursor)
                    .is_some_and(|engine| model.engines.contains(engine))
                    .then_some(index)
            })
            .collect()
    }

    fn enabled_local_models(&self) -> Result<Vec<LocalCandidate>, String> {
        if self.chosen.len() != self.engine_choices.len() {
            return Err("Choose engines before continuing.".into());
        }
        let (models, choices): (Vec<_>, Vec<_>) = self
            .chosen
            .iter()
            .zip(&self.engine_choices)
            .filter(|(_, engines)| !engines.is_empty())
            .map(|(model, engines)| (model.clone(), engines.clone()))
            .unzip();
        setup::select_local_engines(&models, &choices)
    }

    fn begin_engine_install(&mut self) {
        if self.prereq.is_none() {
            self.engine = EnginePhase::Done(Err(
                "System check is still running. Retry once it completes.".into(),
            ));
            return;
        }
        let models = match self.enabled_local_models() {
            Ok(models) => models,
            Err(error) => {
                self.engine = EnginePhase::Done(Err(error));
                return;
            }
        };
        if models.is_empty() && !self.sources[1] {
            self.engine = EnginePhase::Done(Err(
                "Select at least one engine that enables a local model, or enable cloud models."
                    .into(),
            ));
            return;
        }
        let (tx, rx) = channel();
        let gpu = self.gpu_available();
        std::thread::spawn(move || {
            let downloads = tx.clone();
            let result = crate::inference::engines::download::with_progress(
                move |progress| {
                    let _ = downloads.send(LocalPreparationEvent::Download(progress));
                },
                || {
                    setup::prepare_local_models(&models, gpu, |status| {
                        let _ = tx.send(LocalPreparationEvent::Progress(status));
                    })
                },
            );
            let _ = tx.send(LocalPreparationEvent::Finished(result));
        });
        self.engine_status = "Checking local model files".into();
        self.engine_download = None;
        self.engine = EnginePhase::Installing(rx);
    }

    fn key_engine(&mut self, k: KeyEvent) {
        match &self.engine {
            EnginePhase::Pick => match k.code {
                KeyCode::Left | KeyCode::Right => {
                    if k.code == KeyCode::Left {
                        self.engine_cursor = self.engine_cursor.saturating_sub(1);
                    } else {
                        self.engine_cursor = (self.engine_cursor + 1)
                            .min(self.engine_available.len().saturating_sub(1));
                    }
                    self.engine_option = 0;
                }
                KeyCode::Up => self.engine_option = self.engine_option.saturating_sub(1),
                KeyCode::Down => {
                    self.engine_option =
                        (self.engine_option + 1).min(self.engine_models().len().saturating_sub(1))
                }
                KeyCode::Char(' ') => {
                    if let Some(selected) = self.engine_selected.get_mut(self.engine_cursor) {
                        *selected = !*selected;
                        self.refresh_engine_choices();
                    }
                }
                KeyCode::Char('d') => {
                    if let Some(&model_index) = self.engine_models().get(self.engine_option) {
                        self.engine_selected[self.engine_cursor] = true;
                        self.refresh_engine_choices();
                        let engine = self.engine_available[self.engine_cursor];
                        let choices = &mut self.engine_choices[model_index];
                        choices.retain(|candidate| *candidate != engine);
                        choices.insert(0, engine);
                    }
                }
                KeyCode::Enter => self.begin_engine_install(),
                KeyCode::Esc => self.go(Step::Local),
                _ => {}
            },
            EnginePhase::Installing(_) => {}
            EnginePhase::Done(Ok(_)) => {
                if k.code == KeyCode::Enter {
                    self.next_after_engine();
                } else if k.code == KeyCode::Esc {
                    self.engine = EnginePhase::Pick;
                }
            }
            EnginePhase::Done(Err(_)) => {
                if k.code == KeyCode::Enter {
                    self.begin_engine_install();
                } else if k.code == KeyCode::Esc {
                    self.engine = EnginePhase::Pick;
                }
            }
        }
    }

    fn plan(&self) -> Result<SetupPlan, String> {
        Ok(SetupPlan {
            exposure: self.exposure,
            port: self.port.value().trim().parse().unwrap_or(8080),
            server_api_key: self.server_key.clone(),
            cloud: self.cloud_done.clone(),
            local: self.enabled_local_models()?,
            gpu_available: self.gpu_available(),
        })
    }

    fn key_review(&mut self, k: KeyEvent) -> Flow {
        match k.code {
            KeyCode::Esc => {
                self.review_error = None;
                if self.sources[1] {
                    self.go(Step::Cloud);
                } else if self.engine_needed() {
                    self.go(Step::Engine)
                } else {
                    self.go(Step::Local)
                }
            }
            KeyCode::Enter => {
                let plan = match self.plan() {
                    Ok(plan) => plan,
                    Err(error) => {
                        self.review_error = Some(error);
                        return Flow::Stay;
                    }
                };
                if plan.local.is_empty() && plan.cloud.is_empty() {
                    self.review_error =
                        Some("Configure at least one enabled model before saving.".into());
                    return Flow::Stay;
                }
                if !plan.local.is_empty() {
                    match &self.engine {
                        EnginePhase::Done(Ok(prepared)) => {
                            if let Err(error) = prepared.validate(&plan.local, plan.gpu_available) {
                                self.review_error = Some(error);
                                return Flow::Stay;
                            }
                        }
                        _ => {
                            self.review_error = Some(
                                "Every enabled local model must pass a real inference check before saving."
                                    .into(),
                            );
                            return Flow::Stay;
                        }
                    }
                }
                let prepared = match &self.engine {
                    EnginePhase::Done(Ok(proof)) if !plan.local.is_empty() => Some(proof),
                    _ => None,
                };
                match setup::write_verified_setup(
                    &plan,
                    prepared,
                    &self.ws.server_path,
                    &self.ws.catalog_path,
                    &self.ws.env_file(),
                ) {
                    Ok(_) => return Flow::Finish,
                    Err(e) => self.review_error = Some(format!("{e:#}")),
                }
            }
            _ => {}
        }
        Flow::Stay
    }

    // --------------------------------------------------------------- drawing

    fn draw(&self, f: &mut Frame) {
        let area = centered(f.area(), 100, f.area().height.min(32));
        let [title, steps_area, body, footer] = Layout::vertical([
            Constraint::Length(if area.height >= 32 { 14 } else { 1 }),
            Constraint::Length(1),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(area);

        f.render_widget(
            Paragraph::new(if area.height >= 32 {
                format!("{}\n RUNE INFER · SETUP", crate::CLI_LOGO)
            } else {
                "RUNE INFER · SETUP".into()
            }),
            title,
        );

        let steps = self.visible_steps();
        let step_index = steps
            .iter()
            .position(|(_, step)| *step == self.step)
            .unwrap_or(0);
        let mut spans = Vec::new();
        let compact = area.width < 84;
        if compact {
            spans.push(Span::styled(
                format!(
                    "Step {} of {} · {}",
                    step_index + 1,
                    steps.len(),
                    steps[step_index].0
                ),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        }
        for (i, (name, _)) in steps.iter().enumerate().filter(|_| !compact) {
            let cur = i == step_index;
            let done = i < step_index;
            let style = if cur {
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
            } else if done {
                Style::new().fg(OK)
            } else {
                Style::new().fg(DIM)
            };
            let mark = if done {
                "✓"
            } else if cur {
                "●"
            } else {
                "○"
            };
            spans.push(Span::styled(format!("{mark} {name}"), style));
            if i + 1 < steps.len() {
                spans.push(Span::styled("  ─  ", Style::new().fg(DIM)));
            }
        }
        f.render_widget(Paragraph::new(Line::from(spans)), steps_area);

        match self.step {
            Step::Sources => self.draw_sources(f, body),
            Step::Network => self.draw_network(f, body),
            Step::Cloud => self.draw_cloud(f, body),
            Step::Local => self.draw_local(f, body),
            Step::Engine => self.draw_engine(f, body),
            Step::Review => self.draw_review(f, body),
        }

        f.render_widget(Paragraph::new(Line::from(self.hints())), footer);
        terminal_compatibility(f);
    }

    fn hints(&self) -> Vec<Span<'static>> {
        let mut h: Vec<Span<'static>> = Vec::new();
        let mut add = |k: &'static str, w: &'static str| {
            h.extend([
                Span::styled(format!(" {k} "), Style::new().fg(ACCENT)),
                Span::styled(format!("{w}  "), Style::new().fg(DIM)),
            ])
        };
        match self.step {
            Step::Sources => {
                add("Space", "select");
                add("Enter", "next");
                add("Esc", "back");
            }
            Step::Network => {
                add("↑↓", "move");
                add("Space", "toggle");
                add("Enter", "next");
                add("Esc", "quit");
            }
            Step::Cloud => match self.cloud_phase {
                CloudPhase::Pick => {
                    add("Space", "select");
                    add("Enter", "continue");
                    add("Esc", "back");
                }
                CloudPhase::Details => match &self.validation {
                    Validation::Running(_) => {}
                    Validation::Failed(_) => {
                        add("Enter", "retry");
                        add("r", "edit key");
                        add("Esc", "providers");
                    }
                    Validation::Idle => {
                        add("Enter", "fetch models");
                        add("Esc", "providers");
                    }
                },
                CloudPhase::Models => {
                    add("Type", "filter");
                    add("Space", "select");
                    add("Enter", "continue");
                    add("Esc", "key");
                }
            },
            Step::Local => match self.local_phase {
                LocalPhase::Path => {
                    add("Enter", "scan path (empty = continue)");
                    add("Esc", "back");
                }
                LocalPhase::Pick => {
                    add("Space", "toggle");
                    add("a", "all/none");
                    add("Enter", "add selected");
                    add("Esc", "back");
                }
            },
            Step::Engine => match &self.engine {
                EnginePhase::Pick => {
                    add("Left/Right", "engine");
                    add("Up/Down", "model");
                    add("Space", "select engine");
                    add("d", "model default");
                    add("Enter", "install & check");
                    add("Esc", "back");
                }
                EnginePhase::Installing(_) => {}
                EnginePhase::Done(Ok(_)) => {
                    add("Enter", "continue");
                    add("Esc", "engine choices");
                }
                EnginePhase::Done(Err(_)) => {
                    add("Enter", "retry");
                    add("Esc", "engine choices");
                }
            },
            Step::Review => {
                add("Enter", "save & start");
                add("Esc", "back");
            }
        }
        add("Ctrl-C", "cancel");
        h
    }

    fn draw_sources(&self, f: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::from("Select where your models run. Choose at least one."),
            Line::from(""),
        ];
        for (i, label) in ["Local models", "Cloud models"].iter().enumerate() {
            lines.push(Line::styled(
                format!(
                    "{} {} {label}",
                    if i == self.source_cursor { ">" } else { " " },
                    if self.sources[i] { "●" } else { "○" }
                ),
                Style::new().fg(if i == self.source_cursor { ACCENT } else { DIM }),
            ));
        }
        if let Some(error) = &self.source_error {
            lines.push(Line::styled(error.as_str(), Style::new().fg(ERR)));
        }
        f.render_widget(Paragraph::new(lines).block(panel(" Model sources ")), area);
    }

    fn draw_network(&self, f: &mut Frame, area: Rect) {
        let block = panel(" Network & security ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [exp, port, key, msg] = Layout::vertical([
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Min(2),
        ])
        .areas(inner);

        let radio = |sel: bool, label: &str, note: &str| {
            Line::from(vec![
                Span::styled(
                    if sel { "  ◉ " } else { "  ○ " },
                    Style::new().fg(if sel { ACCENT } else { DIM }),
                ),
                Span::raw(label.to_string()),
                Span::styled(format!("  {note}"), Style::new().fg(DIM)),
            ])
        };
        let focus_style = |i: usize| {
            if self.net_focus == i {
                Style::new().fg(ACCENT)
            } else {
                Style::new().fg(DIM)
            }
        };

        f.render_widget(
            Paragraph::new(vec![
                radio(
                    self.exposure == Exposure::Local,
                    "This computer only (127.0.0.1)",
                    "recommended",
                ),
                radio(
                    self.exposure == Exposure::Network,
                    "Local network (0.0.0.0)",
                    "other devices can connect",
                ),
            ])
            .block(
                Block::default()
                    .border_style(focus_style(0))
                    .title(" Who can connect? "),
            ),
            exp,
        );
        render_input(f, port, &self.port, false, self.net_focus == 1, " Port ");
        f.render_widget(
            Paragraph::new(vec![
                radio(
                    self.want_key,
                    "Require an API key",
                    "generated for you, saved to .env",
                ),
                radio(!self.want_key, "No API key", "only safe on localhost"),
            ])
            .block(
                Block::default()
                    .border_style(focus_style(2))
                    .title(" API key "),
            ),
            key,
        );
        if let Some(e) = &self.net_error {
            f.render_widget(
                Paragraph::new(Span::styled(format!("  ✗ {e}"), Style::new().fg(ERR)))
                    .wrap(Wrap { trim: true }),
                msg,
            );
        }
    }

    fn draw_cloud(&self, f: &mut Frame, area: Rect) {
        match self.cloud_phase {
            CloudPhase::Pick => {
                let mut lines = vec![
                    Line::from(""),
                    Line::from("  Route requests to hosted models through the same endpoint."),
                    Line::styled(
                        "  Keys are stored in .env — never in the config file.",
                        Style::new().fg(DIM),
                    ),
                    Line::from(""),
                ];
                for (i, p) in PROVIDERS.iter().enumerate() {
                    let cur = i == self.cloud_cursor;
                    lines.push(Line::from(vec![
                        Span::styled(if cur { "  ❯ " } else { "    " }, Style::new().fg(ACCENT)),
                        Span::styled(
                            if self.cloud_checked[i] {
                                "● "
                            } else {
                                "○ "
                            },
                            Style::new().fg(if self.cloud_checked[i] { OK } else { DIM }),
                        ),
                        Span::styled(
                            p.label,
                            if cur {
                                Style::new().add_modifier(Modifier::BOLD)
                            } else {
                                Style::new()
                            },
                        ),
                        Span::styled(format!("   {}", p.env_var), Style::new().fg(DIM)),
                    ]));
                }
                if let Some(error) = &self.review_error {
                    lines.push(Line::styled(error.as_str(), Style::new().fg(ERR)));
                }
                if !self.cloud_done.is_empty() {
                    lines.push(Line::from(""));
                    lines.push(Line::styled(
                        format!(
                            "  Already configured: {}",
                            self.cloud_done
                                .iter()
                                .map(|c| c.preset.label)
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                        Style::new().fg(OK),
                    ));
                }
                f.render_widget(
                    Paragraph::new(lines).block(panel(" Cloud providers ")),
                    area,
                );
            }
            CloudPhase::Details => {
                let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
                let block = panel(format!(
                    " {} · {}/{} ",
                    preset.label,
                    self.cloud_pos + 1,
                    self.cloud_queue.len()
                ));
                let inner = block.inner(area);
                f.render_widget(block, area);
                let [key, status] =
                    Layout::vertical([Constraint::Length(3), Constraint::Min(3)]).areas(inner);
                render_input(
                    f,
                    key,
                    &self.key_input,
                    true,
                    matches!(self.validation, Validation::Idle),
                    &format!(" API key ({}) ", preset.env_var),
                );
                let line = match &self.validation {
                    Validation::Idle => Line::styled(
                        "Enter validates the key and fetches models. No inference is run.",
                        Style::new().fg(DIM),
                    ),
                    Validation::Running(_) => Line::styled(
                        format!("{} Loading provider models…", spinner(self.tick)),
                        Style::new().fg(WARN),
                    ),
                    Validation::Failed(error) => Line::styled(
                        format!("Failed: {error} Enter retries; r edits the key."),
                        Style::new().fg(ERR),
                    ),
                };
                f.render_widget(Paragraph::new(line).wrap(Wrap { trim: true }), status);
            }
            CloudPhase::Models => self.draw_cloud_models(f, area),
        }
    }

    fn draw_cloud_models(&self, f: &mut Frame, area: Rect) {
        let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
        let selected = self.cloud_model_picks.iter().filter(|p| **p).count();
        let block = panel(format!(" {} models · {} selected ", preset.label, selected));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [filter, list, detail] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(if area.height >= 14 { 7 } else { 3 }),
        ])
        .areas(inner);
        render_input(
            f,
            filter,
            &self.cloud_filter,
            false,
            true,
            " Filter models ",
        );
        let visible = self.filtered_cloud_models();
        let height = list.height as usize;
        let start = self
            .cloud_model_cursor
            .saturating_sub(height.saturating_sub(1));
        let mut rows = Vec::new();
        for (position, i) in visible.iter().enumerate().skip(start).take(height) {
            let model = &self.cloud_models[*i];
            rows.push(Line::styled(
                format!(
                        "{} {} {}{}",
                        if position == self.cloud_model_cursor {
                            ">"
                        } else {
                            " "
                        },
                        if self.cloud_model_picks[*i] {
                            "●"
                        } else {
                            "○"
                        },
                        model.id,
                        if model.selection_error().is_some() {
                            " [unavailable]"
                        } else if model.metadata.as_ref().is_some_and(
                            |metadata| metadata.status == setup::RegistryStatus::Deprecated
                        ) {
                            " [deprecated]"
                        } else {
                            ""
                        }
                    ),
                Style::new().fg(if position == self.cloud_model_cursor {
                    ACCENT
                } else {
                    DIM
                }),
            ));
        }
        if rows.is_empty() {
            rows.push(Line::from("No models match the filter."));
        }
        f.render_widget(Paragraph::new(rows), list);
        let mut lines = Vec::new();
        if let Some(i) = visible.get(self.cloud_model_cursor) {
            let model = &self.cloud_models[*i];
            lines.push(Line::from(model.id.clone()));
            if let Some(error) = model.selection_error() {
                lines.push(Line::styled(error, Style::new().fg(ERR)));
            }
            if let Some(metadata) = &model.metadata {
                lines.push(Line::from(metadata.catalog_capabilities().join(" · ")));
                lines.push(Line::styled(
                    format!(
                        "Context: {} · Max output: {}",
                        metadata
                            .context_tokens
                            .map(|n| n.to_string())
                            .unwrap_or("unknown".into()),
                        metadata
                            .max_output_tokens
                            .map(|n| n.to_string())
                            .unwrap_or("unknown".into())
                    ),
                    Style::new().fg(DIM),
                ));
                if !metadata.notes.is_empty() {
                    lines.push(Line::styled(metadata.notes.as_str(), Style::new().fg(DIM)));
                }
            }
        }
        if let Some(error) = &self.cloud_model_error {
            lines.insert(0, Line::styled(error.as_str(), Style::new().fg(ERR)));
        }
        lines.push(Line::styled(
            "Listing validates discovery, not inference access or quota.",
            Style::new().fg(DIM),
        ));
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), detail);
    }

    fn draw_local(&self, f: &mut Frame, area: Rect) {
        match self.local_phase {
            LocalPhase::Path => {
                let block = panel(" Local models ");
                let inner = block.inner(area);
                f.render_widget(block, area);
                let [intro, input, list] = Layout::vertical([
                    Constraint::Length(4),
                    Constraint::Length(3),
                    Constraint::Min(2),
                ])
                .areas(inner);
                f.render_widget(
                    Paragraph::new(vec![
                        Line::from(""),
                        Line::from("  Point at a .gguf file or a folder of models (e.g. an LM Studio or Hugging Face cache)."),
                        Line::styled("  Vision projectors (mmproj) next to a model are detected automatically.", Style::new().fg(DIM)),
                    ]),
                    intro,
                );
                render_input(f, input, &self.path_input, false, true, " File or folder ");
                let mut lines = Vec::new();
                if let Some(e) = &self.local_error {
                    lines.push(Line::styled(format!("  ✗ {e}"), Style::new().fg(ERR)));
                }
                for c in &self.chosen {
                    lines.push(Line::from(vec![
                        Span::styled("  ✓ ", Style::new().fg(OK)),
                        Span::raw(format!("{}  ", c.display_name)),
                        Span::styled(
                            format!(
                                "{} · {} · {}",
                                c.kind_label,
                                c.engine(),
                                human_bytes(c.size_bytes)
                            ),
                            Style::new().fg(DIM),
                        ),
                    ]));
                }
                f.render_widget(Paragraph::new(lines), list);
            }
            LocalPhase::Pick => {
                let block = panel(format!(
                    " Found {} model(s) · {} file(s) skipped ",
                    self.candidates.len(),
                    self.skipped.len()
                ));
                let inner = block.inner(area);
                f.render_widget(block, area);
                let [list, info] =
                    Layout::vertical([Constraint::Min(3), Constraint::Length(5)]).areas(inner);

                let h = list.height as usize;
                let start = self.local_cursor.saturating_sub(h.saturating_sub(1));
                let mut lines = Vec::new();
                for (i, c) in self
                    .candidates
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(h.max(1))
                {
                    let cur = i == self.local_cursor;
                    let mut spans = vec![
                        Span::styled(if cur { " ❯ " } else { "   " }, Style::new().fg(ACCENT)),
                        Span::styled(
                            if self.picks[i] { "● " } else { "○ " },
                            Style::new().fg(if self.picks[i] { OK } else { DIM }),
                        ),
                        Span::raw(format!("{}  ", c.display_name)),
                        Span::styled(
                            format!(
                                "{} · {} · {}",
                                c.kind_label,
                                c.engine(),
                                human_bytes(c.size_bytes)
                            ),
                            Style::new().fg(DIM),
                        ),
                    ];
                    if !c.is_ready() {
                        spans.push(Span::styled("  ⚠ needs setup", Style::new().fg(WARN)));
                    }
                    lines.push(Line::from(spans));
                }
                f.render_widget(Paragraph::new(lines), list);

                let mut notes: Vec<Line> = Vec::new();
                if let Some(c) = self.candidates.get(self.local_cursor) {
                    if !c.needs.is_empty() {
                        notes.push(Line::styled(
                            format!(
                                " ⚠ add extra.{} to the config before use (unticked by default)",
                                c.needs.join(", extra.")
                            ),
                            Style::new().fg(WARN),
                        ));
                    }
                    for n in &c.notes {
                        notes.push(Line::styled(format!(" ℹ {n}"), Style::new().fg(DIM)));
                    }
                }
                for s in self.skipped.iter().take(2) {
                    let name = s.path.file_name().unwrap_or_default().to_string_lossy();
                    notes.push(Line::styled(
                        format!(" skipped {name}: {}", s.reason),
                        Style::new().fg(DIM),
                    ));
                }
                if self.skipped.len() > 2 {
                    notes.push(Line::styled(
                        format!(" … and {} more skipped", self.skipped.len() - 2),
                        Style::new().fg(DIM),
                    ));
                }
                f.render_widget(Paragraph::new(notes).wrap(Wrap { trim: true }), info);
            }
        }
    }

    fn draw_engine(&self, f: &mut Frame, area: Rect) {
        let enabled = self
            .engine_choices
            .iter()
            .filter(|choices| !choices.is_empty())
            .count();
        let mut lines = vec![
            Line::from("Select engines to install. Models need at least one selected engine."),
            Line::from(""),
        ];
        if let Some(engine) = self.engine_available.get(self.engine_cursor) {
            lines.push(Line::from(format!(
                "Engine {}/{}: {}",
                self.engine_cursor + 1,
                self.engine_available.len(),
                engine
            )));
            lines.push(Line::styled(
                format!(
                    "> [{}] Install {}",
                    if self.engine_selected[self.engine_cursor] {
                        "x"
                    } else {
                        " "
                    },
                    engine
                ),
                Style::new().fg(ACCENT),
            ));
            let models = self.engine_models();
            let height = area.height.saturating_sub(13).max(1) as usize;
            let start = self.engine_option.saturating_sub(height.saturating_sub(1));
            lines.push(Line::from(format!(
                "Supported models ({}-{}/{}):",
                start + 1,
                (start + height).min(models.len()),
                models.len()
            )));
            for (position, &index) in models.iter().enumerate().skip(start).take(height) {
                let model = &self.chosen[index];
                let choices = &self.engine_choices[index];
                let status = match choices.first() {
                    None => "disabled · no selected engine".to_owned(),
                    Some(default) if default == engine => "enabled · default engine".to_owned(),
                    Some(default) if self.engine_selected[self.engine_cursor] => {
                        format!("enabled · default: {default}; this engine untested")
                    }
                    Some(default) => format!("enabled · default: {default}"),
                };
                lines.push(Line::styled(
                    format!(
                        "{} {} · {}",
                        if position == self.engine_option {
                            ">"
                        } else {
                            " "
                        },
                        model.display_name,
                        status
                    ),
                    Style::new().fg(if choices.is_empty() { DIM } else { ACCENT }),
                ));
            }
        }
        lines.push(Line::from(format!(
            "{} enabled · {} disabled models",
            enabled,
            self.chosen.len().saturating_sub(enabled)
        )));
        lines.push(Line::from(""));
        match &self.engine {
            EnginePhase::Pick => lines.push(Line::from(
                "Enter installs selected engines and checks enabled models with their defaults.",
            )),
            EnginePhase::Installing(_) => {
                lines.push(Line::from(format!(
                    "{} {}",
                    spinner(self.tick),
                    self.engine_status
                )));
                if let Some(download) = &self.engine_download {
                    use crate::inference::engines::download::format_bytes;
                    if download.extracting {
                        lines.push(Line::from(format!(
                            "{} Extracting {}",
                            spinner(self.tick),
                            download.filename
                        )));
                    } else {
                        let total = download
                            .total
                            .map(format_bytes)
                            .unwrap_or_else(|| "unknown size".into());
                        let dots = download
                            .total
                            .map(|total| {
                                let filled = (download.downloaded.saturating_mul(16) / total.max(1))
                                    .min(16) as usize;
                                format!("[{}{}] ", "●".repeat(filled), "·".repeat(16 - filled))
                            })
                            .unwrap_or_default();
                        lines.push(Line::from(download.filename.clone()));
                        lines.push(Line::from(format!(
                            "{}{} / {} · {}/s",
                            dots,
                            format_bytes(download.downloaded),
                            total,
                            format_bytes(download.bytes_per_second as u64)
                        )));
                    }
                }
            }
            EnginePhase::Done(Ok(installed)) => {
                lines.push(Line::from(if enabled == 0 {
                    "No local models enabled. Enter to continue with cloud models.".to_owned()
                } else { format!("● {} engines installed; {} defaults passed. Other combinations untested. Enter to continue.", installed.engines.len(), enabled) }));
            }
            EnginePhase::Done(Err(error)) => {
                lines.push(Line::styled(
                    format!("Failed: {error}"),
                    Style::new().fg(ERR),
                ));
                lines.push(Line::from(
                    "Enter retries. Esc returns to engine selection.",
                ));
            }
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" Engines ")),
            area,
        );
    }

    fn draw_review(&self, f: &mut Frame, area: Rect) {
        let p = match self.plan() {
            Ok(plan) => plan,
            Err(error) => {
                f.render_widget(Paragraph::new(error).style(Style::new().fg(ERR)), area);
                return;
            }
        };
        let bullet = |s: String| {
            Line::from(vec![
                Span::styled("  • ", Style::new().fg(ACCENT)),
                Span::raw(s),
            ])
        };
        let mut lines = vec![
            Line::from(""),
            bullet(format!(
                "Listen on {}:{} ({})",
                p.exposure.host(),
                p.port,
                if p.exposure == Exposure::Local {
                    "this computer only"
                } else {
                    "network"
                }
            )),
        ];
        if !self.chosen.is_empty() {
            lines.push(bullet(format!(
                "Local models: {} enabled · {} disabled",
                p.local.len(),
                self.chosen.len().saturating_sub(p.local.len())
            )));
        }
        match &p.server_api_key {
            Some(k) => {
                lines.push(bullet(
                    "Require an API key — copy it now, it is also saved in .env:".into(),
                ));
                lines.push(Line::styled(
                    format!("      {k}"),
                    Style::new().fg(WARN).add_modifier(Modifier::BOLD),
                ));
            }
            None => lines.push(bullet("No API key (localhost only)".into())),
        }
        for c in &p.cloud {
            lines.push(bullet(format!("{} → {}", c.preset.label, c.model.id)));
        }
        for c in &p.local {
            lines.push(bullet(format!(
                "Local: {}  ({} · {} default; {} additional untested)",
                c.display_name,
                c.kind_label,
                c.engine(),
                c.engines.len().saturating_sub(1)
            )));
            if !c.needs.is_empty() {
                lines.push(Line::styled(
                    format!(
                        "      ⚠ add extra.{} to the config before first use",
                        c.needs.join(", extra.")
                    ),
                    Style::new().fg(WARN),
                ));
            }
        }
        for (model, _) in self
            .chosen
            .iter()
            .zip(&self.engine_choices)
            .filter(|(_, engines)| engines.is_empty())
        {
            lines.push(Line::styled(
                format!(
                    "  Disabled: {} · no selected engine (not added)",
                    model.display_name
                ),
                Style::new().fg(DIM),
            ));
        }
        lines.push(Line::from(""));
        lines.push(Line::styled(
            format!("  Catalog: {}", self.ws.catalog_path.display()),
            Style::new().fg(DIM),
        ));
        lines.push(Line::styled(
            format!("  Server:  {}", self.ws.server_path.display()),
            Style::new().fg(DIM),
        ));
        lines.push(Line::styled(
            format!("  Secrets: {}", self.ws.env_file().display()),
            Style::new().fg(DIM),
        ));
        if let Some(e) = &self.review_error {
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!("  ✗ Could not save: {e}"),
                Style::new().fg(ERR),
            ));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" Review ")),
            area,
        );
    }
}

fn render_input(
    f: &mut Frame,
    area: Rect,
    input: &Input,
    masked: bool,
    focused: bool,
    title: &str,
) {
    let block = Block::default()
        .border_style(Style::new().fg(if focused { ACCENT } else { DIM }))
        .title(title.to_string());
    let inner = block.inner(area);
    let width = inner.width.max(1) as usize;
    let scroll = input.visual_scroll(width);
    let shown: String = if masked {
        "•".repeat(input.value().chars().count())
    } else {
        input.value().to_string()
    };
    f.render_widget(
        Paragraph::new(shown)
            .scroll((0, scroll as u16))
            .block(block),
        area,
    );
    if focused {
        f.set_cursor_position((
            inner.x + (input.visual_cursor().max(scroll) - scroll) as u16,
            inner.y,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn type_str(w: &mut Wizard, s: &str) {
        for ch in s.chars() {
            w.on_key(key(KeyCode::Char(ch)));
        }
    }

    fn render_all_sizes(w: &Wizard) {
        for (cols, rows) in [(30, 8), (60, 20), (80, 24), (71, 30), (140, 50)] {
            let mut t = Terminal::new(TestBackend::new(cols, rows)).unwrap();
            t.draw(|f| w.draw(f)).unwrap();
            if cols == 80 && rows == 24 && w.step == Step::Network {
                let text = t
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|cell| cell.symbol())
                    .collect::<String>();
                assert!(
                    text.contains("Who can connect?")
                        && text.contains("Port")
                        && text.contains("API key")
                );
                if w.net_error.is_some() {
                    assert!(text.contains("Port must") || text.contains("Network access"));
                }
            }
            for cell in &t.backend().buffer().content {
                if std::env::var_os("NO_COLOR").is_some() {
                    assert_eq!(cell.fg, Color::Reset);
                    assert_eq!(cell.bg, Color::Reset);
                }
                assert_eq!(cell.bg, Color::Reset);
                assert!(matches!(
                    cell.fg,
                    Color::Reset | Color::White | Color::DarkGray | Color::Red
                ));
                if std::env::var_os("RUNE_ASCII").is_some() {
                    assert!(cell.symbol().is_ascii());
                }
            }
        }
    }

    fn temp_ws() -> Workspace {
        let root = std::env::temp_dir().join(format!("rune-wiz-{}", setup::generate_api_key()));
        std::fs::create_dir_all(&root).unwrap();
        Workspace {
            server_path: root.join("config").join("server.json"),
            catalog_path: root.join("config").join("catalog.json"),
            legacy_config_path: root.join("config").join("models.json"),
            root,
        }
    }

    #[test]
    fn full_flow_writes_a_loadable_config_and_every_screen_renders() {
        let ws = temp_ws();
        let model_dir = ws.root.join("models");
        crate::detect::testing::Gguf::llm("llama", "Tiny").write(&model_dir.join("tiny.gguf"));
        // An image model that still needs companion files, to exercise the warning rendering.
        crate::detect::testing::Gguf::new("lumina2").write(&model_dir.join("img/z-image.gguf"));

        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        render_all_sizes(&w);

        assert!(w.step == Step::Network);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Sources);
        assert!(w.server_key.is_some());
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Sources && w.source_error.is_some());
        w.on_key(key(KeyCode::Char(' ')));
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Local);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Local && w.local_error.is_some());
        // Local: scan a folder, pick everything
        type_str(&mut w, &model_dir.to_string_lossy());
        w.on_key(key(KeyCode::Enter));
        assert!(w.local_phase == LocalPhase::Pick, "{:?}", w.local_error);
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert_eq!(w.chosen.len(), 1);
        // Every local selection must pass installation and inference checks.
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Engine);
        render_all_sizes(&w);
        let models = setup::select_local_engines(&w.chosen, &w.engine_choices).unwrap();
        let fake_binary = ws.root.join("fixture-engine");
        std::fs::write(&fake_binary, "fixture engine").unwrap();
        let prepared = setup::prepare_local_models_with(
            &models,
            w.gpu_available(),
            |_| Ok(fake_binary.clone()),
            |_| Ok(()),
            |_| {},
        )
        .unwrap();
        w.engine = EnginePhase::Done(Ok(prepared));
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Review);
        render_all_sizes(&w);
        // Save
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Finish));

        let cfg = ModelRegistryProbe::load(&ws);
        assert_eq!(cfg.0, 1, "one local model written");
        assert!(ws.env_file().exists());
        std::fs::remove_dir_all(&ws.root).ok();
    }

    /// `RUNE_PROBE_DIR=D:/Models cargo test dump_pick_screen -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_pick_screen() {
        let Ok(dir) = std::env::var("RUNE_PROBE_DIR") else {
            return;
        };
        let ws = temp_ws();
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.step = Step::Local;
        w.local_phase = LocalPhase::Pick;
        let d = setup::discover_models(std::path::Path::new(&dir)).unwrap();
        w.picks = d.models.iter().map(|c| c.is_ready()).collect();
        w.candidates = d.models;
        w.skipped = d.skipped;
        w.local_cursor = 14;
        for (cols, rows) in [(120u16, 32u16), (80, 30)] {
            let mut t = Terminal::new(TestBackend::new(cols, rows)).unwrap();
            t.draw(|f| w.draw(f)).unwrap();
            let buf = t.backend().buffer().clone();
            println!("---- {cols}x{rows}");
            for y in 0..rows {
                let line: String = (0..cols)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect();
                println!("{}", line.trim_end());
            }
        }
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn network_without_key_is_refused_and_bad_port_rejected() {
        let ws = temp_ws();
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.on_key(key(KeyCode::Char(' '))); // exposure -> Network
        w.on_key(key(KeyCode::Down));
        w.on_key(key(KeyCode::Down));
        w.on_key(key(KeyCode::Char(' '))); // key off
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Network && w.net_error.is_some());

        w.on_key(key(KeyCode::Up));
        for _ in 0..4 {
            w.on_key(key(KeyCode::Backspace));
        }
        type_str(&mut w, "99999");
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Network && w.net_error.is_some());
        render_all_sizes(&w);
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn missing_path_shows_error_and_stays() {
        let ws = temp_ws();
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.step = Step::Local;
        type_str(&mut w, "Z:/definitely/not/here");
        w.on_key(key(KeyCode::Enter));
        assert!(w.local_error.is_some() && w.local_phase == LocalPhase::Path);
        render_all_sizes(&w);
        std::fs::remove_dir_all(&ws.root).ok();
    }

    #[test]
    fn source_routes_cloud_retry_and_empty_review_do_not_write() {
        let ws = temp_ws();
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.on_key(key(KeyCode::Enter));
        w.on_key(key(KeyCode::Down));
        w.on_key(key(KeyCode::Char(' ')));
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Cloud);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Cloud && w.review_error.is_some());
        w.on_key(key(KeyCode::Char(' ')));
        w.on_key(key(KeyCode::Enter));
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert!(w.cloud_done.is_empty() && w.cloud_phase == CloudPhase::Details);
        w.validation = Validation::Failed("rejected".into());
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Char('r')));
        assert!(matches!(w.validation, Validation::Idle) && w.cloud_done.is_empty());
        w.key_input = Input::new("fixture-key".into());
        let metadata = setup::registered_model("google", "gemini-3.5-flash-lite")
            .unwrap()
            .unwrap();
        let model = CloudModel {
            id: metadata.id.clone(),
            name: metadata.name.clone(),
            context_tokens: None,
            max_output_tokens: None,
            capabilities: setup::CloudCapabilities::default(),
            metadata: Some(metadata),
        };
        let (tx, rx) = channel();
        w.validation = Validation::Running(rx);
        tx.send(Ok(vec![model])).unwrap();
        w.poll_workers();
        assert!(w.cloud_phase == CloudPhase::Models);
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Char(' ')));
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Review && w.cloud_done.len() == 1);
        render_all_sizes(&w);
        w.cloud_done.clear();
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Stay));
        assert!(!ws.server_path.exists() && !ws.catalog_path.exists() && !ws.env_file().exists());
        w.sources = [true, true];
        w.next_after_local();
        assert!(w.step == Step::Cloud);
        w.sources = [true, false];
        w.next_after_local();
        assert!(w.step == Step::Review);
        std::fs::remove_dir_all(&ws.root).unwrap();
    }

    #[test]
    fn cloud_multiselect_filters_without_losing_selection_and_requires_registry_metadata() {
        let ws = temp_ws();
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.sources = [false, true];
        w.step = Step::Cloud;
        w.cloud_phase = CloudPhase::Models;
        w.cloud_queue = vec![0];
        w.key_input = Input::new("fixture-secret".into());
        for id in ["gemini-3.5-flash-lite", "gemini-3.8-flash"] {
            let metadata = setup::registered_model("google", id).unwrap().unwrap();
            w.cloud_models.push(CloudModel {
                id: id.into(),
                name: metadata.name.clone(),
                context_tokens: None,
                max_output_tokens: None,
                capabilities: setup::CloudCapabilities::default(),
                metadata: Some(metadata),
            });
        }
        w.cloud_models.push(CloudModel {
            id: "unregistered".into(),
            name: "Unregistered".into(),
            context_tokens: None,
            max_output_tokens: None,
            capabilities: setup::CloudCapabilities::default(),
            metadata: None,
        });
        w.cloud_model_picks = vec![false; 3];
        w.on_key(key(KeyCode::Enter));
        assert!(w.cloud_model_error.is_some() && w.cloud_done.is_empty());
        type_str(&mut w, "3.5");
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(w.cloud_model_picks, [true, false, false]);
        render_all_sizes(&w);
        for _ in 0..3 {
            w.on_key(key(KeyCode::Backspace));
        }
        type_str(&mut w, "3.8");
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(w.cloud_model_picks, [true, true, false]);
        for _ in 0..3 {
            w.on_key(key(KeyCode::Backspace));
        }
        type_str(&mut w, "unregistered");
        w.on_key(key(KeyCode::Char(' ')));
        assert!(w.cloud_model_error.is_some());
        assert!(!w.cloud_model_picks[2]);
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Review);
        assert_eq!(w.cloud_done.len(), 2);
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Finish));
        assert_eq!(ModelRegistryProbe::load(&ws).0, 2);
        let env = std::fs::read_to_string(ws.env_file()).unwrap();
        assert_eq!(env.matches("GEMINI_API_KEY=").count(), 1);
        std::fs::remove_dir_all(ws.root).unwrap();
    }

    #[test]
    fn engine_first_selection_disables_only_models_without_a_selected_engine() {
        let ws = temp_ws();
        crate::detect::testing::Gguf::llm("llama", "Tiny").write(&ws.root.join("tiny.gguf"));
        crate::detect::testing::Gguf::new("audiocpp")
            .string("audiocpp.model_spec.family", "audio8_tts")
            .write(&ws.root.join("audio.gguf"));
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.sources = [true, false];
        w.chosen = setup::discover_models(&ws.root).unwrap().models;
        w.next_after_local();
        assert_eq!(
            w.engine_available.len(),
            3,
            "engines are unique across models"
        );
        w.engine_cursor = w
            .engine_available
            .iter()
            .position(|engine| *engine == EngineKind::AudioCpp)
            .unwrap();
        assert_eq!(w.engine_models().len(), 1);
        w.on_key(key(KeyCode::Char(' ')));
        let enabled = w.plan().unwrap().local;
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].engine(), EngineKind::LlamaServer);
        let mut screen = Terminal::new(TestBackend::new(100, 28)).unwrap();
        screen.draw(|f| w.draw_engine(f, f.area())).unwrap();
        let text = screen
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            text.contains("Engine 1/3") && text.contains("disabled") && text.contains("audio-cpp")
        );
        render_all_sizes(&w);
        // Disabling one model cannot lead to its engine being installed or smoke-tested.
        let binary = ws.root.join("fixture-engine");
        std::fs::write(&binary, "fixture").unwrap();
        let mut installed = Vec::new();
        let mut checked = Vec::new();
        let prepared = setup::prepare_local_models_with(
            &enabled,
            false,
            |engine| {
                installed.push(engine);
                Ok(binary.clone())
            },
            |model| {
                checked.push(model.path.clone());
                Ok(())
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(installed, [EngineKind::LlamaServer]);
        assert_eq!(checked, [ws.root.join("tiny.gguf")]);
        w.engine = EnginePhase::Done(Ok(prepared));
        w.step = Step::Review;
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Finish));
        assert_eq!(ModelRegistryProbe::load(&ws).0, 1);
        let catalog = std::fs::read_to_string(&ws.catalog_path).unwrap();
        assert!(!catalog.contains("audio.gguf"));
        // A model with two compatible engines remains enabled through the other one.
        w.engine = EnginePhase::Pick;
        w.step = Step::Engine;
        w.engine_cursor = w
            .engine_available
            .iter()
            .position(|engine| *engine == EngineKind::LlamaServerPrism)
            .unwrap();
        w.on_key(key(KeyCode::Char(' ')));
        w.engine_cursor = w
            .engine_available
            .iter()
            .position(|engine| *engine == EngineKind::LlamaServer)
            .unwrap();
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(
            w.plan().unwrap().local[0].engine(),
            EngineKind::LlamaServerPrism
        );
        // Re-enabling an engine restores its uniquely supported models.
        w.engine_cursor = w
            .engine_available
            .iter()
            .position(|engine| *engine == EngineKind::AudioCpp)
            .unwrap();
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(w.plan().unwrap().local.len(), 2);
        std::fs::remove_dir_all(ws.root).unwrap();
    }

    #[test]
    fn all_local_models_disabled_requires_cloud_or_an_enabled_model() {
        let ws = temp_ws();
        crate::detect::testing::Gguf::llm("llama", "Tiny").write(&ws.root.join("tiny.gguf"));
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.sources = [true, false];
        w.chosen = setup::discover_models(&ws.root).unwrap().models;
        w.next_after_local();
        w.on_key(key(KeyCode::Char(' ')));
        assert!(w.plan().unwrap().local.is_empty());
        w.prereq = Some(Prerequisites {
            os: "test",
            arch: "test",
            home: ws.root.clone(),
            gpu: None,
            ffmpeg: None,
            llama_server_installed: false,
        });
        w.on_key(key(KeyCode::Enter));
        assert!(
            matches!(&w.engine, EnginePhase::Done(Err(error)) if error.contains("at least one"))
        );
        w.step = Step::Review;
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Stay));
        assert!(!ws.catalog_path.exists());
        // Cloud-only completion requires no installation or proof for disabled locals.
        w.sources[1] = true;
        w.cloud_done.push(CloudChoice {
            preset: PROVIDERS[0],
            api_key: "fixture-key".into(),
            model: setup::registered_model("google", "gemini-3.5-flash-lite")
                .unwrap()
                .unwrap(),
        });
        render_all_sizes(&w);
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Finish));
        assert_eq!(ModelRegistryProbe::load(&ws).0, 1);
        assert!(
            !std::fs::read_to_string(&ws.catalog_path)
                .unwrap()
                .contains("tiny.gguf")
        );
        std::fs::remove_dir_all(ws.root).unwrap();
    }

    #[test]
    fn local_engine_choice_failure_progress_and_ready_save_guard() {
        let ws = temp_ws();
        crate::detect::testing::Gguf::llm("llama", "Tiny").write(&ws.root.join("tiny.gguf"));
        let (_tx, rx) = channel();
        let mut w = Wizard::new(&ws, rx);
        w.sources = [true, false];
        w.chosen = setup::discover_models(&ws.root).unwrap().models;
        assert!(w.chosen[0].engines.contains(&EngineKind::LlamaServerPrism));
        w.next_after_local();
        w.on_key(key(KeyCode::Right));
        w.on_key(key(KeyCode::Char('d')));
        w.on_key(key(KeyCode::Left));
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(w.engine_choices, [vec![EngineKind::LlamaServerPrism]]);
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(
            w.plan().unwrap().local[0].engines,
            [EngineKind::LlamaServerPrism, EngineKind::LlamaServer]
        );
        w.on_key(key(KeyCode::Char(' ')));
        assert_eq!(
            w.plan().unwrap().local[0].engines,
            [EngineKind::LlamaServerPrism]
        );
        w.step = Step::Review;
        assert!(matches!(w.on_key(key(KeyCode::Enter)), Flow::Stay));
        assert!(!ws.catalog_path.exists());
        w.step = Step::Engine;
        w.engine = EnginePhase::Done(Err("download failed".into()));
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Char('n')));
        assert!(w.step == Step::Engine);
        w.on_key(key(KeyCode::Esc));
        assert!(matches!(w.engine, EnginePhase::Pick));
        let (tx, rx) = channel();
        w.engine = EnginePhase::Installing(rx);
        tx.send(LocalPreparationEvent::Progress("Loading Tiny".into()))
            .unwrap();
        w.poll_workers();
        assert_eq!(w.engine_status, "Loading Tiny");
        render_all_sizes(&w);
        let download = crate::inference::engines::download::DownloadProgress {
            filename: "engine.zip".into(),
            extracting: false,
            downloaded: 512,
            total: Some(1024),
            bytes_per_second: 128.0,
        };
        tx.send(LocalPreparationEvent::Download(download.clone()))
            .unwrap();
        w.poll_workers();
        assert_eq!(w.engine_download.as_ref().unwrap().downloaded, 512);
        render_all_sizes(&w);
        let mut extraction = download;
        extraction.extracting = true;
        tx.send(LocalPreparationEvent::Download(extraction))
            .unwrap();
        w.poll_workers();
        render_all_sizes(&w);
        tx.send(LocalPreparationEvent::Progress(
            "Checking Tiny default".into(),
        ))
        .unwrap();
        w.poll_workers();
        assert!(w.engine_download.is_none());
        tx.send(LocalPreparationEvent::Finished(Err(
            "inference failed".into()
        )))
        .unwrap();
        w.poll_workers();
        assert!(matches!(w.engine, EnginePhase::Done(Err(_))));
        std::fs::remove_dir_all(ws.root).unwrap();
    }

    struct ModelRegistryProbe(usize);
    impl ModelRegistryProbe {
        fn load(ws: &Workspace) -> Self {
            let secrets = std::fs::read_to_string(ws.env_file()).unwrap();
            let pairs: Vec<(String, String)> = secrets
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let reg = crate::config::ModelRegistry::from_files_with_env(
                &ws.server_path,
                &ws.catalog_path,
                &pairs,
            )
            .unwrap();
            Self(reg.models.len())
        }
    }
}
