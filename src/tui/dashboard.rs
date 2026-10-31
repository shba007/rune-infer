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
}

#[derive(Debug, Clone)]
pub struct DashboardInfo {
    pub version: &'static str,
    pub addr: String,
    pub auth_enabled: bool,
    pub exposed_to_network: bool,
    pub config_path: PathBuf,
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    model_action: Option<ModelAction>,
    model_action_rx: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    model_action_status: Option<String>,
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

impl Drop for App {
    fn drop(&mut self) {
        if let Some(cancel) = &self.update_cancel {
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
            .map(|row| ModelState {
                row,
                status: Status::Idle,
                requests: 0,
                errors: 0,
                tps_sum: 0.0,
                tps_n: 0,
                last_used: None,
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
            model_action: None,
            model_action_rx: None,
            model_action_status: None,
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
        match &self.models.get(self.model_selected)?.status {
            Status::Loaded { .. } => Some(ModelAction::Unload),
            Status::Loading(_) => None,
            Status::Idle | Status::Failed(_) => Some(ModelAction::Load),
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
            KeyCode::Enter if self.tab == Tab::Models && self.model_action_rx.is_none() => {
                self.model_action = self.selected_model_action();
            }
            KeyCode::Up if self.tab == Tab::Models => {
                self.model_selected = self.model_selected.saturating_sub(1);
            }
            KeyCode::Down if self.tab == Tab::Models => {
                self.model_selected =
                    (self.model_selected + 1).min(self.models.len().saturating_sub(1));
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
                                    crate::inference::engines::update::update_sd(
                                        state,
                                        cancel,
                                        engine.latest,
                                        tx.clone(),
                                    )
                                } else if engine.name == "nemo-speech" {
                                    crate::inference::engines::update::update_nemo(
                                        state,
                                        cancel,
                                        engine.latest,
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
                                    crate::inference::engines::update::update_crispasr(
                                        state,
                                        cancel,
                                        engine.latest,
                                        tx.clone(),
                                    )
                                } else {
                                    crate::inference::engines::update::update_audio(
                                        state,
                                        cancel,
                                        engine.latest,
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
        match app.selected_model_action() {
            Some(ModelAction::Load) => spans.extend(key_hint("Enter", "load")),
            Some(ModelAction::Unload) => spans.extend(key_hint("Enter", "unload")),
            None => spans.push(Span::styled(" model busy ", Style::new().fg(DIM))),
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
    published_at: Option<String>,
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
    ];
    let mut rows: Vec<EngineRow> = definitions
        .into_iter()
        .map(|(name, directory, executable, version_supported)| {
            let filename = if name == "litert" {
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
                _ => &[],
            };
            let custom = custom_vars
                .iter()
                .find_map(std::env::var_os)
                .map(PathBuf::from)
                .filter(|p| name == "litert" || p.is_file());
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
                    if matches!(name, "needle" | "litert") {
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
                None => (base.join(&filename), name == "litert", false),
            };
            let version = if name == "litert" && installed {
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
                latest: if name == "litert" {
                    "2.3.0 · pinned".into()
                } else if managed || !installed {
                    "Checking…".into()
                } else {
                    "External".into()
                },
                build_tag: None,
                published_at: None,
                check_detail: (name == "litert").then(|| "Official Windows x64 CPU runtime · pinned SHA-256 · model weights configured separately".into()),
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
        published_at: None,
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
            if rows[index].name == "litert" {
                continue;
            }
            if rows[index].installed && !rows[index].managed {
                continue;
            }
            match check_latest_release(rows[index].name) {
                Ok(release) => {
                    rows[index].latest = release.tag;
                    rows[index].build_tag = release.build_tag;
                    rows[index].published_at = release.published_at;
                    rows[index].check_detail = Some(release.detail);
                }
                Err(error) => {
                    rows[index].latest = if error.contains("GitHub rate limited until") {
                        "Rate limited"
                    } else {
                        "Check failed"
                    }
                    .into();
                    rows[index].build_tag = None;
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
    if engine.name == "litert" {
        if os != "windows" || arch != "x86_64" {
            return unavailable("LiteRT installation currently supports Windows x64 CPU only");
        }
        if !engine.managed {
            return unavailable(
                "External LiteRT runtime: fix or remove LITERT_LIBRARY_PATH to use the managed installation",
            );
        }
        if !engine.installed {
            return EngineAction::Install;
        }
        return unavailable(if engine.version == "LiteRT 2.3.0 CPU" {
            "Pinned LiteRT runtime is installed; no download needed"
        } else {
            "Existing LiteRT DLL is unverified; preserve it before reinstalling"
        });
    }
    if engine.installed && !engine.managed {
        return unavailable("Externally managed; update using its original installation method");
    }
    if engine.installed {
        let supported = match engine.name {
            "audio-cpp" => os == "windows",
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
    if engine.name == "sd-server"
        && crate::inference::engines::sd_server::revision_matches(&engine.version, &engine.latest)
            .is_none()
    {
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
        _ => return None,
    })
}

fn release_url(name: &str) -> Option<&'static str> {
    match name {
        "llama-server" => Some("https://api.github.com/repos/ggml-org/llama.cpp/releases"),
        "llama-server-prism" => Some("https://api.github.com/repos/PrismML-Eng/llama.cpp/releases"),
        "audio-cpp" => Some("https://api.github.com/repos/0xShug0/audio.cpp/releases"),
        "sd-server" => Some("https://api.github.com/repos/leejet/stable-diffusion.cpp/releases"),
        "crispasr" => Some("https://api.github.com/repos/CrispStrobe/CrispASR/releases"),
        "nemo-speech" => Some("https://api.github.com/repos/NVIDIA/NeMo-Speech.cpp/releases"),
        "needle" => Some("https://huggingface.co/api/models/Cactus-Compute/needle3"),
        "whisper-cpp" => Some("https://api.github.com/repos/ggml-org/whisper.cpp/releases"),
        _ => None,
    }
}

struct AvailableRelease {
    tag: String,
    build_tag: Option<String>,
    published_at: Option<String>,
    detail: String,
}

fn check_latest_release(name: &str) -> Result<AvailableRelease, String> {
    use crate::inference::engines::download;
    crate::inference::engines::release_cache::begin_check();
    let url = release_url(name).ok_or("No release source is available for this engine")?;
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
            published_at: value["lastModified"].as_str().map(str::to_owned),
            detail:
                "Approved Needle distribution · revision and SHA-256 verified during installation"
                    .into(),
        });
    }
    let value = if name == "whisper-cpp" {
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
    let channel = if download::is_stable_release(&value) {
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
        published_at,
        detail,
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
    if engine.name == "litert" {
        return if engine.version == "LiteRT 2.3.0 CPU" {
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
        return match crate::inference::engines::sd_server::revision_matches(
            &engine.version,
            &engine.latest,
        ) {
            Some(true) => "Up to date",
            Some(false) => "Different revision",
            None => "Revision unknown",
        };
    }
    if matches!(engine.name, "llama-server" | "llama-server-prism") {
        let available = engine.build_tag.as_deref().unwrap_or(&engine.latest);
        if let Some((installed, published)) =
            engine_build_identity(&engine.version).zip(engine_build_identity(available))
        {
            return match installed.0.cmp(&published.0) {
                std::cmp::Ordering::Less => "New release",
                std::cmp::Ordering::Greater => "Ahead",
                std::cmp::Ordering::Equal => match (installed.1, published.1) {
                    (Some(a), Some(b))
                        if !a.to_ascii_lowercase().starts_with(&b.to_ascii_lowercase())
                            && !b.to_ascii_lowercase().starts_with(&a.to_ascii_lowercase()) =>
                    {
                        "Different revision"
                    }
                    (None, Some(_)) => "Unknown",
                    _ => "Up to date",
                },
            };
        }
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

fn engine_download_line(
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
            Paragraph::new(engine_download_line(progress, app.tick)),
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
    let [table_area, details] =
        Layout::vertical([Constraint::Min(4), Constraint::Length(6)]).areas(area);
    let block = panel(format!(
        " Models Catalog · {} configured ",
        app.models.len()
    ));
    let header = Row::new([
        "",
        "Model ID / Name",
        "Locality",
        "Capabilities",
        "Provider / Engine",
        "Reqs",
        "Status",
    ])
    .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));

    let visible = table_area.height.saturating_sub(3) as usize;
    let start = app.model_selected.saturating_add(1).saturating_sub(visible);
    let rows = app
        .models
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, m)| {
            let (icon, color, status_text) = match &m.status {
                Status::Idle => ("·", DIM, "IDLE"),
                Status::Loading(_) => (spinner(app.tick), ACCENT, "LOADING"),
                Status::Loaded { .. } => ("●", OK, "LOADED"),
                Status::Failed(_) => ("×", ERR, "FAILED"),
            };

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
            let cap_str =
                format!("C:{chat_dot} V:{vision_dot} T:{tools_dot} A:{audio_dot} I:{image_dot}");

            let locality_style = if m.row.locality == "LOCAL" {
                Style::new().fg(OK)
            } else {
                Style::new().fg(ACCENT)
            };

            Row::new([
                Cell::from(if index == app.model_selected {
                    ">"
                } else {
                    icon
                })
                .style(Style::new().fg(color)),
                Cell::from(format!("{} ({})", m.row.id, m.row.name)),
                Cell::from(m.row.locality.clone()).style(locality_style),
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
        });

    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Min(24),
            Constraint::Length(8),
            Constraint::Length(22),
            Constraint::Length(18),
            Constraint::Length(6),
            Constraint::Length(9),
        ],
    )
    .header(header)
    .block(block);
    f.render_widget(table, table_area);
    let selected = app.models.get(app.model_selected);
    let detail = selected
        .map(|model| {
            let status = match &model.status {
                Status::Failed(error) => error.clone(),
                Status::Loaded { detail, .. } => detail.clone(),
                _ => String::new(),
            };
            format!(
                "{}\n{}\n{}\nManual loads still follow idle timeout and memory eviction.",
                model.row.id,
                app.model_action_status
                    .as_deref()
                    .unwrap_or("Enter: toggle load/unload · in-use models cannot be unloaded"),
                status
            )
        })
        .unwrap_or_else(|| "No models configured".into());
    f.render_widget(
        Paragraph::new(detail)
            .wrap(Wrap { trim: false })
            .block(panel(" Model controls ")),
        details,
    );
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
            models: vec![
                ModelRow {
                    id: "a".into(),
                    name: "A".into(),
                    modality: "text".into(),
                    provider: None,
                    engine: Some("llama-server".into()),
                    locality: "LOCAL".into(),
                    capabilities: vec!["Chat".into()],
                },
                ModelRow {
                    id: "g".into(),
                    name: "G".into(),
                    modality: "vision-text".into(),
                    provider: Some("google".into()),
                    engine: None,
                    locality: "CLOUD".into(),
                    capabilities: vec!["Chat".into(), "Vision".into()],
                },
            ],
        };
        App::new(info, Arc::new(Mutex::new(Some((8 << 30, 16 << 30)))))
    }

    fn ts() -> String {
        "00:00:00".into()
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
            published_at: None,
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
                published_at: None,
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
                published_at: None,
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
            published_at: None,
            check_detail: None,
        };
        assert_eq!(release_status(&row), "New release");
        row.latest = "v0.9.2".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "nightly".into();
        assert_eq!(release_status(&row), "Unknown");
        row.name = "whisper-cpp";
        row.version = "whisper.cpp version: 1.9.2".into();
        row.latest = "v1.9.2".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "v1.9.3".into();
        assert_eq!(release_status(&row), "New release");
        row.latest = "v1.9.1".into();
        assert_eq!(release_status(&row), "Ahead");
        row.name = "llama-server";
        row.version = "0.6.0-dev (build 11512, commit abc)".into();
        row.latest = "b11512".into();
        assert_eq!(release_status(&row), "Up to date");
        for (name, version, latest, build, expected) in [
            (
                "llama-server",
                "0.7.0-dev (build 11669, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "Up to date",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11668, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "New release",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11670, commit 620d2225c)",
                "v0.7.0",
                Some("b11669"),
                "Ahead",
            ),
            (
                "llama-server",
                "0.7.0-dev (build 11669, commit 620d2225c)",
                "v0.7.0",
                None,
                "Development build",
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
                "Different revision",
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
            published_at: None,
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
        let line = engine_download_line(progress, 0);
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
    fn engine_status_distinguishes_not_installed_development_and_built_in() {
        let mut row = EngineRow {
            name: "whisper-cpp",
            path: "bin/whisper-cpp".into(),
            managed: false,
            installed: false,
            version: "Not installed".into(),
            latest: "v1.9.2".into(),
            build_tag: None,
            published_at: None,
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
        row.version = "commit be48649".into();
        row.latest = "master-1017-be48649".into();
        assert_eq!(release_status(&row), "Up to date");
        row.latest = "master-1028-37bc88b".into();
        assert_eq!(release_status(&row), "Different revision");
        row.version = "Unknown".into();
        assert_eq!(release_status(&row), "Revision unknown");
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
    fn missing_litert_runtime_offers_install_on_windows_x64_and_preserves_external_files() {
        let mut row = EngineRow {
            name: "litert",
            path: "bin/litert/libLiteRt.dll".into(),
            managed: true,
            installed: false,
            version: "Not installed".into(),
            latest: "2.3.0 · pinned".into(),
            build_tag: None,
            published_at: None,
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
            EngineAction::Unavailable(_)
        ));
        assert_eq!(release_status(&row), "Ready · CPU");
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
            published_at: None,
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
            if name == "sd-server" {
                row.version = "commit be48649".into();
                row.latest = "master-1028-37bc88b".into();
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
                published_at: None,
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
                published_at: None,
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
        assert!(text.contains("Check versions before installing or updating"));
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
