//! First-run setup wizard.
//!
//! Every step is skippable except the review. Nothing touches disk until the
//! user confirms on the final screen.

use std::collections::VecDeque;
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

use super::theme::*;
use crate::config::EngineKind;
use crate::home::Workspace;
use crate::setup::{
    self, CloudChoice, Exposure, LocalCandidate, PROVIDERS, Prerequisites, SetupPlan,
};
use crate::telemetry::UiEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WizardOutcome {
    Completed,
    Cancelled,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Welcome,
    Network,
    Cloud,
    Local,
    Engine,
    Review,
}

const STEPS: [(&str, Step); 6] = [
    ("Welcome", Step::Welcome),
    ("Network", Step::Network),
    ("Cloud", Step::Cloud),
    ("Local models", Step::Local),
    ("Engine", Step::Engine),
    ("Review", Step::Review),
];

enum Flow {
    Stay,
    Cancel,
    Finish,
}

enum Validation {
    Idle,
    Running(Receiver<Result<(), String>>),
    Ok,
    Failed(String),
}

#[derive(PartialEq, Eq)]
enum CloudPhase {
    Pick,
    Details,
}

#[derive(PartialEq, Eq)]
enum LocalPhase {
    Path,
    Pick,
}

enum EnginePhase {
    Ask,
    Installing(Receiver<Result<PathBuf, String>>),
    Done(Result<PathBuf, String>),
}

struct Wizard<'a> {
    ws: &'a Workspace,
    step: Step,
    tick: u64,
    log_tail: VecDeque<String>,

    prereq: Option<Prerequisites>,
    prereq_rx: Receiver<Prerequisites>,

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
    model_input: Input,
    cloud_focus: usize,
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

    // engine
    engine: EnginePhase,

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
        while let Ok(ev) = rx.try_recv() {
            if let UiEvent::Log(l) = ev {
                if l.level <= tracing::Level::INFO {
                    if w.log_tail.len() >= 6 {
                        w.log_tail.pop_front();
                    }
                    w.log_tail.push_back(l.message);
                }
            }
        }
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
            step: Step::Welcome,
            tick: 0,
            log_tail: VecDeque::new(),
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
            model_input: Input::default(),
            cloud_focus: 0,
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
            engine: EnginePhase::Ask,
            review_error: None,
        }
    }

    fn step_index(&self) -> usize {
        STEPS.iter().position(|(_, s)| *s == self.step).unwrap_or(0)
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
        if let Validation::Running(rx) = &self.validation {
            if let Ok(res) = rx.try_recv() {
                self.validation = match res {
                    Ok(()) => Validation::Ok,
                    Err(e) => Validation::Failed(e),
                };
            }
        }
        if let EnginePhase::Installing(rx) = &self.engine {
            if let Ok(res) = rx.try_recv() {
                self.engine = EnginePhase::Done(res);
            }
        }
    }

    // ------------------------------------------------------------------ keys

    fn go(&mut self, step: Step) {
        self.step = step;
    }

    fn engine_needed(&self) -> bool {
        self.chosen
            .iter()
            .any(|c| c.engine() == EngineKind::LlamaServer)
            && self
                .prereq
                .as_ref()
                .is_some_and(|p| !p.llama_server_installed)
    }

    fn next_after_local(&mut self) {
        if self.engine_needed() {
            self.engine = EnginePhase::Ask;
            self.go(Step::Engine);
        } else {
            self.go(Step::Review);
        }
    }

    fn on_key(&mut self, k: KeyEvent) -> Flow {
        match self.step {
            Step::Welcome => match k.code {
                KeyCode::Enter => self.go(Step::Network),
                KeyCode::Esc | KeyCode::Char('q') => return Flow::Cancel,
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
            KeyCode::Esc => self.go(Step::Welcome),
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
                self.go(Step::Cloud);
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
                KeyCode::Esc => self.go(Step::Network),
                KeyCode::Up => self.cloud_cursor = self.cloud_cursor.saturating_sub(1),
                KeyCode::Down => {
                    self.cloud_cursor = (self.cloud_cursor + 1).min(PROVIDERS.len() - 1)
                }
                KeyCode::Char(' ') => {
                    self.cloud_checked[self.cloud_cursor] = !self.cloud_checked[self.cloud_cursor]
                }
                KeyCode::Enter => {
                    self.cloud_done.clear();
                    self.cloud_queue = (0..PROVIDERS.len())
                        .filter(|i| self.cloud_checked[*i])
                        .collect();
                    self.cloud_pos = 0;
                    if self.cloud_queue.is_empty() {
                        self.go(Step::Local);
                    } else {
                        self.begin_provider();
                        self.cloud_phase = CloudPhase::Details;
                    }
                }
                _ => {}
            },
            CloudPhase::Details => self.key_cloud_details(k),
        }
    }

    fn begin_provider(&mut self) {
        let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
        self.key_input = Input::default();
        self.model_input = Input::new(preset.default_model.into());
        self.cloud_focus = 0;
        self.validation = Validation::Idle;
    }

    fn advance_provider(&mut self) {
        self.cloud_pos += 1;
        if self.cloud_pos >= self.cloud_queue.len() {
            self.cloud_phase = CloudPhase::Pick;
            self.go(Step::Local);
        } else {
            self.begin_provider();
        }
    }

    fn key_cloud_details(&mut self, k: KeyEvent) {
        let preset = PROVIDERS[self.cloud_queue[self.cloud_pos]];
        match &self.validation {
            Validation::Running(_) => return,
            Validation::Ok | Validation::Failed(_) => {
                match k.code {
                    KeyCode::Enter => {
                        self.cloud_done.push(CloudChoice {
                            preset,
                            api_key: self.key_input.value().trim().to_string(),
                            upstream_model: self.model_input.value().trim().to_string(),
                        });
                        self.advance_provider();
                    }
                    KeyCode::Char('r') | KeyCode::Esc => {
                        self.validation = Validation::Idle;
                        self.cloud_focus = 0;
                    }
                    _ => {}
                }
                return;
            }
            Validation::Idle => {}
        }

        match k.code {
            KeyCode::Esc => {
                self.cloud_phase = CloudPhase::Pick;
            }
            KeyCode::Tab | KeyCode::Down | KeyCode::Up => self.cloud_focus = 1 - self.cloud_focus,
            KeyCode::Enter => {
                let key = self.key_input.value().trim().to_string();
                if key.is_empty() {
                    // Empty key = skip this provider.
                    self.advance_provider();
                } else if self.cloud_focus == 0 || self.model_input.value().trim().is_empty() {
                    self.cloud_focus = 1;
                } else {
                    let (tx, rx) = channel();
                    std::thread::spawn(move || {
                        let _ = tx.send(setup::validate_provider_key(&preset, &key));
                    });
                    self.validation = Validation::Running(rx);
                }
            }
            _ => {
                let input = if self.cloud_focus == 0 {
                    &mut self.key_input
                } else {
                    &mut self.model_input
                };
                input.handle_event(&Event::Key(k));
            }
        }
    }

    fn key_local(&mut self, k: KeyEvent) {
        match self.local_phase {
            LocalPhase::Path => match k.code {
                KeyCode::Esc => self.go(Step::Cloud),
                KeyCode::Enter => {
                    let raw = strip_quotes(self.path_input.value()).to_string();
                    if raw.is_empty() {
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
                            self.picks = d.models.iter().map(|c| c.is_ready()).collect();
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
            LocalPhase::Pick => match k.code {
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
                            self.chosen.push(c.clone());
                        }
                    }
                    self.path_input = Input::default();
                    self.local_phase = LocalPhase::Path;
                }
                _ => {}
            },
        }
    }

    fn key_engine(&mut self, k: KeyEvent) {
        match &self.engine {
            EnginePhase::Ask => match k.code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    let (tx, rx) = channel();
                    std::thread::spawn(move || {
                        let _ = tx.send(setup::install_llama_server());
                    });
                    self.engine = EnginePhase::Installing(rx);
                }
                KeyCode::Char('n') => self.go(Step::Review),
                KeyCode::Esc => self.go(Step::Local),
                _ => {}
            },
            EnginePhase::Installing(_) => {}
            EnginePhase::Done(_) => {
                if matches!(k.code, KeyCode::Enter) {
                    self.go(Step::Review);
                }
            }
        }
    }

    fn plan(&self) -> SetupPlan {
        SetupPlan {
            exposure: self.exposure,
            port: self.port.value().trim().parse().unwrap_or(8080),
            server_api_key: self.server_key.clone(),
            cloud: self.cloud_done.clone(),
            local: self.chosen.clone(),
            gpu_available: self.gpu_available(),
        }
    }

    fn key_review(&mut self, k: KeyEvent) -> Flow {
        match k.code {
            KeyCode::Esc => {
                self.review_error = None;
                if self.engine_needed() {
                    self.engine = EnginePhase::Ask;
                    self.go(Step::Engine)
                } else {
                    self.go(Step::Local)
                }
            }
            KeyCode::Enter => {
                match setup::write_setup(
                    &self.plan(),
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
        let [title, steps, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .areas(area);

        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    "ᚱ Rune Infer",
                    Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ·  first-time setup", Style::new().fg(DIM)),
            ])),
            title,
        );

        let mut spans = Vec::new();
        let compact = area.width < 84;
        if compact {
            spans.push(Span::styled(
                format!(
                    "Step {} of {} · {}",
                    self.step_index() + 1,
                    STEPS.len(),
                    STEPS[self.step_index()].0
                ),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        }
        for (i, (name, _)) in STEPS.iter().enumerate().filter(|_| !compact) {
            let cur = i == self.step_index();
            let done = i < self.step_index();
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
            if i + 1 < STEPS.len() {
                spans.push(Span::styled("  ─  ", Style::new().fg(DIM)));
            }
        }
        f.render_widget(Paragraph::new(Line::from(spans)), steps);

        match self.step {
            Step::Welcome => self.draw_welcome(f, body),
            Step::Network => self.draw_network(f, body),
            Step::Cloud => self.draw_cloud(f, body),
            Step::Local => self.draw_local(f, body),
            Step::Engine => self.draw_engine(f, body),
            Step::Review => self.draw_review(f, body),
        }

        f.render_widget(Paragraph::new(Line::from(self.hints())), footer);
    }

    fn hints(&self) -> Vec<Span<'static>> {
        let mut h: Vec<Span<'static>> = Vec::new();
        let mut add = |k: &'static str, w: &'static str| h.extend(key_hint(k, w));
        match self.step {
            Step::Welcome => {
                add("Enter", "begin");
                add("Esc", "quit");
            }
            Step::Network => {
                add("↑↓", "move");
                add("Space", "toggle");
                add("Enter", "next");
                add("Esc", "back");
            }
            Step::Cloud => match self.cloud_phase {
                CloudPhase::Pick => {
                    add("Space", "select");
                    add("Enter", "continue (none = skip)");
                    add("Esc", "back");
                }
                CloudPhase::Details => {
                    add("Tab", "switch field");
                    add("Enter", "next / test key");
                    add("empty key + Enter", "skip provider");
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
            Step::Engine => {
                add("Enter", "download");
                add("n", "skip");
            }
            Step::Review => {
                add("Enter", "save & start");
                add("Esc", "back");
            }
        }
        add("Ctrl-C", "cancel");
        h
    }

    fn draw_welcome(&self, f: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::from(""),
            Line::from("  Welcome! This takes about a minute and everything is optional."),
            Line::from("  Nothing is written until the final review.".to_string()),
            Line::from(""),
        ];
        match &self.prereq {
            None => lines.push(Line::styled(
                format!("  {} checking your system…", spinner(self.tick)),
                Style::new().fg(WARN),
            )),
            Some(p) => {
                let ok = |s: String| {
                    Line::from(vec![
                        Span::styled("  ✓ ", Style::new().fg(OK)),
                        Span::raw(s),
                    ])
                };
                let warn = |s: String| {
                    Line::from(vec![
                        Span::styled("  ! ", Style::new().fg(WARN)),
                        Span::raw(s),
                    ])
                };
                lines.push(ok(format!("Platform: {} / {}", p.os, p.arch)));
                lines.push(ok("Rune home:".into()));
                lines.push(Line::styled(
                    format!("      {}", p.home.display()),
                    Style::new().fg(DIM),
                ));
                if self.ws.config_exists() {
                    lines.push(warn(
                        "Existing config found — new entries will be merged, nothing is removed."
                            .into(),
                    ));
                }
                lines.push(match &p.gpu {
                    Some(g) => ok(format!("GPU: {} ({})", g.name, human_bytes(g.total_bytes))),
                    None if p.os == "macos" && p.arch == "aarch64" => {
                        ok("GPU: Apple Silicon (Metal)".into())
                    }
                    None => {
                        warn("No NVIDIA GPU detected — models will run on the CPU (slower).".into())
                    }
                });
                lines.push(match &p.ffmpeg {
                    Some(v) => ok(v.clone()),
                    None => warn("FFmpeg not found — video input will be unavailable. Install it from ffmpeg.org.".into()),
                });
                lines.push(if p.llama_server_installed {
                    ok("llama-server engine is installed".into())
                } else {
                    warn(
                        "llama-server engine not installed yet — we can download it for you later."
                            .into(),
                    )
                });
            }
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" Welcome ")),
            area,
        );
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
                Block::bordered()
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
                Block::bordered()
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
                    Line::from(
                        "  Route requests to hosted models through the same endpoint (optional).",
                    ),
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
                                "[x] "
                            } else {
                                "[ ] "
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
                    Paragraph::new(lines).block(panel(" Cloud providers (optional) ")),
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
                let [key, model, status] = Layout::vertical([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(3),
                ])
                .areas(inner);
                let editing = matches!(self.validation, Validation::Idle);
                render_input(
                    f,
                    key,
                    &self.key_input,
                    true,
                    editing && self.cloud_focus == 0,
                    &format!(" API key ({}) ", preset.env_var),
                );
                render_input(
                    f,
                    model,
                    &self.model_input,
                    false,
                    editing && self.cloud_focus == 1,
                    " Model id ",
                );
                let line = match &self.validation {
                    Validation::Idle => Line::styled(
                        "  Leave the key empty and press Enter to skip this provider. Enter on the model tests the key.",
                        Style::new().fg(DIM),
                    ),
                    Validation::Running(_) => Line::styled(
                        format!("  {} testing key…", spinner(self.tick)),
                        Style::new().fg(WARN),
                    ),
                    Validation::Ok => Line::styled(
                        "  ✓ Key accepted.  Enter = continue   r = re-enter",
                        Style::new().fg(OK),
                    ),
                    Validation::Failed(e) => Line::styled(
                        format!("  ✗ Could not verify: {e}.  Enter = keep anyway   r = re-enter"),
                        Style::new().fg(ERR),
                    ),
                };
                f.render_widget(Paragraph::new(line).wrap(Wrap { trim: true }), status);
            }
        }
    }

    fn draw_local(&self, f: &mut Frame, area: Rect) {
        match self.local_phase {
            LocalPhase::Path => {
                let block = panel(" Local models (optional) ");
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
                            if self.picks[i] { "[x] " } else { "[ ] " },
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
        let mut lines = vec![Line::from("")];
        match &self.engine {
            EnginePhase::Ask => {
                lines.push(Line::from(
                    "  Local GGUF models run on the llama.cpp `llama-server` engine.",
                ));
                lines.push(Line::from(
                    "  Download it now so your first request doesn't have to wait?",
                ));
                lines.push(Line::styled(
                    "  (Skip to download automatically on first use instead.)",
                    Style::new().fg(DIM),
                ));
            }
            EnginePhase::Installing(_) => {
                lines.push(Line::styled(
                    format!(
                        "  {} downloading llama-server… this can take a minute",
                        spinner(self.tick)
                    ),
                    Style::new().fg(WARN),
                ));
                lines.push(Line::from(""));
                for l in &self.log_tail {
                    lines.push(Line::styled(format!("  {l}"), Style::new().fg(DIM)));
                }
            }
            EnginePhase::Done(Ok(p)) => {
                lines.push(Line::styled(
                    format!("  ✓ Installed at {}", p.display()),
                    Style::new().fg(OK),
                ));
                lines.push(Line::styled(
                    "  Press Enter to continue.",
                    Style::new().fg(DIM),
                ));
            }
            EnginePhase::Done(Err(e)) => {
                lines.push(Line::styled(
                    format!("  ✗ Download failed: {e}"),
                    Style::new().fg(ERR),
                ));
                lines.push(Line::from("  Not fatal — it will retry on the first request (check your network or set GH_TOKEN)."));
                lines.push(Line::styled(
                    "  Press Enter to continue.",
                    Style::new().fg(DIM),
                ));
            }
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" Engine ")),
            area,
        );
    }

    fn draw_review(&self, f: &mut Frame, area: Rect) {
        let p = self.plan();
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
        if p.cloud.is_empty() && p.local.is_empty() {
            lines.push(bullet("No models yet — add them later by editing the config or re-running `rune-infer setup`.".into()));
        }
        for c in &p.cloud {
            lines.push(bullet(format!("{} → {}", c.preset.label, c.upstream_model)));
        }
        for c in &p.local {
            lines.push(bullet(format!(
                "Local: {}  ({} · {})",
                c.display_name,
                c.kind_label,
                c.engine()
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
    let block = Block::bordered()
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
        for (cols, rows) in [(30, 8), (60, 20), (71, 30), (140, 50)] {
            let mut t = Terminal::new(TestBackend::new(cols, rows)).unwrap();
            t.draw(|f| w.draw(f)).unwrap();
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

        // Welcome -> Network
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Network);
        render_all_sizes(&w);
        // Network -> Cloud (defaults: localhost, port 8080, generated key)
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Cloud);
        assert!(w.server_key.is_some());
        render_all_sizes(&w);
        // Select Gemini, open details, empty key => skipped, lands on Local
        w.on_key(key(KeyCode::Char(' ')));
        w.on_key(key(KeyCode::Enter));
        assert!(w.cloud_phase == CloudPhase::Details);
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert!(w.step == Step::Local && w.cloud_done.is_empty());
        render_all_sizes(&w);
        // Local: scan a folder, pick everything
        type_str(&mut w, &model_dir.to_string_lossy());
        w.on_key(key(KeyCode::Enter));
        assert!(w.local_phase == LocalPhase::Pick, "{:?}", w.local_error);
        render_all_sizes(&w);
        w.on_key(key(KeyCode::Enter));
        assert_eq!(w.chosen.len(), 1);
        // Empty path => continue (prerequisites unknown in test => Review)
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
        w.on_key(key(KeyCode::Enter)); // -> Network
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
