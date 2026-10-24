//! Live dashboard: model lifecycle, VRAM, throughput, requests and logs.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
    Logs,
}

const MAX_LOGS: usize = 2000;
const MAX_REQUESTS: usize = 100;
const HISTORY: usize = 80;

struct App {
    info: DashboardInfo,
    models: Vec<ModelState>,
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

impl App {
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
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('?') | KeyCode::F(1) => self.help = true,
            KeyCode::Tab | KeyCode::Char('l') => {
                self.tab = if self.tab == Tab::Dashboard {
                    Tab::Logs
                } else {
                    Tab::Dashboard
                }
            }
            KeyCode::Char('m') | KeyCode::Char('M') => {
                self.tab = if self.tab == Tab::Models {
                    Tab::Dashboard
                } else {
                    Tab::Models
                }
            }
            KeyCode::Char('1') => self.tab = Tab::Dashboard,
            KeyCode::Char('2') => self.tab = Tab::Models,
            KeyCode::Char('3') => self.tab = Tab::Logs,
            KeyCode::Char('e') => {
                self.errors_only = !self.errors_only;
                self.follow = true;
            }
            KeyCode::Char('c') => {
                self.logs.clear();
                self.log_scroll = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => self.log_scroll += 1,
            KeyCode::PageUp => {
                self.follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(15);
            }
            KeyCode::PageDown => self.log_scroll += 15,
            KeyCode::End | KeyCode::Char('f') => self.follow = true,
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
    mut rx: UnboundedReceiver<UiEvent>,
) -> Result<()> {
    let vram = Arc::new(Mutex::new(None));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    spawn_vram_poller(vram.clone(), stop.clone());

    let mut app = App::new(info, vram);
    while !app.quit {
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
    spans.extend(key_hint("Tab", "dashboard / logs"));
    spans.extend(key_hint(
        "m",
        if app.tab == Tab::Models {
            "dashboard"
        } else {
            "all models"
        },
    ));
    if app.tab == Tab::Logs {
        spans.extend(key_hint("↑↓ PgUp/Dn", "scroll"));
        spans.extend(key_hint("f", "follow"));
        spans.extend(key_hint(
            "e",
            if app.errors_only {
                "show all"
            } else {
                "errors only"
            },
        ));
        spans.extend(key_hint("c", "clear"));
    }
    spans.extend(key_hint("?", "help"));
    spans.extend(key_hint("q", "quit"));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let [top, mid, bottom] = Layout::vertical([
        Constraint::Percentage(42),
        Constraint::Percentage(30),
        Constraint::Percentage(28),
    ])
    .areas(area);
    let [models, stats] =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).areas(top);

    draw_models(f, models, app);
    draw_stats(f, stats, app);
    draw_requests(f, mid, app);
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

    let rows = app.models.iter().map(|m| {
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
            Cell::from(icon).style(Style::new().fg(color)),
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
    f.render_widget(table, area);
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
    let area = centered(f.area(), 60, 14);
    f.render_widget(Clear, area);
    let text = vec![
        Line::from(""),
        Line::from("  Tab / 1 / 2 / 3 switch Dashboard, Models, and Logs"),
        Line::from("  m               toggle full models catalog"),
        Line::from("  ↑ ↓  PgUp PgDn  scroll the log view"),
        Line::from("  f               follow the newest log line"),
        Line::from("  e               errors and warnings only"),
        Line::from("  c               clear logs"),
        Line::from("  q  Esc  Ctrl-C  stop the server and quit"),
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

    fn draw_all(app: &mut App) {
        for (w, h) in [(20, 6), (60, 20), (120, 40)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            for tab in [Tab::Dashboard, Tab::Models, Tab::Logs] {
                app.tab = tab;
                t.draw(|f| draw(f, app)).unwrap();
            }
        }
        app.tab = Tab::Dashboard;
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
        a.on_key(KeyCode::Char('e'), KeyModifiers::NONE);
        for code in [
            KeyCode::Up,
            KeyCode::PageUp,
            KeyCode::PageDown,
            KeyCode::Down,
            KeyCode::End,
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
