use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

pub const ACCENT: Color = Color::Cyan;
pub const OK: Color = Color::White;
pub const WARN: Color = Color::DarkGray;
pub const ERR: Color = Color::Red;
pub const DIM: Color = Color::DarkGray;

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub const BRAILLE_BAR: ratatui::symbols::bar::Set = ratatui::symbols::bar::Set {
    full: "⣿",
    seven_eighths: "⣶",
    three_quarters: "⣴",
    five_eighths: "⣤",
    half: "⣄",
    three_eighths: "⣀",
    one_quarter: "⡀",
    one_eighth: "⠄",
    empty: " ",
};

pub const ASCII_BAR: ratatui::symbols::bar::Set = ratatui::symbols::bar::Set {
    full: "#",
    seven_eighths: "#",
    three_quarters: "=",
    five_eighths: "=",
    half: "-",
    three_eighths: "-",
    one_quarter: ".",
    one_eighth: ".",
    empty: " ",
};

pub fn is_no_color() -> bool {
    std::env::var_os("NO_COLOR").is_some()
}

pub fn sparkline_bar_set() -> ratatui::symbols::bar::Set<'static> {
    if is_no_color() {
        ASCII_BAR
    } else {
        BRAILLE_BAR
    }
}

pub fn spinner(tick: u64) -> &'static str {
    SPINNER[(tick as usize / 2) % SPINNER.len()]
}

pub fn panel<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(DIM))
        .title(title.into())
        .title_style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
}

pub fn key_hint<'a>(key: &'a str, what: &'a str) -> Vec<Span<'a>> {
    vec![
        Span::styled(format!(" {key} "), Style::new().fg(Color::Black).bg(DIM)),
        Span::styled(format!(" {what}  "), Style::new().fg(DIM)),
    ]
}

/// Center a box of at most `w`×`h` inside `area`.
pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let [v] = Layout::vertical([Constraint::Max(h)])
        .flex(Flex::Center)
        .areas(area);
    let [r] = Layout::horizontal([Constraint::Max(w)])
        .flex(Flex::Center)
        .areas(v);
    r
}

pub fn human_bytes(b: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if b as f64 >= GIB {
        format!("{:.1} GiB", b as f64 / GIB)
    } else {
        format!("{:.0} MiB", b as f64 / MIB)
    }
}

pub fn human_duration(secs: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}",
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}
