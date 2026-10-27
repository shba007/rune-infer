//! Terminal UI: live dashboard and first-run setup wizard.
//!
//! Enabled by the default `tui` cargo feature. Library users who only want the
//! HTTP server can depend on this crate with `default-features = false`.

mod dashboard;
mod theme;
mod wizard;

pub use dashboard::{DashboardInfo, ModelRow, run_dashboard, run_dashboard_with_state};
pub use wizard::{WizardOutcome, run_wizard};

use ratatui::DefaultTerminal;

/// Enter the alternate screen / raw mode. A panic hook that restores the
/// terminal is installed so a crash never leaves the shell unusable.
pub fn init_terminal() -> DefaultTerminal {
    ratatui::init()
}

pub fn restore_terminal() {
    ratatui::restore();
}

/// Whether both stdin and stdout are interactive terminals.
pub fn is_interactive() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}
