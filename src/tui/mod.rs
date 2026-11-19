mod dashboard;
mod theme;
mod wizard;

pub use dashboard::{DashboardInfo, ModelRow, run_dashboard, run_dashboard_with_state};
pub use wizard::{WizardOutcome, run_wizard};

use ratatui::DefaultTerminal;

pub fn init_terminal() -> DefaultTerminal {
    ratatui::init()
}

pub fn restore_terminal() {
    ratatui::restore();
}

pub fn is_interactive() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}
