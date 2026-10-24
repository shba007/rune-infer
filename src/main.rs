#![allow(clippy::too_many_arguments, clippy::collapsible_if)]

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};

use rune_infer::config::ModelRegistry;
use rune_infer::home::Workspace;

use tracing_subscriber::filter::{EnvFilter, filter_fn};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

mod self_update;

#[derive(Parser, Debug)]
#[command(name = "rune-infer")]
#[command(author = "Rune Infer Team")]
#[command(before_help = rune_infer::CLI_LOGO)]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "A standalone model server with OpenAI-compatible API and Cloud AI Gateway")]
#[command(after_help = "Data lives in ~/.rune/infer (override with RUNE_HOME). \
Precedence: --config, RUNE_HOME, local ./config (development checkouts), default home.")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Configuration directory, server.json, catalog.json or legacy models.json (overrides RUNE_HOME)
    #[arg(short, long, global = true)]
    config: Option<String>,

    #[arg(long, global = true)]
    host: Option<String>,

    #[arg(short, long, global = true)]
    port: Option<u16>,

    /// Max models kept loaded at once, 0 = unlimited [default: from config]
    #[arg(long, global = true)]
    max_loaded_models: Option<usize>,

    /// Unload a model after this many idle seconds, 0 = never [default: from config]
    #[arg(long, global = true)]
    idle_timeout: Option<u64>,

    /// Run as a plain server without the terminal dashboard (implied when there is no TTY)
    #[arg(long, global = true)]
    headless: bool,

    /// Allow binding to a non-localhost address without an API key (INSECURE)
    #[arg(long, global = true)]
    insecure_no_auth: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Update this executable from a verified release archive
    SelfUpdate {
        /// Check for an update without replacing the executable
        #[arg(long)]
        check: bool,
        /// Install a specific stable version (default: latest, or RUNE_VERSION)
        #[arg(long)]
        version: Option<String>,
    },
    /// Start the inference server (dashboard on a TTY; --headless for background use)
    Serve,
    /// Run the interactive setup wizard (API keys, local models, engines)
    Setup,
    /// Upgrade a schema 1 models.json to the current schema (keeps a .v1.bak backup)
    Migrate {
        /// Show what would change without writing anything
        #[arg(long, global = true)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.command.is_none() {
        Args::command().print_help()?;
        println!();
        return Ok(());
    }
    if let Some(Command::SelfUpdate { check, version }) = &args.command {
        return self_update::run(*check, version.as_deref()).await;
    }

    let ws =
        Workspace::resolve(args.config.as_deref()).context("Cannot determine working folder")?;

    if let Some(Command::Migrate { dry_run }) = &args.command {
        return rune_infer::migrate::run(&ws.legacy_config_path, *dry_run);
    }

    ws.enter()
        .with_context(|| format!("Cannot use folder {}", ws.root.display()))?;
    let env_result = dotenvy::from_path(ws.env_file()).map(|_| ws.env_file());

    #[cfg(feature = "tui")]
    let tui_mode = !args.headless && rune_infer::tui::is_interactive();
    #[cfg(not(feature = "tui"))]
    let tui_mode = false;

    let wants_setup = matches!(args.command, Some(Command::Setup));
    let needs_setup = wants_setup || (!ws.config_exists() && !ws.legacy_config_exists());

    if !ws.config_exists() && ws.legacy_config_exists() {
        anyhow::bail!(
            "Found legacy configuration at {}.\n\
             Schema 3 splits configuration into server.json and catalog.json.\n\
             Run `rune-infer migrate` to automatically upgrade your configuration.",
            ws.legacy_config_path.display()
        );
    }

    if needs_setup && !tui_mode {
        if wants_setup {
            anyhow::bail!("`rune-infer setup` needs an interactive terminal.");
        }
        anyhow::bail!(
            "No configuration found at {} and {}.\n\
             Run `rune-infer setup` in a terminal to create one, or copy an example configuration there.",
            ws.server_path.display(),
            ws.catalog_path.display()
        );
    }

    std::fs::create_dir_all("logs").context("Failed to create 'logs' directory")?;
    let file_appender = tracing_appender::rolling::never("logs", "rune-infer.log");

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "rune_infer=info,audit=info,llama_server=info,nemo_speech=info,tower_http=info",
        )
    });

    // Persistent file log: records everything, including child process output.
    let file_layer = fmt::layer()
        .with_writer(file_appender)
        .with_ansi(false)
        .with_target(true);

    // Colored console log only when there is no dashboard to draw over.
    let stdout_layer = (!tui_mode).then(|| {
        fmt::layer()
            .with_writer(std::io::stdout)
            .with_ansi(true)
            .with_filter(filter_fn(|meta| meta.target() != "llama_server"))
    });

    // Structured events for the dashboard / wizard.
    let (ui_tx, ui_rx) = tokio::sync::mpsc::unbounded_channel();
    let ui_layer = tui_mode.then(|| rune_infer::telemetry::EventLayer::new(ui_tx));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(file_layer)
        .with(stdout_layer)
        .with(ui_layer)
        .init();

    tracing::info!("Starting Rune Infer v{}...", env!("CARGO_PKG_VERSION"));
    tracing::info!("Working folder: {}", ws.root.display());
    rune_infer::inference::process::cleanup_orphaned_pids();

    match env_result {
        Ok(path) => tracing::info!(
            "✓ Loaded environment configurations from {}",
            path.display()
        ),
        Err(e) if e.not_found() => {
            tracing::info!("No .env file; using system environment variables")
        }
        Err(e) => {
            tracing::warn!("Failed to load .env file: {e}; continuing with system environment")
        }
    }

    run(args, ws, needs_setup, tui_mode, ui_rx).await
}

#[cfg(feature = "tui")]
fn finish_terminal(active: bool) {
    if active {
        rune_infer::tui::restore_terminal();
    }
}

#[cfg(not(feature = "tui"))]
fn finish_terminal(_active: bool) {}

async fn run(
    args: Args,
    ws: Workspace,
    needs_setup: bool,
    tui_mode: bool,
    #[cfg_attr(not(feature = "tui"), allow(unused_mut, unused_variables))]
    mut ui_rx: tokio::sync::mpsc::UnboundedReceiver<rune_infer::telemetry::UiEvent>,
) -> Result<()> {
    #[cfg(feature = "tui")]
    let mut terminal = tui_mode.then(rune_infer::tui::init_terminal);

    #[cfg(feature = "tui")]
    if needs_setup {
        if let Some(term) = terminal.as_mut() {
            let outcome =
                tokio::task::block_in_place(|| rune_infer::tui::run_wizard(term, &ws, &mut ui_rx));
            match outcome {
                Ok(rune_infer::tui::WizardOutcome::Completed) => {
                    // Pick up any keys the wizard just saved.
                    let _ = dotenvy::from_path_override(ws.env_file());
                }
                Ok(rune_infer::tui::WizardOutcome::Cancelled) => {
                    finish_terminal(true);
                    println!("Setup cancelled. Any downloaded engines are kept.");
                    return Ok(());
                }
                Err(e) => {
                    finish_terminal(true);
                    return Err(e);
                }
            }
        }
    }
    #[cfg(not(feature = "tui"))]
    let _ = needs_setup;

    // Everything below may fail with a user-facing message; restore the
    // terminal first so that message is actually readable.
    let prepared = prepare(&args, &ws).await;
    let (config, listener, addr) = match prepared {
        Ok(p) => p,
        Err(e) => {
            finish_terminal(tui_mode);
            return Err(e);
        }
    };

    let auth_enabled = config
        .server
        .api_key
        .clone()
        .or_else(|| std::env::var("RUNE_API_KEY").ok())
        .or_else(|| std::env::var("API_KEY").ok())
        .is_some_and(|k| !k.trim().is_empty());
    let host = addr.rsplit_once(':').map(|(h, _)| h).unwrap_or("");
    let exposed = !matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]");

    if exposed && !auth_enabled {
        let allow_insecure = args.insecure_no_auth
            || std::env::var("RUNE_INSECURE_NO_AUTH")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

        if !allow_insecure {
            finish_terminal(tui_mode);
            anyhow::bail!(
                "SECURITY ERROR: Server is configured to listen on '{addr}' without an API key.\n\
                 Anyone on your network or internet can access your models and resources.\n\
                 To fix:\n\
                   1. Bind to localhost: set server.host to \"127.0.0.1\" (or pass --host 127.0.0.1)\n\
                   2. Set an API key: set server.api_key (e.g. \"{{{{RUNE_API_KEY}}}}\") or RUNE_API_KEY in .env\n\
                   3. Explicitly override (UNSAFE): pass --insecure-no-auth"
            );
        }
        tracing::warn!(
            "Listening on {addr} with NO API key (--insecure-no-auth active) — anyone on the network can use your models."
        );
    }
    tracing::info!("Server listening on {addr}");

    #[cfg(feature = "tui")]
    let dashboard_data = rune_infer::tui::DashboardInfo {
        version: env!("CARGO_PKG_VERSION"),
        addr: addr.clone(),
        auth_enabled,
        exposed_to_network: exposed,
        config_path: ws.catalog_path.clone(),
        models: config
            .models
            .iter()
            .map(|m| rune_infer::tui::ModelRow {
                id: m.id.clone(),
                name: m.name.clone(),
                modality: m.modality.as_str().to_string(),
                provider: (!m.is_local()).then(|| m.provider.as_str().to_string()),
                engine: m
                    .is_local()
                    .then(|| m.engine_plan()[0].as_str().to_string()),
                locality: if m.is_local() {
                    "LOCAL".to_string()
                } else {
                    "CLOUD".to_string()
                },
                capabilities: m.capabilities.clone().unwrap_or_default(),
            })
            .collect(),
    };

    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(rune_infer::server::serve_with_paths(
        config,
        Some(ws.server_path.clone()),
        Some(ws.catalog_path.clone()),
        listener,
        async {
            let _ = stop_rx.await;
        },
    ));

    #[cfg(feature = "tui")]
    if let Some(term) = terminal.as_mut() {
        let result = tokio::task::block_in_place(|| {
            rune_infer::tui::run_dashboard(term, dashboard_data, ui_rx)
        });
        finish_terminal(true);
        let _ = stop_tx.send(());
        shutdown(server).await;
        println!("Rune Infer stopped.");
        return result;
    }

    // Headless: run until Ctrl-C or SIGTERM (or the server dies).
    let mut server = server;
    tokio::select! {
        _ = wait_for_shutdown_signal() => {
            tracing::info!("Graceful shutdown initiated...");
            let _ = stop_tx.send(());
            shutdown(server).await;
        }
        res = &mut server => {
            res.context("server task panicked")??;
        }
    }
    tracing::info!("Server stopped");
    Ok(())
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received SIGINT (Ctrl-C) shutdown signal");
            }
            _ = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {
                tracing::info!("Received SIGTERM shutdown signal");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received Ctrl-C shutdown signal");
    }
}

async fn shutdown(server: tokio::task::JoinHandle<Result<()>>) {
    // In-flight streams can hold the graceful shutdown open; allow a sensible 10s drain timeout.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(10), server).await;
    // Clean up any remaining registered child engine processes on shutdown
    rune_infer::inference::process::cleanup_orphaned_pids();
}

async fn prepare(
    args: &Args,
    ws: &Workspace,
) -> Result<(ModelRegistry, tokio::net::TcpListener, String)> {
    let mut config =
        ModelRegistry::from_files(&ws.server_path, &ws.catalog_path).with_context(|| {
            format!(
                "Failed to load config from {} and {}",
                ws.server_path.display(),
                ws.catalog_path.display()
            )
        })?;
    if let Some(n) = args.max_loaded_models {
        config.server.max_loaded_models = n;
    }
    if let Some(secs) = args.idle_timeout {
        config.server.idle_unload_seconds = secs;
    }

    let host = args
        .host
        .clone()
        .unwrap_or_else(|| config.server.host.clone());
    let port = args.port.unwrap_or(config.server.port);
    let listener = rune_infer::server::bind(&host, port).await?;
    Ok((config, listener, format!("{host}:{port}")))
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    #[test]
    fn serve_accepts_global_server_options() {
        let args =
            Args::try_parse_from(["rune-infer", "serve", "--headless", "--port", "8089"]).unwrap();
        assert!(matches!(args.command, Some(Command::Serve)));
        assert!(args.headless);
        assert_eq!(args.port, Some(8089));
        let args = Args::try_parse_from(["rune-infer", "--headless", "serve"]).unwrap();
        assert!(matches!(args.command, Some(Command::Serve)));
        assert!(args.headless);
    }
}
