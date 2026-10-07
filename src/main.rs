use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;

use rune_infer::api::{AppState, create_router};
use rune_infer::config::ModelRegistry;
use rune_infer::inference::AppState as InferenceAppState;
use rune_infer::proxy::ProxyService;

use tracing_subscriber::filter::{EnvFilter, filter_fn};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "rune-infer")]
#[command(author = "Rune Infer Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "A standalone model server with OpenAI-compatible API and Cloud AI Gateway")]
struct Args {
    #[arg(short, long, default_value = "config/models.json")]
    config: String,

    #[arg(long)]
    host: Option<String>,

    #[arg(short, long)]
    port: Option<u16>,

    #[arg(long, default_value = "1")]
    max_loaded_models: usize,

    #[arg(long, default_value = "300")]
    idle_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let env_result = dotenvy::dotenv();

    std::fs::create_dir_all("logs").context("Failed to create 'logs' directory")?;
    let file_appender = tracing_appender::rolling::never("logs", "rune-infer.log");

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "rune_infer=info,audit=info,llama_server=info,nemo_speech=info,tower_http=info",
        )
    });

    // Layer 1: Clean persistent file logging (records everything, including child process outputs)
    let file_layer = fmt::layer()
        .with_writer(file_appender)
        .with_ansi(false)
        .with_target(true);

    // Layer 2: Colored terminal console logging (filters out verbose llama_server logs)
    let console_filter = filter_fn(|meta| meta.target() != "llama_server");
    let stdout_layer = fmt::layer()
        .with_writer(std::io::stdout)
        .with_ansi(true)
        .with_filter(console_filter);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(file_layer)
        .with(stdout_layer)
        .init();

    tracing::info!("Starting Rune Infer (with Cloud AI Gateway)...");
    tracing::info!("Persistent log audit active at logs/rune-infer.log");

    // Clean up any stale child processes from abnormal previous exits
    rune_infer::inference::process::cleanup_orphaned_pids();

    match env_result {
        Ok(path) => tracing::info!(
            "✓ Loaded environment configurations from {}",
            path.display()
        ),
        Err(e) if e.not_found() => tracing::warn!(
            "No .env file found in root; continuing with system environment variables"
        ),
        Err(e) => tracing::warn!(
            "Failed to load .env file: {e}; continuing with system environment variables"
        ),
    }

    let config_path = std::path::Path::new(&args.config);
    let config = ModelRegistry::from_file(config_path).context("Failed to load config")?;

    let host = args
        .host
        .clone()
        .unwrap_or_else(|| config.server.host.clone());
    let port = args.port.unwrap_or(config.server.port);

    let inference_state = Arc::new(InferenceAppState::new(&config));
    let proxy_service = Arc::new(ProxyService::new(&config));

    let state = AppState {
        inference: inference_state.clone(),
        proxy: proxy_service,
        config,
    };

    let app = create_router(state)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower::ServiceBuilder::new().layer(tower_http::cors::CorsLayer::permissive()));

    let addr = format!("{}:{}", host, port);
    tracing::info!(
        "Server listening on {} (host={} port={} from config)",
        addr,
        host,
        port
    );

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    tracing::info!("Server stopped");

    Ok(())
}
