use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber;

use rune_infer::api::{AppState, create_router};
use rune_infer::config::ModelRegistry;
use rune_infer::inference::AppState as InferenceAppState;

#[derive(Parser, Debug)]
#[command(name = "rune-infer")]
#[command(author = "Rune Infer Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "A standalone model server with OpenAI-compatible API")]
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

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rune_infer=info".parse().unwrap()),
        )
        .init();

    tracing::info!("Starting Rune Infer...");

    let config_path = std::path::Path::new(&args.config);
    let config = ModelRegistry::from_file(config_path).context("Failed to load config")?;

    let host = args
        .host
        .clone()
        .unwrap_or_else(|| config.server.host.clone());
    let port = args.port.unwrap_or(config.server.port);

    let inference_state = Arc::new(InferenceAppState::new(&config));

    let state = AppState {
        inference: inference_state.clone(),
        config,
    };

    let app = create_router(state.clone());

    let app = app
        .layer(tower::ServiceBuilder::new().layer(tower_http::cors::CorsLayer::permissive()))
        .with_state(Arc::new(state));

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
