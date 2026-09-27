use anyhow::{Context, Result};
use clap::Parser;
use std::io::Write;
use std::sync::{Arc, Mutex};
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

#[derive(Clone)]
struct LogWriter {
    file: Arc<Mutex<std::fs::File>>,
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stdout().write(buf);
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(buf);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let _ = std::io::stdout().flush();
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogWriter {
    type Writer = LogWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let env_result = dotenvy::dotenv();

    std::fs::create_dir_all("logs").context("Failed to create 'logs' directory")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/rune-infer.log")
        .context("Failed to open 'logs/rune-infer.log'")?;

    let log_writer = LogWriter {
        file: Arc::new(Mutex::new(log_file)),
    };

    tracing_subscriber::fmt()
        .with_writer(log_writer)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rune_infer=info".parse().unwrap())
                .add_directive("audit=info".parse().unwrap()),
        )
        .init();

    tracing::info!("Starting Rune Infer...");
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
