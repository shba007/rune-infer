//! Embeddable server entry point shared by the CLI and library users.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::TcpListener;

use crate::api::{AppState, create_router};
use crate::config::ModelRegistry;
use crate::inference::AppState as InferenceAppState;
use crate::proxy::ProxyService;

use std::path::{Path, PathBuf};

/// Bind `host:port`, turning the common failure modes into actionable errors.
pub async fn bind(host: &str, port: u16) -> Result<TcpListener> {
    let addr = format!("{host}:{port}");
    TcpListener::bind(&addr).await.map_err(|e| match e.kind() {
        std::io::ErrorKind::AddrInUse => anyhow::anyhow!(
            "Cannot listen on {addr}: the port is already in use. \
             Stop the other process or choose another port with --port."
        ),
        std::io::ErrorKind::PermissionDenied => anyhow::anyhow!(
            "Cannot listen on {addr}: permission denied (ports below 1024 may need elevated rights)."
        ),
        _ => anyhow::Error::new(e).context(format!("Cannot listen on {addr}")),
    })
}

/// Serve the API on `listener` until `shutdown` resolves.
pub async fn serve(
    config: ModelRegistry,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_with_paths(config, None, None, listener, shutdown).await
}

/// Serve the API with optional configuration paths to watch for live catalog hot-reloading.
pub async fn serve_with_paths(
    config: ModelRegistry,
    server_path: Option<PathBuf>,
    catalog_path: Option<PathBuf>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let inference = Arc::new(InferenceAppState::new(&config));
    let reaper = spawn_idle_reaper(inference.clone(), config.server.idle_unload_seconds);
    let state = AppState::new(inference, Arc::new(ProxyService::new(&config)), config);

    let watcher = match (server_path, catalog_path) {
        (Some(srv), Some(cat)) => Some(spawn_config_watcher(state.clone(), srv, cat)),
        _ => None,
    };

    let app = create_router(state)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower::ServiceBuilder::new().layer(tower_http::cors::CorsLayer::permissive()));

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .context("HTTP server error");
    if let Some(handle) = reaper {
        handle.abort();
    }
    if let Some(handle) = watcher {
        handle.abort();
    }
    result
}

fn get_file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn get_max_mtime(srv: &Path, cat: &Path) -> Option<std::time::SystemTime> {
    match (get_file_mtime(srv), get_file_mtime(cat)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn spawn_config_watcher(
    state: AppState,
    server_path: PathBuf,
    catalog_path: PathBuf,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut last_mtime = get_max_mtime(&server_path, &catalog_path);

        loop {
            interval.tick().await;
            let current_mtime = get_max_mtime(&server_path, &catalog_path);
            if current_mtime.is_some() && current_mtime != last_mtime {
                last_mtime = current_mtime;
                match ModelRegistry::from_files(&server_path, &catalog_path) {
                    Ok(new_config) => {
                        state.reload_catalog(new_config);
                        tracing::info!(
                            "Configuration hot-reloaded successfully from {} and {}",
                            server_path.display(),
                            catalog_path.display()
                        );
                    }
                    Err(err) => {
                        tracing::error!("Failed to reload configuration: {err:#}");
                    }
                }
            }
        }
    })
}

/// Periodically unload models that have not been used for `idle_secs`
/// (0 disables). Models serving a request are never unloaded.
fn spawn_idle_reaper(
    inference: Arc<InferenceAppState>,
    idle_secs: u64,
) -> Option<tokio::task::JoinHandle<()>> {
    if idle_secs == 0 {
        return None;
    }
    let idle = Duration::from_secs(idle_secs);
    let tick = Duration::from_secs((idle_secs / 4).clamp(5, 30));
    Some(tokio::spawn(async move {
        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let state = inference.clone();
            if let Ok(unloaded) = tokio::task::spawn_blocking(move || state.evict_idle(idle)).await
            {
                for id in unloaded {
                    tracing::info!("Unloaded idle model '{id}' (no requests for {idle_secs}s)");
                }
            }
        }
    }))
}
