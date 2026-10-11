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
    let inference = Arc::new(InferenceAppState::new(&config));
    let reaper = spawn_idle_reaper(inference.clone(), config.server.idle_unload_seconds);
    let state = AppState {
        inference,
        proxy: Arc::new(ProxyService::new(&config)),
        config,
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
    result
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
