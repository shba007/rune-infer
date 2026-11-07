//! Authenticated catalog management for servers with explicit config paths.
use super::AppState;
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub(crate) struct CatalogStore {
    server: PathBuf,
    catalog: PathBuf,
    lock: Mutex<()>,
}
impl CatalogStore {
    pub(crate) fn new(server: PathBuf, catalog: PathBuf) -> Self {
        Self {
            server,
            catalog,
            lock: Mutex::new(()),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct CatalogDocument {
    revision: String,
    catalog: String,
}
type Reply<T> = Result<T, (StatusCode, &'static str)>;

fn store(state: &AppState) -> Reply<&CatalogStore> {
    let configured = state
        .config()
        .server
        .api_key
        .clone()
        .or_else(|| std::env::var("RUNE_API_KEY").ok())
        .or_else(|| std::env::var("API_KEY").ok());
    if configured.as_deref().is_none_or(|s| s.trim().is_empty()) {
        return Err((
            StatusCode::FORBIDDEN,
            "Catalog management requires an API key",
        ));
    }
    state.management.as_deref().ok_or((
        StatusCode::NOT_IMPLEMENTED,
        "Catalog management is unavailable",
    ))
}

fn references_only(raw: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    value
        .get("models")
        .and_then(|v| v.as_array())
        .is_some_and(|models| {
            models.iter().all(|model| {
                let Some(remote) = model.get("remote").filter(|v| !v.is_null()) else {
                    return true;
                };
                remote
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .is_some_and(|key| {
                        key.starts_with("{{")
                            && key.ends_with("}}")
                            && key.len() > 4
                            && !key[2..key.len() - 2].contains(['{', '}'])
                    })
            })
        })
}

pub(super) async fn read(State(state): State<Arc<AppState>>) -> Reply<Json<CatalogDocument>> {
    let store = store(&state)?;
    let _guard = store
        .lock
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Catalog unavailable"))?;
    let bytes = crate::model_registry::read_regular(&store.catalog)
        .ok()
        .flatten()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Cannot read catalog"))?;
    if bytes.len() > 1_048_576 {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "Catalog is too large to edit",
        ));
    }
    let raw = std::str::from_utf8(&bytes)
        .map_err(|_| (StatusCode::UNPROCESSABLE_ENTITY, "Catalog is not UTF-8"))?;
    if !references_only(raw) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Catalog contains inline credentials or invalid JSON",
        ));
    }
    Ok(Json(CatalogDocument {
        revision: crate::catalog_management::revision(&bytes),
        catalog: raw.to_owned(),
    }))
}

pub(super) async fn write(
    State(state): State<Arc<AppState>>,
    Json(document): Json<CatalogDocument>,
) -> Reply<Json<CatalogDocument>> {
    use std::io::Write;
    let store = store(&state)?;
    if document.catalog.len() > 1_048_576 || !references_only(&document.catalog) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid catalog or inline credentials",
        ));
    }
    let _guard = store
        .lock
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Catalog unavailable"))?;
    let current = crate::model_registry::read_regular(&store.catalog)
        .ok()
        .flatten()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Cannot read catalog"))?;
    if crate::catalog_management::revision(&current) != document.revision {
        return Err((
            StatusCode::CONFLICT,
            "Catalog changed; reload before saving",
        ));
    }
    let server = std::fs::read_to_string(&store.server).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Cannot read server configuration",
        )
    })?;
    let env_path = store
        .server
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(".env"));
    let mut variables = std::collections::HashMap::new();
    if let Some(path) = env_path.filter(|p| p.exists()) {
        for entry in dotenvy::from_path_iter(path).map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cannot read Infer environment",
            )
        })? {
            let (key, value) = entry.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Invalid Infer environment",
                )
            })?;
            variables.insert(key, value);
        }
    }
    crate::catalog_management::validate_update(
        &current,
        &document.revision,
        &server,
        &document.catalog,
        &|key| {
            std::env::var(key)
                .ok()
                .or_else(|| variables.get(key).cloned())
        },
    )
    .map_err(|_| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            "Catalog validation failed",
        )
    })?;
    let parent = store
        .catalog
        .parent()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Invalid catalog path"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Clock error"))?
        .as_nanos();
    let temp = parent.join(format!(".catalog-{}-{stamp}.tmp", std::process::id()));
    let backup = parent.join(format!(".catalog-{}-{stamp}.backup", std::process::id()));
    let result = (|| -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(document.catalog.as_bytes())?;
        file.sync_all()?;
        drop(file);
        if crate::model_registry::read_regular(&store.catalog)?.as_deref()
            != Some(current.as_slice())
        {
            anyhow::bail!("catalog changed");
        }
        crate::model_registry::move_no_replace(&store.catalog, &backup)?;
        let publish = (|| -> anyhow::Result<()> {
            anyhow::ensure!(
                crate::model_registry::read_regular(&backup)?.as_deref()
                    == Some(current.as_slice()),
                "catalog changed during save"
            );
            crate::model_registry::move_no_replace(&temp, &store.catalog)?;
            Ok(())
        })();
        if let Err(error) = publish {
            if crate::model_registry::move_no_replace(&backup, &store.catalog).is_err() {
                return Err(error.context(format!(
                    "Previous catalog preserved at {}",
                    backup.display()
                )));
            }
            return Err(error);
        }
        Ok(())
    })();
    if let Err(error) = result {
        tracing::warn!(%error, "Catalog replacement failed");
        let _ = std::fs::remove_file(&temp);
        return Err((
            StatusCode::CONFLICT,
            "Catalog could not be saved; reload and retry",
        ));
    }
    Ok(Json(CatalogDocument {
        revision: crate::catalog_management::revision(document.catalog.as_bytes()),
        catalog: document.catalog,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        extract::Request,
    };
    use tower::ServiceExt;

    #[test]
    fn rejects_inline_provider_credentials() {
        assert!(!references_only(
            r#"{"models":[{"remote":{"api_key":"secret"}}]}"#
        ));
        assert!(references_only(
            r#"{"models":[{"remote":{"api_key":"{{PROVIDER_KEY}}"}}]}"#
        ));
        assert!(!references_only(
            r#"{"models":[{"remote":{"api_key":"{{KEY}}suffix"}}]}"#
        ));
    }

    #[tokio::test]
    async fn management_requires_auth_and_rejects_stale_updates() {
        let dir = std::env::temp_dir().join(format!(
            "infer-catalog-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let server = dir.join("server.json");
        let catalog = dir.join("catalog.json");
        std::fs::write(&server, r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"api_key":"test-key","max_loaded_models":1,"idle_unload_seconds":300}}"#).unwrap();
        let original = r#"{"schema_version":3,"models":[]}"#;
        std::fs::write(&catalog, original).unwrap();
        let mut config = crate::config::ModelRegistry::new();
        config.server.api_key = Some("test-key".into());
        let mut state = AppState::new(
            Arc::new(crate::inference::AppState::new(&config)),
            Arc::new(crate::proxy::ProxyService::new(&config)),
            config,
        );
        state.management = Some(Arc::new(CatalogStore::new(server.clone(), catalog.clone())));
        let router = crate::api::create_router(state);
        let request = |key: bool| {
            let builder = Request::builder().uri("/v1/management/catalog");
            let builder = if key {
                builder.header("authorization", "Bearer test-key")
            } else {
                builder
            };
            builder.body(Body::empty()).unwrap()
        };
        assert_eq!(
            router
                .clone()
                .oneshot(request(false))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
        let response = router.clone().oneshot(request(true)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let document: CatalogDocument =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        std::fs::write(&catalog, format!("{original}\n")).unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/v1/management/catalog")
                    .header("authorization", "Bearer test-key")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&document).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            std::fs::read_to_string(&catalog).unwrap(),
            format!("{original}\n")
        );
        let current = std::fs::read(&catalog).unwrap();
        let updated = CatalogDocument {
            revision: crate::catalog_management::revision(&current),
            catalog: "{\"schema_version\":3,\"models\":[],\"aliases\":{\"alias\":\"example\"}}\n"
                .into(),
        };
        let response = router
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/v1/management/catalog")
                    .header("authorization", "Bearer test-key")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&updated).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(std::fs::read_to_string(&catalog).unwrap(), updated.catalog);
        std::fs::remove_file(server).unwrap();
        std::fs::remove_file(catalog).unwrap();
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(dir).unwrap();
    }
}
