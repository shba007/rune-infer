//! Validation helpers for editing the raw Infer catalog.
//!
//! These functions do not write files. Callers must serialize writes and
//! compare the revision again at the point where a change is committed.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// A digest of the exact file bytes, including whitespace and placeholders.
pub fn revision(raw: &[u8]) -> String {
    hex::encode(Sha256::digest(raw))
}

/// Validate a proposed catalog without replacing its environment references.
pub fn validate_update(
    current: &[u8],
    expected_revision: &str,
    server_raw: &str,
    catalog_raw: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<()> {
    if revision(current) != expected_revision {
        bail!("Catalog changed; reload it before saving");
    }
    let server = crate::config::interpolate_with(server_raw, lookup)?;
    let catalog = crate::config::interpolate_with(catalog_raw, lookup)?;
    let server: crate::schema::ServerFile =
        serde_json::from_str(&server).context("Invalid server configuration")?;
    let catalog: crate::schema::CatalogFile =
        serde_json::from_str(&catalog).context("Invalid catalog")?;
    if server.schema_version != crate::schema::SCHEMA_VERSION
        || catalog.schema_version != crate::schema::SCHEMA_VERSION
    {
        bail!("Configuration schema version is not supported");
    }
    let models = catalog
        .models
        .into_iter()
        .map(TryInto::try_into)
        .collect::<Result<Vec<_>>>()?;
    let registry = crate::config::ModelRegistry {
        schema_version: crate::schema::SCHEMA_VERSION,
        server: server.server,
        models,
        roles: catalog.roles,
        aliases: catalog.aliases,
    };
    registry.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVER: &str = r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"api_key":null,"max_loaded_models":1,"idle_unload_seconds":300}}"#;
    const CATALOG: &str = r#"{"schema_version":3,"models":[]}"#;

    #[test]
    fn rejects_stale_and_invalid_updates() {
        let rev = revision(CATALOG.as_bytes());
        assert!(validate_update(CATALOG.as_bytes(), &rev, SERVER, CATALOG, &|_| None).is_ok());
        assert!(validate_update(b"changed", &rev, SERVER, CATALOG, &|_| None).is_err());
        assert!(validate_update(CATALOG.as_bytes(), &rev, SERVER, "{}", &|_| None).is_err());
    }

    #[test]
    fn validates_with_environment_without_changing_raw_input() {
        let server = SERVER.replace("127.0.0.1", "{{BIND}}");
        let rev = revision(CATALOG.as_bytes());
        assert!(
            validate_update(CATALOG.as_bytes(), &rev, &server, CATALOG, &|name| {
                (name == "BIND").then(|| "127.0.0.1".to_owned())
            })
            .is_ok()
        );
        assert!(server.contains("{{BIND}}"));
        assert!(validate_update(CATALOG.as_bytes(), &rev, &server, CATALOG, &|_| None).is_err());
    }
}
