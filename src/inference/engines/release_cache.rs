//! Public GitHub release metadata only; credentials and engine packages are never cached here.
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    error::Error,
    io::{Read, Write},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

const TTL: u64 = 3600;
const MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    checked_at: u64,
    status: u16,
    value: Option<Value>,
}

#[derive(Clone, Serialize, Deserialize)]
struct Backoff {
    until: u64,
}

struct Reply {
    status: u16,
    headers: reqwest::header::HeaderMap,
    value: Option<Value>,
    checked_at: Option<u64>,
}

struct Outcome {
    entry: Entry,
    cached: bool,
    warning: Option<String>,
}

#[derive(Default)]
struct CheckNote {
    oldest_cached: Option<u64>,
    warning: Option<String>,
}

thread_local! {
    static NOTE: std::cell::RefCell<CheckNote> = std::cell::RefCell::new(CheckNote::default());
}

pub(crate) fn begin_check() {
    NOTE.with(|note| *note.borrow_mut() = CheckNote::default());
}

#[cfg(feature = "tui")]
pub(crate) fn check_note() -> Option<String> {
    NOTE.with(|note| {
        let note = note.borrow();
        let cached = note.oldest_cached.map(|checked| {
            let seconds = now().saturating_sub(checked);
            let age = if seconds < 60 {
                format!("{seconds}s")
            } else if seconds < 3600 {
                format!("{}m", seconds / 60)
            } else {
                format!("{}h", seconds / 3600)
            };
            format!("Cached release metadata · checked {age} ago")
        });
        match (cached, &note.warning) {
            (Some(cached), Some(warning)) => Some(format!("{cached} · {warning}")),
            (Some(cached), None) => Some(cached),
            (None, warning) => warning.clone(),
        }
    })
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn rate_message(until: u64) -> String {
    let timestamp = i64::try_from(until)
        .ok()
        .and_then(|until| chrono::DateTime::from_timestamp(until, 0))
        .map(|time| time.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| until.to_string());
    format!("GitHub rate limited until {timestamp}")
}

struct Store {
    root: Option<PathBuf>,
    entries: HashMap<String, Entry>,
    backoffs: HashMap<String, Backoff>,
}

impl Store {
    fn new(root: Option<PathBuf>) -> Self {
        Self {
            root,
            entries: HashMap::new(),
            backoffs: HashMap::new(),
        }
    }

    fn read<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        let file = std::fs::File::open(self.root.as_ref()?.join(format!("{key}.json"))).ok()?;
        if file.metadata().ok()?.len() > MAX_BYTES {
            return None;
        }
        serde_json::from_reader(file.take(MAX_BYTES)).ok()
    }

    fn save(&self, key: &str, value: &impl Serialize) {
        let Some(root) = &self.root else { return };
        let result = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(root)?;
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let temporary = root.join(format!(
                ".{key}-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            let result = (|| {
                file.write_all(&serde_json::to_vec(value)?)?;
                drop(file);
                std::fs::rename(&temporary, root.join(format!("{key}.json")))
            })();
            if result.is_err() {
                let _ = std::fs::remove_file(temporary);
            }
            result
        })();
        if let Err(error) = result {
            tracing::warn!("Cannot persist engine release cache: {error}");
        }
    }

    fn get(
        &mut self,
        url: &str,
        scope: &str,
        now: u64,
        fetch: impl FnOnce() -> Result<Reply, String>,
    ) -> Result<Outcome, String> {
        let key = hex::encode(Sha256::digest(url.as_bytes()));
        let quota_key = format!("quota-{scope}");
        if self
            .entries
            .get(&key)
            .is_none_or(|entry| now.saturating_sub(entry.checked_at) >= TTL)
        {
            if let Some(entry) = self.read::<Entry>(&key).filter(|entry| {
                matches!(entry.status, 200 | 404)
                    && entry.checked_at <= now
                    && (entry.status == 404 || entry.value.is_some())
            }) {
                if self
                    .entries
                    .get(&key)
                    .is_none_or(|existing| entry.checked_at >= existing.checked_at)
                {
                    self.entries.insert(key.clone(), entry);
                }
            }
        }
        if let Some(backoff) = self.read::<Backoff>(&quota_key) {
            if self
                .backoffs
                .get(scope)
                .is_none_or(|existing| backoff.until > existing.until)
            {
                self.backoffs.insert(scope.to_owned(), backoff);
            }
        }
        let cached = self.entries.get(&key).cloned();
        let blocked = self
            .backoffs
            .get(scope)
            .filter(|backoff| backoff.until > now)
            .map(|backoff| rate_message(backoff.until));
        if let Some(entry) = cached
            .as_ref()
            .filter(|entry| now.saturating_sub(entry.checked_at) < TTL)
        {
            return Ok(Outcome {
                entry: entry.clone(),
                cached: true,
                warning: blocked,
            });
        }
        let fallback = |reason: String| {
            cached
                .map(|entry| Outcome {
                    entry,
                    cached: true,
                    warning: Some(reason.clone()),
                })
                .ok_or(reason)
        };
        if let Some(reason) = blocked {
            return fallback(reason);
        }
        let reply = match fetch() {
            Ok(reply) => reply,
            Err(error) => return fallback(error),
        };
        let header = |name: &str| {
            reply
                .headers
                .get(name)
                .and_then(|value| value.to_str().ok())
        };
        let exhausted = header("x-ratelimit-remaining") == Some("0");
        let limited = reply.status == 429
            || exhausted
            || (reply.status == 403 && header("retry-after").is_some());
        let warning = if limited {
            let reset = if exhausted {
                header("x-ratelimit-reset").and_then(|value| value.parse::<u64>().ok())
            } else {
                None
            };
            let retry = header("retry-after").and_then(|value| {
                value
                    .parse::<u64>()
                    .ok()
                    .map(|delay| now.saturating_add(delay))
                    .or_else(|| {
                        chrono::DateTime::parse_from_rfc2822(value)
                            .ok()
                            .and_then(|time| u64::try_from(time.timestamp()).ok())
                    })
            });
            let until = reset
                .into_iter()
                .chain(retry)
                .max()
                .filter(|until| *until > now)
                .unwrap_or(now.saturating_add(60));
            let backoff = Backoff { until };
            self.save(&quota_key, &backoff);
            self.backoffs.insert(scope.to_owned(), backoff);
            Some(rate_message(until))
        } else {
            None
        };
        if matches!(reply.status, 200 | 404) && (reply.status == 404 || reply.value.is_some()) {
            let entry = Entry {
                checked_at: reply.checked_at.unwrap_or(now),
                status: reply.status,
                value: reply.value,
            };
            self.save(&key, &entry);
            self.entries.insert(key, entry.clone());
            return Ok(Outcome {
                entry,
                cached: false,
                warning,
            });
        }
        fallback(warning.unwrap_or_else(|| {
            if reply.status == 200 {
                "GitHub returned invalid release metadata".into()
            } else {
                format!("GitHub release check failed (HTTP {})", reply.status)
            }
        }))
    }
}

fn credentials() -> (Option<String>, String) {
    let token = ["GITHUB_TOKEN", "GH_TOKEN"]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok())
        .map(|value| value.trim().to_owned())
        .find(|value| !value.is_empty());
    let scope = token
        .as_ref()
        .map(|token| hex::encode(Sha256::digest(token.as_bytes())))
        .unwrap_or_else(|| "anonymous".into());
    (token, scope)
}

fn cache_root() -> Option<PathBuf> {
    crate::home::rune_home()
        .ok()
        .map(|root| root.join("cache/engine-releases"))
}

fn remember(outcome: Outcome) -> Option<Value> {
    NOTE.with(|note| {
        let mut note = note.borrow_mut();
        if outcome.cached {
            note.oldest_cached = Some(
                note.oldest_cached
                    .map_or(outcome.entry.checked_at, |oldest| {
                        oldest.min(outcome.entry.checked_at)
                    }),
            );
        }
        if outcome.warning.is_some() {
            note.warning = outcome.warning;
        }
    });
    outcome.entry.value.filter(|_| outcome.entry.status != 404)
}

/// Save complete successful resolutions too, so a partial failed refresh cannot
/// replace a known release with "Check failed" or refetch build-pointer files.
pub(crate) fn resolved_release(
    key: &str,
    work: impl FnOnce() -> Result<Value, Box<dyn Error>>,
) -> Result<Value, Box<dyn Error>> {
    begin_check();
    let (_, scope) = credentials();
    static RESOLVED: OnceLock<Mutex<Store>> = OnceLock::new();
    let store = RESOLVED.get_or_init(|| Mutex::new(Store::new(cache_root())));
    let outcome = store
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&format!("resolved:{key}"), &scope, now(), || {
            let value = work().map_err(|error| error.to_string())?;
            let checked_at = NOTE.with(|note| note.borrow().oldest_cached);
            Ok(Reply {
                status: 200,
                headers: Default::default(),
                value: Some(value),
                checked_at,
            })
        })?;
    remember(outcome).ok_or_else(|| "Cached release metadata is missing".into())
}

pub(crate) fn github_metadata(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<Option<Value>, Box<dyn Error>> {
    if !url.starts_with("https://api.github.com/repos/") {
        return Err("Invalid GitHub release metadata source".into());
    }
    let (token, scope) = credentials();
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    let store = STORE.get_or_init(|| Mutex::new(Store::new(cache_root())));
    let outcome = store
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(url, &scope, now(), || {
            let mut request = client
                .get(url)
                .header("Accept", "application/vnd.github+json");
            if let Some(token) = token {
                request = request.bearer_auth(token);
            }
            let response = request
                .send()
                .map_err(|error| format!("Cannot reach GitHub release source: {error}"))?;
            let status = response.status().as_u16();
            let headers = response.headers().clone();
            let value: Option<Value> = serde_json::from_reader(response.take(MAX_BYTES)).ok();
            let value = value.filter(|value| value.is_object() || value.is_array());
            Ok(Reply {
                status,
                headers,
                value,
                checked_at: None,
            })
        })?;
    Ok(remember(outcome))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "rune-release-cache-{}",
                crate::setup::generate_api_key()
            )))
        }
        fn store(&self) -> Store {
            Store::new(Some(self.0.clone()))
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn reply(status: u16, headers: &[(&str, &str)]) -> Reply {
        Reply {
            status,
            headers: headers
                .iter()
                .map(|(key, value)| {
                    (
                        reqwest::header::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                        value.parse().unwrap(),
                    )
                })
                .collect(),
            value: (status == 200).then(|| serde_json::json!({"tag_name":"v1.0.0"})),
            checked_at: None,
        }
    }

    #[test]
    fn persistent_cache_reuses_metadata_across_restarts_and_expires_without_losing_it() {
        let directory = Directory::new();
        let mut store = directory.store();
        assert!(
            !store
                .get("release", "anonymous", 1000, || Ok(reply(200, &[])))
                .unwrap()
                .cached
        );
        let cached = directory
            .store()
            .get("release", "anonymous", 1001, || {
                panic!("restart must use cache")
            })
            .unwrap();
        assert!(cached.cached && cached.entry.checked_at == 1000);
        assert!(
            !directory
                .store()
                .get("release", "anonymous", 1000 + TTL, || Ok(reply(200, &[])))
                .unwrap()
                .cached
        );
        let stale = directory
            .store()
            .get("release", "anonymous", 1000 + TTL * 2, || {
                Err("offline".into())
            })
            .unwrap();
        assert!(stale.cached);
        assert_eq!(stale.warning.as_deref(), Some("offline"));
        assert_eq!(stale.entry.value.unwrap()["tag_name"], "v1.0.0");
        assert_eq!(
            store
                .get("missing-latest", "anonymous", 1000, || Ok(reply(404, &[])))
                .unwrap()
                .entry
                .status,
            404
        );
        assert_eq!(
            directory
                .store()
                .get("missing-latest", "anonymous", 1001, || panic!(
                    "404 latest lookup is cached"
                ))
                .unwrap()
                .entry
                .status,
            404
        );
        let key = hex::encode(Sha256::digest(b"release"));
        std::fs::write(directory.0.join(format!("{key}.json")), "invalid JSON").unwrap();
        assert!(
            !directory
                .store()
                .get("release", "anonymous", 1002, || Ok(reply(200, &[])))
                .unwrap()
                .cached
        );
    }

    #[test]
    fn rate_limit_is_shared_persistent_and_keeps_stale_values_until_reset() {
        let directory = Directory::new();
        let mut store = directory.store();
        store
            .get("release", "anonymous", 1000, || Ok(reply(200, &[])))
            .unwrap();
        let stale = store
            .get("release", "anonymous", 5000, || {
                Ok(reply(
                    403,
                    &[
                        ("x-ratelimit-remaining", "0"),
                        ("x-ratelimit-reset", "9000"),
                    ],
                ))
            })
            .unwrap();
        assert!(stale.cached && stale.warning.unwrap().contains("GitHub rate limited until"));
        let mut restarted = directory.store();
        assert!(
            restarted
                .get("another-repository", "anonymous", 5001, || panic!(
                    "quota applies across repositories and restarts"
                ))
                .err()
                .unwrap()
                .contains("GitHub rate limited until")
        );
        assert!(
            restarted
                .get("release", "anonymous", 5001, || panic!(
                    "stale cache remains usable"
                ))
                .unwrap()
                .cached
        );
        assert!(
            !restarted
                .get("another-repository", "authenticated-scope", 5001, || Ok(
                    reply(200, &[])
                ))
                .unwrap()
                .cached
        );
        assert!(
            !restarted
                .get("after-reset", "anonymous", 9000, || Ok(reply(200, &[])))
                .unwrap()
                .cached
        );
        assert!(
            !std::fs::read_to_string(directory.0.join("quota-anonymous.json"))
                .unwrap()
                .contains("token")
        );
        directory
            .store()
            .save("quota-anonymous", &Backoff { until: 11000 });
        assert!(
            restarted
                .get("other-process-limit", "anonymous", 9001, || panic!(
                    "must observe another process's cooldown"
                ))
                .is_err()
        );
        let mut stale_resolution = reply(200, &[]);
        stale_resolution.checked_at = Some(1000);
        let outcome = restarted
            .get("resolved-snapshot", "authenticated-scope", 5000, || {
                Ok(stale_resolution)
            })
            .unwrap();
        assert_eq!(
            outcome.entry.checked_at, 1000,
            "composing cached metadata must not reset its age"
        );
        let retained = restarted
            .get("resolved-snapshot", "authenticated-scope", 5001, || {
                Err("partial refresh failed".into())
            })
            .unwrap();
        assert!(retained.cached);
        assert_eq!(retained.entry.value.unwrap()["tag_name"], "v1.0.0");
    }

    #[test]
    fn retry_after_and_last_successful_response_stop_further_requests() {
        for (status, headers, until) in [
            (
                200,
                vec![
                    ("x-ratelimit-remaining", "0"),
                    ("x-ratelimit-reset", "2000"),
                ],
                2000,
            ),
            (
                429,
                vec![
                    ("x-ratelimit-remaining", "20"),
                    ("x-ratelimit-reset", "9000"),
                    ("retry-after", "90"),
                ],
                1090,
            ),
            (
                403,
                vec![("retry-after", "Thu, 01 Jan 1970 00:20:00 GMT")],
                1200,
            ),
            (429, vec![], 1060),
        ] {
            let mut store = Store::new(None);
            let result = store.get("release", "anonymous", 1000, || Ok(reply(status, &headers)));
            assert_eq!(result.is_ok(), status == 200);
            assert_eq!(store.backoffs["anonymous"].until, until);
            assert!(
                store
                    .get("other", "anonymous", until - 1, || panic!(
                        "must honor reset / Retry-After"
                    ))
                    .is_err()
            );
        }
        let mut store = Store::new(None);
        assert!(
            store
                .get("release", "anonymous", 1000, || Ok(reply(403, &[])))
                .is_err()
        );
        assert!(
            store.backoffs.is_empty(),
            "permission errors must not be called rate limits"
        );
    }
}
