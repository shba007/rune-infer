//! Shared engine release lookup for the dashboard and management API.
use crate::engine_inventory::{EngineRow, install_engine_kind};
use std::time::Duration;

fn release_url(name: &str) -> Option<&'static str> {
    match name {
        "whisper-cpp" => Some("https://api.github.com/repos/ggml-org/whisper.cpp/releases"),
        _ => None,
    }
}

#[cfg_attr(not(feature = "tui"), allow(dead_code))]
pub(crate) struct AvailableRelease {
    pub(crate) tag: String,
    pub(crate) build_tag: Option<String>,
    pub(crate) registry_versions: Option<std::collections::BTreeMap<u64, u64>>,
    pub(crate) registry_revisions: Option<std::collections::BTreeMap<String, u64>>,
    pub(crate) published_at: Option<String>,
    pub(crate) detail: String,
    pub(crate) whisper_gpu: Option<crate::inference::engines::whisper::PackageIdentity>,
}

pub(crate) fn apply_release(
    engine: &mut EngineRow,
    release: AvailableRelease,
    root: &std::path::Path,
) {
    engine.latest = release.tag;
    engine.build_tag = release.build_tag;
    engine.registry_versions = release.registry_versions;
    engine.registry_revisions = release.registry_revisions;
    engine.published_at = release.published_at;
    let mut whisper_verified = false;
    if engine.name == "whisper-cpp" && engine.installed && engine.managed {
        if let Some(package) = &release.whisper_gpu {
            let base = root.join("bin/whisper-cpp");
            let in_managed_folder = engine
                .path
                .canonicalize()
                .ok()
                .zip(base.canonicalize().ok())
                .is_some_and(|(path, root)| path.starts_with(root));
            whisper_verified = in_managed_folder
                && crate::inference::engines::whisper::installation_matches(&base, package);
            engine.gpu_upgrade =
                in_managed_folder && !whisper_verified && release_status(engine) == "Up to date";
        }
    }
    engine.check_detail = Some(if let Some(package) = release.whisper_gpu {
        format!(
            "{} · GPU package: {} · {}",
            release.detail,
            package.archive,
            if whisper_verified {
                "installed package verified"
            } else if !engine.installed {
                "not installed"
            } else if engine.gpu_upgrade {
                "installed package unverified; verified GPU refresh available"
            } else {
                "installed package unverified; version rules apply"
            }
        )
    } else {
        release.detail
    });
}

pub(crate) fn check_latest_release(name: &str) -> Result<AvailableRelease, String> {
    use crate::inference::engines::download;
    crate::inference::engines::release_cache::begin_check();
    if name == "needle" {
        let runtime = crate::inference::engines::registry::needle_runtime()
            .map_err(|error| error.to_string())?;
        return Ok(AvailableRelease {
            tag: format!("revision {}", runtime.revision),
            build_tag: None,
            registry_versions: None,
            registry_revisions: None,
            published_at: None,
            detail: format!(
                "Reviewed Cactus runtime {} · checksum verified during installation",
                runtime.version
            ),
            whisper_gpu: None,
        });
    }
    let url = if matches!(
        name,
        "llama-server"
            | "llama-server-prism"
            | "nemo-speech"
            | "audio-cpp"
            | "crispasr"
            | "sd-server"
    ) {
        ""
    } else {
        release_url(name).ok_or("No release source is available for this engine")?
    };
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .user_agent(concat!("rune-infer/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|_| "Could not create release client")?;
    let value = if name == "llama-server" {
        crate::inference::engines::registry::llama_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "llama-server-prism" {
        crate::inference::engines::registry::prism_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "nemo-speech" {
        crate::inference::engines::registry::nemo_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "audio-cpp" {
        crate::inference::engines::registry::audio_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "crispasr" {
        crate::inference::engines::registry::crispasr_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "sd-server" {
        crate::inference::engines::registry::sd_release(&client)
            .map_err(|e| format!("Registry check failed: {e}; press r to retry"))?
    } else if name == "whisper-cpp" {
        crate::inference::engines::whisper::resolve_release(&client)
            .map_err(|e| format!("Release check failed: {e}; press r to retry"))?
    } else {
        download::resolve_release(&client, url)
            .map_err(|e| format!("Release check failed: {e}; press r to retry"))?
    };
    let tag = value["tag_name"]
        .as_str()
        .ok_or("Release tag missing")?
        .to_owned();
    let channel = if value["rune_registry"].as_bool() == Some(true) {
        "Reviewed registry build"
    } else if download::is_stable_release(&value) {
        "Stable release"
    } else {
        "Latest published build · no stable release available"
    };
    let mut detail = if let Some(build) = value["rune_build_tag"].as_str() {
        format!("{channel} {tag} · upstream binary build {build}")
    } else {
        format!("{channel} {tag}")
    };
    if let Some(note) = crate::inference::engines::release_cache::check_note() {
        detail.push_str(" · ");
        detail.push_str(&note);
    }
    let published_at = value["rune_build_published_at"]
        .as_str()
        .or_else(|| value["published_at"].as_str())
        .map(str::to_owned);
    Ok(AvailableRelease {
        tag,
        build_tag: value["rune_build_tag"].as_str().map(str::to_owned),
        registry_versions: if name == "llama-server" {
            Some(
                serde_json::from_value(value["rune_registry_versions"].clone())
                    .map_err(|_| "Invalid registry version order")?,
            )
        } else {
            None
        },
        registry_revisions: if matches!(
            name,
            "nemo-speech" | "audio-cpp" | "crispasr" | "sd-server" | "llama-server-prism"
        ) {
            Some(
                serde_json::from_value(value["rune_registry_revisions"].clone())
                    .map_err(|_| "Invalid registry version order")?,
            )
        } else {
            None
        },
        published_at,
        detail,
        whisper_gpu: if name == "whisper-cpp" {
            crate::inference::engines::whisper::platform_asset(&value)
                .and_then(|asset| {
                    crate::inference::engines::whisper::PackageIdentity::from_asset(asset).ok()
                })
                .filter(|package| package.is_gpu())
        } else {
            None
        },
    })
}

pub(crate) fn engine_build_identity(text: &str) -> Option<(u64, Option<&str>)> {
    let (build, commit) = if let Some((_, tail)) = text.split_once("build ") {
        let build = tail.split(|c: char| !c.is_ascii_digit()).next()?;
        let commit = tail.split_once("commit ").map(|(_, commit)| commit);
        (build, commit)
    } else {
        let tail = text
            .strip_prefix("prism-")
            .unwrap_or(text)
            .strip_prefix('b')?;
        tail.split_once('-')
            .map_or((tail, None), |(build, commit)| (build, Some(commit)))
    };
    let commit = commit.and_then(|value| {
        let value = value.split(|c: char| !c.is_ascii_hexdigit()).next()?;
        (value.len() >= 7).then_some(value)
    });
    Some((build.parse().ok()?, commit))
}

pub(crate) fn perform_update(
    state: std::sync::Arc<crate::inference::AppState>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    engine: crate::engine_inventory::EngineRow,
    tx: std::sync::mpsc::Sender<crate::inference::engines::update::UpdateEvent>,
) -> Result<String, String> {
    if matches!(engine.name, "llama-server" | "llama-server-prism") {
        let expected_registry_release = if engine.name == "llama-server-prism" {
            engine
                .registry_revisions
                .as_ref()
                .and_then(|versions| versions.get(&engine.latest).copied())
        } else {
            engine.registry_versions.as_ref().and_then(|versions| {
                engine_build_identity(&engine.latest)
                    .and_then(|(build, _)| versions.get(&build).copied())
            })
        };
        crate::inference::engines::update::update_llama(
            state,
            if engine.name == "llama-server-prism" {
                crate::config::EngineKind::LlamaServerPrism
            } else {
                crate::config::EngineKind::LlamaServer
            },
            cancel,
            engine.latest,
            engine.build_tag,
            expected_registry_release,
            tx.clone(),
        )
    } else if engine.name == "needle" {
        crate::inference::engines::update::update_needle(state, cancel, engine.latest, tx.clone())
    } else if engine.name == "sd-server" {
        let expected_release = engine.registry_revisions.as_ref().and_then(|versions| {
            crate::inference::engines::sd_server::release_commit(&engine.latest)
                .and_then(|commit| versions.get(commit).copied())
        });
        crate::inference::engines::update::update_sd(
            state,
            cancel,
            engine.latest,
            expected_release,
            tx.clone(),
        )
    } else if engine.name == "nemo-speech" {
        let expected_release = engine
            .registry_revisions
            .as_ref()
            .and_then(|versions| versions.get(engine.latest.trim_start_matches('v')).copied());
        crate::inference::engines::update::update_nemo(
            state,
            cancel,
            engine.latest,
            expected_release,
            tx.clone(),
        )
    } else if engine.name == "whisper-cpp" {
        crate::inference::engines::update::update_whisper(state, cancel, engine.latest, tx.clone())
    } else if engine.name == "crispasr" {
        let expected_release = engine
            .registry_revisions
            .as_ref()
            .and_then(|versions| versions.get(engine.latest.trim_start_matches('v')).copied());
        crate::inference::engines::update::update_crispasr(
            state,
            cancel,
            engine.latest,
            expected_release,
            tx.clone(),
        )
    } else {
        let expected_release = engine
            .registry_revisions
            .as_ref()
            .and_then(|versions| versions.get(engine.latest.trim_start_matches('v')).copied());
        crate::inference::engines::update::update_audio(
            state,
            cancel,
            engine.latest,
            expected_release,
            tx.clone(),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum EngineAction {
    Install,
    Update,
    Unavailable(String),
}

pub(crate) fn engine_action(engine: &EngineRow) -> EngineAction {
    engine_action_on(engine, std::env::consts::OS, std::env::consts::ARCH)
}

// Share eligibility between the action hint and dispatch. The installer still
// selects and verifies a compatible package before replacing engine files.
pub(crate) fn engine_action_on(engine: &EngineRow, os: &str, arch: &str) -> EngineAction {
    let unavailable = |reason: &str| EngineAction::Unavailable(reason.into());
    if engine.name == "encoder" {
        return unavailable("Built in; no installation or update required");
    }
    if engine.name == "onnx" {
        if os != "windows" || arch != "x86_64" {
            return unavailable("ONNX Runtime installation currently supports Windows x64 only");
        }
        if !engine.managed {
            return unavailable(
                "External ONNX Runtime: remove ONNX_RUNTIME_LIBRARY_PATH to use the managed installation",
            );
        }
        if !engine.installed || engine.version == "ONNX Runtime 1.31.1 CPU" {
            return EngineAction::Install;
        }
        return if engine.version == "ONNX Runtime 1.31.1 CPU + CUDA" {
            unavailable("Pinned ONNX CPU and CUDA components are installed")
        } else {
            unavailable("Existing ONNX Runtime is unverified; preserve it before reinstalling")
        };
    }
    if engine.name == "litert" {
        if os != "windows" || arch != "x86_64" {
            return unavailable("LiteRT installation currently supports Windows x64 only");
        }
        if !engine.managed {
            return unavailable(
                "External LiteRT runtime: fix or remove LITERT_LIBRARY_PATH to use the managed installation",
            );
        }
        if !engine.installed {
            return EngineAction::Install;
        }
        return if engine.version == "LiteRT 2.3.0 CPU" {
            EngineAction::Install
        } else if engine.version == "LiteRT 2.3.0 GPU + CPU" {
            unavailable("Pinned CPU and GPU runtime components are installed")
        } else {
            unavailable("Existing LiteRT DLL is unverified; preserve it before reinstalling")
        };
    }
    if engine.installed && !engine.managed {
        return unavailable("Externally managed; update using its original installation method");
    }
    if engine.installed {
        let supported = match engine.name {
            "audio-cpp" => os == "windows" && arch == "x86_64",
            "llama-server" | "llama-server-prism" | "sd-server" | "nemo-speech" | "whisper-cpp"
            | "crispasr" | "needle" => os == "windows" && arch == "x86_64",
            _ => return unavailable("Automatic update is not implemented for this engine"),
        };
        if !supported {
            return unavailable("Automatic update is unavailable on this platform");
        }
    } else if install_engine_kind(engine.name).is_none() {
        return unavailable("Installation is unavailable for this engine");
    } else if !matches!(os, "windows" | "linux" | "android" | "macos") {
        return unavailable("Installation is unavailable on this platform");
    }
    match engine.latest.as_str() {
        "Checking…" => {
            return unavailable("Checking release metadata; wait for the check to finish");
        }
        "Rate limited" => {
            return unavailable(engine.check_detail.as_deref().unwrap_or(
                "GitHub rate limited; wait for the reset before checking versions again",
            ));
        }
        "Check failed" => {
            return unavailable(
                engine
                    .check_detail
                    .as_deref()
                    .unwrap_or("Release check failed; press r to check versions again"),
            );
        }
        "" | "Unknown" | "External" | "Not checked" => {
            return unavailable("Check versions before installing or updating");
        }
        _ => {}
    }
    if !engine.installed {
        return EngineAction::Install;
    }
    if engine.name == "needle" && release_status(engine) == "Revision unknown" {
        return unavailable("Cannot verify Needle installation receipt; update refused");
    }
    if engine.name == "sd-server" && release_status(engine) == "Unknown" {
        return unavailable(
            "Cannot verify SD-server commit identities; check versions before updating",
        );
    }
    if matches!(engine.name, "llama-server" | "llama-server-prism")
        && (engine_build_identity(&engine.version).is_none()
            || engine_build_identity(engine.build_tag.as_deref().unwrap_or(&engine.latest))
                .is_none())
    {
        return unavailable(
            "Cannot verify llama-server build identities; check versions before updating",
        );
    }
    if matches!(engine.name, "llama-server" | "llama-server-prism")
        && release_status(engine) == "Unknown"
    {
        return unavailable("Installed llama-server build is not in the registry; update refused");
    }
    if matches!(engine.name, "nemo-speech" | "audio-cpp" | "crispasr")
        && release_status(engine) == "Unknown"
    {
        return unavailable("Installed engine version is not in the registry; update refused");
    }
    match release_status(engine) {
        "Up to date" => unavailable("Already up to date; no download needed"),
        "Ahead" => unavailable("Installed version is newer; no downgrade will be performed"),
        _ => EngineAction::Update,
    }
}

pub(crate) fn release_status(engine: &EngineRow) -> &'static str {
    if engine.name == "encoder" {
        return "Built in · synthetic";
    }
    if !engine.installed {
        return "Not installed";
    }
    if engine.name == "onnx" {
        return if engine.version == "ONNX Runtime 1.31.1 CPU + CUDA" {
            "Ready · CUDA + CPU"
        } else if engine.version == "ONNX Runtime 1.31.1 CPU" {
            "Ready · CPU"
        } else {
            "Unverified runtime"
        };
    }
    if engine.name == "litert" {
        return if engine.version == "LiteRT 2.3.0 GPU + CPU" {
            "Ready · GPU + CPU"
        } else if engine.version == "LiteRT 2.3.0 CPU" {
            "Ready · CPU"
        } else {
            "Unverified runtime"
        };
    }
    if engine.name == "needle" {
        if engine.version.starts_with("revision ") && engine.latest.starts_with("revision ") {
            return if engine.version == engine.latest {
                "Up to date"
            } else {
                "Different revision"
            };
        }
        return "Revision unknown";
    }
    if engine.latest == "Check failed" {
        return "Check failed";
    }
    if engine.latest == "Rate limited" {
        return "Rate limited";
    }
    if engine.name == "sd-server" {
        return match engine.registry_revisions.as_ref().and_then(|versions| {
            crate::inference::engines::registry::compare_sd_registry(
                versions,
                &engine.version,
                &engine.latest,
            )
        }) {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if matches!(engine.name, "nemo-speech" | "audio-cpp" | "crispasr") {
        if engine.name == "audio-cpp" {
            let Some((version, custom)) =
                crate::inference::engines::registry::audio_identity(&engine.version)
            else {
                return "Unknown";
            };
            let order = engine.registry_revisions.as_ref().and_then(|versions| {
                crate::inference::engines::registry::compare_nemo_revision(
                    versions,
                    version,
                    &engine.latest,
                )
            });
            return match order {
                Some(std::cmp::Ordering::Less) => "New release",
                Some(std::cmp::Ordering::Equal) if custom => "Different revision",
                Some(std::cmp::Ordering::Equal) => "Up to date",
                Some(std::cmp::Ordering::Greater) => "Ahead",
                None => "Unknown",
            };
        }
        let installed = engine
            .version
            .trim()
            .strip_prefix("nemo-speech ")
            .unwrap_or(engine.version.trim())
            .split_whitespace()
            .next()
            .unwrap_or("");
        let order = engine.registry_revisions.as_ref().and_then(|versions| {
            crate::inference::engines::registry::compare_nemo_revision(
                versions,
                installed,
                &engine.latest,
            )
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if engine.name == "llama-server" {
        let order = engine.registry_versions.as_ref().and_then(|versions| {
            let installed = engine_build_identity(&engine.version)?.0;
            let selected = engine_build_identity(&engine.latest)?.0;
            crate::inference::engines::registry::compare_llama_build(versions, installed, selected)
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    if engine.name == "llama-server-prism" {
        let order = engine.registry_revisions.as_ref().and_then(|versions| {
            let (build, commit) = engine_build_identity(&engine.version)?;
            crate::inference::engines::registry::compare_prism_revision(
                versions,
                build,
                commit?,
                &engine.latest,
            )
        });
        return match order {
            Some(std::cmp::Ordering::Less) => "New release",
            Some(std::cmp::Ordering::Equal) => "Up to date",
            Some(std::cmp::Ordering::Greater) => "Ahead",
            None => "Unknown",
        };
    }
    let lower = engine.version.to_ascii_lowercase();
    if lower.contains("nightly") || lower.contains("-dev") {
        return "Development build";
    }
    let compare = if matches!(engine.name, "llama-server" | "llama-server-prism")
        && engine.latest.starts_with('b')
    {
        let installed = engine
            .version
            .split("build ")
            .nth(1)
            .and_then(|tail| tail.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|v| v.parse::<u64>().ok());
        let published = engine
            .latest
            .strip_prefix('b')
            .and_then(|v| v.parse::<u64>().ok());
        installed.zip(published).map(|(a, b)| a.cmp(&b))
    } else {
        let parse = |text: &str| {
            let text = text.trim_start_matches('v');
            let mut parts = text.split('.');
            let major = parts.next()?.parse::<u64>().ok()?;
            let minor = parts.next()?.parse::<u64>().ok()?;
            let patch = parts.next()?.parse::<u64>().ok()?;
            if parts.next().is_some() {
                return None;
            }
            Some((major, minor, patch))
        };
        let installed = engine.version.split_whitespace().find_map(parse);
        installed.zip(parse(&engine.latest)).map(|(a, b)| a.cmp(&b))
    };
    match compare {
        Some(std::cmp::Ordering::Equal) if engine.name == "whisper-cpp" && engine.gpu_upgrade => {
            "GPU build available"
        }
        Some(std::cmp::Ordering::Equal) => "Up to date",
        Some(std::cmp::Ordering::Less) => "New release",
        Some(std::cmp::Ordering::Greater) => "Ahead",
        None => "Unknown",
    }
}
