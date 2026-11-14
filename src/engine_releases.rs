//! Shared engine release lookup for the dashboard and management API.
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
