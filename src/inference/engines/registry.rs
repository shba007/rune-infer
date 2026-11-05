//! Reviewed remote engine distribution metadata.
use serde::Deserialize;
use std::{collections::BTreeMap, error::Error, io::Read};

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/shba007/rune-infer/main/registry/engines.json";
const MAX_REGISTRY_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema_version: u32,
    engines: Vec<Engine>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Engine {
    id: String,
    name: String,
    kind: String,
    latest: String,
    versions: BTreeMap<String, Version>,
    #[serde(default)]
    comparison_url: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Version {
    release_number: u64,
    artifacts: Vec<Artifact>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    platform: Platform,
    files: Vec<File>,
    executable: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Platform {
    os: String,
    arch: String,
    accelerator: String,
    #[serde(default)]
    cuda_version: Option<String>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct File {
    pub(crate) id: String,
    pub(crate) url: String,
    pub(crate) sha256: String,
    pub(crate) size_bytes: u64,
    pub(crate) format: String,
}

pub(crate) fn llama_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    llama_release_for_host(client, crate::embedding::root().is_some())
}

fn llama_release_for_host(
    client: &reqwest::blocking::Client,
    embedded: bool,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if embedded {
        return bundled_llama_release();
    }
    parse_llama(&fetch_document(client)?)
}

fn bundled_llama_release() -> Result<serde_json::Value, Box<dyn Error>> {
    parse_llama(include_bytes!("../../../registry/engines.json"))
}

pub(crate) fn prism_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_llama_engine(&fetch_document(client)?, "llama-server-prism")
}

fn fetch_document(client: &reqwest::blocking::Client) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut body = Vec::new();
    client
        .get(REGISTRY_URL)
        .send()?
        .error_for_status()?
        .take(MAX_REGISTRY_BYTES + 1)
        .read_to_end(&mut body)?;
    if body.len() as u64 > MAX_REGISTRY_BYTES {
        return Err("Engine registry exceeds size limit".into());
    }
    Ok(body)
}

pub(crate) fn litert_files(
    client: &reqwest::blocking::Client,
) -> Result<BTreeMap<String, File>, Box<dyn Error>> {
    parse_litert(&fetch_document(client)?)
}

pub(crate) fn nemo_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_nemo(&fetch_document(client)?)
}

pub(crate) fn audio_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_audio(&fetch_document(client)?)
}

pub(crate) fn crispasr_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_crispasr(&fetch_document(client)?)
}

pub(crate) fn sd_release(
    client: &reqwest::blocking::Client,
) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_sd(&fetch_document(client)?)
}

pub(crate) fn parse_sd(body: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut entries = registry
        .engines
        .iter()
        .filter(|entry| entry.id == "sd-server");
    let engine = entries
        .next()
        .ok_or("SD-server missing from engine registry")?;
    let comparison = engine
        .comparison_url
        .as_deref()
        .ok_or("SD-server comparison source missing")?;
    let comparison_url = reqwest::Url::parse(comparison)?;
    if comparison_url.scheme() != "https"
        || comparison_url.host_str() != Some("api.github.com")
        || !comparison_url.username().is_empty()
        || comparison_url.password().is_some()
        || comparison_url.query().is_some()
        || comparison_url.fragment().is_some()
        || !comparison_url.path().starts_with("/repos/")
        || !comparison_url.path().ends_with("/compare")
        || comparison_url.path().split('/').count() != 5
    {
        return Err("Invalid SD-server comparison source".into());
    }
    if entries.next().is_some() || engine.kind != "managed" || engine.name.trim().is_empty() {
        return Err("Invalid SD-server registry entry".into());
    }
    let selected = engine
        .versions
        .get(&engine.latest)
        .ok_or("Missing SD-server release")?;
    let mut versions = BTreeMap::new();
    let mut numbers = std::collections::HashSet::new();
    for (tag, version) in &engine.versions {
        let commit =
            super::sd_server::release_commit(tag).ok_or("Invalid SD-server release identity")?;
        if version.release_number == 0
            || version.release_number > selected.release_number
            || !numbers.insert(version.release_number)
            || versions
                .insert(commit.to_ascii_lowercase(), version.release_number)
                .is_some()
        {
            return Err("Invalid SD-server registry order".into());
        }
    }
    let mut assets = Vec::new();
    let mut platforms = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for artifact in &selected.artifacts {
        let p = &artifact.platform;
        if !matches!(p.os.as_str(), "windows" | "linux" | "macos")
            || !matches!(p.arch.as_str(), "x86_64" | "aarch64")
            || !matches!(p.accelerator.as_str(), "cpu" | "cuda" | "vulkan" | "metal")
            || (p.accelerator == "cuda") != p.cuda_version.is_some()
            || artifact.executable
                != if p.os == "windows" {
                    "sd-server.exe"
                } else {
                    "sd-server"
                }
            || artifact.files.len()
                != if p.os == "windows" && p.accelerator == "cuda" {
                    2
                } else {
                    1
                }
            || !platforms.insert((&p.os, &p.arch, &p.accelerator, &p.cuda_version))
        {
            return Err("Invalid SD-server platform".into());
        }
        let mut ids = std::collections::HashSet::new();
        for file in &artifact.files {
            let url = reqwest::Url::parse(&file.url)?;
            let name = url
                .path_segments()
                .and_then(|mut p| p.next_back())
                .filter(|s| !s.is_empty())
                .ok_or("Missing SD-server archive name")?;
            if !matches!(file.id.as_str(), "engine" | "runtime")
                || !ids.insert(&file.id)
                || file.format != "zip"
                || !name.ends_with(".zip")
                || name.contains('%')
                || file.sha256.len() != 64
                || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || file.size_bytes == 0
                || url.scheme() != "https"
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
                || !names.insert(name.to_owned())
            {
                return Err("Invalid SD-server artifact".into());
            }
            assets.push(serde_json::json!({"name":name,"browser_download_url":file.url,"digest":format!("sha256:{}",file.sha256),"size":file.size_bytes}));
        }
        if !ids.contains(&"engine".to_string())
            || (artifact.files.len() == 2 && !ids.contains(&"runtime".to_string()))
        {
            return Err("Incomplete SD-server bundle".into());
        }
    }
    if assets.is_empty() {
        return Err("No SD-server artifacts listed".into());
    }
    Ok(
        serde_json::json!({"tag_name":engine.latest,"draft":false,"prerelease":false,"assets":assets,"rune_registry":true,"rune_registry_release":selected.release_number,"rune_registry_revisions":versions,"rune_comparison_url":comparison}),
    )
}

#[cfg(feature = "tui")]
pub(crate) fn compare_sd_registry(
    versions: &BTreeMap<String, u64>,
    installed: &str,
    selected: &str,
) -> Option<std::cmp::Ordering> {
    let commit = installed.strip_prefix("commit ")?;
    if !(7..=40).contains(&commit.len()) || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let selected = super::sd_server::release_commit(selected)?;
    let mut matches = versions
        .iter()
        .filter(|(known, _)| known.starts_with(commit) || commit.starts_with(known.as_str()));
    let (_, installed_number) = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(installed_number.cmp(versions.get(selected)?))
}

pub(crate) fn parse_crispasr(body: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut entries = registry.engines.iter().filter(|e| e.id == "crispasr");
    let engine = entries
        .next()
        .ok_or("CrispASR missing from engine registry")?;
    if entries.next().is_some() || engine.kind != "managed" || engine.name.trim().is_empty() {
        return Err("Invalid CrispASR registry entry".into());
    }
    let selected = engine
        .versions
        .get(&engine.latest)
        .ok_or("Missing CrispASR release")?;
    let mut numbers = std::collections::HashSet::new();
    let mut revisions = BTreeMap::new();
    for (tag, version) in &engine.versions {
        let identity = tag.strip_prefix('v').unwrap_or(tag);
        if identity.is_empty()
            || identity.chars().any(char::is_whitespace)
            || version.release_number == 0
            || version.release_number > selected.release_number
            || !numbers.insert(version.release_number)
            || revisions
                .insert(identity.to_owned(), version.release_number)
                .is_some()
        {
            return Err("Invalid CrispASR release order".into());
        }
    }
    let mut platforms = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    let mut assets = Vec::new();
    for artifact in &selected.artifacts {
        let p = &artifact.platform;
        if !matches!(p.os.as_str(), "windows" | "linux" | "macos")
            || !matches!(p.arch.as_str(), "x86_64" | "aarch64")
            || !matches!(
                p.accelerator.as_str(),
                "cpu" | "cpu-legacy" | "cuda" | "metal" | "vulkan"
            )
            || (p.accelerator == "cuda") != p.cuda_version.is_some()
            || artifact.executable
                != if p.os == "windows" {
                    "crispasr.exe"
                } else {
                    "crispasr"
                }
            || artifact.files.len() != 1
            || !platforms.insert((&p.os, &p.arch, &p.accelerator, &p.cuda_version))
        {
            return Err("Invalid CrispASR platform".into());
        }
        let file = &artifact.files[0];
        let url = reqwest::Url::parse(&file.url)?;
        let name = url
            .path_segments()
            .and_then(|mut p| p.next_back())
            .filter(|s| !s.is_empty())
            .ok_or("Missing CrispASR archive name")?;
        if name.contains('%')
            || !name.ends_with(&format!(".{}", file.format))
            || file.id != "engine"
            || file.format != if p.os == "windows" { "zip" } else { "tar.gz" }
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || file.size_bytes == 0
            || url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || !names.insert(name.to_string())
        {
            return Err("Invalid CrispASR artifact".into());
        }
        assets.push(serde_json::json!({"name":name,"browser_download_url":file.url,"digest":format!("sha256:{}",file.sha256),"size":file.size_bytes,"rune_platform": {"os":p.os,"arch":p.arch,"accelerator":p.accelerator,"cuda_version":p.cuda_version}}));
    }
    if assets.is_empty() {
        return Err("No CrispASR artifacts listed".into());
    }
    Ok(
        serde_json::json!({"tag_name":engine.latest,"draft":false,"prerelease":false,"assets":assets,"rune_registry":true,"rune_registry_release":selected.release_number,"rune_registry_revisions":revisions}),
    )
}

#[cfg(feature = "tui")]
pub(crate) fn audio_identity(output: &str) -> Option<(&str, bool)> {
    let line = output.lines().next()?.trim().strip_prefix("audio.cpp ")?;
    let mut words = line.split_whitespace();
    let identity = words.next()?;
    let suffix = words.collect::<Vec<_>>().join(" ");
    match suffix.as_str() {
        "" => Some((identity, false)),
        "(rune-gguf1)" if identity == "0.9.3" => Some((identity, true)),
        _ => None,
    }
}

fn parse_audio(body: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut entries = registry
        .engines
        .iter()
        .filter(|entry| entry.id == "audio-cpp");
    let engine = entries
        .next()
        .ok_or("audio.cpp missing from engine registry")?;
    if entries.next().is_some() || engine.kind != "managed" || engine.name.trim().is_empty() {
        return Err("Invalid audio.cpp registry entry".into());
    }
    let selected = engine
        .versions
        .get(&engine.latest)
        .ok_or("Missing audio.cpp release")?;
    let mut numbers = std::collections::HashSet::new();
    let mut revisions = BTreeMap::new();
    for (tag, version) in &engine.versions {
        let identity = tag.strip_prefix('v').unwrap_or(tag);
        if identity.is_empty()
            || identity.chars().any(char::is_whitespace)
            || version.release_number == 0
            || version.release_number > selected.release_number
            || !numbers.insert(version.release_number)
            || revisions
                .insert(identity.to_owned(), version.release_number)
                .is_some()
        {
            return Err("Invalid audio.cpp release order".into());
        }
    }
    let mut assets = Vec::new();
    let mut platforms = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for artifact in &selected.artifacts {
        let p = &artifact.platform;
        if !matches!(p.os.as_str(), "windows" | "linux" | "macos")
            || !matches!(p.arch.as_str(), "x86_64" | "aarch64")
            || !matches!(p.accelerator.as_str(), "cpu" | "cuda" | "metal")
            || (p.accelerator == "cuda") != p.cuda_version.is_some()
            || artifact.executable
                != if p.os == "windows" {
                    "audiocpp_cli.exe"
                } else {
                    "audiocpp_cli"
                }
            || !platforms.insert((&p.os, &p.arch, &p.accelerator, &p.cuda_version))
            || artifact.files.len()
                != if p.os == "windows" && p.accelerator == "cuda" {
                    2
                } else {
                    1
                }
        {
            return Err("Invalid audio.cpp platform".into());
        }
        let mut ids = std::collections::HashSet::new();
        for file in &artifact.files {
            let url = reqwest::Url::parse(&file.url)?;
            let name = url
                .path_segments()
                .and_then(|mut path| path.next_back())
                .filter(|name| !name.is_empty())
                .ok_or("Missing audio.cpp archive name")?;
            let arch = if p.arch == "x86_64" { "x64" } else { "arm64" };
            let os = if p.os == "linux" { "ubuntu" } else { &p.os };
            let backend = if p.accelerator == "cuda" {
                format!("cuda{}", p.cuda_version.as_deref().unwrap_or(""))
            } else {
                p.accelerator.clone()
            };
            let role = match file.id.as_str() {
                "engine" => "bin",
                "runtime" => "cudart",
                _ => return Err("Invalid audio.cpp artifact role".into()),
            };
            let expected = format!("audio-{}-{role}-{os}-{arch}-{backend}", engine.latest);
            if !ids.insert(&file.id)
                || !name.starts_with(&expected)
                || !matches!(file.format.as_str(), "zip" | "tar.gz")
                || !name.ends_with(&format!(".{}", file.format))
                || file.sha256.len() != 64
                || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || file.size_bytes == 0
                || url.scheme() != "https"
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
                || name.contains('%')
                || !names.insert(name.to_owned())
            {
                return Err("Invalid audio.cpp artifact".into());
            }
            assets.push(serde_json::json!({"name":name,"browser_download_url":file.url,"digest":format!("sha256:{}",file.sha256),"size":file.size_bytes}));
        }
        if !ids.contains(&"engine".to_string())
            || (artifact.files.len() == 2 && !ids.contains(&"runtime".to_string()))
        {
            return Err("Incomplete audio.cpp bundle".into());
        }
    }
    if assets.is_empty() {
        return Err("No audio.cpp artifacts listed".into());
    }
    Ok(
        serde_json::json!({"tag_name":engine.latest,"draft":false,"prerelease":false,"assets":assets,"rune_registry":true,"rune_registry_release":selected.release_number,"rune_registry_revisions":revisions}),
    )
}

pub(crate) fn parse_nemo(body: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut entries = registry
        .engines
        .iter()
        .filter(|entry| entry.id == "nemo-speech");
    let engine = entries
        .next()
        .ok_or("NeMo Speech missing from engine registry")?;
    if entries.next().is_some() || engine.kind != "managed" || engine.name.trim().is_empty() {
        return Err("Invalid NeMo Speech registry entry".into());
    }
    let selected = engine
        .versions
        .get(&engine.latest)
        .ok_or("Missing selected NeMo Speech release")?;
    let mut numbers = std::collections::HashSet::new();
    let mut versions = BTreeMap::new();
    for (tag, entry) in &engine.versions {
        let version = tag.strip_prefix('v').unwrap_or(tag);
        let parts: Vec<_> = version.split('.').collect();
        if parts.len() != 3
            || parts.iter().any(|part| {
                part.parse::<u64>()
                    .ok()
                    .is_none_or(|n| n.to_string() != *part)
            })
            || entry.release_number == 0
            || !numbers.insert(entry.release_number)
            || entry.release_number > selected.release_number
            || versions
                .insert(version.to_string(), entry.release_number)
                .is_some()
        {
            return Err("Invalid NeMo Speech release order".into());
        }
    }
    let mut assets = Vec::new();
    let mut platforms = std::collections::HashSet::new();
    let mut names = std::collections::HashSet::new();
    for artifact in &selected.artifacts {
        let platform = &artifact.platform;
        if !matches!(platform.os.as_str(), "windows" | "linux" | "macos")
            || !matches!(platform.arch.as_str(), "x86_64" | "aarch64")
            || !matches!(
                platform.accelerator.as_str(),
                "cpu" | "cuda" | "vulkan" | "metal"
            )
            || (platform.accelerator != "cuda" && platform.cuda_version.is_some())
            || artifact.executable
                != if platform.os == "windows" {
                    "nemo-speech.exe"
                } else {
                    "nemo-speech"
                }
            || artifact.files.len() != 1
            || !platforms.insert((
                &platform.os,
                &platform.arch,
                &platform.accelerator,
                &platform.cuda_version,
            ))
        {
            return Err("Invalid NeMo Speech platform".into());
        }
        let file = &artifact.files[0];
        let url = reqwest::Url::parse(&file.url)?;
        let name = url
            .path_segments()
            .and_then(|mut path| path.next_back())
            .filter(|name| !name.is_empty())
            .ok_or("Missing NeMo archive name")?;
        if file.id != "engine"
            || !matches!(file.format.as_str(), "zip" | "tar.gz")
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || file.size_bytes == 0
            || url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || name.contains('%')
            || !name.contains(&platform.os)
            || !name.contains(&platform.arch)
            || !names.insert(name.to_string())
        {
            return Err("Invalid NeMo Speech artifact".into());
        }
        assets.push(serde_json::json!({"name":name,"browser_download_url":file.url,"digest":format!("sha256:{}",file.sha256),"size":file.size_bytes}));
    }
    if assets.is_empty() {
        return Err("No NeMo Speech artifacts listed".into());
    }
    Ok(
        serde_json::json!({"tag_name":engine.latest,"draft":false,"prerelease":false,"assets":assets,"rune_registry":true,"rune_registry_release":selected.release_number,"rune_registry_revisions":versions}),
    )
}

#[cfg(feature = "tui")]
pub(crate) fn compare_nemo_revision(
    versions: &BTreeMap<String, u64>,
    installed: &str,
    selected: &str,
) -> Option<std::cmp::Ordering> {
    Some(
        versions
            .get(installed.trim_start_matches('v'))?
            .cmp(versions.get(selected.trim_start_matches('v'))?),
    )
}

fn parse_litert(body: &[u8]) -> Result<BTreeMap<String, File>, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut engines = registry
        .engines
        .iter()
        .filter(|engine| engine.id == "litert");
    let engine = engines
        .next()
        .ok_or("LiteRT missing from engine registry")?;
    if engines.next().is_some()
        || engine.kind != "managed"
        || engine.name.trim().is_empty()
        || engine.latest != "2.3.0"
    {
        return Err("Unsupported LiteRT registry entry".into());
    }
    let version = engine
        .versions
        .get(&engine.latest)
        .ok_or("LiteRT version missing")?;
    let mut numbers = std::collections::HashSet::new();
    if engine.versions.values().any(|value| {
        value.release_number == 0
            || !numbers.insert(value.release_number)
            || value.release_number > version.release_number
    }) {
        return Err("Invalid LiteRT release order".into());
    }
    if version.artifacts.len() != 1 {
        return Err("Invalid LiteRT artifact bundle".into());
    }
    let artifact = &version.artifacts[0];
    if artifact.platform.os != "windows"
        || artifact.platform.arch != "x86_64"
        || artifact.platform.accelerator != "cpu"
        || artifact.platform.cuda_version.is_some()
        || artifact.executable != "libLiteRt.dll"
        || artifact.files.len() != 3
    {
        return Err("Invalid LiteRT platform bundle".into());
    }
    let mut files = BTreeMap::new();
    for file in &artifact.files {
        let (hash, format) = match file.id.as_str() {
            "cpu" => (
                "6fe42cfa518879dd7bae4500244cf371d949bf0c8559cb60be207d81112da16d",
                "dll",
            ),
            "gpu" => (
                "ade56eff38715f3e7b3ada0cfb736db4cccb6bed4668a891a37ebcf4eee38548",
                "dll",
            ),
            "dxc" => (
                "9ad895a6b039e3a8f8c22a1009f866800b840a74b50db9218d13319e215ea8a4",
                "zip",
            ),
            _ => return Err("Unknown LiteRT component".into()),
        };
        let url = reqwest::Url::parse(&file.url)?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
            || url.query().is_some()
            || file.sha256 != hash
            || file.size_bytes == 0
            || file.format != format
            || files.insert(file.id.clone(), file.clone()).is_some()
        {
            return Err("Invalid or unreviewed LiteRT component".into());
        }
    }
    Ok(files)
}

#[cfg(any(windows, test))]
pub(crate) fn onnx_files(
    client: &reqwest::blocking::Client,
) -> Result<BTreeMap<String, File>, Box<dyn Error>> {
    parse_onnx(&fetch_document(client)?)
}

#[cfg(any(windows, test))]
fn parse_onnx(body: &[u8]) -> Result<BTreeMap<String, File>, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut engines = registry.engines.iter().filter(|engine| engine.id == "onnx");
    let engine = engines.next().ok_or("ONNX missing from engine registry")?;
    if engines.next().is_some()
        || engine.kind != "managed"
        || engine.name.trim().is_empty()
        || engine.latest != "1.31.1"
    {
        return Err("Unsupported ONNX registry entry".into());
    }
    let version = engine
        .versions
        .get(&engine.latest)
        .ok_or("ONNX version missing")?;
    let mut numbers = std::collections::HashSet::new();
    if engine.versions.values().any(|value| {
        value.release_number == 0
            || !numbers.insert(value.release_number)
            || value.release_number > version.release_number
    }) {
        return Err("Invalid ONNX release order".into());
    }
    if version.artifacts.len() != 1 {
        return Err("Invalid ONNX artifact bundle".into());
    }
    let artifact = &version.artifacts[0];
    if artifact.platform.os != "windows"
        || artifact.platform.arch != "x86_64"
        || artifact.platform.accelerator != "cuda"
        || artifact.platform.cuda_version.as_deref() != Some("12")
        || artifact.executable != "onnxruntime.dll"
        || artifact.files.len() != 4
    {
        return Err("Invalid ONNX platform bundle".into());
    }
    let mut files = BTreeMap::new();
    for file in &artifact.files {
        let (hash, format) = match file.id.as_str() {
            "ort" => (
                "5cbf65edc93dc7bb3296249b1b6e1b2228e575902af64d6163d0da2103cef9d5",
                "zip",
            ),
            "cudnn" => (
                "06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142",
                "whl",
            ),
            "cublas" => (
                "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
                "whl",
            ),
            "cudart" => (
                "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
                "whl",
            ),
            _ => return Err("Unknown ONNX component".into()),
        };
        let url = reqwest::Url::parse(&file.url)?;
        if url.scheme() != "https"
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
            || url.query().is_some()
            || file.sha256 != hash
            || file.size_bytes == 0
            || file.format != format
            || files.insert(file.id.clone(), file.clone()).is_some()
        {
            return Err("Invalid or unreviewed ONNX component".into());
        }
    }
    Ok(files)
}

fn parse_llama(body: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    parse_llama_engine(body, "llama-server")
}

fn parse_llama_engine(body: &[u8], engine_id: &str) -> Result<serde_json::Value, Box<dyn Error>> {
    let registry: Registry = serde_json::from_slice(body)?;
    if registry.schema_version != 1 {
        return Err("Unsupported engine registry schema".into());
    }
    let mut ids = std::collections::HashSet::new();
    if registry.engines.iter().any(|e| !ids.insert(&e.id)) {
        return Err("Duplicate engine registry ID".into());
    }
    let engine = registry
        .engines
        .iter()
        .find(|e| e.id == engine_id)
        .ok_or("llama-server missing from engine registry")?;
    if engine.kind != "managed" || engine.name.trim().is_empty() {
        return Err("Invalid llama-server registry entry".into());
    }
    let tag = &engine.latest;
    let version = engine
        .versions
        .get(tag)
        .ok_or("Missing selected engine version")?;
    let mut release_numbers = std::collections::HashSet::new();
    let mut registry_versions = BTreeMap::new();
    let mut registry_revisions = BTreeMap::new();
    for (id, entry) in &engine.versions {
        let number = if engine_id == "llama-server-prism" {
            let (build, commit) = prism_identity(id).ok_or("Invalid Prism build identity")?;
            registry_revisions.insert(id.clone(), entry.release_number);
            if commit.is_empty() {
                return Err("Invalid Prism commit".into());
            }
            build
        } else {
            id.strip_prefix('b')
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|number| format!("b{number}") == *id)
                .ok_or("Invalid llama-server build identity")?
        };
        if registry_versions
            .insert(number, entry.release_number)
            .is_some()
        {
            return Err("Duplicate llama-server build identity".into());
        }
    }
    if engine
        .versions
        .values()
        .any(|v| v.release_number == 0 || !release_numbers.insert(v.release_number))
        || engine
            .versions
            .values()
            .any(|v| v.release_number > version.release_number)
    {
        return Err("Invalid registry release order".into());
    }
    if engine_id == "llama-server"
        && tag
            .strip_prefix('b')
            .is_none_or(|n| n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err("Invalid engine build".into());
    }
    let mut assets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut platforms = std::collections::HashSet::new();
    for artifact in &version.artifacts {
        let platform = &artifact.platform;
        if !matches!(platform.os.as_str(), "windows" | "linux" | "macos")
            || !matches!(platform.arch.as_str(), "x86_64" | "aarch64")
            || !matches!(platform.accelerator.as_str(), "cpu" | "vulkan" | "cuda")
            || (platform.accelerator == "cuda") != platform.cuda_version.is_some()
            || artifact.files.is_empty()
            || artifact.executable
                != if platform.os == "windows" {
                    "llama-server.exe"
                } else {
                    "llama-server"
                }
        {
            return Err("Invalid engine artifact platform".into());
        }
        if !platforms.insert((
            &platform.os,
            &platform.arch,
            &platform.accelerator,
            &platform.cuda_version,
        )) {
            return Err("Duplicate engine platform".into());
        }
        if artifact.files.iter().filter(|f| f.id == "engine").count() != 1
            || artifact.files.iter().filter(|f| f.id == "runtime").count()
                != usize::from(platform.accelerator == "cuda" && platform.os == "windows")
        {
            return Err("Invalid engine artifact bundle".into());
        }
        for file in &artifact.files {
            let url = reqwest::Url::parse(&file.url)?;
            let name = url
                .path_segments()
                .and_then(|mut p| p.next_back())
                .filter(|n| !n.is_empty())
                .ok_or("Missing archive name")?;
            if url.scheme() != "https"
                || url.username() != ""
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
                || name.contains('%')
                || !matches!(file.format.as_str(), "zip" | "tar.gz")
                || file.sha256.len() != 64
                || !file.sha256.bytes().all(|c| c.is_ascii_hexdigit())
                || file.size_bytes == 0
                || !matches!(file.id.as_str(), "engine" | "runtime")
            {
                return Err("Invalid engine artifact metadata".into());
            }
            if !seen.insert(name.to_string()) {
                return Err("Duplicate engine archive name".into());
            }
            if platform.os == std::env::consts::OS && platform.arch == std::env::consts::ARCH {
                assets.push(serde_json::json!({"name": name, "browser_download_url":file.url, "digest":format!("sha256:{}", file.sha256), "size":file.size_bytes}));
            }
        }
    }
    if assets.is_empty() {
        return Err("No engine artifacts listed".into());
    }
    Ok(
        serde_json::json!({"tag_name":tag,"draft":false,"prerelease":true,"assets":assets,"rune_registry":true,"rune_registry_release":version.release_number,"rune_registry_versions":registry_versions,"rune_registry_revisions":registry_revisions}),
    )
}

pub(crate) fn prism_identity(tag: &str) -> Option<(u64, &str)> {
    let (build, commit) = tag.strip_prefix("prism-b")?.split_once('-')?;
    let number = build.parse::<u64>().ok()?;
    if format!("{number}") != build
        || !(7..=40).contains(&commit.len())
        || !commit.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return None;
    }
    Some((number, commit))
}

pub(crate) fn compare_prism_revision(
    versions: &BTreeMap<String, u64>,
    installed_build: u64,
    installed_commit: &str,
    selected: &str,
) -> Option<std::cmp::Ordering> {
    let mut matches = versions.iter().filter(|(tag, _)| {
        prism_identity(tag).is_some_and(|(build, commit)| {
            build == installed_build
                && installed_commit.len() >= 7
                && installed_commit
                    .to_ascii_lowercase()
                    .starts_with(&commit.to_ascii_lowercase())
        })
    });
    let (_, installed) = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(installed.cmp(versions.get(selected)?))
}

/// Compare a verified upstream build using registry order. Unknown builds cannot
/// be ordered safely, even when their upstream build number happens to be larger.
pub(crate) fn compare_llama_build(
    versions: &BTreeMap<u64, u64>,
    installed: u64,
    selected: u64,
) -> Option<std::cmp::Ordering> {
    Some(versions.get(&installed)?.cmp(versions.get(&selected)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    const SEED: &[u8] = include_bytes!("../../../registry/engines.json");
    #[test]
    fn embedded_llama_uses_the_reviewed_bundled_registry() {
        let client = reqwest::blocking::Client::new();
        let actual = llama_release_for_host(&client, true).unwrap();
        let expected = parse_llama(SEED).unwrap();
        assert_eq!(actual, expected);
        assert!(
            actual["assets"]
                .as_array()
                .is_some_and(|assets| !assets.is_empty())
        );
    }
    #[test]
    fn prism_registry_orders_only_matching_build_and_commit() {
        let release = parse_llama_engine(SEED, "llama-server-prism").unwrap();
        let versions: BTreeMap<String, u64> =
            serde_json::from_value(release["rune_registry_revisions"].clone()).unwrap();
        let latest = "prism-b12128-d7b2896";
        assert_eq!(
            compare_prism_revision(&versions, 10831, "9d9efea11", latest),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_prism_revision(&versions, 12128, "d7b289683", latest),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_prism_revision(&versions, 12128, "aaaaaaa", latest),
            None
        );
        assert_eq!(
            compare_prism_revision(&versions, 42, "d7b289683", latest),
            None
        );
        assert!(
            release["assets"]
                .as_array()
                .is_some_and(|assets| !assets.is_empty())
        );
    }
    #[test]
    fn sd_registry_contains_verified_bundles_and_ordered_commits() {
        let release = parse_sd(SEED).unwrap();
        assert_eq!(release["tag_name"], "master-1043-3be2378");
        assert_eq!(release["rune_registry_revisions"]["c6dfa9d"], 1);
        assert_eq!(release["rune_registry_revisions"]["3be2378"], 2);
        #[cfg(feature = "tui")]
        {
            let revisions =
                serde_json::from_value(release["rune_registry_revisions"].clone()).unwrap();
            assert_eq!(
                compare_sd_registry(
                    &revisions,
                    "commit c6dfa9def12fd0cb36ac7149ba30a75792bedbf0",
                    "master-1043-3be2378"
                ),
                Some(std::cmp::Ordering::Less)
            );
            assert_eq!(
                compare_sd_registry(&revisions, "commit 3be2378", "master-1043-3be2378"),
                Some(std::cmp::Ordering::Equal)
            );
            assert_eq!(
                compare_sd_registry(&revisions, "commit ffffffff", "master-1043-3be2378"),
                None
            );
        }
        assert_eq!(release["assets"].as_array().unwrap().len(), 7);
        let mut bad: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let engine = bad["engines"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == "sd-server")
            .unwrap();
        engine["versions"]["master-1043-3be2378"]["artifacts"][2]["files"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(parse_sd(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    #[test]
    fn crispasr_registry_validates_artifacts_and_release_order() {
        let release = parse_crispasr(SEED).unwrap();
        assert_eq!(release["tag_name"], "v0.8.49");
        assert_eq!(release["rune_registry_revisions"]["0.8.48"], 1);
        assert_eq!(release["rune_registry_revisions"]["0.8.49"], 2);
        assert_eq!(release["assets"].as_array().unwrap().len(), 12);
        let mut bad: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let engine = bad["engines"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|e| e["id"] == "crispasr")
            .unwrap();
        engine["versions"]["v0.8.49"]["artifacts"][0]["files"][0]["sha256"] = "bad".into();
        assert!(parse_crispasr(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    #[test]
    fn audio_registry_requires_complete_verified_bundles() {
        let release = parse_audio(SEED).unwrap();
        assert_eq!(release["tag_name"], "v0.9.3");
        assert_eq!(release["rune_registry_revisions"]["0.9.3"], 1);
        assert_eq!(release["assets"].as_array().unwrap().len(), 8);
        let mut bad: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let engine = bad["engines"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == "audio-cpp")
            .unwrap();
        engine["versions"]["v0.9.3"]["artifacts"][4]["files"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(parse_audio(&serde_json::to_vec(&bad).unwrap()).is_err());
    }

    #[cfg(feature = "tui")]
    #[test]
    fn audio_identity_recognizes_only_known_custom_build() {
        assert_eq!(
            audio_identity("audio.cpp 0.9.3\ngit: abc"),
            Some(("0.9.3", false))
        );
        assert_eq!(
            audio_identity("audio.cpp 0.9.3 (rune-gguf1)\ngit: 41c1f8f"),
            Some(("0.9.3", true))
        );
        assert_eq!(audio_identity("audio.cpp 0.9.3 (other)"), None);
        assert_eq!(audio_identity("audio.cpp 0.9.4 (rune-gguf1)"), None);
    }
    #[test]
    fn nemo_registry_contains_verified_cross_platform_packages() {
        let release = parse_nemo(SEED).unwrap();
        assert_eq!(release["tag_name"], "v0.3.0");
        assert_eq!(release["rune_registry_revisions"]["0.3.0"], 1);
        assert!(release["assets"].as_array().unwrap().len() >= 10);
        let mut bad: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let engine = bad["engines"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|engine| engine["id"] == "nemo-speech")
            .unwrap();
        engine["versions"]["v0.3.0"]["artifacts"][0]["files"][0]["sha256"] = "bad".into();
        assert!(parse_nemo(&serde_json::to_vec(&bad).unwrap()).is_err());
    }
    #[test]
    fn litert_registry_pins_all_runtime_components() {
        let files = parse_litert(SEED).unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files["cpu"].size_bytes, 12173824);
        let mut value: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        value["engines"][1]["versions"]["2.3.0"]["artifacts"][0]["files"][0]["sha256"] =
            "bad".into();
        assert!(parse_litert(&serde_json::to_vec(&value).unwrap()).is_err());
        value["engines"][1]["latest"] = "2.4.0".into();
        assert!(parse_litert(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    #[test]
    fn onnx_registry_pins_all_runtime_components() {
        let files = parse_onnx(SEED).unwrap();
        assert_eq!(files.len(), 4);
        assert_eq!(files["ort"].size_bytes, 392924457);
        let mut value: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        value["engines"][2]["versions"]["1.31.1"]["artifacts"][0]["files"][1]["sha256"] =
            "bad".into();
        assert!(parse_onnx(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    #[test]
    fn seed_is_valid_and_pinned() {
        let r = parse_llama(SEED).unwrap();
        assert_eq!(r["tag_name"], "b12027");
        let order: BTreeMap<u64, u64> =
            serde_json::from_value(r["rune_registry_versions"].clone()).unwrap();
        assert_eq!(
            compare_llama_build(&order, 12027, 12027),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            compare_llama_build(&order, 11960, 12027),
            Some(std::cmp::Ordering::Less)
        );
        assert!(!r["assets"].as_array().unwrap().is_empty());
    }

    #[test]
    fn registry_order_is_independent_of_upstream_build_numbers() {
        let order = BTreeMap::from([(900, 1), (3, 2), (100, 3)]);
        use std::cmp::Ordering;
        assert_eq!(compare_llama_build(&order, 900, 3), Some(Ordering::Less));
        assert_eq!(compare_llama_build(&order, 3, 3), Some(Ordering::Equal));
        assert_eq!(compare_llama_build(&order, 100, 3), Some(Ordering::Greater));
        assert_eq!(compare_llama_build(&order, 42, 3), None);
        assert_eq!(compare_llama_build(&order, 3, 42), None);
    }
    #[test]
    fn rejects_invalid_registry_data() {
        let mut v: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        v["engines"][0]["versions"]["b12027"]["artifacts"][0]["files"][0]["sha256"] = "bad".into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
        v["schema_version"] = 9.into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
    }

    #[test]
    fn latest_is_explicit_not_lexicographically_selected() {
        let mut v: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let version = v["engines"][0]["versions"]["b12027"].clone();
        v["engines"][0]["versions"]["b9"] = version;
        v["engines"][0]["versions"]["b9"]["release_number"] = 3.into();
        v["engines"][0]["latest"] = "b9".into();
        assert_eq!(
            parse_llama(&serde_json::to_vec(&v).unwrap()).unwrap()["tag_name"],
            "b9"
        );
        v["engines"][0]["latest"] = "b10".into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
    }

    #[test]
    fn release_numbers_must_be_unique_positive_and_latest_must_be_newest() {
        let mut v: serde_json::Value = serde_json::from_slice(SEED).unwrap();
        let version = v["engines"][0]["versions"]["b12027"].clone();
        v["engines"][0]["versions"]["b9"] = version;
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
        v["engines"][0]["versions"]["b9"]["release_number"] = 3.into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
        v["engines"][0]["latest"] = "b9".into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_ok());
        v["engines"][0]["versions"]["b12027"]["release_number"] = 0.into();
        assert!(parse_llama(&serde_json::to_vec(&v).unwrap()).is_err());
    }
}
