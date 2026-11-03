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
    parse_llama(&fetch_document(client)?)
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
        .find(|e| e.id == "llama-server")
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
    for (id, entry) in &engine.versions {
        let number = id
            .strip_prefix('b')
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|number| format!("b{number}") == *id)
            .ok_or("Invalid llama-server build identity")?;
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
    if tag
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
                != usize::from(platform.accelerator == "cuda")
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
        serde_json::json!({"tag_name":tag,"draft":false,"prerelease":true,"assets":assets,"rune_registry":true,"rune_registry_release":version.release_number,"rune_registry_versions":registry_versions}),
    )
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
