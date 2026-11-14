use serde::Serialize;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub id: String,
    pub name: String,
    pub license: String,
    pub bytes: u64,
    pub available_bytes: Option<u64>,
    pub revision: String,
    pub manifest: String,
    pub source_url: String,
    pub recommended: bool,
    pub recommendation_basis: Option<String>,
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub phase: String,
    pub model: Option<String>,
    pub filename: Option<String>,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub error: Option<String>,
}

fn manifest(version: &crate::model_registry::Version) -> String {
    version
        .files
        .iter()
        .map(|file| {
            format!(
                "{}:{}:{}:{}:{}",
                file.id, file.filename, file.url, file.sha256, file.size_bytes
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Models supported by the managed installer, optionally limited to computer use.
pub fn managed_candidates(
    computer_use: bool,
    memory: Option<u64>,
    gpu: bool,
    disk: Option<u64>,
) -> anyhow::Result<Vec<Candidate>> {
    let ids = crate::model_registry::bundled()?
        .into_iter()
        .filter(|model| {
            model.engine == "llama-server"
                && (!computer_use || model.catalog.as_ref().is_some_and(supports_computer_use))
        })
        .map(|model| model.id)
        .collect::<Vec<_>>();
    candidates(&ids, memory, gpu, disk)
}

fn supports_computer_use(catalog: &crate::model_registry::CatalogDefaults) -> bool {
    let vision = if catalog.tasks.is_empty() {
        catalog
            .capabilities
            .iter()
            .any(|cap| cap.eq_ignore_ascii_case("vision"))
    } else {
        catalog
            .tasks
            .iter()
            .any(|task| task == "text-image-to-text")
    };
    let tools = if let Some(support) = &catalog.tool_call_support {
        support.status == crate::audio_operations::SupportStatus::Supported
            && catalog.tasks.iter().any(|task| task == "text-to-text")
            && support
                .verified_with
                .iter()
                .any(|proof| proof.engine == crate::config::EngineKind::LlamaServer)
    } else {
        catalog.capabilities.iter().any(|cap| {
            ["tool call", "tool_calls", "tools"]
                .iter()
                .any(|alias| cap.eq_ignore_ascii_case(alias))
        })
    };
    vision && tools
}

pub fn candidates(
    ids: &[String],
    memory: Option<u64>,
    gpu: bool,
    disk: Option<u64>,
) -> anyhow::Result<Vec<Candidate>> {
    let models = crate::model_registry::bundled()?;
    let usable =
        memory.map(|bytes| bytes.saturating_sub(if gpu { 2 } else { 4 } * 1024 * 1024 * 1024));
    let basis = memory.map(|bytes| {
        format!(
            "Estimated from {} GiB of {} memory{}",
            bytes / (1024 * 1024 * 1024),
            if gpu { "GPU" } else { "available system" },
            if gpu {
                ""
            } else {
                "; GPU memory could not be detected"
            }
        )
    });
    let mut output: Vec<_> = models
        .into_iter()
        .filter(|model| ids.contains(&model.id))
        .filter_map(|model| {
            let version = model.versions.get(&model.latest)?;
            let bytes = version
                .files
                .iter()
                .try_fold(0u64, |sum, file| sum.checked_add(file.size_bytes))?;
            let need = bytes.saturating_add(1024 * 1024 * 1024);
            let fits = usable.is_some_and(|memory| memory >= need)
                && disk.is_none_or(|space| space >= bytes.saturating_add(2 * 1024 * 1024 * 1024));
            let params = model
                .catalog
                .as_ref()
                .and_then(|catalog| catalog.local.params)
                .unwrap_or(0);
            Some((
                params,
                fits,
                Candidate {
                    id: model.id.clone(),
                    name: model.name.clone(),
                    license: model.license.clone(),
                    bytes,
                    available_bytes: disk,
                    revision: model.latest.clone(),
                    manifest: manifest(version),
                    source_url: model.source_url.clone(),
                    recommended: false,
                    recommendation_basis: basis.clone(),
                },
            ))
        })
        .collect();
    if let Some((index, _)) = output
        .iter()
        .enumerate()
        .filter(|(_, (_, fits, _))| *fits)
        .max_by_key(|(_, (params, _, _))| *params)
    {
        output[index].2.recommended = true;
    }
    Ok(output
        .into_iter()
        .map(|(_, _, candidate)| candidate)
        .collect())
}

pub fn detected_memory() -> (Option<u64>, bool) {
    #[cfg(not(target_os = "linux"))]
    if let Some(memory) = crate::inference::budget::detect_total_gpu_vram() {
        return (Some(memory), true);
    }
    #[cfg(target_os = "linux")]
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            if let Ok(mib) = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .parse::<u64>()
            {
                return (mib.checked_mul(1024 * 1024), true);
            }
        }
    }
    (system_memory(), false)
}

#[cfg(windows)]
fn system_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullAvailPhys)
}
#[cfg(target_os = "linux")]
fn system_memory() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kb| kb * 1024)
        })
}
#[cfg(not(any(windows, target_os = "linux")))]
fn system_memory() -> Option<u64> {
    None
}

#[cfg(windows)]
pub fn available_disk(root: &std::path::Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let mut path = root;
    while !path.exists() {
        path = path.parent()?;
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0;
    (unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } != 0)
        .then_some(available)
}
#[cfg(not(windows))]
pub fn available_disk(_root: &std::path::Path) -> Option<u64> {
    None
}

#[derive(Default)]
pub struct Downloads {
    state: Arc<Mutex<Status>>,
    cancel: Mutex<Option<Arc<AtomicBool>>>,
}
impl Downloads {
    pub fn status(&self) -> Status {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn completed_model(&self) -> Option<String> {
        let state = self.status();
        (state.phase == "complete").then_some(state.model).flatten()
    }
    pub fn cancel(&self) {
        if let Some(token) = self
            .cancel
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            token.store(true, Ordering::Relaxed);
        }
    }
    pub fn acknowledge(&self) {
        if matches!(
            self.status().phase.as_str(),
            "complete" | "failed" | "cancelled"
        ) {
            self.cancel.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
    }
    pub fn start(
        &self,
        root: std::path::PathBuf,
        id: String,
        revision: String,
        reviewed_manifest: String,
    ) -> Result<(), String> {
        if !root.is_absolute() {
            return Err("Infer home must be absolute".into());
        }
        let model = crate::model_registry::bundled()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|model| model.id == id)
            .ok_or("Model is no longer available")?;
        if model.engine != "llama-server" {
            return Err(
                "This embedded installer currently supports llama-server models only".into(),
            );
        }
        if model.latest != revision
            || model
                .versions
                .get(&revision)
                .is_none_or(|version| manifest(version) != reviewed_manifest)
        {
            return Err(
                "The model changed since you reviewed it. Reload the list and review it again."
                    .into(),
            );
        }
        let mut slot = self.cancel.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Err("A model download is already in progress".into());
        }
        let cancel = Arc::new(AtomicBool::new(false));
        *slot = Some(cancel.clone());
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = Status {
            phase: "preparing".into(),
            model: Some(id),
            ..Status::default()
        };
        let state = self.state.clone();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let report = |state: Arc<Mutex<Status>>| {
                    move |progress: crate::inference::engines::download::DownloadProgress| {
                        let mut status = state.lock().unwrap_or_else(|e| e.into_inner());
                        status.filename = Some(progress.filename);
                        status.downloaded = progress.downloaded;
                        status.total = progress.total;
                    }
                };
                state.lock().unwrap_or_else(|e| e.into_inner()).phase = "downloading".into();
                crate::model_registry::install_observed(
                    &model,
                    &root,
                    &root.join("config/server.json"),
                    &root.join("config/catalog.json"),
                    cancel.clone(),
                    report(state.clone()),
                )
                .map_err(|e| format!("Model installation failed: {e:#}"))?;
                if cancel.load(Ordering::Relaxed) {
                    return Err("Installation cancelled".into());
                }
                {
                    let mut status = state.lock().unwrap_or_else(|e| e.into_inner());
                    status.phase = "engine".into();
                    status.filename = None;
                    status.downloaded = 0;
                    status.total = None;
                }
                crate::embedding::install_llama_observed(
                    &root,
                    cancel.clone(),
                    report(state.clone()),
                )
                .map_err(|e| format!("Engine preparation failed: {e}"))?;
                Ok::<(), String>(())
            }))
            .unwrap_or_else(|_| Err("Model installer stopped unexpectedly".into()));
            let mut status = state.lock().unwrap_or_else(|e| e.into_inner());
            status.phase = if result.is_ok() {
                "complete"
            } else if cancel.load(Ordering::Relaxed) {
                "cancelled"
            } else {
                "failed"
            }
            .into();
            status.error = result.err();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_candidates_come_from_the_registry_and_computer_use_requires_capabilities() {
        let memory = Some(32 * 1024 * 1024 * 1024);
        let all = managed_candidates(false, memory, true, None).unwrap();
        let computer = managed_candidates(true, memory, true, None).unwrap();
        let registry = crate::model_registry::bundled().unwrap();
        assert!(!all.is_empty() && !computer.is_empty());
        assert!(all.iter().all(|candidate| {
            registry
                .iter()
                .any(|model| model.id == candidate.id && model.engine == "llama-server")
        }));
        assert!(computer.iter().all(|candidate| {
            registry.iter().any(|model| {
                model.id == candidate.id
                    && model.catalog.as_ref().is_some_and(supports_computer_use)
            })
        }));
        assert!(
            computer
                .iter()
                .all(|candidate| all.iter().any(|model| model.id == candidate.id))
        );
        assert!(all.iter().filter(|model| model.recommended).count() <= 1);
        assert!(computer.iter().filter(|model| model.recommended).count() <= 1);
    }

    #[test]
    fn computer_use_respects_explicit_tool_support_over_legacy_capabilities() {
        let mut model = crate::model_registry::bundled()
            .unwrap()
            .into_iter()
            .find(|model| model.id == "holo3.1-4b-q4")
            .unwrap();
        let catalog = model.catalog.as_mut().unwrap();
        assert!(supports_computer_use(catalog));
        catalog.tool_call_support = Some(crate::audio_operations::OperationSupport {
            status: crate::audio_operations::SupportStatus::Unsupported,
            ..Default::default()
        });
        assert!(!supports_computer_use(catalog));
    }
    #[test]
    fn lists_only_requested_models_and_recommends_one_that_fits() {
        let ids = ["holo3.1-4b-q4", "holo3.1-9b-q4", "holo4-27b-q4"].map(str::to_owned);
        let gib = 1024 * 1024 * 1024;
        let models = candidates(&ids, Some(8 * gib), true, Some(20 * gib)).unwrap();
        assert_eq!(models.len(), 3);
        assert_eq!(
            models.iter().find(|model| model.recommended).unwrap().id,
            "holo3.1-4b-q4"
        );
        assert_eq!(
            candidates(&ids, None, false, None)
                .unwrap()
                .iter()
                .filter(|model| model.recommended)
                .count(),
            0
        );
        assert!(
            models
                .iter()
                .all(|model| model.manifest.contains("https://huggingface.co/"))
        );
    }
    #[test]
    fn unreviewed_or_unknown_model_cannot_start() {
        let downloads = Downloads::default();
        assert!(
            downloads
                .start(
                    std::env::temp_dir(),
                    "not-a-model".into(),
                    "rev".into(),
                    "manifest".into()
                )
                .is_err()
        );
        assert!(downloads.status().phase.is_empty());
    }
}
