use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[cfg(windows)]
#[derive(Clone, Copy)]
struct JobHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for JobHandle {}
#[cfg(windows)]
unsafe impl Sync for JobHandle {}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateJobObjectW(
        lpjobattributes: *const std::ffi::c_void,
        lpname: *const u16,
    ) -> windows_sys::Win32::Foundation::HANDLE;
}

#[cfg(windows)]
static GLOBAL_JOB_OBJECT: std::sync::OnceLock<Option<JobHandle>> = std::sync::OnceLock::new();

#[cfg(windows)]
fn get_job_object() -> Option<windows_sys::Win32::Foundation::HANDLE> {
    GLOBAL_JOB_OBJECT
        .get_or_init(|| unsafe {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::JobObjects::{
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectExtendedLimitInformation, SetInformationJobObject,
            };

            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                eprintln!("[Supervisor] Failed to create JobObject");
                return None;
            }

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let res = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );

            if res == 0 {
                eprintln!("[Supervisor] Failed to set JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE");
                CloseHandle(job);
                return None;
            }

            Some(JobHandle(job))
        })
        .map(|h| h.0)
}

pub fn configure_death_signal(_cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            _cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            _cmd.pre_exec(|| {
                // Spawns child into its own process group on macOS and BSD
                libc::setpgid(0, 0);
                Ok(())
            });
        }
    }
}

pub fn attach_to_supervisor(child: &Child) {
    #[cfg(windows)]
    {
        if let Some(job) = get_job_object() {
            unsafe {
                use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;
                let handle = child.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
                AssignProcessToJobObject(job, handle);
            }
        }
    }
    register_pid(child.id());
}

pub fn register_pid(pid: u32) {
    let _ = fs::create_dir_all("logs");
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/.active_pids")
    {
        let _ = writeln!(file, "{}", pid);
    }
}

pub fn unregister_pid(pid: u32) {
    let path = Path::new("logs/.active_pids");
    if !path.exists() {
        return;
    }
    if let Ok(content) = fs::read_to_string(path) {
        let remaining: Vec<&str> = content
            .lines()
            .filter(|line| line.trim() != pid.to_string())
            .collect();
        let updated = if remaining.is_empty() {
            String::new()
        } else {
            remaining.join("\n") + "\n"
        };
        let _ = fs::write(path, updated);
    }
}

pub fn cleanup_orphaned_pids() {
    let path = Path::new("logs/.active_pids");
    if !path.exists() {
        return;
    }
    if let Ok(content) = fs::read_to_string(path) {
        for line in content.lines() {
            if let Ok(pid) = line.trim().parse::<u32>() {
                kill_pid(pid);
            }
        }
    }
    let _ = fs::remove_file(path);
}

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
    #[cfg(unix)]
    {
        unsafe {
            let pid_t = pid as libc::pid_t;
            let _ = libc::kill(-pid_t, libc::SIGKILL);
            let _ = libc::kill(pid_t, libc::SIGKILL);
        }
    }
}

pub struct ProcessGuard {
    child: Child,
    label: String,
}

impl ProcessGuard {
    pub fn new(child: Child, label: impl Into<String>) -> Arc<Mutex<Self>> {
        let label = label.into();
        attach_to_supervisor(&child);
        tracing::info!(
            "[Supervisor] Registered managed process '{}' (PID: {})",
            label,
            child.id()
        );
        Arc::new(Mutex::new(Self { child, label }))
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        let pid = self.child.id();
        tracing::warn!(
            "[Supervisor] Shutting down {} (PID: {})...",
            self.label,
            pid
        );
        let _ = self.child.kill();
        let _ = self.child.wait();
        unregister_pid(pid);
        println!(
            "[Supervisor] Process {} (PID: {}) terminated and VRAM released.",
            self.label, pid
        );
    }
}
