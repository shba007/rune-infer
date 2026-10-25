use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

/// Resolve Windows system utilities outside CWD/engine search paths.
#[doc(hidden)]
pub fn system_command(name: &str) -> Command {
    #[cfg(windows)]
    {
        let root = std::env::var_os("SystemRoot")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "C:/Windows".into());
        let relative = match name {
            "tar" => "System32/tar.exe",
            "taskkill" => "System32/taskkill.exe",
            "powershell" => "System32/WindowsPowerShell/v1.0/powershell.exe",
            _ => unreachable!("only fixed system utilities are resolved here"),
        };
        Command::new(root.join(relative))
    }
    #[cfg(not(windows))]
    {
        Command::new(name)
    }
}

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
                tracing::warn!("[Supervisor] Failed to create JobObject");
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
                tracing::warn!("[Supervisor] Failed to set JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE");
                CloseHandle(job);
                return None;
            }

            Some(JobHandle(job))
        })
        .map(|h| h.0)
}

static PID_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn configure_death_signal(_cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            _cmd.pre_exec(|| {
                // Ensure child process is placed in its own process group
                libc::setpgid(0, 0);
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
    let _lock = PID_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    let _lock = PID_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    let _lock = PID_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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

pub fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = crate::inference::process::system_command("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output();
    }
    #[cfg(unix)]
    {
        unsafe {
            let pid_t = pid as libc::pid_t;
            if pid_t > 1 {
                let _ = libc::kill(-pid_t, libc::SIGKILL);
                let _ = libc::kill(pid_t, libc::SIGKILL);
            }
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
        tracing::info!(
            "[Supervisor] Process {} (PID: {}) terminated and VRAM released.",
            self.label,
            pid
        );
    }
}

/// Wrapper around `reqwest::blocking::Client` that prevents Tokio worker panics on drop.
///
/// Dropping `reqwest::blocking::Client` on an active Tokio worker thread panics Tokio's
/// runtime shutdown guard (`Cannot drop a runtime in a context where blocking is not allowed`).
/// When dropped inside a Tokio runtime context, `SafeBlockingClient` offloads the inner client's
/// drop to an independent OS thread.
pub struct SafeBlockingClient(Option<reqwest::blocking::Client>);

impl SafeBlockingClient {
    pub fn new(client: reqwest::blocking::Client) -> Self {
        Self(Some(client))
    }
}

impl std::ops::Deref for SafeBlockingClient {
    type Target = reqwest::blocking::Client;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("SafeBlockingClient alive")
    }
}

impl Drop for SafeBlockingClient {
    fn drop(&mut self) {
        if let Some(client) = self.0.take() {
            if tokio::runtime::Handle::try_current().is_ok() {
                let _ = std::thread::Builder::new()
                    .name("safe-client-drop".into())
                    .spawn(move || drop(client));
            } else {
                drop(client);
            }
        }
    }
}

/// Lightweight TCP health/readiness probe for localhost services without spinning up HTTP clients.
///
/// Sends `GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n` and
/// returns `true` if the status code starts with `2` (e.g. 200 OK).
pub fn http_probe_success(port: u16, path: &str) -> bool {
    let Ok(addr) = format!("127.0.0.1:{port}").parse::<std::net::SocketAddr>() else {
        return false;
    };
    let Ok(mut stream) =
        std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300))
    else {
        return false;
    };
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(300)));
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));

    use std::io::{Read, Write};
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let _ = stream.flush();

    let mut buf = [0u8; 64];
    // TCP may deliver the status line in several reads.
    let Ok(()) = stream.read_exact(&mut buf[..12]) else {
        return false;
    };
    let response_header = String::from_utf8_lossy(&buf[..12]);
    if let Some(status_part) = response_header.split_whitespace().nth(1) {
        status_part.starts_with('2')
    } else {
        false
    }
}
