#![cfg(windows)]

use std::process::{Command, Stdio};

#[test]
fn embedded_guard_owns_only_its_child_and_does_not_use_cwd_pid_registry() {
    use rune_infer::inference::process::{ProcessGuard, cleanup_orphaned_pids};
    use std::os::windows::process::CommandExt;

    let root = std::env::temp_dir().join(format!("rune-embedded-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let cwd = std::env::current_dir().unwrap();
    let scratch = root.join("unrelated-cwd");
    std::fs::create_dir(&scratch).unwrap();
    std::env::set_current_dir(&scratch).unwrap();
    std::fs::create_dir(root.join("config")).unwrap();
    std::fs::create_dir(root.join("models")).unwrap();
    std::fs::write(root.join("models/model.gguf"), b"fixture").unwrap();
    std::fs::write(root.join("models/projector.gguf"), b"fixture").unwrap();
    std::fs::write(root.join("config/server.json"), r#"{"schema_version":3,"server":{"host":"127.0.0.1","port":8080,"max_loaded_models":1,"idle_unload_seconds":300,"api_key":null}}"#).unwrap();
    std::fs::write(root.join("config/catalog.json"), r#"{"schema_version":3,"models":[{"id":"fixture","name":"Fixture","modality":"VisionText","engine":{"supported":["llama-server"]},"local":{"path":"models/model.gguf"},"extra":{"mmproj_path":"models/projector.gguf"}}]}"#).unwrap();
    let config = rune_infer::embedding::load_local_model(&root, "fixture").unwrap();
    assert_eq!(
        std::path::Path::new(&config.models[0].model_path),
        root.join("models/model.gguf")
    );
    assert_eq!(
        std::path::Path::new(config.models[0].mmproj_path.as_ref().unwrap()),
        root.join("models/projector.gguf")
    );
    assert!(rune_infer::embedding::load_local_model(&root, "missing").is_err());
    std::fs::remove_file(root.join("models/projector.gguf")).unwrap();
    assert!(rune_infer::embedding::load_local_model(&root, "fixture").is_err());
    std::fs::write(root.join("models/projector.gguf"), b"fixture").unwrap();
    assert!(rune_infer::embedding::initialize(std::path::Path::new("relative")).is_err());
    rune_infer::embedding::initialize(&root).unwrap();
    assert!(rune_infer::embedding::initialize(&root).is_ok());
    assert!(rune_infer::embedding::initialize(&root.join("other")).is_err());

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(rune_infer::server::serve(config, listener, async {
            let _ = stopped.await;
        }));
        let response = reqwest::get(format!("http://{address}/health"))
            .await
            .unwrap();
        assert!(response.status().is_success());
        let models: serde_json::Value = reqwest::get(format!("http://{address}/v1/models"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(models["data"][0]["id"], "fixture");
        stop.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    });

    let spawn = || {
        Command::new(rune_infer::inference::process::system_command("powershell").get_program())
            .args(["-NoProfile", "-Command", "Start-Sleep -Seconds 60"])
            .creation_flags(0x08000000)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let mut independent = spawn();
    let owned = ProcessGuard::try_new(spawn(), "embedded-test").unwrap();
    let pid = owned.lock().unwrap().id();
    let handle = unsafe {
        windows_sys::Win32::System::Threading::OpenProcess(
            0x0010_0000, // SYNCHRONIZE
            0,
            pid,
        )
    };
    assert!(!handle.is_null());
    cleanup_orphaned_pids();
    assert!(independent.try_wait().unwrap().is_none());
    assert_eq!(std::env::current_dir().unwrap(), scratch);
    assert!(!scratch.join("logs").exists());
    drop(owned);
    assert_eq!(
        unsafe { windows_sys::Win32::System::Threading::WaitForSingleObject(handle, 1000) },
        0
    );
    unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) };
    assert!(independent.try_wait().unwrap().is_none());
    independent.kill().unwrap();
    independent.wait().unwrap();
    std::env::set_current_dir(cwd).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
