//! Offline acceptance for the actual bootstrap scripts; never changes the user's PATH.
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}
fn bash() -> Option<PathBuf> {
    if cfg!(windows) {
        let path = PathBuf::from(env::var_os("ProgramFiles").unwrap_or("C:/Program Files".into()))
            .join("Git/bin/bash.exe");
        path.exists().then_some(path)
    } else {
        Some("bash".into())
    }
}
fn shell_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) && text.as_bytes().get(1) == Some(&b':') {
        format!("/{}{}", text[..1].to_ascii_lowercase(), &text[2..])
    } else {
        text
    }
}
#[derive(Default, Clone, Copy)]
struct Case {
    windows: bool,
    custom: bool,
    bad_checksum: bool,
    conflict: bool,
    rollback: bool,
}

fn exercise(case: Case) {
    let temp = tempfile::Builder::new()
        .prefix("infer installation spaces ")
        .tempdir()
        .unwrap();
    let home = temp.path().join("owner's home");
    let shared = home.join("rune");
    let target = if case.custom {
        temp.path().join("custom installation")
    } else {
        shared.clone()
    };
    fs::create_dir_all(shared.join("tools/plugins")).unwrap();
    fs::write(shared.join("rune-kit.exe"), b"kit-owned executable").unwrap();
    fs::write(shared.join("tools/plugins/keep"), b"kit-owned plugin").unwrap();
    let legacy = home.join(".rune-infer");
    fs::create_dir(&legacy).unwrap();
    fs::write(legacy.join(".env"), b"KEEP_ORIGINAL=private-fixture\n").unwrap();
    let assets = temp.path().join("assets");
    let stage = temp.path().join("stage");
    fs::create_dir(&assets).unwrap();
    fs::create_dir(&stage).unwrap();
    let name = if case.windows {
        "rune-infer.exe"
    } else {
        "rune-infer"
    };
    let version = infer_xtask::version(&root()).unwrap().to_string();
    let archive_name = if case.windows {
        "rune-infer-windows-x64.zip"
    } else {
        "rune-infer-linux-x64.tar.gz"
    };
    let archive = assets.join(archive_name);
    if case.windows {
        let binary = root().join("target/debug/rune-infer.exe");
        assert!(
            binary.is_file(),
            "Run cargo build --locked before installer acceptance"
        );
        fs::copy(binary, stage.join(name)).unwrap();
        let mut pack = Command::new("powershell.exe");
        for (key, _) in env::vars_os() {
            if key.to_string_lossy().eq_ignore_ascii_case("PSModulePath") {
                pack.env_remove(key);
            }
        }
        let output = pack.args(["-NoProfile", "-Command", "$ErrorActionPreference='Stop'; Compress-Archive -LiteralPath $env:INFER_FIXTURE_BINARY -DestinationPath $env:INFER_FIXTURE_ARCHIVE"])
            .env("INFER_FIXTURE_BINARY", stage.join(name)).env("INFER_FIXTURE_ARCHIVE", &archive).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    } else {
        let failure = if case.rollback {
            "case \"$0\" in */rune/rune-infer) exit 1 ;; esac\n"
        } else {
            ""
        };
        fs::write(
            stage.join(name),
            format!("#!/bin/sh\n{failure}printf 'rune-infer {version}\\n'\n"),
        )
        .unwrap();
        let output = Command::new(bash().unwrap())
            .args([
                "--noprofile",
                "--norc",
                "-c",
                "tar -C \"$INFER_FIXTURE_STAGE\" -czf \"$INFER_FIXTURE_ARCHIVE\" rune-infer",
            ])
            .env("INFER_FIXTURE_STAGE", shell_path(&stage))
            .env("INFER_FIXTURE_ARCHIVE", shell_path(&archive))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let digest = if case.bad_checksum {
        "0".repeat(64)
    } else {
        Sha256::digest(fs::read(&archive).unwrap())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    };
    fs::write(
        assets.join(format!("{archive_name}.sha256")),
        format!("{digest}  {archive_name}\n"),
    )
    .unwrap();
    if case.conflict || case.rollback {
        fs::create_dir_all(&target).unwrap();
    }
    let previous = b"#!/bin/sh\nprintf 'rune-infer 0.7.0\\n'\n";
    if case.conflict {
        fs::write(target.join(name), b"unrelated destination data").unwrap();
    }
    if case.rollback {
        fs::write(target.join(name), previous).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(target.join(name), fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    let old_bin = if case.windows {
        temp.path().join("old local app data/Programs/rune-infer")
    } else {
        home.join(".local/bin")
    };
    fs::create_dir_all(&old_bin).unwrap();
    fs::write(old_bin.join(name), b"retained old executable").unwrap();
    let failing = case.bad_checksum || case.conflict || case.rollback;
    for _ in 0..if failing { 1 } else { 2 } {
        let mut command = if case.windows {
            let mut c = Command::new("powershell.exe");
            c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
                .arg(root().join("install.ps1"));
            c
        } else {
            let mut c = Command::new(bash().unwrap());
            c.args(["--noprofile", "--norc"])
                .arg(shell_path(&root().join("install.sh")));
            c
        };
        for (key, _) in env::vars_os() {
            let text = key.to_string_lossy().to_ascii_uppercase();
            if text.starts_with("RUNE_") || text == "PSMODULEPATH" {
                command.env_remove(key);
            }
        }
        command
            .current_dir(temp.path())
            .env("USERPROFILE", &home)
            .env(
                "HOME",
                if case.windows {
                    home.to_string_lossy().into_owned()
                } else {
                    shell_path(&home)
                },
            )
            .env("LOCALAPPDATA", temp.path().join("old local app data"))
            .env("RUNE_VERSION", format!("v{version}"))
            .env(
                "RUNE_PLATFORM",
                if case.windows {
                    "windows-x64"
                } else {
                    "linux-x64"
                },
            )
            .env(
                "RUNE_BASE_URL",
                if case.windows {
                    assets.to_string_lossy().into_owned()
                } else {
                    shell_path(&assets)
                },
            )
            .env("RUNE_NO_MODIFY_PATH", if case.windows { "1" } else { "0" })
            .env("SHELL", "/bin/bash");
        if case.custom {
            command.env(
                "RUNE_INSTALL_DIR",
                if case.windows {
                    target.to_string_lossy().into_owned()
                } else {
                    shell_path(&target)
                },
            );
        }
        let output = command.output().unwrap();
        assert_eq!(
            output.status.success(),
            !failing,
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read(shared.join("rune-kit.exe")).unwrap(),
            b"kit-owned executable"
        );
        assert_eq!(
            fs::read(shared.join("tools/plugins/keep")).unwrap(),
            b"kit-owned plugin"
        );
        assert_eq!(
            fs::read(legacy.join(".env")).unwrap(),
            b"KEEP_ORIGINAL=private-fixture\n"
        );
        assert_eq!(
            fs::read(old_bin.join(name)).unwrap(),
            b"retained old executable"
        );
    }
    if case.rollback {
        assert_eq!(fs::read(target.join(name)).unwrap(), previous);
    } else if case.conflict {
        assert_eq!(
            fs::read(target.join(name)).unwrap(),
            b"unrelated destination data"
        );
    } else if case.bad_checksum {
        assert!(!target.join(name).exists());
    } else {
        assert!(target.join(name).is_file());
        if !case.windows {
            assert_eq!(
                fs::read_to_string(home.join(".profile"))
                    .unwrap()
                    .matches("# Rune shared PATH")
                    .count(),
                1
            );
            let output = Command::new(bash().unwrap())
                .args([
                    "--noprofile",
                    "--norc",
                    "-c",
                    ". \"$HOME/.profile\"; . \"$HOME/.profile\"; printf '%s' \"$PATH\"",
                ])
                .env("HOME", shell_path(&home))
                .output()
                .unwrap();
            assert!(output.status.success());
            assert_eq!(
                String::from_utf8(output.stdout)
                    .unwrap()
                    .split(':')
                    .filter(|p| *p == shell_path(&target))
                    .count(),
                1
            );
        }
    }
}

#[test]
fn posix_bootstrap_preserves_data_and_rolls_back() {
    if bash().is_none() {
        eprintln!("POSIX acceptance unavailable: install Git Bash on Windows");
        return;
    }
    for case in [
        Case::default(),
        Case {
            custom: true,
            ..Case::default()
        },
        Case {
            bad_checksum: true,
            ..Case::default()
        },
        Case {
            conflict: true,
            ..Case::default()
        },
        Case {
            rollback: true,
            ..Case::default()
        },
    ] {
        exercise(case);
    }
}
#[test]
#[cfg(windows)]
fn windows_bootstrap_preserves_data() {
    for case in [
        Case::default(),
        Case {
            custom: true,
            ..Case::default()
        },
        Case {
            bad_checksum: true,
            ..Case::default()
        },
        Case {
            conflict: true,
            ..Case::default()
        },
    ] {
        exercise(Case {
            windows: true,
            ..case
        });
    }
}
